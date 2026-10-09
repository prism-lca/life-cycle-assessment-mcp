"""Brightway 2.5 calculation and inventory engine.

Accepts a product graph as a YAML string.
Returns a structured dict with LCI totals, LCIA scores, and scaling vector.
No external server required — all computation runs in-process via Brightway.

Run scripts/setup_databases.py once before using this module.
"""

import hashlib
import json
import logging
import math
import os
import pathlib
import re
import tarfile
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Must be set before bw2data is imported; the configured directory must exist.
_bw_dir = pathlib.Path(os.environ.get("BRIGHTWAY2_DIR", ROOT / "brightway_data"))
_bw_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("BRIGHTWAY2_DIR", str(_bw_dir))

import yaml
import numpy as np
import bw2data as bd
import bw2calc as bc

from .models import ContributionBatchResult, LcaCoreResult
from .contribution_graph import (
    ADJOINT_SCORE_ABS_TOLERANCE,
    ADJOINT_SCORE_REL_TOLERANCE,
    build_contribution_graph,
    factorize_adjoint,
)
from .mock_database import DATABASE_NAME as MOCK_BACKGROUND_DB
from .mock_database import ensure_mock_background_database

BRIGHTWAY_PROJECT = os.environ.get("BRIGHTWAY_PROJECT", "lca_server")
BIOSPHERE_DB = "biosphere3"
LEGACY_FOREGROUND_DB = "foreground"
FOREGROUND_DB_PREFIX = "foreground_request_"

NUMERIC_ABS_TOLERANCE = 1e-12
NUMERIC_REL_TOLERANCE = 1e-9

# URL of the pre-built Brightway database tarball on GitHub Releases.
# To update: build a new tarball (see docs/bafu_database_setup.md) and bump this URL.
TARBALL_URL = (
    "https://github.com/prism-lca/life-cycle-assessment-mcp"
    "/releases/download/lca-data-v2/brightway_bafu_v1.tar.gz"
)

_db_lock = threading.Lock()
_calculation_lock = threading.RLock()
_startup_databases_ready = False
_performance_logger = logging.getLogger("lca.performance")
_performance_logger.setLevel(logging.INFO)
if not _performance_logger.handlers:
    _performance_handler = logging.StreamHandler()
    _performance_handler.setFormatter(logging.Formatter("%(message)s"))
    _performance_logger.addHandler(_performance_handler)
_performance_logger.propagate = False
_logger = logging.getLogger(__name__)


def _elapsed_seconds(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def _add_phase(phases: dict, name: str, started: float) -> float:
    elapsed = _elapsed_seconds(started)
    phases[name] = round(phases.get(name, 0.0) + elapsed, 6)
    return elapsed


def _emit_performance_log(operation: str, started: float, phases: dict) -> None:
    record = {
        "event": "lca_engine_performance",
        "operation": operation,
        "total_seconds": _elapsed_seconds(started),
        "phases": phases,
    }
    _performance_logger.info(
        json.dumps(record, separators=(",", ":"), sort_keys=True)
    )


def _ensure_search_projection():
    """Build the disposable search database when it is missing or stale."""
    from .search import build_search_database, get_projection_status

    database_names = ["bafu", MOCK_BACKGROUND_DB]
    status = get_projection_status(project=BRIGHTWAY_PROJECT)
    indexed_databases = set(status.get("source_databases", []))
    if status.get("fresh") and set(database_names).issubset(indexed_databases):
        return status

    reason = status.get("reason", "Search projection is unavailable")
    print(f"[lca_engine] {reason} — rebuilding search projection...")
    build_search_database(databases=database_names, project=BRIGHTWAY_PROJECT)
    status = get_projection_status(project=BRIGHTWAY_PROJECT)
    indexed_databases = set(status.get("source_databases", []))
    if not status.get("fresh") or not set(database_names).issubset(indexed_databases):
        raise RuntimeError(
            "Search projection build completed but freshness validation failed: "
            f"{status.get('reason', 'unknown reason')}"
        )
    return status


def _ensure_databases():
    """Ensure Brightway data and its searchable projection are production-ready."""
    global _startup_databases_ready
    if _startup_databases_ready:
        return

    with _db_lock:
        if _startup_databases_ready:
            return

        bd.projects.set_current(BRIGHTWAY_PROJECT)
        if "bafu" not in bd.databases:
            bw_dir = pathlib.Path(os.environ["BRIGHTWAY2_DIR"])
            tarball = bw_dir / "brightway_bafu_v1.tar.gz"
            print(f"[lca_engine] bafu database not found — downloading from GitHub releases...")
            urllib.request.urlretrieve(TARBALL_URL, tarball)
            print(
                f"[lca_engine] Downloaded {tarball.stat().st_size // 1024 // 1024} MB "
                "— extracting..."
            )
            with tarfile.open(tarball, "r:gz") as tf:
                tf.extractall(bw_dir.parent)
            tarball.unlink()
            # bw2data loaded its metadata stores before the release archive
            # existed. Re-selecting the project reloads the extracted database
            # and method metadata in this same process, so first boot does not
            # rely on a container restart.
            bd.projects.set_current(BRIGHTWAY_PROJECT)
            print(f"[lca_engine] Database ready — {len(bd.Database('bafu'))} bafu processes.")

        # Versions before result schema 2 reused this persistent scratch
        # database. The historical release archive can contain its metadata,
        # so cleanup must run after a possible first-boot extraction. It holds
        # no authoritative user data and must not survive now that foreground
        # calculations are request-isolated.
        if LEGACY_FOREGROUND_DB in bd.databases:
            del bd.databases[LEGACY_FOREGROUND_DB]

        mock_status = ensure_mock_background_database(
            bd, biosphere_database=BIOSPHERE_DB
        )
        if mock_status["changed"]:
            print(
                "[lca_engine] Installed bundled mock background database — "
                f"{mock_status['activities']} processes."
            )

        _ensure_search_projection()
        from . import background_intensity

        cache_started = time.perf_counter()
        try:
            requests = _startup_background_intensity_requests()
            background_intensity.warm(bd, bc, requests)
            print(
                "[lca_engine] Background intensity cache ready — "
                f"{len(requests)} database/category combinations in "
                f"{_elapsed_seconds(cache_started):.3f}s."
            )
        except Exception as exc:
            background_intensity.disable(str(exc))
        _startup_databases_ready = True

# Index: (lowercase name, compartment) → activity key — built once on first lookup
_FLOW_INDEX: dict | None = None

# Common-name → ecoinvent/biosphere3 name aliases (case-insensitive, applied at lookup time)
_FLOW_ALIASES: dict[str, str] = {
    "co2": "carbon dioxide, fossil",
    "carbon dioxide": "carbon dioxide, fossil",
    "ch4": "methane, fossil",
    "methane": "methane, fossil",
    "n2o": "dinitrogen monoxide",
    "nitrous oxide": "dinitrogen monoxide",
    "nox": "nitrogen oxides",
    "sox": "sulfur dioxide",
}


def _ensure_project():
    _ensure_databases()
    bd.projects.set_current(BRIGHTWAY_PROJECT)


def _load_spec(product_graph_yaml: str) -> dict:
    text = product_graph_yaml.strip()
    if text.startswith("---"):
        _, fm, _ = text.split("---", 2)
        spec = yaml.safe_load(fm)
    else:
        spec = yaml.safe_load(text)
    _validate_spec(spec)
    return spec


def _require_finite(value, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number.")
    return result


def _validate_spec(spec: dict) -> None:
    """Validate identity and numeric invariants needed by every calculation."""
    if not isinstance(spec, dict):
        raise ValueError("product_graph must contain a YAML mapping.")

    processes = spec.get("processes")
    if not isinstance(processes, list) or not processes:
        raise ValueError("product_graph.processes must be a non-empty list.")

    functional_unit = spec.get("functional_unit")
    if not isinstance(functional_unit, dict):
        raise ValueError("product_graph.functional_unit must be a mapping.")
    _require_finite(functional_unit.get("amount"), "functional_unit.amount")
    if not functional_unit.get("unit"):
        raise ValueError("functional_unit.unit is required.")

    names: set[str] = set()
    output_flows: set[str] = set()
    for proc_index, proc in enumerate(processes):
        path = f"processes[{proc_index}]"
        if not isinstance(proc, dict):
            raise ValueError(f"{path} must be a mapping.")
        name = proc.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string.")
        if name in names:
            raise ValueError(f"Duplicate process name '{name}' is not allowed.")
        names.add(name)

        reference_output = proc.get("reference_output")
        if not isinstance(reference_output, dict):
            raise ValueError(f"{path}.reference_output must be a mapping.")
        flow = reference_output.get("flow")
        if not isinstance(flow, str) or not flow.strip():
            raise ValueError(f"{path}.reference_output.flow is required.")
        if flow in output_flows:
            raise ValueError(
                f"Product flow '{flow}' has more than one foreground provider."
            )
        output_flows.add(flow)
        output_amount = _require_finite(
            reference_output.get("amount"), f"{path}.reference_output.amount"
        )
        if abs(output_amount) <= NUMERIC_ABS_TOLERANCE:
            raise ValueError(f"{path}.reference_output.amount must be non-zero.")

        for collection in ("inputs", "emissions", "resources"):
            rows = proc.get(collection, [])
            if not isinstance(rows, list):
                raise ValueError(f"{path}.{collection} must be a list.")
            for row_index, row in enumerate(rows):
                row_path = f"{path}.{collection}[{row_index}]"
                if not isinstance(row, dict):
                    raise ValueError(f"{row_path} must be a mapping.")
                if not isinstance(row.get("flow"), str) or not row["flow"].strip():
                    raise ValueError(f"{row_path}.flow is required.")
                _require_finite(row.get("amount"), f"{row_path}.amount")

    reference_process = spec.get("reference_process")
    if reference_process not in names:
        raise ValueError(
            f"Reference process '{reference_process}' does not match a process name."
        )
    lcia = spec.get("lcia")
    if not isinstance(lcia, dict) or not lcia.get("method_name"):
        raise ValueError("lcia.method_name is required.")
    contribution_graph = lcia.get("contribution_graph")
    if (
        contribution_graph is not None
        and not isinstance(contribution_graph, dict)
    ):
        raise ValueError("lcia.contribution_graph must be a mapping.")
    categories = lcia.get("categories")
    if categories is None and isinstance(contribution_graph, dict):
        # Backwards compatibility for product graphs created before LCIA
        # category selection was separated from graph traversal settings.
        categories = contribution_graph.get("categories")
    if (
        not isinstance(categories, list)
        or not categories
        or any(
            not isinstance(category, str) or not category.strip()
            for category in categories
        )
    ):
        raise ValueError(
            "lcia.categories must be a non-empty list of impact category names."
        )
    if contribution_graph is not None:
        for key in ("cutoff", "biosphere_cutoff"):
            if key in contribution_graph:
                value = _require_finite(
                    contribution_graph[key], f"lcia.contribution_graph.{key}"
                )
                if not 0 < value < 1:
                    raise ValueError(
                        f"lcia.contribution_graph.{key} must be between 0 and 1."
                    )
        for key in ("max_depth", "max_calculations"):
            if key in contribution_graph:
                value = contribution_graph[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    raise ValueError(
                        f"lcia.contribution_graph.{key} must be a positive integer."
                    )
        if "include_flows" in contribution_graph and not isinstance(
            contribution_graph["include_flows"], bool
        ):
            raise ValueError(
                "lcia.contribution_graph.include_flows must be true or false."
            )


def _configured_impact_categories(spec: dict) -> list[str]:
    lcia = spec["lcia"]
    configured = lcia.get("categories")
    if configured is None:
        configured = lcia["contribution_graph"]["categories"]
    return configured


def _impact_category_config(spec: dict) -> dict:
    return {"categories": _configured_impact_categories(spec)}


def _contribution_graph_config(spec: dict) -> dict | None:
    configured = spec["lcia"].get("contribution_graph")
    if configured is None:
        return None
    return {
        "categories": _configured_impact_categories(spec),
        "cutoff": float(configured.get("cutoff", 0.005)),
        "biosphere_cutoff": float(
            configured.get("biosphere_cutoff", 0.0001)
        ),
        "max_depth": configured.get("max_depth"),
        "max_calculations": configured.get("max_calculations", 1000),
        "include_flows": configured.get("include_flows", True),
    }


def _resolve_contribution_graph_methods(
    method_tuples: list[tuple], config: dict | None
) -> set[tuple]:
    if config is None:
        return set()

    resolved: set[tuple] = set()
    labels = {
        method_tuple: " | ".join(method_tuple[1:])
        for method_tuple in method_tuples
    }
    for requested in config["categories"]:
        query = requested.casefold().strip()
        exact = [
            method_tuple
            for method_tuple, label in labels.items()
            if label.casefold() == query
        ]
        component = [
            method_tuple
            for method_tuple, label in labels.items()
            if label.split(" | ", 1)[0].casefold() == query
        ]
        substring = [
            method_tuple
            for method_tuple, label in labels.items()
            if query in label.casefold()
        ]
        matches = exact or component or substring
        if not matches:
            available = ", ".join(labels.values())
            raise ValueError(
                f"Contribution graph category '{requested}' was not found "
                f"in LCIA method '{method_tuples[0][0]}'. "
                f"Available categories: {available}."
            )
        if len(matches) > 1:
            options = ", ".join(labels[item] for item in matches)
            raise ValueError(
                f"Contribution graph category '{requested}' is ambiguous; "
                f"matches: {options}."
            )
        resolved.add(matches[0])
    return resolved


def _startup_background_intensity_requests() -> set[tuple[tuple[str, ...], tuple]]:
    from .background_intensity import database_names_from_spec

    requests: set[tuple[tuple[str, ...], tuple]] = set()
    candidates = []
    for directory in (ROOT / "product-graphs", ROOT / "mock_examples"):
        if directory.exists():
            candidates.extend(sorted(directory.glob("*.yaml")))
    for path in candidates:
        spec = _load_spec(path.read_text())
        database_names = database_names_from_spec(spec)
        if not database_names:
            continue
        method_name = spec["lcia"]["method_name"]
        available = sorted(
            [method for method in bd.methods if method[0] == method_name],
            key=lambda method: method[-1],
        )
        resolved = _resolve_contribution_graph_methods(
            available,
            _impact_category_config(spec),
        )
        requests.update((database_names, method) for method in resolved)
    return requests


def _cached_request_cumulative_intensities(
    *,
    lca,
    spec: dict,
    activities: dict,
    method: tuple,
    transpose_lu,
):
    from . import background_intensity

    try:
        cached = background_intensity.assemble_request_cumulative_intensities(
            bd=bd,
            bc=bc,
            lca=lca,
            activities=activities,
            database_names=background_intensity.database_names_from_spec(spec),
            method=method,
        )
    except Exception as exc:
        _logger.warning(
            "Background intensity cache fallback for %s: %s",
            " | ".join(method),
            exc,
        )
        background_intensity.disable(str(exc))
        if transpose_lu is None:
            transpose_lu = factorize_adjoint(lca)
        return None, transpose_lu

    demand = np.asarray(lca.demand_array).ravel()
    cached_score = float(cached @ demand)
    score_matches = math.isclose(
        cached_score,
        float(lca.score),
        rel_tol=ADJOINT_SCORE_REL_TOLERANCE,
        abs_tol=ADJOINT_SCORE_ABS_TOLERANCE,
    )
    if not score_matches:
        reason = (
            "cached cumulative intensities do not reconcile with the full "
            f"Brightway score for {' | '.join(method)}: "
            f"{cached_score} != {float(lca.score)}"
        )
        _logger.warning(reason)
        background_intensity.disable(reason)
        if transpose_lu is None:
            transpose_lu = factorize_adjoint(lca)
        return None, transpose_lu
    return cached, transpose_lu


def _stable_id(kind: str, *parts: object) -> str:
    canonical = "\x1f".join(str(part) for part in parts)
    slug_source = str(parts[0]) if parts else kind
    slug = re.sub(r"[^a-z0-9]+", "-", slug_source.lower()).strip("-")[:48]
    slug = slug or "item"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{slug}:{digest}"


def _background_link_rows(
    spec: dict, background_providers: dict
) -> tuple[list[dict], list]:
    """Describe every foreground->background link in stable spec order.

    ``background_providers`` is keyed by ``(process_index, input_index)``, which
    is exactly the identity a client needs to match a row back to its own YAML.
    """
    rows: list[dict] = []
    providers: list = []
    for (proc_index, input_index) in sorted(background_providers):
        provider = background_providers[(proc_index, input_index)]
        proc = spec["processes"][proc_index]
        exchange = proc["inputs"][input_index]
        rows.append(
            {
                "link_id": _stable_id(
                    "background-link", proc["name"], proc_index, input_index
                ),
                "process_index": proc_index,
                "input_index": input_index,
                "process_name": proc["name"],
                "flow": exchange["flow"],
                "database": provider.get("database", exchange.get("database", "")),
                "code": provider.get("code", ""),
                "location": provider.get("location"),
                "amount": float(exchange["amount"]),
                "unit": exchange.get("unit") or provider.get("unit") or "",
                "intensities": {},
            }
        )
        providers.append(provider)
    return rows, providers


def _attach_background_link_intensities(
    *,
    rows: list[dict],
    providers: list,
    database_names: tuple[str, ...],
    method: tuple,
    label: str,
) -> bool:
    """Add this category's cached provider intensity to every link row.

    Returns ``False`` when the cache cannot supply the category, in which case
    the caller omits the whole field rather than publishing a partial payload.
    """
    from . import background_intensity

    if not background_intensity.enabled():
        return False
    if not rows:
        return True
    try:
        cached = background_intensity.get_background_y(
            bd, bc, database_names, method
        )
        values = [float(cached[provider.id]) for provider in providers]
    except Exception as exc:
        _logger.warning(
            "Background link intensities unavailable for %s: %s", label, exc
        )
        return False
    for row, value in zip(rows, values, strict=True):
        row["intensities"][label] = value
    return True


def _result_id(spec: dict) -> str:
    """Return a deterministic identity for one normalized calculation input."""
    normalized = yaml.safe_dump(spec, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _process_ids(spec: dict) -> dict[str, str]:
    return {
        proc["name"]: _stable_id("process", proc["name"])
        for proc in spec["processes"]
    }


def _declared_flow_units(spec: dict) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    product_units = {
        item["name"]: item["unit"]
        for item in spec.get("products", [])
        if isinstance(item, dict) and item.get("name") and item.get("unit")
    }
    elementary_units: dict[tuple[str, str], str] = {}
    elementary = spec.get("elementary_flows", {})
    if isinstance(elementary, dict):
        for kind in ("emissions", "resources"):
            for item in elementary.get(kind, []):
                if isinstance(item, dict) and item.get("name") and item.get("unit"):
                    elementary_units[(kind, item["name"])] = item["unit"]
    return product_units, elementary_units


def _exchange_unit(
    exchange: dict,
    *,
    path: str,
    declared_units: dict[str, str],
) -> str:
    unit = exchange.get("unit") or declared_units.get(exchange["flow"])
    if not isinstance(unit, str) or not unit:
        raise ValueError(
            f"Unit for {path} flow '{exchange['flow']}' is not declared."
        )
    return unit


def _sub_compartment_priority(flow) -> int:
    """Score a flow's sub-compartment: higher = preferred winner in _FLOW_INDEX.

    "unspecified" is the preferred biosphere3 sub-compartment that LCIA methods
    consistently characterize.  "indoor" and stratospheric sub-compartments
    are rarely included in LCIA CF tables and must not win the index slot.
    """
    cats = [c.lower() for c in flow.get("categories", [])]
    sub = cats[-1] if cats else ""
    if "unspecified" in sub:
        return 2
    if "indoor" in sub or "stratosphere" in sub:
        return 0
    return 1


def _build_flow_index():
    global _FLOW_INDEX
    _FLOW_INDEX = {}
    _priority: dict = {}
    for flow in bd.Database(BIOSPHERE_DB):
        name_key = flow.get("name", "").lower()
        compartment = flow.get("compartment", "")
        if not compartment:
            cats = [c.lower() for c in flow.get("categories", [])]
            cat_str = " ".join(cats)
            if "air" in cat_str:
                compartment = "air"
            elif "water" in cat_str or "freshwater" in cat_str:
                compartment = "water"
            elif "soil" in cat_str or "ground" in cat_str:
                compartment = "ground"
            elif "resource" in cat_str:
                compartment = "resource"
            else:
                compartment = "other"
        key = (name_key, compartment)
        score = _sub_compartment_priority(flow)
        if score > _priority.get(key, -1):
            _FLOW_INDEX[key] = flow.key
            _priority[key] = score


def _find_biosphere_flow(name: str, compartment: str):
    """Look up a flow in biosphere3 by name and compartment."""
    global _FLOW_INDEX
    if _FLOW_INDEX is None:
        _build_flow_index()
    canonical = _FLOW_ALIASES.get(name.lower(), name.lower())
    key = _FLOW_INDEX.get((canonical, compartment.lower()))
    if key:
        try:
            return bd.get_activity(key)
        except Exception:
            pass
    return None


def _compartment_for_emission(em: dict) -> str:
    return em.get("compartment", "air")


def _compartment_for_resource(res: dict) -> str:
    comp = res.get("compartment", "water")
    return comp


def _build_foreground_db(spec: dict, database_name: str) -> tuple[dict, dict, dict]:
    """Build (or rebuild) the foreground database from a parsed spec.

    Returns foreground activities, product providers, and the exact background
    provider selected for each ``(process_index, input_index)`` pair.
    """
    fg = bd.Database(database_name)
    fg.register()

    activities: dict = {}
    product_to_activity: dict = {}
    background_providers: dict = {}

    for proc_index, proc in enumerate(spec["processes"]):
        ref = proc["reference_output"]
        act = fg.new_activity(
            code=proc["name"],
            name=proc["name"],
            unit=ref.get("unit", "kg"),
            location="GLO",
        )
        act.save()
        activities[proc["name"]] = act
        product_to_activity[ref["flow"]] = act

    for proc_index, proc in enumerate(spec["processes"]):
        act = activities[proc["name"]]
        ref = proc["reference_output"]

        act.new_exchange(
            input=act,
            amount=float(ref["amount"]),
            type="production",
        ).save()

        for input_index, inp in enumerate(proc.get("inputs", [])):
            db_name = inp.get("database")
            if db_name:
                bg_db = bd.Database(db_name)
                location = inp.get("location")
                code = inp.get("code")
                if code:
                    try:
                        provider = bd.get_activity((db_name, code))
                    except Exception as exc:
                        raise ValueError(
                            f"Background activity code '{code}' not found "
                            f"in database '{db_name}'."
                        ) from exc
                else:
                    matches = [
                        activity
                        for activity in bg_db
                        if activity["name"] == inp["flow"]
                        and (
                            location is None
                            or activity.get("location") == location
                        )
                    ]
                    if len(matches) > 1:
                        raise ValueError(
                            f"Background flow '{inp['flow']}' [{location}] is "
                            f"ambiguous in database '{db_name}'; specify code."
                        )
                    provider = matches[0] if matches else None
                if provider is None:
                    raise ValueError(
                        f"Background flow '{inp['flow']}' [{location}] not found "
                        f"in database '{db_name}'."
                    )
                background_providers[(proc_index, input_index)] = provider
            else:
                provider = product_to_activity.get(inp["flow"])
                if provider is None:
                    raise ValueError(
                        f"Input flow '{inp['flow']}' in process '{proc['name']}' "
                        f"has no provider in this product graph."
                    )
            act.new_exchange(
                input=provider,
                amount=float(inp["amount"]),
                type="technosphere",
            ).save()

        for em in proc.get("emissions", []):
            compartment = _compartment_for_emission(em)
            flow = _find_biosphere_flow(em["flow"], compartment)
            if flow is None:
                raise ValueError(
                    f"Emission flow '{em['flow']}' (compartment: {compartment}) "
                    f"not found in '{BIOSPHERE_DB}'. "
                    f"Check the flow name and compartment in your product graph."
                )
            act.new_exchange(
                input=flow,
                amount=float(em["amount"]),
                type="biosphere",
            ).save()

        for res in proc.get("resources", []):
            compartment = _compartment_for_resource(res)
            flow = _find_biosphere_flow(res["flow"], compartment)
            if flow is None:
                raise ValueError(
                    f"Resource flow '{res['flow']}' (compartment: {compartment}) "
                    f"not found in '{BIOSPHERE_DB}'. "
                    f"Check the flow name and compartment in your product graph."
                )
            act.new_exchange(
                input=flow,
                amount=float(res["amount"]),
                type="biosphere",
            ).save()

    return activities, product_to_activity, background_providers


@contextmanager
def _request_foreground(spec: dict, *, phases: dict | None = None):
    """Create isolated foreground state and always remove it after the request.

    Brightway project and metadata state is process-global. The lock covers the
    complete calculation, while the unique database name prevents a failed or
    interrupted request from reusing another request's foreground activities.
    """
    database_name = f"{FOREGROUND_DB_PREFIX}{uuid.uuid4().hex}"
    with _calculation_lock:
        try:
            started = time.perf_counter()
            foreground = _build_foreground_db(spec, database_name)
            if phases is not None:
                _add_phase(phases, "temporary_foreground_creation", started)
            yield foreground
        finally:
            cleanup_started = time.perf_counter()
            if database_name in bd.databases:
                del bd.databases[database_name]
            if phases is not None:
                _add_phase(
                    phases, "temporary_foreground_creation", cleanup_started
                )


def _contribution_category(
    lca,
    spec: dict,
    activities: dict,
    label: str,
    unit: str,
    background_metadata: dict[object, tuple[str, str, str]],
) -> dict:
    """Build exact, exclusive direct scores for all foreground/background activities."""
    column_totals = np.asarray(lca.characterized_inventory.sum(axis=0)).ravel()
    act_dict = lca.dicts.activity if hasattr(lca, "dicts") else lca.activity_dict
    process_ids = _process_ids(spec)
    total_score = float(lca.score)
    process_rows: list[dict] = []
    foreground_node_ids: set[object] = set()

    for proc in spec["processes"]:
        name = proc["name"]
        act = activities[name]
        node_id = act.id if hasattr(lca, "dicts") else act.key
        foreground_node_ids.add(node_id)
        column = act_dict.get(node_id)
        direct_score = (
            float(column_totals[column])
            if column is not None and column < len(column_totals)
            else 0.0
        )
        percentage = (
            None
            if abs(total_score) <= NUMERIC_ABS_TOLERANCE
            else direct_score / total_score * 100.0
        )
        process_rows.append(
            {
                "process_id": process_ids[name],
                "process_name": name,
                "direct_score": direct_score,
                "percentage": percentage,
                "scope": "foreground",
            }
        )

    if hasattr(lca, "dicts"):
        background_rows: list[dict] = []
        for column, direct_score_value in enumerate(column_totals):
            direct_score = float(direct_score_value)
            node_id = lca.dicts.activity.reversed[column]
            if node_id in foreground_node_ids or abs(direct_score) <= NUMERIC_ABS_TOLERANCE:
                continue
            metadata = background_metadata.get(node_id)
            if metadata is None:
                activity = bd.get_node(id=node_id)
                metadata = (
                    activity.get("database", ""),
                    activity.get("code", ""),
                    activity.get("name", str(activity.key)),
                )
                background_metadata[node_id] = metadata
            database, code, process_name = metadata
            background_rows.append(
                {
                    "process_id": _stable_id(
                        "background-process", database, code
                    ),
                    "process_name": process_name,
                    "direct_score": direct_score,
                    "percentage": (
                        None
                        if abs(total_score) <= NUMERIC_ABS_TOLERANCE
                        else direct_score / total_score * 100.0
                    ),
                    "scope": "background",
                }
            )
        process_rows.extend(
            sorted(
                background_rows,
                key=lambda row: (
                    -abs(row["direct_score"]),
                    row["process_name"],
                    row["process_id"],
                ),
            )
        )

    direct_total = sum(row["direct_score"] for row in process_rows)
    residual_score = total_score - direct_total
    if abs(residual_score) <= (
        max(abs(total_score), 1.0) * NUMERIC_REL_TOLERANCE
        + NUMERIC_ABS_TOLERANCE
    ):
        residual_score = 0.0
    if abs(direct_total + residual_score - total_score) > (
        max(abs(total_score), 1.0) * NUMERIC_REL_TOLERANCE
        + NUMERIC_ABS_TOLERANCE
    ):
        raise RuntimeError(f"Process contributions do not reconcile for '{label}'.")

    return {
        "id": _stable_id("impact", spec["lcia"]["method_name"], label),
        "label": label,
        "unit": unit,
        "total_score": total_score,
        "processes": process_rows,
        "residual_score": residual_score,
    }


def _build_sankey(
    spec: dict,
    scaling_vector: dict[str, float],
    background_providers: dict,
) -> dict:
    """Build a renderer-neutral graph from YAML using the solved scaling state."""
    process_ids = _process_ids(spec)
    product_units, elementary_units = _declared_flow_units(spec)
    product_providers = {
        proc["reference_output"]["flow"]: proc["name"]
        for proc in spec["processes"]
    }
    nodes: list[dict] = []
    links: list[dict] = []
    node_ids: set[str] = set()
    link_occurrences: dict[tuple[str, str, str, str], int] = {}

    def add_node(node: dict) -> None:
        if node["id"] not in node_ids:
            node_ids.add(node["id"])
            nodes.append(node)

    def add_link(
        *,
        source: str,
        target: str,
        kind: str,
        flow_name: str,
        amount: float,
        unit: str,
    ) -> None:
        if abs(amount) <= NUMERIC_ABS_TOLERANCE:
            return
        identity = (source, target, kind, flow_name)
        occurrence = link_occurrences.get(identity, 0)
        link_occurrences[identity] = occurrence + 1
        links.append(
            {
                "id": _stable_id("link", kind, source, target, flow_name, occurrence),
                "source": source,
                "target": target,
                "kind": kind,
                "flow_name": flow_name,
                "amount": amount,
                "unit": unit,
            }
        )

    for proc in spec["processes"]:
        name = proc["name"]
        add_node(
            {
                "id": process_ids[name],
                "label": name,
                "kind": "process",
                "process_name": name,
                "scope": "foreground",
            }
        )

    for proc_index, proc in enumerate(spec["processes"]):
        name = proc["name"]
        target = process_ids[name]
        scale = scaling_vector.get(name, 0.0)

        for input_index, inp in enumerate(proc.get("inputs", [])):
            db_name = inp.get("database")
            if db_name:
                provider = background_providers[(proc_index, input_index)]
                provider_key = provider.key
                provider_name = provider.get("name", inp["flow"])
                source = _stable_id(
                    "background-process", provider_key[0], provider_key[1]
                )
                add_node(
                    {
                        "id": source,
                        "label": provider_name,
                        "kind": "process",
                        "process_name": provider_name,
                        "scope": "background",
                    }
                )
            else:
                provider_name = product_providers.get(inp["flow"])
                if provider_name is None:
                    raise ValueError(
                        f"Input flow '{inp['flow']}' in process '{name}' has no provider."
                    )
                source = process_ids[provider_name]
            unit = _exchange_unit(
                inp,
                path=f"processes[{proc_index}].inputs[{input_index}]",
                declared_units=product_units,
            )
            add_link(
                source=source,
                target=target,
                kind="technosphere",
                flow_name=inp["flow"],
                amount=_require_finite(inp["amount"], "input amount") * scale,
                unit=unit,
            )

        for collection, node_kind, link_kind in (
            ("resources", "resource", "extraction"),
            ("emissions", "emission", "emission"),
        ):
            declared = {
                flow: unit
                for (kind, flow), unit in elementary_units.items()
                if kind == collection
            }
            for row_index, row in enumerate(proc.get(collection, [])):
                unit = _exchange_unit(
                    row,
                    path=f"processes[{proc_index}].{collection}[{row_index}]",
                    declared_units=declared,
                )
                compartment = row.get(
                    "compartment", "water" if collection == "resources" else "air"
                )
                flow_node_id = _stable_id(
                    node_kind, row["flow"], compartment, unit
                )
                add_node(
                    {
                        "id": flow_node_id,
                        "label": row["flow"],
                        "kind": node_kind,
                        "flow_name": row["flow"],
                    }
                )
                amount = _require_finite(row["amount"], f"{collection} amount") * scale
                add_link(
                    source=flow_node_id if link_kind == "extraction" else target,
                    target=target if link_kind == "extraction" else flow_node_id,
                    kind=link_kind,
                    flow_name=row["flow"],
                    amount=amount,
                    unit=unit,
                )

    reference_name = spec["reference_process"]
    reference_proc = next(
        proc for proc in spec["processes"] if proc["name"] == reference_name
    )
    final_flow = reference_proc["reference_output"]["flow"]
    final_node_id = _stable_id("final-product", final_flow, spec.get("name", ""))
    add_node(
        {
            "id": final_node_id,
            "label": spec["functional_unit"].get("description", final_flow),
            "kind": "final_product",
            "flow_name": final_flow,
        }
    )
    add_link(
        source=process_ids[reference_name],
        target=final_node_id,
        kind="final_product",
        flow_name=final_flow,
        amount=_require_finite(
            spec["functional_unit"]["amount"], "functional_unit.amount"
        ),
        unit=spec["functional_unit"]["unit"],
    )

    available_units = sorted({link["unit"] for link in links})
    return {"nodes": nodes, "links": links, "available_units": available_units}


def _ensure_finite_result(value, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number.")
    if isinstance(value, dict):
        for key, item in value.items():
            _ensure_finite_result(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite_result(item, f"{path}[{index}]")


def _run_analysis(
    product_graph_yaml: str,
    *,
    include_contribution_graphs: bool,
    performance_phases: dict | None = None,
) -> LcaCoreResult:
    phases = performance_phases
    started = time.perf_counter()
    spec = _load_spec(product_graph_yaml)
    if phases is not None:
        _add_phase(phases, "yaml_parsing_and_validation", started)

    with _calculation_lock:
        started = time.perf_counter()
        _ensure_project()
        if phases is not None:
            _add_phase(phases, "brightway_project_readiness", started)
        with _request_foreground(spec, phases=phases) as (
            activities,
            _,
            background_providers,
        ):
            ref_proc_name = spec["reference_process"]
            ref_act = activities[ref_proc_name]
            fu_amount = float(spec["functional_unit"]["amount"])

            method_name = spec["lcia"]["method_name"]
            method_tuples = sorted(
                [m for m in bd.methods if len(m) >= 2 and m[0] == method_name],
                key=lambda m: m[-1],
            )
            if not method_tuples:
                raise ValueError(
                    f"LCIA method '{method_name}' not found. "
                    f"Run scripts/setup_databases.py to load methods."
                )
            contribution_config = _contribution_graph_config(spec)
            configured_methods = _resolve_contribution_graph_methods(
                method_tuples, _impact_category_config(spec)
            )
            calculation_methods = [
                method_tuple
                for method_tuple in method_tuples
                if method_tuple in configured_methods
            ]
            contribution_methods = (
                configured_methods
                if include_contribution_graphs and contribution_config is not None
                else set()
            )

            # Compute inventory and the scaling solution exactly once.
            started = time.perf_counter()
            lca = bc.LCA(
                demand={ref_act: fu_amount},
                method=calculation_methods[0],
            )
            if phases is not None:
                _add_phase(phases, "lca_construction", started)
            started = time.perf_counter()
            # Each request solves one forward demand exactly once. Retaining
            # that LU factorization adds substantial cost and is never reused;
            # contribution traversal has its own A.T factorization below.
            lca.lci()
            if phases is not None:
                _add_phase(phases, "lci_factorization", started)
            from . import background_intensity

            transpose_lu = None
            if contribution_methods and not background_intensity.enabled():
                started = time.perf_counter()
                transpose_lu = factorize_adjoint(lca)
                if phases is not None:
                    _add_phase(phases, "adjoint_transpose_factorization", started)

            base_result_started = time.perf_counter()
            scaling_vector: dict[str, float] = {}
            act_dict = (
                lca.dicts.activity if hasattr(lca, "dicts") else lca.activity_dict
            )
            for act_name, act in activities.items():
                node_id = act.id if hasattr(lca, "dicts") else act.key
                idx = act_dict.get(node_id)
                if idx is not None:
                    scaling_vector[act_name] = float(lca.supply_array[idx])

            # LCI totals retain their existing contract and meaning.
            lci: dict = {}
            total_inv = np.asarray(lca.inventory.sum(axis=1)).ravel()
            bio_dict = (
                lca.dicts.biosphere
                if hasattr(lca, "dicts")
                else lca.biosphere_dict
            )
            bio_db = bd.Database(BIOSPHERE_DB)
            for flow in bio_db:
                node_id = flow.id if hasattr(lca, "dicts") else flow.key
                idx = bio_dict.get(node_id)
                if idx is not None and idx < len(total_inv):
                    amount = float(total_inv[idx])
                    if abs(amount) > 1e-15:
                        lci[flow["name"]] = {
                            "amount": amount,
                            "unit": flow.get("unit", "kg"),
                            "type": flow.get("type", "emission"),
                        }

            lcia_results: dict = {}
            contribution_categories: list[dict] = []
            contribution_graphs: list[dict] = []
            background_metadata: dict[object, tuple[str, str, str]] = {}
            foreground_ids = _process_ids(spec)
            foreground_metadata = {
                activity.id: {
                    "activity_id": foreground_ids[name],
                    "process_name": name,
                    "database": "foreground",
                    "code": name,
                    "location": activity.get("location"),
                }
                for name, activity in activities.items()
            }
            background_link_rows, background_link_providers = (
                _background_link_rows(spec, background_providers)
            )
            background_database_names = (
                background_intensity.database_names_from_spec(spec)
            )
            background_links_complete = True
            background_link_timings: list[dict] = []
            if phases is not None:
                _add_phase(
                    phases, "inventory_base_result_construction", base_result_started
                )
            lcia_timings = []
            traversal_timings = []
            background_cache_timings = []
            for method_index, method_tuple in enumerate(calculation_methods):
                category_started = time.perf_counter()
                if method_index:
                    lca.switch_method(method_tuple)
                lca.lcia()
                label = " | ".join(method_tuple[1:])
                unit = bd.methods[method_tuple].get("unit", "")
                lcia_results[label] = {"score": float(lca.score), "unit": unit}
                link_started = time.perf_counter()
                background_links_complete = (
                    _attach_background_link_intensities(
                        rows=background_link_rows,
                        providers=background_link_providers,
                        database_names=background_database_names,
                        method=method_tuple,
                        label=label,
                    )
                    and background_links_complete
                )
                background_link_timings.append(
                    {"category": label, "seconds": _elapsed_seconds(link_started)}
                )
                category = _contribution_category(
                    lca,
                    spec,
                    activities,
                    label=label,
                    unit=unit,
                    background_metadata=background_metadata,
                )
                lcia_timings.append(
                    {"category": label, "seconds": _elapsed_seconds(category_started)}
                )
                if method_tuple in contribution_methods:
                    cumulative_intensities = None
                    if background_intensity.enabled():
                        cache_started = time.perf_counter()
                        cumulative_intensities, transpose_lu = (
                            _cached_request_cumulative_intensities(
                                lca=lca,
                                spec=spec,
                                activities=activities,
                                method=method_tuple,
                                transpose_lu=transpose_lu,
                            )
                        )
                        background_cache_timings.append(
                            {
                                "category": label,
                                "seconds": _elapsed_seconds(cache_started),
                            }
                        )
                    traversal_started = time.perf_counter()
                    graph = build_contribution_graph(
                        lca=lca,
                        spec=spec,
                        label=label,
                        unit=unit,
                        config=contribution_config,
                        foreground_metadata=foreground_metadata,
                        transpose_lu=transpose_lu,
                        cumulative_intensities=cumulative_intensities,
                    )
                    contribution_graphs.append(graph)
                    traversal_timings.append(
                        {"category": label, "seconds": _elapsed_seconds(traversal_started)}
                    )
                contribution_categories.append(category)
            if phases is not None:
                phases["lcia_calculation_and_direct_contributions"] = {
                    "total_seconds": round(
                        sum(item["seconds"] for item in lcia_timings), 6
                    ),
                    "categories": lcia_timings,
                }
                if traversal_timings:
                    phases["contribution_traversal_per_category"] = {
                        "total_seconds": round(
                            sum(item["seconds"] for item in traversal_timings), 6
                        ),
                        "categories": traversal_timings,
                    }
                if background_cache_timings:
                    phases["background_intensity_cache_per_category"] = {
                        "total_seconds": round(
                            sum(
                                item["seconds"]
                                for item in background_cache_timings
                            ),
                            6,
                        ),
                        "categories": background_cache_timings,
                    }

            if phases is not None and background_link_timings:
                phases["background_link_intensities_per_category"] = {
                    "total_seconds": round(
                        sum(item["seconds"] for item in background_link_timings), 6
                    ),
                    "categories": background_link_timings,
                }

            base_result_started = time.perf_counter()
            fu_spec = spec["functional_unit"]
            result: LcaCoreResult = {
                "result_id": _result_id(spec),
                "name": spec.get("name", ""),
                "method": method_name,
                "functional_unit": (
                    f"{fu_amount} {fu_spec['unit']} — {fu_spec['description']}"
                ),
                "lci": lci,
                "lcia": lcia_results,
                "scaling_vector": scaling_vector,
                "result_schema_version": 3,
                "process_contributions": {
                    "categories": contribution_categories
                },
                "contribution_graphs": contribution_graphs,
                "sankey": _build_sankey(
                    spec, scaling_vector, background_providers
                ),
            }
            if background_links_complete:
                result["background_link_intensities"] = background_link_rows
            if phases is not None:
                _add_phase(
                    phases, "inventory_base_result_construction", base_result_started
                )
            started = time.perf_counter()
            _ensure_finite_result(result)
            if phases is not None:
                _add_phase(phases, "result_validation", started)
            return result


def run_analysis(product_graph_yaml: str) -> LcaCoreResult:
    """Run the backwards-compatible full calculation."""
    return _run_analysis(
        product_graph_yaml,
        include_contribution_graphs=True,
    )


def run_base_analysis(product_graph_yaml: str) -> LcaCoreResult:
    """Run totals and direct contributions without cumulative impact graphs."""
    request_started = time.perf_counter()
    phases: dict = {}
    try:
        return _run_analysis(
            product_graph_yaml,
            include_contribution_graphs=False,
            performance_phases=phases,
        )
    finally:
        _emit_performance_log("base", request_started, phases)


def _run_contribution_analysis(
    product_graph_yaml: str,
    categories: list[str],
    *,
    result_id: str | None = None,
    performance_phases: dict | None = None,
) -> ContributionBatchResult:
    """Build cumulative contribution graphs for a requested category batch."""
    phases = performance_phases
    started = time.perf_counter()
    spec = _load_spec(product_graph_yaml)
    if phases is not None:
        _add_phase(phases, "yaml_parsing_and_validation", started)
    if (
        not isinstance(categories, list)
        or not categories
        or any(
            not isinstance(category, str) or not category.strip()
            for category in categories
        )
    ):
        raise ValueError("categories must be a non-empty list of category names.")

    actual_result_id = _result_id(spec)
    if result_id is not None and result_id != actual_result_id:
        raise ValueError(
            "result_id does not match product_graph; recalculate the base result."
        )

    configured = _contribution_graph_config(spec)
    config = {
        "categories": categories,
        "cutoff": configured["cutoff"] if configured else 0.005,
        "biosphere_cutoff": (
            configured["biosphere_cutoff"] if configured else 0.0001
        ),
        "max_depth": configured["max_depth"] if configured else None,
        "max_calculations": (
            configured["max_calculations"] if configured else 1000
        ),
        "include_flows": configured["include_flows"] if configured else True,
    }

    with _calculation_lock:
        started = time.perf_counter()
        _ensure_project()
        if phases is not None:
            _add_phase(phases, "brightway_project_readiness", started)
        with _request_foreground(spec, phases=phases) as (activities, _, _):
            method_name = spec["lcia"]["method_name"]
            method_tuples = sorted(
                [method for method in bd.methods if method[0] == method_name],
                key=lambda method: method[-1],
            )
            if not method_tuples:
                raise ValueError(
                    f"LCIA method '{method_name}' not found. "
                    "Run scripts/setup_databases.py to load methods."
                )
            configured_methods = _resolve_contribution_graph_methods(
                method_tuples, _impact_category_config(spec)
            )
            requested_methods = _resolve_contribution_graph_methods(
                method_tuples, config
            )
            if not requested_methods.issubset(configured_methods):
                requested_labels = ", ".join(
                    " | ".join(method_tuple[1:])
                    for method_tuple in method_tuples
                    if method_tuple in requested_methods - configured_methods
                )
                raise ValueError(
                    "Requested contribution categories are not listed in "
                    "lcia.categories: "
                    f"{requested_labels}."
                )
            first_requested_method = next(
                method for method in method_tuples if method in requested_methods
            )

            reference = activities[spec["reference_process"]]
            amount = float(spec["functional_unit"]["amount"])
            started = time.perf_counter()
            lca = bc.LCA(
                demand={reference: amount},
                method=first_requested_method,
            )
            if phases is not None:
                _add_phase(phases, "lca_construction", started)
            started = time.perf_counter()
            # The forward system is solved once; only the separate transpose
            # factorization is reused by contribution traversal.
            lca.lci()
            if phases is not None:
                _add_phase(phases, "lci_factorization", started)
            from . import background_intensity

            started = time.perf_counter()
            transpose_lu = None
            if not background_intensity.enabled():
                transpose_lu = factorize_adjoint(lca)
                if phases is not None:
                    _add_phase(phases, "adjoint_transpose_factorization", started)
            process_ids = _process_ids(spec)
            foreground_metadata = {
                activity.id: {
                    "activity_id": process_ids[name],
                    "process_name": name,
                    "database": "foreground",
                    "code": name,
                    "location": activity.get("location"),
                }
                for name, activity in activities.items()
            }

            graphs = []
            lcia_timings = []
            traversal_timings = []
            background_cache_timings = []
            for method_tuple in method_tuples:
                if method_tuple not in requested_methods:
                    continue
                category_started = time.perf_counter()
                lca.switch_method(method_tuple)
                lca.lcia()
                label = " | ".join(method_tuple[1:])
                unit = bd.methods[method_tuple].get("unit", "")
                lcia_timings.append(
                    {"category": label, "seconds": _elapsed_seconds(category_started)}
                )
                cumulative_intensities = None
                if background_intensity.enabled():
                    cache_started = time.perf_counter()
                    cumulative_intensities, transpose_lu = (
                        _cached_request_cumulative_intensities(
                            lca=lca,
                            spec=spec,
                            activities=activities,
                            method=method_tuple,
                            transpose_lu=transpose_lu,
                        )
                    )
                    background_cache_timings.append(
                        {
                            "category": label,
                            "seconds": _elapsed_seconds(cache_started),
                        }
                    )
                traversal_started = time.perf_counter()
                graphs.append(
                    build_contribution_graph(
                        lca=lca,
                        spec=spec,
                        label=label,
                        unit=unit,
                        config=config,
                        foreground_metadata=foreground_metadata,
                        transpose_lu=transpose_lu,
                        cumulative_intensities=cumulative_intensities,
                    )
                )
                traversal_timings.append(
                    {"category": label, "seconds": _elapsed_seconds(traversal_started)}
                )
            if phases is not None:
                phases["lcia_calculation_and_direct_contributions"] = {
                    "total_seconds": round(
                        sum(item["seconds"] for item in lcia_timings), 6
                    ),
                    "categories": lcia_timings,
                }
                phases["contribution_traversal_per_category"] = {
                    "total_seconds": round(
                        sum(item["seconds"] for item in traversal_timings), 6
                    ),
                    "categories": traversal_timings,
                }
                if background_cache_timings:
                    phases["background_intensity_cache_per_category"] = {
                        "total_seconds": round(
                            sum(
                                item["seconds"]
                                for item in background_cache_timings
                            ),
                            6,
                        ),
                        "categories": background_cache_timings,
                    }

            result: ContributionBatchResult = {
                "result_id": actual_result_id,
                "method": method_name,
                "contribution_graphs": graphs,
            }
            started = time.perf_counter()
            _ensure_finite_result(result)
            if phases is not None:
                _add_phase(phases, "result_validation", started)
            return result


def run_contribution_analysis(
    product_graph_yaml: str,
    categories: list[str],
    *,
    result_id: str | None = None,
) -> ContributionBatchResult:
    """Build contribution graphs and emit one request timing record."""
    request_started = time.perf_counter()
    phases: dict = {}
    try:
        return _run_contribution_analysis(
            product_graph_yaml,
            categories,
            result_id=result_id,
            performance_phases=phases,
        )
    finally:
        _emit_performance_log("contribution", request_started, phases)


def get_contributions(product_graph_yaml: str, method_name: str, top_n: int = 10) -> dict:
    """
    Run contribution analysis for a single named impact category.

    Runs a fresh LCA and returns the top processes driving impact for the
    specified category, ranked by absolute score.

    method_name must substring-match a key in the LCIA results
    (e.g. "climate change", "acidification", "water use").
    Returns {method, score, unit, processes: [{activity, location, score, fraction}]}.
    """
    spec = _load_spec(product_graph_yaml)
    with _calculation_lock:
        _ensure_project()
        with _request_foreground(spec) as (activities, _, _):
            import bw2analyzer as ba

            method_name_full = spec["lcia"]["method_name"]
            method_tuples = sorted(
                [m for m in bd.methods if len(m) >= 2 and m[0] == method_name_full],
                key=lambda m: m[-1],
            )
            if not method_tuples:
                raise ValueError(f"LCIA method '{method_name_full}' not found.")

            target = next(
                (
                    m
                    for m in method_tuples
                    if method_name.lower() in " | ".join(m[1:]).lower()
                ),
                None,
            )
            if target is None:
                available = [" | ".join(m[1:]) for m in method_tuples]
                raise ValueError(
                    f"Method '{method_name}' not found. Available: {available}"
                )

            ref_act = activities[spec["reference_process"]]
            fu_amount = float(spec["functional_unit"]["amount"])
            lca = bc.LCA({ref_act: fu_amount}, target)
            lca.lci(factorize=True)
            lca.lcia()

            total = lca.score
            ca = ba.ContributionAnalysis()
            processes = []
            for score, _, act in ca.annotated_top_processes(lca, limit=top_n):
                try:
                    name = act["name"]
                    location = act.get("location", "")
                except Exception:
                    name = str(act)
                    location = ""
                processes.append({
                    "activity": name,
                    "location": location,
                    "score": float(score),
                    "fraction": float(score / total) if total else 0.0,
                })

            return {
                "method": " | ".join(target[1:]),
                "score": float(total),
                "unit": bd.methods[target].get("unit", ""),
                "processes": processes,
            }


def query_database(sql: str, limit: int = 100) -> dict:
    """Run read-only SQL against the searchable projection, never databases.db."""
    _ensure_project()
    from .search import query_search_database

    return query_search_database(sql, limit=limit, project=BRIGHTWAY_PROJECT)


def get_database_schema() -> dict:
    """Return the schema and freshness of the searchable projection."""
    _ensure_project()
    from .search import get_search_schema

    return get_search_schema(project=BRIGHTWAY_PROJECT)


def list_databases() -> list:
    """Return all databases installed in the current Brightway project."""
    _ensure_project()
    results = []
    for name in sorted(bd.databases):
        meta = bd.databases[name]
        results.append({
            "name": name,
            "size": meta.get("number", len(bd.Database(name))),
            "backend": meta.get("backend", "sqlite"),
            "depends": meta.get("depends", []),
        })
    return results


def search_database(query: str, database: str = "biosphere3", limit: int = 25) -> list:
    """Search projection activities without querying Brightway's SQLite file."""
    _ensure_project()
    from .search import search_activities

    results = search_activities(
        query,
        database=database,
        limit=limit,
        project=BRIGHTWAY_PROJECT,
    )
    return [
        {
            "name": result["name"],
            "reference_product": result.get("reference_product"),
            "location": result.get("location"),
            "categories": (result.get("categories_text") or "").split("::")
            if result.get("categories_text")
            else [],
            "unit": result.get("unit") or "",
            "type": result.get("type") or "",
            "key": [result["database"], result["code"]],
        }
        for result in results
    ]


def top_emissions(product_graph_yaml: str, method_name: str, top_n: int = 15) -> list:
    """
    Return the top biosphere flows (emissions/resources) driving impact for one
    LCIA category, ranked by absolute direct impact score.

    method_name must substring-match a key in the LCIA results
    e.g. "climate change", "acidification", "water use".

    Returns list of {flow, categories, unit, score, fraction}.
    """
    spec = _load_spec(product_graph_yaml)
    with _calculation_lock:
        _ensure_project()
        with _request_foreground(spec) as (activities, _, _):
            import bw2analyzer as ba

            method_name_full = spec["lcia"]["method_name"]
            method_tuples = sorted(
                [m for m in bd.methods if len(m) >= 2 and m[0] == method_name_full],
                key=lambda m: m[-1],
            )
            target = next(
                (
                    m
                    for m in method_tuples
                    if method_name.lower() in " | ".join(m[1:]).lower()
                ),
                None,
            )
            if target is None:
                available = [" | ".join(m[1:]) for m in method_tuples]
                raise ValueError(
                    f"Method '{method_name}' not found. Available: {available}"
                )

            ref_act = activities[spec["reference_process"]]
            fu_amount = float(spec["functional_unit"]["amount"])
            lca = bc.LCA({ref_act: fu_amount}, target)
            lca.lci(factorize=True)
            lca.lcia()

            total = lca.score
            ca = ba.ContributionAnalysis()
            rows = []
            for score, _, flow in ca.annotated_top_emissions(lca, limit=top_n):
                try:
                    name = flow["name"]
                    categories = list(flow.get("categories", []))
                    unit = flow.get("unit", "")
                except Exception:
                    name = str(flow)
                    categories = []
                    unit = ""
                rows.append({
                    "flow": name,
                    "categories": categories,
                    "unit": unit,
                    "score": float(score),
                    "fraction": float(score / total) if total else 0.0,
                })
            return rows


def compare_activities(
    activity_names: list,
    method_name: str,
    database: str = "bafu",
    location: str | None = None,
    amount: float = 1.0,
    method_family: str = "EF v3.1",
) -> list:
    """
    Compare multiple background database activities on a single LCIA method.

    activity_names: list of process names to compare (must exist in `database`)
    method_name: substring match against category e.g. "climate change", "acidification"
    database: Brightway database to search (default "bafu")
    location: optional location filter e.g. "RER", "GLO"
    amount: functional unit amount (default 1.0 kg)
    method_family: top-level method family to search within (default "EF v3.1")

    Returns list of {activity, location, score, unit, fraction} sorted by score descending.
    """
    _ensure_project()
    bg_db = bd.Database(database)

    family_methods = [m for m in bd.methods if len(m) >= 2 and m[0] == method_family]
    if not family_methods:
        raise ValueError(f"Method family '{method_family}' not found.")
    target = next(
        (m for m in sorted(family_methods, key=lambda m: m[-1])
         if method_name.lower() in " | ".join(m[1:]).lower()),
        None,
    )
    if target is None:
        categories = [" | ".join(m[1:]) for m in family_methods]
        raise ValueError(
            f"No category matching '{method_name}' in '{method_family}'. "
            f"Available: {categories[:10]}"
        )

    rows = []
    for name in activity_names:
        act = next(
            (a for a in bg_db
             if a["name"] == name
             and (location is None or a.get("location") == location)),
            None,
        )
        if act is None:
            rows.append({"activity": name, "location": location or "?", "score": None,
                         "unit": "", "fraction": None, "error": "not found"})
            continue
        lca = bc.LCA({act: amount}, target)
        lca.lci()
        lca.lcia()
        rows.append({
            "activity": act["name"],
            "location": act.get("location", ""),
            "score": float(lca.score),
            "unit": bd.methods[target].get("unit", ""),
            "fraction": None,
        })

    # Compute fractions relative to max score
    valid = [r for r in rows if r["score"] is not None]
    if valid:
        max_score = max(r["score"] for r in valid)
        for r in valid:
            r["fraction"] = float(r["score"] / max_score) if max_score else 0.0

    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
    return rows


def list_methods() -> list:
    """Return all LCIA methods registered in the current Brightway project."""
    _ensure_project()
    seen = set()
    results = []
    for m in sorted(bd.methods):
        top = m[0]
        if top not in seen:
            seen.add(top)
            results.append({"name": top, "categories": []})
        results[-1]["categories"].append(m[-1])
    return results


def check_brightway() -> dict:
    """Return Brightway project status."""
    try:
        _ensure_project()
        result = {
            "running": True,
            "engine": "brightway2.5",
            "project": BRIGHTWAY_PROJECT,
            "databases": list(bd.databases),
            "methods": len(list(bd.methods)),
        }
        from .search import get_projection_status

        result["search_database"] = get_projection_status(project=BRIGHTWAY_PROJECT)
        return result
    except Exception as exc:
        return {"running": False, "error": str(exc)}
