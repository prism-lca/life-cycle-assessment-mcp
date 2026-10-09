#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE_SCRIPT="$SCRIPT_DIR/deploy_lca_server_remote.sh"

usage() {
    cat <<'EOF'
Deploy a pushed Git ref to the standalone LCA Docker server.

Usage:
  scripts/deploy_lca_server.sh [REF]
  scripts/deploy_lca_server.sh --rollback

REF defaults to main and can be a branch, tag, or commit SHA available from
the remote Git repository.

Required environment:
  LCA_DEPLOY_HOST       Server hostname, IP address, or SSH config alias

Optional environment:
  LCA_DEPLOY_USER       SSH user (default: root)
  LCA_DEPLOY_PORT       SSH port (default: 22)
  LCA_DEPLOY_SSH_KEY    Path to a private SSH key
  LCA_DEPLOY_REPO_URL   Git repository cloned/fetched on the server
  LCA_REMOTE_REPO_DIR   Managed server checkout (default: /opt/lca-benchmark)
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$REMOTE_SCRIPT" ]]; then
    echo "Remote deployment script not found: $REMOTE_SCRIPT" >&2
    exit 1
fi

: "${LCA_DEPLOY_HOST:?Set LCA_DEPLOY_HOST to the server hostname, IP, or SSH alias.}"

action=deploy
ref=${1:-main}
if [[ "$ref" == "--rollback" ]]; then
    action=rollback
    ref=unused
elif [[ "$ref" == -* ]]; then
    echo "Unknown option: $ref" >&2
    usage >&2
    exit 2
fi

deploy_user=${LCA_DEPLOY_USER:-root}
deploy_port=${LCA_DEPLOY_PORT:-22}
repo_url=${LCA_DEPLOY_REPO_URL:-https://github.com/prism-lca/life-cycle-assessment-mcp.git}
remote_repo_dir=${LCA_REMOTE_REPO_DIR:-/opt/lca-benchmark}
container_name=${LCA_DEPLOY_CONTAINER:-lca-benchmark}
volume_name=${LCA_DEPLOY_VOLUME:-lca_benchmark_brightway}
image_name=${LCA_DEPLOY_IMAGE:-lca-benchmark}
port_binding=${LCA_DEPLOY_PORT_BINDING:-127.0.0.1:9000:9000}
health_url=${LCA_DEPLOY_HEALTH_URL:-http://127.0.0.1:9000/api/health}

if [[ ! "$deploy_port" =~ ^[0-9]+$ ]]; then
    echo "LCA_DEPLOY_PORT must be numeric." >&2
    exit 2
fi
ssh_args=(-p "$deploy_port")
if [[ -n ${LCA_DEPLOY_SSH_KEY:-} ]]; then
    ssh_args+=(-i "$LCA_DEPLOY_SSH_KEY")
fi

printf -v remote_command \
    'bash -s -- %q %q %q %q %q %q %q %q %q' \
    "$action" \
    "$ref" \
    "$repo_url" \
    "$remote_repo_dir" \
    "$container_name" \
    "$volume_name" \
    "$image_name" \
    "$port_binding" \
    "$health_url"

echo "Target: ${deploy_user}@${LCA_DEPLOY_HOST}:${deploy_port}"
if [[ "$action" == "deploy" ]]; then
    echo "Deploying Git ref: $ref"
else
    echo "Rolling back to the previously deployed container"
fi

ssh "${ssh_args[@]}" "${deploy_user}@${LCA_DEPLOY_HOST}" "$remote_command" \
    < "$REMOTE_SCRIPT"
