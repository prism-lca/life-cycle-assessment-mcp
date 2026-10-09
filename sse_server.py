"""
sse_server.py — HTTP entry point for the Life Cycle Assessment MCP server.

Runs FastMCP directly (same pattern as MacFLStudioMCP) so that FastMCP owns
the uvicorn server and handles all routing including the MCP endpoint.

MCP endpoint: /mcp  (Streamable HTTP — MCP spec 2025-03-26)
REST API:     /api/* (custom routes registered in lca_server.py)

Usage:
    python3 sse_server.py
"""

import os

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

from lca_server import mcp

CORS_MIDDLEWARE = [
    Middleware(
        CORSMiddleware,
        allow_origins=[
            "https://calvinw.github.io",
            "https://prism-lca.github.io",
            "https://catiehe.github.io",
            "http://localhost:5173",
        ],
        allow_credentials=False,
        # DELETE lets a browser client terminate its MCP session explicitly
        # instead of leaving it to expire.
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        # Mcp-Session-Id is needed in both directions: expose_headers lets
        # JavaScript read the id that `initialize` returns, allow_headers lets
        # the preflight pass so it can be sent back on every later request.
        # Either one alone still leaves a browser client stuck after
        # `initialize`. Mcp-Protocol-Version is sent by clients on MCP spec
        # 2025-06-18 and later.
        allow_headers=["Content-Type", "Mcp-Session-Id", "Mcp-Protocol-Version"],
        expose_headers=["Mcp-Session-Id"],
    )
]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
        middleware=CORS_MIDDLEWARE,
    )
