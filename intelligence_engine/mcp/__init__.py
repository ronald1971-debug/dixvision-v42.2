"""Model Context Protocol (MCP) servers exposed by the intelligence engine.

Currently only :mod:`intelligence_engine.mcp.opennews` is shipped — a
deterministic, dependency-free MCP-style JSON-RPC server that exposes
the news knowledge index over stdio. Designed to be embeddable in
the same process (in-memory dispatch via
:func:`OpenNewsServer.handle_request`) or hosted as a child process
via :func:`OpenNewsServer.serve_stdio`.
"""

from intelligence_engine.mcp.opennews import (
    OPENNEWS_PROTOCOL_VERSION,
    OPENNEWS_SERVER_NAME,
    OpenNewsServer,
)

__all__ = [
    "OPENNEWS_PROTOCOL_VERSION",
    "OPENNEWS_SERVER_NAME",
    "OpenNewsServer",
]
