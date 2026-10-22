"""MCP server exposing the three governed maintenance tools over stdio.

The PoC's main UI runs the agent in-process. This script demonstrates the same
tool contract through the Model Context Protocol so another MCP-capable client
can consume it.

Note: the ``langchain_mcp_adapters.create_mcp_server`` helper mentioned in the
target architecture is not present in the published adapter versions. The
script attempts it for forward compatibility and otherwise falls back to the
official MCP Python SDK ``FastMCP`` server. Both paths expose the same wrapped
``PreloopTool``/local-governed tools.

Approval mode is controlled by ``APPROVAL_MODE``. For non-interactive MCP
clients, use ``auto_approve`` or ``auto_reject``; ``cli`` will prompt on the
server's stdin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools import TOOLS  # noqa: E402

_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def _invoke(tool_name: str, arguments: dict[str, Any]) -> str:
    """Invoke a governed tool and return its textual result."""
    tool = _TOOLS_BY_NAME[tool_name]
    result = tool.invoke(arguments)
    return result if isinstance(result, str) else str(result)


def _build_server() -> Any:
    """Build an MCP server exposing the governed tools."""
    try:  # Forward-compatible path requested by the target architecture.
        from langchain_mcp_adapters import create_mcp_server  # type: ignore[attr-defined]
    except Exception:
        create_mcp_server = None  # type: ignore[assignment]

    if create_mcp_server is not None:  # pragma: no cover - not in published versions
        try:
            return create_mcp_server("industrial-agent-harness", tools=TOOLS)
        except Exception:
            pass

    from mcp.server.fastmcp import FastMCP

    server = FastMCP("industrial-agent-harness")

    @server.tool()
    def query_knowledge_graph(question: str) -> str:
        """Query the industrial GraphRAG knowledge graph for an equipment symptom."""
        return _invoke("query_knowledge_graph", {"question": question})

    @server.tool()
    def propose_maintenance_action(equipment_id: str, action: str) -> str:
        """Propose a maintenance action without executing it."""
        return _invoke(
            "propose_maintenance_action",
            {"equipment_id": equipment_id, "action": action},
        )

    @server.tool()
    def execute_maintenance_action(equipment_id: str, action: str) -> str:
        """Execute a maintenance action after the mandatory human approval."""
        return _invoke(
            "execute_maintenance_action",
            {"equipment_id": equipment_id, "action": action},
        )

    return server


def main() -> None:
    """Start the MCP server on stdio."""
    os.environ.setdefault("APPROVAL_MODE", "cli")
    server = _build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
