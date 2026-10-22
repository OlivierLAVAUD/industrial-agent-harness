"""Tests for the agent graph, tool ordering and approval gating."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.graph import ScriptedToolCallingModel, build_agent_graph, run_agent
from agent.tools import TOOLS
from harness import governance
from harness.audit import get_events
from harness.governance import get_reasoning_context, govern_tool, local_govern
from harness.policy import get_policy_for_tool

DEMO_QUESTION = "L'équipement P-204 présente une vibration anormale. Que faire ?"


class QueryInput(BaseModel):
    question: str = Field(...)


class ActionInput(BaseModel):
    equipment_id: str = Field(...)
    action: str = Field(...)


def _fake_tools(calls: list[str]) -> list[StructuredTool]:
    """Build three recording tools with the production tool names."""

    def query(question: str) -> str:
        """Return a GraphRAG-like context."""
        calls.append("query")
        return (
            "Contexte GraphRAG — Équipement P-204 (Pompe centrifuge)\n"
            "Procédures recommandées :\n"
            "- REMPLACER_ROULEMENT — Remplacer le roulement\n"
            "Sources : P-204, S-VIB-01, PR-ROUL-01"
        )

    def propose(equipment_id: str, action: str) -> str:
        """Record a proposal."""
        calls.append("propose")
        return f"PROPOSITION_ENREGISTREE: {action} sur {equipment_id}"

    def execute(equipment_id: str, action: str) -> str:
        """Simulate execution."""
        calls.append("execute")
        return f"EXECUTION_SIMULEE: {action} sur {equipment_id}"

    return [
        StructuredTool.from_function(
            query,
            name="query_knowledge_graph",
            description="query",
            args_schema=QueryInput,
        ),
        StructuredTool.from_function(
            propose,
            name="propose_maintenance_action",
            description="propose",
            args_schema=ActionInput,
        ),
        StructuredTool.from_function(
            execute,
            name="execute_maintenance_action",
            description="execute",
            args_schema=ActionInput,
        ),
    ]


def test_tool_contract_and_policies() -> None:
    """The exposed tools match the expected names and policies."""
    assert [tool.name for tool in TOOLS] == [
        "query_knowledge_graph",
        "propose_maintenance_action",
        "execute_maintenance_action",
    ]
    assert get_policy_for_tool("query_knowledge_graph") == "auto"
    assert get_policy_for_tool("propose_maintenance_action") == "auto"
    assert get_policy_for_tool("execute_maintenance_action") == "require_approval"


def test_langgraph_runs_tools_in_function_calling_order() -> None:
    """The scripted function-calling model drives query → propose → execute."""
    calls: list[str] = []
    governed_tools = [govern_tool(tool) for tool in _fake_tools(calls)]
    messages = run_agent(
        DEMO_QUESTION,
        llm=ScriptedToolCallingModel(),
        tools=governed_tools,
    )

    assert calls == ["query", "propose", "execute"]
    final = messages[-1]
    assert "P-204" in final.content
    assert "REMPLACER_ROULEMENT" in final.content

    event_types = [event["event_type"] for event in get_events()]
    assert "agent_run_started" in event_types
    assert "agent_run_completed" in event_types
    assert "approval_requested" in event_types
    assert "approval_decided" in event_types
    assert event_types.count("tool_call") == 3


def test_build_graph_exposes_agent_and_tools_nodes() -> None:
    """The compiled graph contains the two required nodes."""
    calls: list[str] = []
    graph = build_agent_graph(llm=ScriptedToolCallingModel(), tools=_fake_tools(calls))
    node_names = set(graph.get_graph().nodes)
    assert {"agent", "tools"}.issubset(node_names)


def test_governed_execution_requests_approval_and_blocks_on_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected approval prevents the underlying tool from running."""
    executed: list[dict[str, Any]] = []

    def execute(equipment_id: str, action: str) -> str:
        """Underlying business tool."""
        executed.append({"equipment_id": equipment_id, "action": action})
        return "EXECUTION_SIMULEE"

    raw = StructuredTool.from_function(
        execute,
        name="execute_maintenance_action",
        description="execute",
        args_schema=ActionInput,
    )
    governed = local_govern(raw)

    requests: list[dict[str, Any]] = []

    def fake_request_approval(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        return {
            "id": "APR-REJECT",
            "status": "rejected",
            "reason": "Risque trop élevé",
            "approver_id": "supervisor",
        }

    monkeypatch.setattr(governance, "request_approval", fake_request_approval)

    result = governed.invoke(
        {"equipment_id": "P-204", "action": "REMPLACER_ROULEMENT"}
    )

    assert requests and requests[0]["equipment_id"] == "P-204"
    assert requests[0]["tool_name"] == "execute_maintenance_action"
    assert executed == []
    assert "ACTION_BLOQUEE" in result
    assert any(event["event_type"] == "tool_blocked" for event in get_events())


def test_governed_execution_runs_only_after_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approved request executes the tool and records the approver."""
    executed: list[dict[str, Any]] = []

    def execute(equipment_id: str, action: str) -> str:
        """Underlying business tool."""
        executed.append({"equipment_id": equipment_id, "action": action})
        return "EXECUTION_SIMULEE"

    raw = StructuredTool.from_function(
        execute,
        name="execute_maintenance_action",
        description="execute",
        args_schema=ActionInput,
    )
    governed = local_govern(raw)

    monkeypatch.setattr(
        governance,
        "request_approval",
        lambda **_kwargs: {
            "id": "APR-OK",
            "status": "approved",
            "reason": "",
            "approver_id": "human-supervisor",
        },
    )

    result = governed.invoke(
        {"equipment_id": "P-204", "action": "REMPLACER_ROULEMENT"}
    )

    assert executed == [{"equipment_id": "P-204", "action": "REMPLACER_ROULEMENT"}]
    assert "APPROUVE_PAR_HUMAIN" in result
    assert "human-supervisor" in result
    results = [event for event in get_events() if event["event_type"] == "tool_result"]
    assert results and results[-1]["payload"]["approved_by"] == "human-supervisor"
    assert get_reasoning_context()["sources"] == []


def test_read_tools_do_not_require_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query and propose tools execute without an approval request."""
    calls: list[str] = []
    tools = {tool.name: tool for tool in _fake_tools(calls)}

    def fail_approval(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("read-only tools must not request approval")

    monkeypatch.setattr(governance, "request_approval", fail_approval)

    query = local_govern(tools["query_knowledge_graph"])
    propose = local_govern(tools["propose_maintenance_action"])

    assert "Contexte GraphRAG" in query.invoke({"question": DEMO_QUESTION})
    assert "PROPOSITION_ENREGISTREE" in propose.invoke(
        {"equipment_id": "P-204", "action": "REMPLACER_ROULEMENT"}
    )
    assert calls == ["query", "propose"]
