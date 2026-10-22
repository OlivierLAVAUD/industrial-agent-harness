"""End-to-end integration tests (require a seeded Neo4j instance)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agent.graph import run_agent
from harness.approval import clear_approvals
from harness.audit import get_events, verify_audit_chain
from harness.governance import reset_reasoning_context

ROOT = Path(__file__).resolve().parents[1]

DEMO_QUESTION = "L'équipement P-204 présente une vibration anormale. Que faire ?"


def _final_answer(messages: list) -> str:
    """Return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return ""


def test_full_supervised_scenario(seeded_neo4j: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Question → GraphRAG → proposal → approval → execution → audit."""
    monkeypatch.setenv("FORCE_FAKE_LLM", "1")
    monkeypatch.setenv("APPROVAL_MODE", "auto_approve")
    clear_approvals()
    reset_reasoning_context()

    messages = run_agent(DEMO_QUESTION)
    answer = _final_answer(messages)

    assert "P-204" in answer
    assert "REMPLACER_ROULEMENT" in answer
    assert "approuv" in answer.lower() or "exécutée" in answer.lower()

    event_types = [event["event_type"] for event in get_events(limit=200)]
    assert "agent_run_started" in event_types
    assert "tool_call" in event_types
    assert "approval_requested" in event_types
    assert "approval_decided" in event_types
    assert "tool_result" in event_types
    assert "agent_run_completed" in event_types

    verification = verify_audit_chain()
    assert verification["valid"] is True
    assert verification["events_checked"] > 0


def test_rejected_action_is_not_executed(
    seeded_neo4j: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected approval blocks execution and leaves an audit trace."""
    monkeypatch.setenv("FORCE_FAKE_LLM", "1")
    monkeypatch.setenv("APPROVAL_MODE", "auto_reject")
    clear_approvals()
    reset_reasoning_context()

    messages = run_agent(DEMO_QUESTION)
    answer = _final_answer(messages)

    assert "P-204" in answer
    assert "n'a pas été autorisée" in answer or "n'a pas été exécutée" in answer

    events = get_events(limit=200)
    assert any(event["event_type"] == "tool_blocked" for event in events)
    blocked = next(event for event in events if event["event_type"] == "tool_blocked")
    assert blocked["payload"]["status"] == "rejected"
    assert verify_audit_chain()["valid"] is True


def test_streamlit_approval_loop(
    seeded_neo4j: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Streamlit console launches the agent and records a human decision."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FORCE_FAKE_LLM", "1")
    clear_approvals()
    reset_reasoning_context()

    app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception

    start = next(button for button in app.button if "Lancer" in button.label)
    start.click().run()

    for _ in range(200):
        app.run()
        if any("Approuver" in button.label for button in app.button):
            break
        time.sleep(0.05)
    assert any("Approuver" in button.label for button in app.button)

    approve = next(button for button in app.button if "Approuver" in button.label)
    approve.click().run()

    for _ in range(200):
        app.run()
        if any("Exécution terminée" in success.value for success in app.success):
            break
        time.sleep(0.05)

    assert not app.exception
    assert any("Exécution terminée" in success.value for success in app.success)
    assert any("P-204" in info.value for info in app.info)
    assert verify_audit_chain()["valid"] is True
    assert any(event["event_type"] == "approval_decided" for event in get_events(limit=200))
