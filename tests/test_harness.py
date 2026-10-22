"""Tests for the policy, approval and audit governance layer."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from harness import approval, policy
from harness.approval import (
    get_pending_approvals,
    record_decision,
    request_approval,
)
from harness.audit import get_events, verify_audit_chain
from harness.policy import LocalPolicyManager, get_policy_for_tool


def test_local_policy_file_mapping() -> None:
    """policy.yaml maps read tools to auto and execution to approval."""
    manager = LocalPolicyManager()

    assert manager.get_policy_for_tool("query_knowledge_graph") == "auto"
    assert manager.get_policy_for_tool("propose_maintenance_action") == "auto"
    assert manager.get_policy_for_tool("execute_maintenance_action") == "require_approval"
    assert manager.get_policy_for_tool("unknown_tool") == "auto"


def test_policy_delegates_to_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_policy_for_tool delegates to the configured Preloop-shaped manager."""

    class FakePolicyManager:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_policy_for_tool(self, tool_name: str) -> str:
            self.calls.append(tool_name)
            return "require_approval" if tool_name == "execute_maintenance_action" else "auto"

    fake = FakePolicyManager()
    monkeypatch.setattr(policy, "_MANAGER", fake)

    assert get_policy_for_tool("execute_maintenance_action") == "require_approval"
    assert get_policy_for_tool("query_knowledge_graph") == "auto"
    assert fake.calls == ["execute_maintenance_action", "query_knowledge_graph"]


class _FakeApprovalManager:
    """Preloop-shaped approval manager double."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.waited: list[tuple[str, float | None]] = []
        self.decisions: list[tuple[str, str, str, str]] = []
        self._record: dict[str, Any] | None = None

    def create_request(self, **kwargs: Any) -> dict[str, Any]:
        record = {
            "id": "APR-TEST",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00+00:00",
            **kwargs,
        }
        self.created.append(kwargs)
        self._record = record
        return dict(record)

    def get_request(self, approval_id: str) -> dict[str, Any] | None:
        return dict(self._record) if self._record else None

    def list_pending(self) -> list[dict[str, Any]]:
        return [dict(self._record)] if self._record and self._record["status"] == "pending" else []

    def wait_for_decision(self, approval_id: str, timeout: float | None = None) -> dict[str, Any]:
        self.waited.append((approval_id, timeout))
        assert self._record is not None
        record = dict(self._record)
        record["status"] = "approved"
        record["approver_id"] = "human-supervisor"
        self._record = record
        return dict(record)

    def record_decision(
        self, approval_id: str, decision: str, approver_id: str, reason: str = ""
    ) -> dict[str, Any]:
        self.decisions.append((approval_id, decision, approver_id, reason))
        assert self._record is not None
        record = dict(self._record)
        record.update(
            {"status": decision, "approver_id": approver_id, "reason": reason}
        )
        self._record = record
        return dict(record)


def test_request_approval_delegates_creation_and_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """request_approval creates the request and waits through the manager."""
    fake = _FakeApprovalManager()
    monkeypatch.setattr(approval, "_MANAGER", fake)
    monkeypatch.setenv("APPROVAL_MODE", "streamlit")
    monkeypatch.setenv("APPROVAL_TIMEOUT_SECONDS", "12")

    result = request_approval(
        tool_name="execute_maintenance_action",
        equipment_id="P-204",
        action="REMPLACER_ROULEMENT",
        reasoning_path="P-204 → vibration anormale → procédure REMPLACER_ROULEMENT",
        sources=["P-204", "S-VIB-01"],
    )

    assert result["status"] == "approved"
    assert fake.created[0]["equipment_id"] == "P-204"
    assert fake.created[0]["policy"] == "require_approval"
    assert fake.waited == [("APR-TEST", 12.0)]
    assert any(event["event_type"] == "approval_requested" for event in get_events())


def test_record_decision_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """record_decision delegates to the manager and journals the decision."""
    fake = _FakeApprovalManager()
    monkeypatch.setattr(approval, "_MANAGER", fake)
    created = fake.create_request(
        tool_name="execute_maintenance_action",
        equipment_id="P-204",
        action="REMPLACER_ROULEMENT",
        reasoning_path="trace",
        sources=["P-204"],
    )
    result = record_decision(created["id"], "approved", "supervisor", "OK")

    assert result["status"] == "approved"
    assert fake.decisions[-1] == (created["id"], "approved", "supervisor", "OK")
    decided = [event for event in get_events() if event["event_type"] == "approval_decided"]
    assert decided and decided[-1]["actor"] == "supervisor"


def test_streamlit_mode_waits_for_the_ui_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent thread blocks until the UI records the decision."""
    monkeypatch.setenv("APPROVAL_MODE", "streamlit")
    monkeypatch.setenv("APPROVAL_TIMEOUT_SECONDS", "5")
    result: dict[str, Any] = {}

    def worker() -> None:
        result["record"] = request_approval(
            tool_name="execute_maintenance_action",
            equipment_id="P-204",
            action="REMPLACER_ROULEMENT",
            reasoning_path="trace",
            sources=["P-204", "S-VIB-01"],
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    pending: list[dict[str, Any]] = []
    for _ in range(200):
        pending = get_pending_approvals()
        if pending:
            break
        time.sleep(0.01)

    assert pending, "the approval request should be visible to the UI"
    record_decision(pending[0]["id"], "approved", "ui-supervisor", "Validé")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result["record"]["status"] == "approved"
    assert result["record"]["approver_id"] == "ui-supervisor"
    assert verify_audit_chain()["valid"] is True


def test_cli_mode_uses_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI mode records the approval through the same Preloop-shaped manager."""
    monkeypatch.setenv("APPROVAL_MODE", "cli")
    monkeypatch.setenv("APPROVER_ID", "cli-tester")
    monkeypatch.setattr("builtins.input", lambda _prompt="": "o")

    result = request_approval(
        tool_name="execute_maintenance_action",
        equipment_id="P-204",
        action="REMPLACER_ROULEMENT",
        reasoning_path="trace",
        sources=["P-204"],
    )

    assert result["status"] == "approved"
    assert result["approver_id"] == "cli-tester"
