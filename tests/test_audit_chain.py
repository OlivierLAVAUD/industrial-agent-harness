"""Tests for the HMAC-chained audit ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.audit import (
    LocalAuditLogger,
    create_tampered_copy,
    log_event,
    verify_audit_chain,
)


def test_valid_chain(tmp_path: Path) -> None:
    """A freshly written chain verifies and counts every event."""
    ledger = LocalAuditLogger(tmp_path / "audit.jsonl", key="key-one")
    ledger.log_event("agent_started", actor="agent", payload={"question": "P-204"})
    ledger.log_event("tool_call", actor="agent", tool="query_knowledge_graph")
    ledger.log_event("approval_decided", actor="supervisor", tool="execute_maintenance_action")

    result = ledger.verify_chain()

    assert result == {"valid": True, "events_checked": 3, "error": ""}


def test_payload_tampering_is_detected(tmp_path: Path) -> None:
    """Changing a payload without the HMAC key invalidates the chain."""
    path = tmp_path / "audit.jsonl"
    ledger = LocalAuditLogger(path, key="secret")
    ledger.log_event("approval_decided", actor="supervisor", payload={"decision": "approved"})
    ledger.log_event("tool_result", actor="agent", payload={"status": "executed"})

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["decision"] = "rejected"
    lines[0] = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = ledger.verify_chain()

    assert result["valid"] is False
    assert "HMAC" in result["error"]
    assert result["events_checked"] == 1


def test_deleted_record_breaks_the_chain(tmp_path: Path) -> None:
    """Deleting a record in the middle breaks sequence and hash linkage."""
    path = tmp_path / "audit.jsonl"
    ledger = LocalAuditLogger(path, key="secret")
    for index in range(3):
        ledger.log_event("event", actor="agent", payload={"index": index})

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    result = ledger.verify_chain()

    assert result["valid"] is False
    assert result["events_checked"] == 1


def test_wrong_hmac_key_invalidates_chain(tmp_path: Path) -> None:
    """A ledger verified with another key is reported as tampered."""
    path = tmp_path / "audit.jsonl"
    writer = LocalAuditLogger(path, key="right-key")
    writer.log_event("event", actor="agent", payload={"value": 1})

    verifier = LocalAuditLogger(path, key="wrong-key")
    result = verifier.verify_chain()

    assert result["valid"] is False
    assert result["events_checked"] == 1


@pytest.mark.parametrize("mode", ["modify", "delete", "reorder"])
def test_tampered_copy_is_detected(tmp_path: Path, mode: str) -> None:
    """The demo copy is detected as invalid while the live ledger stays intact."""
    source = tmp_path / "source.jsonl"
    ledger = LocalAuditLogger(source, key="demo-key")
    for index in range(4):
        ledger.log_event("event", actor="agent", payload={"index": index})

    copy_path = create_tampered_copy(
        source_path=source,
        destination_path=tmp_path / f"{mode}.jsonl",
        mode=mode,
    )

    tampered = LocalAuditLogger(copy_path, key="demo-key").verify_chain()
    assert tampered["valid"] is False
    assert LocalAuditLogger(source, key="demo-key").verify_chain()["valid"] is True


def test_public_helpers_use_the_global_ledger() -> None:
    """The module-level helpers delegate to the configured global ledger."""
    log_event("tool_call", actor="agent", tool="query_knowledge_graph", payload={"x": 1})
    log_event("tool_result", actor="agent", tool="query_knowledge_graph", payload={"ok": True})

    result = verify_audit_chain()

    assert result["valid"] is True
    assert result["events_checked"] == 2
