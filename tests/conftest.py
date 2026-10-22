"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph.connection import get_driver
from harness import approval, audit, policy
from harness.governance import reset_reasoning_context


@pytest.fixture(autouse=True)
def isolated_governance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate audit, approvals, policies and environment for every test."""
    ledger = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_HMAC_KEY", "test-audit-key")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(ledger))
    monkeypatch.setenv("APPROVAL_MODE", "auto_approve")
    monkeypatch.setenv("APPROVAL_TIMEOUT_SECONDS", "3")
    monkeypatch.delenv("PRELOOP_URL", raising=False)
    monkeypatch.delenv("PRELOOP_TOKEN", raising=False)
    monkeypatch.delenv("FORCE_FAKE_LLM", raising=False)

    audit.reset_audit_logger(path=ledger, key="test-audit-key")
    approval.reset_approval_manager()
    policy.reset_policy_manager()
    reset_reasoning_context()

    yield

    approval.clear_approvals()
    policy.reset_policy_manager()
    audit.reset_audit_logger(path=ledger, key="test-audit-key")
    reset_reasoning_context()


@pytest.fixture
def seeded_neo4j():
    """Skip a test unless a reachable, seeded Neo4j instance is available."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        with driver.session() as session:
            row = session.run("MATCH (n:Equipment) RETURN count(n) AS count").single()
            if not row or row["count"] == 0:
                pytest.skip("Neo4j is reachable but the graph is not seeded")
    except Exception as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")
    return True
