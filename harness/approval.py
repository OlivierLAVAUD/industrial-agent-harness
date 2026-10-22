"""Human approval workflow for sensitive agent actions.

The public API mirrors the control-plane primitives expected by the PoC:

* :func:`request_approval` creates a request with the full reasoning context and
  waits for a human decision;
* :func:`get_pending_approvals` feeds the Streamlit approval panel;
* :func:`record_decision` records an approve/reject decision.

Preloop does not ship a standalone ``ApprovalManager`` SDK class in the PyPI
distribution, so ``LocalApprovalManager`` is the explicit in-memory fallback
used by the demo. It is process-local by design (as required for Streamlit's
same-process agent/UI loop) and is not a replacement for Preloop's out-of-band
approval workflows.
"""

from __future__ import annotations

import copy
import os
import threading
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv

from .audit import log_event
from .policy import get_policy_for_tool

load_dotenv()

try:  # pragma: no cover - depends on the installed Preloop distribution
    from preloop import ApprovalManager as _PreloopApprovalManager  # type: ignore[attr-defined]
except Exception:
    _PreloopApprovalManager = None

_APPROVAL_METHODS = ("create_request", "get_request", "list_pending", "record_decision", "wait_for_decision")


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _normalize_decision(decision: str) -> str:
    """Normalize an approval decision.

    Args:
        decision: Free-form decision label.

    Returns:
        Either ``"approved"`` or ``"rejected"``.
    """
    normalized = str(decision).strip().lower()
    if normalized in {"approved", "approve", "accept", "accepted", "oui", "ok", "y", "yes"}:
        return "approved"
    return "rejected"


class LocalApprovalManager:
    """Thread-safe, in-memory approval store used by the PoC fallback."""

    def __init__(self) -> None:
        """Create an empty approval store."""
        self._requests: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._condition = threading.Condition(threading.RLock())

    def create_request(
        self,
        tool_name: str,
        equipment_id: str,
        action: str,
        reasoning_path: str,
        sources: list[str],
        policy: str = "require_approval",
        requested_approver_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a pending approval request.

        Args:
            tool_name: Tool requiring approval.
            equipment_id: Equipment targeted by the action.
            action: Proposed maintenance action.
            reasoning_path: Human-readable GraphRAG trace shown to the approver.
            sources: Graph node identifiers consulted by the agent.
            policy: Policy that triggered the approval.
            requested_approver_id: Optional intended approver.

        Returns:
            A copy of the created request.
        """
        request_id = f"APR-{uuid.uuid4().hex[:10].upper()}"
        record = {
            "id": request_id,
            "tool_name": tool_name,
            "equipment_id": equipment_id,
            "action": action,
            "reasoning_path": reasoning_path,
            "sources": list(sources),
            "policy": policy,
            "status": "pending",
            "created_at": _utc_now(),
            "decided_at": None,
            "approver_id": None,
            "requested_approver_id": requested_approver_id,
            "reason": "",
        }
        with self._condition:
            self._requests[request_id] = record
            return copy.deepcopy(record)

    def get_request(self, approval_id: str) -> dict[str, Any] | None:
        """Return a request by id, or ``None``."""
        with self._condition:
            record = self._requests.get(approval_id)
            return copy.deepcopy(record) if record is not None else None

    def list_pending(self) -> list[dict[str, Any]]:
        """Return pending requests, newest first."""
        with self._condition:
            records = [
                copy.deepcopy(record)
                for record in reversed(self._requests.values())
                if record["status"] == "pending"
            ]
        return records

    def list_all(self) -> list[dict[str, Any]]:
        """Return every request, newest first."""
        with self._condition:
            return [copy.deepcopy(record) for record in reversed(self._requests.values())]

    def record_decision(
        self,
        approval_id: str,
        decision: str,
        approver_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Record a human decision for a pending request.

        Args:
            approval_id: Approval request identifier.
            decision: ``approved``/``rejected`` (free-form accepted).
            approver_id: Simulated or real approver identity.
            reason: Justification, mandatory in the UI for a rejection.

        Returns:
            A copy of the updated request.

        Raises:
            KeyError: If the approval id is unknown.
        """
        with self._condition:
            if approval_id not in self._requests:
                raise KeyError(f"Unknown approval request: {approval_id}")
            record = self._requests[approval_id]
            if record["status"] != "pending":
                return copy.deepcopy(record)

            record["status"] = _normalize_decision(decision)
            record["approver_id"] = approver_id
            record["reason"] = reason
            record["decided_at"] = _utc_now()
            self._condition.notify_all()
            return copy.deepcopy(record)

    def wait_for_decision(self, approval_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Block until a request is decided or the timeout expires.

        Args:
            approval_id: Approval request identifier.
            timeout: Maximum wait in seconds.

        Returns:
            A copy of the request. On timeout its status becomes ``expired``.

        Raises:
            KeyError: If the approval id is unknown.
        """
        with self._condition:
            if approval_id not in self._requests:
                raise KeyError(f"Unknown approval request: {approval_id}")
            record = self._requests[approval_id]
            self._condition.wait_for(lambda: record["status"] != "pending", timeout=timeout)
            if record["status"] == "pending":
                record["status"] = "expired"
                record["decided_at"] = _utc_now()
                record["reason"] = "Délai d'approbation dépassé."
                self._condition.notify_all()
            return copy.deepcopy(record)

    def clear(self) -> None:
        """Remove every request (tests and explicit resets only)."""
        with self._condition:
            self._requests.clear()
            self._condition.notify_all()


_MANAGER: Any = None
_MANAGER_LOCK = threading.Lock()


def _build_manager() -> Any:
    """Return a Preloop manager if it implements the expected API, else local."""
    if _PreloopApprovalManager is not None:  # pragma: no cover - improbable today
        try:
            candidate = _PreloopApprovalManager()
            if all(hasattr(candidate, name) for name in _APPROVAL_METHODS):
                return candidate
        except Exception:
            pass
    return LocalApprovalManager()


def get_approval_manager() -> Any:
    """Return the process-wide approval manager singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = _build_manager()
        return _MANAGER


def reset_approval_manager() -> None:
    """Clear and replace the approval manager singleton (tests only)."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = LocalApprovalManager()


def _print_request(record: dict[str, Any]) -> None:
    """Print an approval request for the CLI mode."""
    print("\n" + "=" * 78)
    print("APPROBATION HUMAINE REQUISE")
    print("=" * 78)
    print(f"Demande      : {record['id']}")
    print(f"Outil        : {record['tool_name']}")
    print(f"Équipement   : {record['equipment_id']}")
    print(f"Action       : {record['action']}")
    print(f"Politique   : {record['policy']}")
    print(f"Sources      : {', '.join(record['sources']) or 'aucune'}")
    print("\nChemin de raisonnement :")
    print(record["reasoning_path"] or "(non fourni)")
    print("=" * 78)


def _handle_cli(record: dict[str, Any], approver_id: str | None) -> dict[str, Any]:
    """Handle an approval interactively through stdin/stdout."""
    _print_request(record)
    approver = approver_id or os.getenv("APPROVER_ID", "cli-operator")
    try:
        answer = input("Approuver cette action ? [o/N] ").strip().lower()
    except EOFError:
        answer = "n"
        reason = "Entrée standard fermée : rejet par défaut."
    else:
        reason = ""

    if answer in {"o", "oui", "y", "yes"}:
        return record_decision(record["id"], "approved", approver, reason or "Approuvé en CLI.")

    if not reason:
        try:
            reason = input("Raison du rejet : ").strip()
        except EOFError:
            reason = ""
    return record_decision(
        record["id"],
        "rejected",
        approver,
        reason or "Rejeté en CLI.",
    )


def request_approval(
    tool_name: str,
    equipment_id: str,
    action: str,
    reasoning_path: str,
    sources: list[str],
    approver_id: str | None = None,
) -> dict[str, Any]:
    """Create an approval request and wait for a decision.

    Behaviour is controlled by ``APPROVAL_MODE``:

    * ``streamlit`` (default): blocks until the UI records a decision.
    * ``cli``: prompts on stdin (useful for the headless demo/tests).
    * ``auto_approve`` / ``auto_reject``: deterministic automation mode.

    Args:
        tool_name: Tool requiring approval.
        equipment_id: Equipment targeted by the action.
        action: Proposed maintenance action.
        reasoning_path: GraphRAG trace shown to the human approver.
        sources: Source node identifiers.
        approver_id: Optional approver identity override.

    Returns:
        The final approval record with status ``approved``, ``rejected`` or
        ``expired``.
    """
    manager = get_approval_manager()
    policy = get_policy_for_tool(tool_name)
    normalized_sources = [str(source) for source in (sources or [])]
    record = manager.create_request(
        tool_name=tool_name,
        equipment_id=equipment_id,
        action=action,
        reasoning_path=reasoning_path,
        sources=normalized_sources,
        policy=policy,
        requested_approver_id=approver_id,
    )
    log_event(
        "approval_requested",
        actor="agent",
        tool=tool_name,
        payload={
            "approval_id": record["id"],
            "equipment_id": equipment_id,
            "action": action,
            "reasoning_path": reasoning_path,
            "sources": normalized_sources,
            "policy": policy,
        },
    )

    mode = os.getenv("APPROVAL_MODE", "streamlit").strip().lower()
    if mode == "auto_approve":
        return record_decision(
            record["id"],
            "approved",
            approver_id or "auto-approver",
            "Approbation automatique de démonstration.",
        )
    if mode == "auto_reject":
        return record_decision(
            record["id"],
            "rejected",
            approver_id or "auto-approver",
            "Rejet automatique de démonstration.",
        )
    if mode == "cli":
        return _handle_cli(record, approver_id)

    timeout = float(os.getenv("APPROVAL_TIMEOUT_SECONDS", "600"))
    result = manager.wait_for_decision(record["id"], timeout=timeout)
    if result["status"] == "expired":
        log_event(
            "approval_expired",
            actor="system",
            tool=tool_name,
            payload={"approval_id": record["id"], "timeout_seconds": timeout},
        )
    return result


def record_decision(
    approval_id: str,
    decision: str,
    approver_id: str,
    reason: str = "",
) -> dict[str, Any]:
    """Record an approval decision through the configured manager.

    Args:
        approval_id: Approval request identifier.
        decision: ``approved`` or ``rejected``.
        approver_id: Approver identity (simulated in the PoC).
        reason: Decision justification.

    Returns:
        The updated approval record.
    """
    manager = get_approval_manager()
    before: dict[str, Any] | None = None
    getter = getattr(manager, "get_request", None)
    if callable(getter):
        before = getter(approval_id)

    result = manager.record_decision(approval_id, decision, approver_id, reason)
    was_pending = before is None or before.get("status") == "pending"
    if was_pending:
        log_event(
            "approval_decided",
            actor=approver_id,
            tool=result.get("tool_name"),
            payload={
                "approval_id": approval_id,
                "decision": result.get("status"),
                "reason": reason,
                "equipment_id": result.get("equipment_id"),
                "action": result.get("action"),
                "policy": result.get("policy"),
            },
        )
    return result


def get_pending_approvals() -> list[dict[str, Any]]:
    """Return pending approval requests for the Streamlit panel."""
    return get_approval_manager().list_pending()


def get_approval(approval_id: str) -> dict[str, Any] | None:
    """Return a single approval request by id."""
    getter = getattr(get_approval_manager(), "get_request", None)
    return getter(approval_id) if callable(getter) else None


def clear_approvals() -> None:
    """Clear every approval request (tests and UI resets only)."""
    manager = get_approval_manager()
    clearer = getattr(manager, "clear", None)
    if callable(clearer):
        clearer()
