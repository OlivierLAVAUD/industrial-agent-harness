"""Tool-level governance boundary.

The agent's raw tools contain business logic only. This module wraps each tool
with the governance layer:

* when a real Preloop endpoint is configured (``PRELOOP_URL`` +
  ``PRELOOP_TOKEN``), the tool is wrapped with ``langchain_preloop``'s
  ``wrap_with_preloop`` (aliased ``PreloopTool``);
* otherwise the explicit local fallback evaluates ``policy.yaml``, requests
  human approval when required, writes the HMAC audit ledger and only then
  executes the underlying tool.

Keeping this logic outside ``agent/tools.py`` is deliberate: tools never contain
approval or policy code, which is exactly the separation the PoC demonstrates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from .approval import request_approval
from .audit import log_event
from .policy import get_policy_for_tool

logger = logging.getLogger(__name__)

try:  # Real Preloop integration (installed through langchain-preloop)
    from langchain_preloop import wrap_with_preloop as PreloopTool
except Exception:  # pragma: no cover - optional dependency
    PreloopTool = None  # type: ignore[assignment]

_CONTEXT: dict[str, Any] = {
    "reasoning_path": "",
    "sources": [],
    "equipment_id": "",
    "action": "",
}
_CONTEXT_LOCK = threading.RLock()


def update_reasoning_context(
    *,
    reasoning_path: str | None = None,
    sources: list[str] | None = None,
    equipment_id: str | None = None,
    action: str | None = None,
) -> None:
    """Update the context shown to the human approver.

    Args:
        reasoning_path: Formatted GraphRAG path.
        sources: Graph node identifiers consulted by the agent.
        equipment_id: Equipment currently under investigation.
        action: Maintenance action proposed by the agent.
    """
    with _CONTEXT_LOCK:
        if reasoning_path is not None:
            _CONTEXT["reasoning_path"] = reasoning_path
        if sources is not None:
            _CONTEXT["sources"] = list(dict.fromkeys(str(source) for source in sources))
        if equipment_id:
            _CONTEXT["equipment_id"] = equipment_id
        if action:
            _CONTEXT["action"] = action


def get_reasoning_context() -> dict[str, Any]:
    """Return a copy of the current approval context."""
    with _CONTEXT_LOCK:
        return {
            "reasoning_path": _CONTEXT["reasoning_path"],
            "sources": list(_CONTEXT["sources"]),
            "equipment_id": _CONTEXT["equipment_id"],
            "action": _CONTEXT["action"],
        }


def reset_reasoning_context() -> None:
    """Reset the approval context (tests and new demo runs)."""
    with _CONTEXT_LOCK:
        _CONTEXT.update(
            {"reasoning_path": "", "sources": [], "equipment_id": "", "action": ""}
        )


def preloop_enabled() -> bool:
    """Return whether a real Preloop endpoint is configured."""
    return bool(
        PreloopTool is not None
        and os.getenv("PRELOOP_URL")
        and os.getenv("PRELOOP_TOKEN")
    )


def _serializable(value: Any) -> Any:
    """Convert a value to a JSON-serializable structure for the audit ledger."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _run_governed(tool: BaseTool, kwargs: dict[str, Any]) -> Any:
    """Apply policy, approval and audit around a single tool invocation."""
    log_event(
        "tool_call",
        actor="agent",
        tool=tool.name,
        payload={"arguments": _serializable(kwargs)},
    )

    policy = get_policy_for_tool(tool.name)
    approved_by = None

    if policy == "require_approval":
        context = get_reasoning_context()
        equipment_id = str(
            kwargs.get("equipment_id") or context.get("equipment_id") or "inconnu"
        )
        action = str(kwargs.get("action") or context.get("action") or "action non précisée")
        decision = request_approval(
            tool_name=tool.name,
            equipment_id=equipment_id,
            action=action,
            reasoning_path=str(context.get("reasoning_path") or ""),
            sources=list(context.get("sources") or []),
        )
        if decision.get("status") != "approved":
            log_event(
                "tool_blocked",
                actor="system",
                tool=tool.name,
                payload={
                    "approval_id": decision.get("id"),
                    "status": decision.get("status"),
                    "reason": decision.get("reason", ""),
                    "equipment_id": equipment_id,
                    "action": action,
                },
            )
            reason = decision.get("reason") or "approbation refusée ou expirée"
            return (
                f"ACTION_BLOQUEE: l'action '{action}' sur {equipment_id} n'a pas été exécutée "
                f"(statut: {decision.get('status')}; raison: {reason})."
            )
        approved_by = decision.get("approver_id")

    try:
        result = tool.invoke(kwargs)
    except Exception as exc:  # pragma: no cover - external failures
        logger.exception("Tool %s failed", tool.name)
        log_event(
            "tool_error",
            actor="agent",
            tool=tool.name,
            payload={"arguments": _serializable(kwargs), "error": str(exc)},
        )
        return f"ERREUR_OUTIL: {exc}"

    log_event(
        "tool_result",
        actor="agent",
        tool=tool.name,
        payload={
            "arguments": _serializable(kwargs),
            "result": _serializable(result),
            "approved_by": approved_by,
        },
    )
    if approved_by:
        return f"APPROUVE_PAR_HUMAIN ({approved_by}) :: {result}"
    return result


def local_govern(tool: BaseTool) -> BaseTool:
    """Wrap a tool with the explicit local policy/approval/audit fallback."""

    def _run(**kwargs: Any) -> Any:
        return _run_governed(tool, kwargs)

    async def _arun(**kwargs: Any) -> Any:
        return await asyncio.to_thread(_run_governed, tool, kwargs)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=tool.name,
        description=(
            f"{tool.description}\n\n"
            "[Gouvernance: cet outil est évalué par policy.yaml et peut nécessiter "
            "une approbation humaine avant exécution.]"
        ),
        args_schema=tool.args_schema,
    )


def govern_tool(tool: BaseTool) -> BaseTool:
    """Wrap a tool with Preloop when configured, else the local fallback.

    Args:
        tool: Business tool to govern.

    Returns:
        A tool with the same name and schema, protected by the governance layer.
    """
    if preloop_enabled():
        reasoning = (
            "Action de maintenance industrielle proposée après raisonnement GraphRAG. "
            "Vérifier l'équipement, le chemin de raisonnement et les sources avant approbation."
        )
        logger.info("Using real Preloop wrapper for tool %s", tool.name)
        return PreloopTool(tool, reasoning=reasoning)  # type: ignore[misc]

    return local_govern(tool)
