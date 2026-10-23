"""Policy resolution for governed tools.

In production, policy evaluation belongs to Preloop's control plane. The
``preloop`` distribution available on PyPI is the self-hosted server and does
not expose a ``PolicyManager`` class; the harness therefore attempts to import
it for forward compatibility and falls back to an explicit, local YAML adapter
(``LocalPolicyManager``). The fallback only reads ``policy.yaml``; it does not
reimplement Preloop's CEL engine.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_PACKAGE_POLICY_FILE = Path(__file__).resolve().parents[1] / "policy.yaml"
_WORKDIR_POLICY_FILE = Path.cwd() / "policy.yaml"
POLICY_FILE = Path(
    os.getenv("POLICY_FILE")
    or (_WORKDIR_POLICY_FILE if _WORKDIR_POLICY_FILE.exists() else _PACKAGE_POLICY_FILE)
)

try:  # pragma: no cover - depends on the installed Preloop distribution
    from preloop import PolicyManager as _PreloopPolicyManager  # type: ignore[attr-defined]
except Exception:  # ImportError or a Preloop version without this SDK type
    _PreloopPolicyManager = None

_MANAGER: Any = None
_LOCK = threading.Lock()

_VALID_POLICIES = {"auto", "require_approval"}


def _normalize_policy(value: Any) -> str:
    """Normalize a policy value returned by Preloop or by the YAML fallback.

    Args:
        value: String, dictionary or object describing a policy decision.

    Returns:
        Either ``"auto"`` or ``"require_approval"``.
    """
    if value is None:
        return "auto"
    if isinstance(value, dict):
        for key in ("policy", "decision", "effect", "action"):
            if key in value:
                return _normalize_policy(value[key])
        return "auto"
    if not isinstance(value, str):
        for attr in ("policy", "decision", "effect", "action"):
            if hasattr(value, attr):
                return _normalize_policy(getattr(value, attr))
        return "auto"

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"require_approval", "approval_required", "approval", "human"}:
        return "require_approval"
    if normalized in {"auto", "allow", "allowed", "auto_approve"}:
        return "auto"
    if normalized in _VALID_POLICIES:
        return normalized
    return "auto"


class LocalPolicyManager:
    """Explicit YAML fallback when Preloop's PolicyManager is unavailable."""

    def __init__(self, policy_file: str | Path | None = None) -> None:
        """Load tool policies from a YAML file.

        Args:
            policy_file: Path to the policy file. Defaults to ``POLICY_FILE``.
        """
        self.policy_file = Path(policy_file) if policy_file else POLICY_FILE
        with self.policy_file.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        self._tools: dict[str, Any] = document.get("tools", {}) or {}

    def get_policy_for_tool(self, tool_name: str) -> str:
        """Return the normalized policy for a tool name.

        Args:
            tool_name: Name exposed to the LLM.

        Returns:
            ``"auto"`` or ``"require_approval"``.
        """
        entry = self._tools.get(tool_name, {})
        if isinstance(entry, dict):
            entry = entry.get("policy", "auto")
        return _normalize_policy(entry)


def _build_manager() -> Any:
    """Create the real Preloop manager when available, else the local adapter."""
    if _PreloopPolicyManager is not None:
        try:  # pragma: no cover - depends on Preloop's evolving constructor
            return _PreloopPolicyManager(policy_path=str(POLICY_FILE))
        except TypeError:
            try:
                return _PreloopPolicyManager()
            except Exception:
                pass
        except Exception:
            pass
    return LocalPolicyManager(POLICY_FILE)


def get_policy_manager() -> Any:
    """Return the singleton policy manager."""
    global _MANAGER
    with _LOCK:
        if _MANAGER is None:
            _MANAGER = _build_manager()
        return _MANAGER


def reset_policy_manager() -> None:
    """Clear the singleton policy manager (used by tests)."""
    global _MANAGER
    with _LOCK:
        _MANAGER = None


def get_policy_for_tool(tool_name: str) -> str:
    """Return the policy applying to a tool.

    Args:
        tool_name: Tool name as exposed to the LLM.

    Returns:
        ``"auto"`` or ``"require_approval"``.
    """
    manager = get_policy_manager()
    result: Any | None = None
    for method_name in ("get_policy_for_tool", "get_policy", "policy_for_tool"):
        method = getattr(manager, method_name, None)
        if callable(method):
            result = method(tool_name)
            break
    return _normalize_policy(result)
