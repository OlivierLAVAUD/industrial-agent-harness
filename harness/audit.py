"""Tamper-evident audit ledger for the PoC.

The ledger is an append-only JSONL file. Every event is authenticated with
HMAC-SHA256 and chained to the previous event, so altering, deleting or
reordering a record invalidates the whole chain.

Preloop's control plane provides a tamper-evident audit ledger, but the
``preloop`` Python distribution available on PyPI is the server itself and does
not expose an ``AuditLogger`` SDK class. ``LocalAuditLogger`` is therefore the
explicit fallback required by the PoC, with the HMAC key supplied by
``AUDIT_HMAC_KEY``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

GENESIS_HASH = "0" * 64
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "var" / "audit_chain.jsonl"
_DEV_KEY = "dev-insecure-audit-key-change-me"

try:  # pragma: no cover - depends on the installed Preloop distribution
    from preloop import AuditLogger as _PreloopAuditLogger  # type: ignore[attr-defined]
except Exception:
    _PreloopAuditLogger = None

_LOCK = threading.RLock()
_LOGGER: LocalAuditLogger | None = None
_WARNED_DEV_KEY = False


def _canonical_json(value: Any) -> str:
    """Serialize a value deterministically for HMAC computation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


class LocalAuditLogger:
    """HMAC-chained, append-only JSONL audit logger."""

    def __init__(
        self,
        path: str | Path | None = None,
        key: str | bytes | None = None,
    ) -> None:
        """Initialise the ledger.

        Args:
            path: Ledger path. Defaults to ``AUDIT_LOG_PATH`` or ``var/audit_chain.jsonl``.
            key: HMAC key. Defaults to ``AUDIT_HMAC_KEY``.
        """
        self.path = Path(path or os.getenv("AUDIT_LOG_PATH", _DEFAULT_PATH))
        self.path.parent.mkdir(parents=True, exist_ok=True)

        global _WARNED_DEV_KEY
        raw_key = key if key is not None else os.getenv("AUDIT_HMAC_KEY")
        if not raw_key:
            if not _WARNED_DEV_KEY:
                warnings.warn(
                    "AUDIT_HMAC_KEY is not set; using the documented development key. "
                    "Configure a strong key before any production use.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED_DEV_KEY = True
            raw_key = _DEV_KEY
        self._key = raw_key.encode("utf-8") if isinstance(raw_key, str) else raw_key
        self._seq = 0
        self._last_hash = GENESIS_HASH
        self._load_tail()

    def _load_tail(self) -> None:
        """Read the last valid record to continue the chain after a restart."""
        if not self.path.exists():
            return
        last_line = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return
        try:
            record = json.loads(last_line)
            self._seq = int(record["seq"])
            self._last_hash = str(record["hash"])
        except (ValueError, KeyError, TypeError):
            # A corrupt tail is left untouched; verification will flag it.
            self._seq = 0
            self._last_hash = GENESIS_HASH

    def _compute_hash(self, body: dict[str, Any]) -> str:
        """Compute the HMAC-SHA256 of a record body."""
        return hmac.new(self._key, _canonical_json(body).encode("utf-8"), hashlib.sha256).hexdigest()

    def log_event(
        self,
        event_type: str,
        actor: str = "system",
        tool: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an authenticated event to the ledger.

        Args:
            event_type: Event category, e.g. ``tool_call`` or ``approval_decided``.
            actor: Human or system originating the event.
            tool: Tool involved, when relevant.
            payload: Structured event details.

        Returns:
            The written record, including its HMAC ``hash``.
        """
        with _LOCK:
            seq = self._seq + 1
            body = {
                "seq": seq,
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": event_type,
                "actor": actor,
                "tool": tool,
                "payload": payload or {},
                "prev_hash": self._last_hash,
            }
            record = {**body, "hash": self._compute_hash(body)}

            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            self._seq = seq
            self._last_hash = record["hash"]
            return record

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read events from the ledger.

        Args:
            limit: Maximum number of most recent events to return.

        Returns:
            Parsed records, oldest first. Malformed lines are represented by an
            ``{"_malformed": ...}`` entry so verification can report them.
        """
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    records.append({"_malformed": str(exc), "raw": line.rstrip("\n")})
        return records[-limit:] if limit is not None else records

    def verify_chain(self) -> dict[str, Any]:
        """Verify integrity, ordering and HMAC signatures of the whole ledger.

        Returns:
            ``{"valid": bool, "events_checked": int, "error": str}``.
        """
        records = self.read_events()
        if not records:
            return {"valid": True, "events_checked": 0, "error": ""}

        previous_hash = GENESIS_HASH
        for index, record in enumerate(records, start=1):
            if "_malformed" in record:
                return {
                    "valid": False,
                    "events_checked": index - 1,
                    "error": f"Malformed JSON at record {index}: {record['_malformed']}",
                }
            if record.get("seq") != index:
                return {
                    "valid": False,
                    "events_checked": index - 1,
                    "error": f"Sequence mismatch at record {index}: got {record.get('seq')}",
                }
            if record.get("prev_hash") != previous_hash:
                return {
                    "valid": False,
                    "events_checked": index - 1,
                    "error": f"Broken chain link at record {index}",
                }

            body = {
                key: value
                for key, value in record.items()
                if key != "hash"
            }
            expected = self._compute_hash(body)
            if not hmac.compare_digest(str(record.get("hash", "")), expected):
                return {
                    "valid": False,
                    "events_checked": index,
                    "error": f"HMAC mismatch at record {index}; the ledger was modified",
                }
            previous_hash = record["hash"]

        return {"valid": True, "events_checked": len(records), "error": ""}

    def reset(self) -> None:
        """Delete the ledger and restart the chain (tests and demo resets only)."""
        with _LOCK:
            if self.path.exists():
                self.path.unlink()
            self._seq = 0
            self._last_hash = GENESIS_HASH


def _build_logger() -> Any:
    """Return a Preloop logger if a compatible SDK exists, else the local fallback."""
    if _PreloopAuditLogger is not None:  # pragma: no cover - improbable today
        try:
            candidate = _PreloopAuditLogger()
            if hasattr(candidate, "log_event") and hasattr(candidate, "verify_chain"):
                return candidate
        except Exception:
            pass
    return LocalAuditLogger()


def get_audit_logger() -> Any:
    """Return the process-wide audit logger singleton."""
    global _LOGGER
    with _LOCK:
        if _LOGGER is None:
            _LOGGER = _build_logger()
        return _LOGGER


def reset_audit_logger(path: str | Path | None = None, key: str | bytes | None = None) -> Any:
    """Replace the audit logger singleton.

    Args:
        path: New ledger path. Defaults to the environment/default location.
        key: New HMAC key. Defaults to ``AUDIT_HMAC_KEY``.

    Returns:
        The new logger instance.
    """
    global _LOGGER
    with _LOCK:
        _LOGGER = LocalAuditLogger(path=path, key=key)
        return _LOGGER


def log_event(
    event_type: str,
    actor: str = "system",
    tool: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an event to the global ledger.

    Args:
        event_type: Event category.
        actor: Human or system originating the event.
        tool: Tool involved, when relevant.
        payload: Structured event details.

    Returns:
        The written audit record.
    """
    return get_audit_logger().log_event(event_type, actor=actor, tool=tool, payload=payload)


def verify_audit_chain() -> dict[str, Any]:
    """Verify the global ledger.

    Returns:
        ``{"valid": bool, "events_checked": int, "error": str}``.
    """
    logger = get_audit_logger()
    return logger.verify_chain()


def get_events(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent audit events for display."""
    return get_audit_logger().read_events(limit=limit)
