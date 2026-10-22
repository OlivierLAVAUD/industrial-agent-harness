"""Headless CLI demonstration of the full supervised maintenance scenario.

Examples::

    python scripts/demo.py --auto-approve
    python scripts/demo.py --reject
    python scripts/demo.py                 # interactive CLI approval

The script runs the LangGraph agent, prints its trace, then verifies the HMAC
audit chain and exits non-zero if the ledger is invalid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_QUESTION = "L'équipement P-204 présente une vibration anormale. Que faire ?"


def _render_messages(messages: list) -> None:
    """Print a compact trace of the agent conversation."""
    for message in messages:
        message_type = type(message).__name__
        content = getattr(message, "content", "")
        if message_type == "AIMessage":
            for call in getattr(message, "tool_calls", []) or []:
                print(f"[AGENT] -> {call.get('name')}({json.dumps(call.get('args', {}), ensure_ascii=False)})")
            if content:
                print(f"[AGENT] {content}")
        elif message_type == "ToolMessage":
            print(f"[TOOL:{getattr(message, 'name', '?')}] {content}")
        elif message_type == "HumanMessage":
            print(f"[HUMAN] {content}")


def main() -> int:
    """Run the headless demo and return a process exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Industrial agent harness headless demo")
    parser.add_argument("--question", default=DEMO_QUESTION, help="Maintenance question")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve the human approval request automatically",
    )
    parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject the human approval request automatically",
    )
    parser.add_argument(
        "--reset-audit",
        action="store_true",
        help="Delete the audit ledger before the run (demo only)",
    )
    args = parser.parse_args()

    if args.auto_approve and args.reject:
        parser.error("--auto-approve and --reject are mutually exclusive")

    os.environ.setdefault("APPROVAL_MODE", "cli")
    if args.auto_approve:
        os.environ["APPROVAL_MODE"] = "auto_approve"
    elif args.reject:
        os.environ["APPROVAL_MODE"] = "auto_reject"

    from agent.graph import run_agent
    from harness.audit import reset_audit_logger, verify_audit_chain

    if args.reset_audit:
        reset_audit_logger().reset()

    print(f"Question : {args.question}")
    print(f"Mode d'approbation : {os.environ['APPROVAL_MODE']}")
    print("-" * 78)

    try:
        messages = run_agent(args.question)
    except Exception as exc:
        print(f"ECHEC DE L'AGENT : {type(exc).__name__}: {exc}")
        return 2

    _render_messages(messages)

    verification = verify_audit_chain()
    print("-" * 78)
    print(f"Audit : valid={verification['valid']} events={verification['events_checked']}")
    if verification["error"]:
        print(f"Audit error : {verification['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
