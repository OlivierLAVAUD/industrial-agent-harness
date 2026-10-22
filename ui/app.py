"""Streamlit supervision console for the industrial maintenance agent.

The agent runs in a background thread inside the Streamlit process. When the
governed execution tool requests approval, the same process stores the request
in memory and this page displays it. The human decision unblocks the agent
thread, so the full loop remains observable without any external queue.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The UI owns the approval loop. Force this mode even if .env was prepared for
# the CLI demo, then import the harness so python-dotenv does not override it.
os.environ["APPROVAL_MODE"] = "streamlit"

from agent.graph import run_agent  # noqa: E402
from harness.approval import (  # noqa: E402
    clear_approvals,
    get_pending_approvals,
    record_decision,
)
from harness.audit import get_events, verify_audit_chain  # noqa: E402
from harness.governance import reset_reasoning_context  # noqa: E402

DEMO_QUESTION = "L'équipement P-204 présente une vibration anormale. Que faire ?"


class RunState:
    """Thread-safe state shared between the agent worker and Streamlit reruns."""

    def __init__(self) -> None:
        """Create an idle run state."""
        self.lock = threading.Lock()
        self.running = False
        self.finished = False
        self.messages: list[Any] = []
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None


def _execute_agent(question: str, state: RunState) -> None:
    """Run the agent in a worker thread and store its result.

    Args:
        question: Question sent to the agent.
        state: Shared run state updated under a lock.
    """
    try:
        messages = run_agent(question)
        with state.lock:
            state.messages = list(messages)
    except Exception as exc:  # pragma: no cover - displayed in the UI
        with state.lock:
            state.error = f"{type(exc).__name__}: {exc}"
    finally:
        with state.lock:
            state.running = False
            state.finished = True
            state.finished_at = time.time()


def _message_content(message: Any) -> str:
    """Render a LangChain message content as text."""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _render_messages(messages: list[Any]) -> None:
    """Render the agent conversation, including tool calls and results."""
    for message in messages:
        if isinstance(message, HumanMessage):
            st.markdown(f"**Question :** {_message_content(message)}")
        elif isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                st.markdown(f"**Appel d'outil — `{call.get('name')}`**")
                st.code(json.dumps(call.get("args", {}), ensure_ascii=False, indent=2), language="json")
            content = _message_content(message)
            if content:
                st.markdown("**Réponse de l'agent :**")
                st.info(content)
        elif isinstance(message, ToolMessage):
            with st.expander(f"Résultat outil — {message.name}", expanded=False):
                st.code(_message_content(message), language="text")


def _render_audit_panel() -> None:
    """Display the audit-chain verification and the latest events."""
    verification = verify_audit_chain()
    if verification["valid"]:
        st.success(
            f"Chaîne d'audit valide — {verification['events_checked']} événement(s) vérifié(s)."
        )
    else:
        st.error(f"Chaîne d'audit invalide : {verification['error']}")

    events = get_events(limit=25)
    with st.expander("Derniers événements du registre d'audit", expanded=False):
        if events:
            st.code(json.dumps(events, indent=2, ensure_ascii=False), language="json")
        else:
            st.caption("Aucun événement journalisé pour le moment.")


def _render_approval_panel(pending: list[dict[str, Any]]) -> None:
    """Display pending approval requests with approve/reject controls."""
    if not pending:
        return

    st.subheader("Approbation humaine requise")
    for request in pending:
        with st.container(border=True):
            st.markdown(
                f"### {request['equipment_id']} — `{request['action']}`"
            )
            st.caption(
                f"Demande {request['id']} · outil `{request['tool_name']}` · "
                f"politique `{request['policy']}`"
            )
            st.markdown("**Chemin de raisonnement**")
            st.code(request["reasoning_path"] or "(non fourni)", language="text")
            st.markdown("**Sources consultées :** " + (", ".join(request["sources"]) or "aucune"))
            st.markdown(f"**Créée le :** {request['created_at']}")

            default_approver = os.getenv("APPROVER_ID", "supervisor-demo")
            approver_id = st.text_input(
                "Identité de l'approbateur (simulée)",
                value=default_approver,
                key=f"approver_{request['id']}",
            )
            reason = st.text_input(
                "Raison (obligatoire pour un rejet)",
                key=f"reason_{request['id']}",
            )

            approve_col, reject_col = st.columns(2)
            if approve_col.button(
                "Approuver",
                type="primary",
                key=f"approve_{request['id']}",
                use_container_width=True,
            ):
                record_decision(
                    request["id"],
                    "approved",
                    approver_id or default_approver,
                    "Approuvé depuis la console Streamlit.",
                )
                st.rerun()

            if reject_col.button(
                "Rejeter",
                key=f"reject_{request['id']}",
                use_container_width=True,
            ):
                if not reason.strip():
                    st.error("Une raison est requise pour rejeter l'action.")
                else:
                    record_decision(
                        request["id"],
                        "rejected",
                        approver_id or default_approver,
                        reason.strip(),
                    )
                    st.rerun()


def _render_dynamic_area(state: RunState) -> None:
    """Render status, approvals, agent trace and audit for the current run."""
    with state.lock:
        running = state.running
        finished = state.finished
        messages = list(state.messages)
        error = state.error

    if running:
        st.info("L'agent raisonne et peut demander une approbation humaine…")
    elif finished:
        st.success("Exécution terminée.")
    else:
        st.caption("Prêt à exécuter le scénario P-204.")

    pending = get_pending_approvals()
    _render_approval_panel(pending)

    if error:
        st.error(f"Erreur d'exécution de l'agent : {error}")

    if messages:
        with st.expander("Trace complète de l'agent", expanded=True):
            _render_messages(messages)

    st.divider()
    _render_audit_panel()


@st.fragment(run_every=2)
def _live_area(state: RunState) -> None:
    """Poll in the same session while the agent or an approval is active."""
    _render_dynamic_area(state)
    with state.lock:
        finished = state.finished
        running = state.running
    if finished and not running:
        st.rerun()


def _static_area(state: RunState) -> None:
    """Render the finished/idle area without polling."""
    _render_dynamic_area(state)


def main() -> None:
    """Render the Streamlit supervision console."""
    st.set_page_config(
        page_title="Industrial Agent Harness",
        layout="wide",
    )

    st.title("Industrial Agent Harness")
    st.caption(
        "Agent GenAI de maintenance supervisé — GraphRAG Neo4j, LangGraph, "
        "gouvernance des outils et registre d'audit HMAC chaîné."
    )

    if "run_state" not in st.session_state:
        st.session_state.run_state = RunState()
    if "agent_thread" not in st.session_state:
        st.session_state.agent_thread = None

    state: RunState = st.session_state.run_state
    with state.lock:
        running = state.running

    with st.sidebar:
        st.header("Configuration")
        llm_mode = "OpenAI" if os.getenv("OPENAI_API_KEY", "").strip() else "Modèle scripté (hors-ligne)"
        st.markdown(f"**LLM :** {llm_mode}")
        st.markdown("**Mode d'approbation :** Streamlit (même processus)")
        st.markdown(f"**Approbateur :** `{os.getenv('APPROVER_ID', 'supervisor-demo')}`")
        st.markdown(f"**Audit :** `{os.getenv('AUDIT_LOG_PATH', 'var/audit_chain.jsonl')}`")
        preloop_mode = (
            "Preloop réel configuré"
            if os.getenv("PRELOOP_URL") and os.getenv("PRELOOP_TOKEN")
            else "Fallback local gouverné"
        )
        st.markdown(f"**Preloop :** {preloop_mode}")
        st.divider()
        st.markdown(
            "**Parcours de démonstration**\n"
            "1. Lancer la démonstration\n"
            "2. Lire le chemin GraphRAG\n"
            "3. Approuver ou rejeter\n"
            "4. Vérifier la chaîne d'audit"
        )
        if st.button("Réinitialiser les demandes", disabled=running):
            clear_approvals()
            reset_reasoning_context()
            st.session_state.run_state = RunState()
            st.session_state.agent_thread = None
            st.rerun()

    question = st.text_input("Question de maintenance", value=DEMO_QUESTION)

    if st.button("Lancer la démonstration", type="primary", disabled=running):
        clear_approvals()
        reset_reasoning_context()
        new_state = RunState()
        with new_state.lock:
            new_state.running = True
        thread = threading.Thread(
            target=_execute_agent,
            args=(question, new_state),
            name="industrial-agent-worker",
            daemon=True,
        )
        st.session_state.run_state = new_state
        st.session_state.agent_thread = thread
        thread.start()
        st.rerun()

    if running or get_pending_approvals():
        _live_area(state)
    else:
        _static_area(state)


if __name__ == "__main__":
    main()
