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
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The UI owns the approval loop. Force this mode even if .env was prepared for
# the CLI demo, then import the harness so python-dotenv does not override it.
os.environ["APPROVAL_MODE"] = "streamlit"

from agent.graph import run_agent  # noqa: E402
from graph.queries import get_reasoning_subgraph  # noqa: E402
from harness.approval import (  # noqa: E402
    clear_approvals,
    get_pending_approvals,
    record_decision,
)
from harness.audit import (  # noqa: E402
    LocalAuditLogger,
    create_tampered_copy,
    get_audit_log_path,
    get_events,
    verify_audit_chain,
)
from harness.governance import reset_reasoning_context  # noqa: E402
from ui.visuals import (  # noqa: E402
    build_audit_view,
    build_reasoning_view,
    render_approval_alert,
)

DEMO_QUESTION = "L'équipement P-204 présente une vibration anormale. Que faire ?"

SCENARIOS: dict[str, str] = {
    "P-204 — vibration anormale": DEMO_QUESTION,
    "C-101 — surchauffe": "L'équipement C-101 présente une surchauffe. Que faire ?",
    "M-310 — bruit métallique": "Le moteur M-310 émet un bruit métallique. Que faire ?",
    "V-450 — fuite d'huile": "La vanne V-450 présente une fuite d'huile. Que faire ?",
    "E-120 — débit insuffisant": "L'échangeur E-120 a un débit insuffisant. Que faire ?",
}

_TAMPER_MODES = {
    "Modifier un payload": "modify",
    "Supprimer un événement": "delete",
    "Réordonner deux événements": "reorder",
}


class RunState:
    """Thread-safe state shared between the agent worker and Streamlit reruns."""

    def __init__(self) -> None:
        """Create an idle run state."""
        self.lock = threading.Lock()
        self.running = False
        self.finished = False
        self.messages: list[Any] = []
        self.question = ""
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None


def _execute_agent(question: str, state: RunState) -> None:
    """Run the agent in a worker thread and store its result.

    Args:
        question: Question sent to the agent.
        state: Shared run state updated under a lock.
    """
    def publish(messages: list[Any]) -> None:
        """Publish an intermediate LangGraph snapshot to the UI."""
        with state.lock:
            state.messages = list(messages)

    try:
        messages = run_agent(question, on_update=publish)
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


def _audit_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format audit events for the live data table."""
    rows: list[dict[str, Any]] = []
    for event in reversed(events):
        timestamp = str(event.get("timestamp", "")).replace("T", " ")[:19]
        rows.append(
            {
                "#": event.get("seq"),
                "Horodatage": timestamp,
                "Événement": event.get("event_type"),
                "Acteur": event.get("actor"),
                "Outil": event.get("tool") or "—",
                "Hash": str(event.get("hash", ""))[:16] + "…",
            }
        )
    return rows


def _render_tamper_demo(has_events: bool) -> None:
    """Render the isolated audit-tampering demonstration."""
    st.markdown("#### Preuve d'intégrité : simuler une falsification")
    st.caption(
        "L'altération est appliquée à une **copie isolée** du registre. "
        "Le journal réel n'est jamais modifié."
    )

    mode_label = st.selectbox(
        "Type d'altération",
        list(_TAMPER_MODES),
        key="tamper_mode",
    )
    action_col, clean_col = st.columns(2)

    if action_col.button(
        "Simuler une falsification",
        key="tamper_button",
        disabled=not has_events,
        use_container_width=True,
    ):
        try:
            copy_path = create_tampered_copy(mode=_TAMPER_MODES[mode_label])
            result = LocalAuditLogger(copy_path).verify_chain()
            st.session_state["tamper_result"] = {
                "valid": result["valid"],
                "error": result["error"],
                "events_checked": result["events_checked"],
                "path": str(copy_path),
                "mode": mode_label,
            }
        except Exception as exc:  # pragma: no cover - shown in the UI
            st.session_state["tamper_result"] = {
                "valid": None,
                "error": f"{type(exc).__name__}: {exc}",
                "events_checked": 0,
                "path": "",
                "mode": mode_label,
            }
        st.rerun()

    if clean_col.button(
        "Nettoyer la copie",
        key="tamper_clean",
        use_container_width=True,
    ):
        previous = st.session_state.pop("tamper_result", None) or {}
        previous_path = previous.get("path")
        if previous_path:
            Path(previous_path).unlink(missing_ok=True)
        st.rerun()

    tamper = st.session_state.get("tamper_result")
    if not tamper:
        return

    if tamper["valid"] is False:
        st.error("INTÉGRITÉ COMPROMISE — la falsification est détectée.")
        st.code(
            f"mode={tamper['mode']}\n"
            f"verify_audit_chain(path='{tamper['path']}') -> valid=False\n"
            f"events_checked={tamper['events_checked']}\n"
            f"error={tamper['error']}",
            language="text",
        )
    elif tamper["valid"] is True:
        st.warning(
            "La falsification n'a pas été détectée. Le mode choisi ne casse peut-être "
            "pas la chaîne ; réessayez avec un autre mode."
        )
    else:
        st.error(f"Falsification impossible : {tamper['error']}")


def _render_audit_panel() -> None:
    """Display the live HMAC chain, its events and the tampering demo."""
    verification = verify_audit_chain()
    events = get_events(limit=30)

    if verification["valid"]:
        st.success(
            f"Chaîne d'audit valide — {verification['events_checked']} événement(s) vérifié(s)."
        )
    else:
        st.error(f"Chaîne d'audit invalide : {verification['error']}")

    if events:
        audit_html, audit_height = build_audit_view(events[-6:], verification["valid"])
        components.html(audit_html, height=audit_height, scrolling=False)
    else:
        st.caption("Aucun événement journalisé pour le moment.")

    with st.expander("Événements du registre d'audit", expanded=False):
        if events:
            st.dataframe(
                _audit_rows(events),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"Journal : `{get_audit_log_path()}`")
        else:
            st.caption("Aucun événement journalisé pour le moment.")

    _render_tamper_demo(bool(events))


def _query_completed(messages: list[Any]) -> bool:
    """Return whether the GraphRAG query already produced a tool result."""
    return any(
        isinstance(message, ToolMessage) and message.name == "query_knowledge_graph"
        for message in messages
    )


@st.cache_data(ttl=60, show_spinner=False)
def _load_graph_payload(question: str) -> dict[str, Any]:
    """Fetch and cache the structured subgraph for a question."""
    return get_reasoning_subgraph(question)


def _render_graph_panel(state: RunState, messages: list[Any]) -> dict[str, Any] | None:
    """Render the animated GraphRAG subgraph, or a waiting state."""
    with state.lock:
        question = state.question
        running = state.running

    st.markdown("### Graphe de raisonnement")
    if not question:
        st.caption("Lancez un scénario pour afficher le chemin GraphRAG.")
        return None
    if not _query_completed(messages):
        if running:
            st.caption("En attente du résultat de `query_knowledge_graph`…")
        else:
            st.caption("Aucun chemin GraphRAG chargé pour cette exécution.")
        return None

    try:
        payload = _load_graph_payload(question)
    except Exception as exc:  # pragma: no cover - Neo4j availability
        st.warning(f"Visualisation du graphe indisponible : {exc}")
        return None

    graph_html, graph_height = build_reasoning_view(payload)
    components.html(graph_html, height=graph_height, scrolling=False)
    if payload.get("nodes"):
        st.caption(
            f"{len(payload['nodes'])} nœuds · {len(payload['edges'])} relations · "
            "animation dans l'ordre du raisonnement"
        )
        with st.expander("Requête Cypher exécutée", expanded=False):
            st.code(payload.get("cypher", ""), language="text")
        with st.expander("Propriétés des nœuds consultés", expanded=False):
            st.json(payload.get("nodes", []))
    return payload


def _render_approval_panel(
    pending: list[dict[str, Any]],
    graph_payload: dict[str, Any] | None = None,
) -> None:
    """Display pending approval requests with approve/reject controls."""
    if not pending:
        return

    equipment = (graph_payload or {}).get("equipment") or {}
    criticality = equipment.get("criticality", "inconnue")
    equipment_name = equipment.get("name", "?")
    st.subheader("Décision de supervision")
    st.caption(
        "L'exécution est matériellement bloquée tant que la décision n'est pas enregistrée."
    )
    for request in pending:
        with st.container(border=True):
            st.markdown(f"### {request['equipment_id']} — `{request['action']}`")
            st.caption(
                f"Demande {request['id']} · outil `{request['tool_name']}` · "
                f"politique `{request['policy']}`"
            )

            metric_cols = st.columns(2)
            metric_cols[0].metric("Criticité", criticality)
            metric_cols[1].metric("Statut", "BLOQUÉ")

            st.markdown("**Contexte complet de la décision**")
            st.markdown(
                f"- **Équipement :** `{request['equipment_id']}` — {equipment_name}\n"
                f"- **Action proposée :** `{request['action']}`\n"
                f"- **Politique :** `{request['policy']}`\n"
                f"- **Niveau de criticité :** {criticality}\n"
                f"- **Demande créée le :** {request['created_at']}"
            )
            st.markdown("**Chemin de raisonnement**")
            st.code(request["reasoning_path"] or "(non fourni)", language="text")
            st.markdown(
                "**Sources Neo4j consultées :** "
                + (", ".join(f"`{source}`" for source in request["sources"]) or "aucune")
            )
            st.caption(
                "La décision, l'identité de l'approbateur, l'horodatage et le motif "
                "seront inscrits dans la chaîne d'audit HMAC."
            )

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
        st.caption("Prêt à lancer un scénario supervisé.")

    pending = get_pending_approvals()
    if pending:
        st.markdown(render_approval_alert(pending), unsafe_allow_html=True)
        graph_col, decision_col = st.columns([1.55, 1], gap="large")
        with graph_col:
            graph_payload = _render_graph_panel(state, messages)
        with decision_col:
            _render_approval_panel(pending, graph_payload)
    else:
        _render_graph_panel(state, messages)

    if error:
        st.error(f"Erreur d'exécution de l'agent : {error}")

    if messages:
        with st.expander("Trace complète de l'agent", expanded=not pending):
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
        st.markdown("**Preuves AI Act**")
        st.markdown("- `Art. 12` — journalisation\n- `Art. 14` — supervision humaine")
        st.divider()
        st.markdown(
            "**Parcours de démonstration**\n"
            "1. Choisir un scénario et lancer\n"
            "2. Lire le chemin GraphRAG animé\n"
            "3. Approuver ou rejeter\n"
            "4. Vérifier la chaîne d'audit\n"
            "5. Simuler une falsification"
        )
        if st.button("Réinitialiser les demandes", disabled=running):
            clear_approvals()
            reset_reasoning_context()
            st.session_state.run_state = RunState()
            st.session_state.agent_thread = None
            st.session_state.pop("tamper_result", None)
            st.rerun()

    scenario = st.selectbox(
        "Scénario de démonstration",
        list(SCENARIOS),
        key="scenario",
        help="Chaque scénario utilise le même pipeline et la même politique d'approbation.",
    )
    question = st.text_input(
        "Question de maintenance",
        value=SCENARIOS[scenario],
        key=f"question_{scenario}",
    )

    if st.button("Lancer la démonstration", type="primary", disabled=running):
        clear_approvals()
        reset_reasoning_context()
        st.session_state.pop("tamper_result", None)
        new_state = RunState()
        new_state.question = question.strip()
        with new_state.lock:
            new_state.running = True
        thread = threading.Thread(
            target=_execute_agent,
            args=(question.strip(), new_state),
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
