"""LangGraph maintenance agent using native LLM function calling.

The graph is deliberately minimal and explicit::

    START -> agent -> (tools -> agent)* -> END

``agent`` calls a chat model with bound tools. ``tools`` is LangGraph's
``ToolNode``. The loop ends when the model returns a message without tool calls.

If ``OPENAI_API_KEY`` is absent (or ``FORCE_FAKE_LLM=1``), a deterministic
scripted chat model reproduces the function-calling protocol. That keeps the
PoC and its tests runnable offline without changing the graph.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import TOOLS
from harness.audit import log_event

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un agent GenAI de maintenance industrielle supervisé.

Méthode obligatoire :
1. Appelle d'abord `query_knowledge_graph` pour collecter le contexte GraphRAG
   (équipement, symptôme, historique, procédure, capteurs). N'invente jamais une
   procédure ou un historique.
2. Appelle ensuite `propose_maintenance_action` avec l'équipement et l'action
   recommandée par le graphe.
3. Appelle `execute_maintenance_action` pour demander l'exécution. Cet appel
   déclenche une approbation humaine : si elle est refusée, n'exécute rien et
   explique-le.
4. Termine par une réponse concise en français : diagnostic, action proposée,
   sources consultées, et statut de l'approbation.

Tu ne dois jamais contourner l'approbation ni prétendre qu'une action a été
exécutée sans le résultat de l'outil d'exécution.
"""

_PROCEDURE_PRIORITY = (
    "REMPLACER_ROULEMENT",
    "REMPLACER_JOINT",
    "GRAISSER_PALIER",
    "REALIGNER_ARBRE",
    "VIDANGER_HUILE",
    "CALIBRER_CAPTEUR",
)
_PROCEDURE_RE = re.compile(r"\b(" + "|".join(_PROCEDURE_PRIORITY) + r")\b")
_EQUIPMENT_RE = re.compile(r"\b([A-Z]-\d{3})\b", re.IGNORECASE)


def _message_text(message: BaseMessage) -> str:
    """Return a message's content as text regardless of its concrete type."""
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


class ScriptedToolCallingModel(BaseChatModel):
    """Deterministic chat model that follows the demo's tool-call sequence.

    This is not a reimplementation of an LLM: it is an offline test double that
    emits the same ``AIMessage(tool_calls=...)`` structures as a real
    function-calling model, so the LangGraph/ToolNode path is exercised exactly
    as in production.
    """

    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain."""
        return "scripted-tool-calling"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Model parameters exposed to LangChain."""
        return {"model": "scripted-tool-calling"}

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolCallingModel:
        """Return itself, mirroring ``bind_tools`` on a real chat model."""
        return self

    def _tool_call(self, name: str, args: dict[str, Any], call_id: str) -> ChatResult:
        """Build a ChatResult containing a single tool call."""
        message = AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Emit the next scripted function call or the final answer."""
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        used_tools = {message.name for message in tool_messages}

        human_messages = [message for message in messages if isinstance(message, HumanMessage)]
        question = _message_text(human_messages[0]) if human_messages else ""

        if "query_knowledge_graph" not in used_tools:
            return self._tool_call(
                "query_knowledge_graph",
                {"question": question},
                "call-query-knowledge-graph",
            )

        query_result = next(
            (message for message in tool_messages if message.name == "query_knowledge_graph"),
            None,
        )
        query_text = _message_text(query_result) if query_result else ""
        equipment_match = _EQUIPMENT_RE.search(query_text)
        equipment_id = equipment_match.group(1).upper() if equipment_match else "P-204"

        procedure_match = _PROCEDURE_RE.search(query_text)
        action = procedure_match.group(1) if procedure_match else "INSPECTION_VIBRATOIRE"

        if "propose_maintenance_action" not in used_tools:
            return self._tool_call(
                "propose_maintenance_action",
                {"equipment_id": equipment_id, "action": action},
                "call-propose-maintenance-action",
            )

        if "execute_maintenance_action" not in used_tools:
            return self._tool_call(
                "execute_maintenance_action",
                {"equipment_id": equipment_id, "action": action},
                "call-execute-maintenance-action",
            )

        execute_result = next(
            (message for message in tool_messages if message.name == "execute_maintenance_action"),
            None,
        )
        execution_text = _message_text(execute_result) if execute_result else ""

        if "ACTION_BLOQUEE" in execution_text:
            answer = (
                f"Diagnostic pour {equipment_id} : {action}. "
                "L'exécution n'a pas été autorisée par l'approbateur humain. "
                f"Détail de la gouvernance : {execution_text}"
            )
        else:
            answer = (
                f"Diagnostic pour {equipment_id} : action {action} recommandée à partir du "
                "chemin GraphRAG Équipement → Symptôme → Historique → Procédure. "
                f"Après approbation humaine, l'action a été exécutée : {execution_text}"
            )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=answer))]
        )


def get_llm() -> BaseChatModel:
    """Return the configured chat model, or the offline scripted fallback.

    Returns:
        A LangChain chat model with function-calling support.
    """
    force_fake = os.getenv("FORCE_FAKE_LLM", "").strip().lower() in {"1", "true", "yes"}
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if force_fake or not api_key:
        logger.info("Using the offline scripted function-calling model")
        return ScriptedToolCallingModel()

    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    except Exception as exc:  # pragma: no cover - optional provider
        logger.warning("OpenAI model unavailable (%s); falling back to scripted model", exc)
        return ScriptedToolCallingModel()


class AgentState(TypedDict):
    """State shared between LangGraph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]


def build_agent_graph(
    llm: BaseChatModel | None = None,
    tools: list[Any] | None = None,
) -> Any:
    """Compile the LangGraph agent.

    Args:
        llm: Optional chat model override (tests/injection).
        tools: Optional governed tools override (tests/injection).

    Returns:
        A compiled LangGraph application.
    """
    model = llm or get_llm()
    tool_list = list(tools) if tools is not None else list(TOOLS)
    model_with_tools = model.bind_tools(tool_list)

    def call_model(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Invoke the model with the system prompt and the running conversation."""
        prompt = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = model_with_tools.invoke(prompt)
        return {"messages": [response]}

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tool_list))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


def run_agent(
    question: str,
    llm: BaseChatModel | None = None,
    tools: list[Any] | None = None,
) -> list[AnyMessage]:
    """Run the maintenance agent to completion.

    Args:
        question: Natural language maintenance question.
        llm: Optional chat model override.
        tools: Optional tool override.

    Returns:
        The complete list of messages exchanged by the graph, ending with the
        agent's final answer.
    """
    graph = build_agent_graph(llm=llm, tools=tools)
    log_event("agent_run_started", actor="agent", payload={"question": question})
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": int(os.getenv("AGENT_RECURSION_LIMIT", "30"))},
        )
    except Exception as exc:
        log_event("agent_run_failed", actor="agent", payload={"question": question, "error": str(exc)})
        raise

    messages = list(result["messages"])
    final = next(
        (message for message in reversed(messages) if isinstance(message, AIMessage) and message.content),
        None,
    )
    log_event(
        "agent_run_completed",
        actor="agent",
        payload={
            "question": question,
            "messages": len(messages),
            "final_answer": _message_text(final) if final else "",
        },
    )
    return messages
