"""LangChain tools exposed to the maintenance agent.

The three business tools are intentionally free of any policy or approval
logic. Each one is wrapped by :func:`harness.governance.govern_tool`, which
applies Preloop (when configured) or the explicit local fallback.
"""

from __future__ import annotations

import uuid

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from graph.queries import (
    collect_sources,
    format_reasoning_path,
    query_knowledge_graph_with_paths,
)
from harness.governance import govern_tool, update_reasoning_context


class QueryKnowledgeGraphInput(BaseModel):
    """Input schema for the GraphRAG query tool."""

    question: str = Field(
        ...,
        description="Question en langage naturel, contenant si possible l'identifiant d'équipement (ex. P-204).",
    )


class ProposeMaintenanceActionInput(BaseModel):
    """Input schema for the maintenance proposal tool."""

    equipment_id: str = Field(..., description="Identifiant de l'équipement, ex. P-204.")
    action: str = Field(
        ...,
        description="Code d'action de maintenance, ex. REMPLACER_ROULEMENT.",
    )


class ExecuteMaintenanceActionInput(BaseModel):
    """Input schema for the maintenance execution tool."""

    equipment_id: str = Field(..., description="Identifiant de l'équipement, ex. P-204.")
    action: str = Field(
        ...,
        description="Code d'action de maintenance à exécuter après approbation humaine.",
    )


class QueryKnowledgeGraphTool(BaseTool):
    """Read-only multi-hop query over the industrial knowledge graph."""

    name: str = "query_knowledge_graph"
    description: str = (
        "Interroge le graphe de connaissance de maintenance (GraphRAG). "
        "Retourne le chemin Équipement → Symptôme → Historique → Procédure, "
        "ainsi que les capteurs associés. À appeler en premier."
    )
    args_schema: type[BaseModel] = QueryKnowledgeGraphInput

    def _run(self, question: str) -> str:
        """Query Neo4j and store the reasoning context for later approval."""
        try:
            context, paths, equipment_id = query_knowledge_graph_with_paths(question)
        except Exception as exc:  # pragma: no cover - Neo4j availability
            return f"ERREUR_GRAPHE: impossible d'interroger le graphe de connaissance ({exc})."

        if paths:
            update_reasoning_context(
                reasoning_path=format_reasoning_path(paths),
                sources=collect_sources(paths),
                equipment_id=equipment_id or "",
            )
        elif equipment_id:
            update_reasoning_context(equipment_id=equipment_id, sources=[equipment_id])
        return context


class ProposeMaintenanceActionTool(BaseTool):
    """Record a maintenance proposal without executing it."""

    name: str = "propose_maintenance_action"
    description: str = (
        "Propose une action de maintenance pour un équipement, sans l'exécuter. "
        "À appeler après query_knowledge_graph pour formaliser la recommandation."
    )
    args_schema: type[BaseModel] = ProposeMaintenanceActionInput

    def _run(self, equipment_id: str, action: str) -> str:
        """Return an acknowledgement and enrich the approval context."""
        normalized_equipment = equipment_id.strip().upper()
        normalized_action = action.strip().upper()
        update_reasoning_context(equipment_id=normalized_equipment, action=normalized_action)
        return (
            f"PROPOSITION_ENREGISTREE: action '{normalized_action}' proposée pour "
            f"{normalized_equipment}. Aucune exécution effectuée à ce stade."
        )


class ExecuteMaintenanceActionTool(BaseTool):
    """Simulated maintenance execution, only reachable through governance."""

    name: str = "execute_maintenance_action"
    description: str = (
        "Exécute l'action de maintenance proposée. Cet outil est soumis à une "
        "approbation humaine obligatoire : le harness bloque l'exécution tant que "
        "l'approbateur n'a pas validé la demande."
    )
    args_schema: type[BaseModel] = ExecuteMaintenanceActionInput

    def _run(self, equipment_id: str, action: str) -> str:
        """Simulate a maintenance execution after the governance layer allowed it."""
        normalized_equipment = equipment_id.strip().upper()
        normalized_action = action.strip().upper()
        ticket = f"MNT-{uuid.uuid4().hex[:8].upper()}"
        return (
            f"EXECUTION_SIMULEE: action '{normalized_action}' réalisée sur "
            f"{normalized_equipment}. Ticket {ticket}. Intervention clôturée."
        )


_RAW_TOOLS: list[BaseTool] = [
    QueryKnowledgeGraphTool(),
    ProposeMaintenanceActionTool(),
    ExecuteMaintenanceActionTool(),
]

# Only the governed tools are exposed to the agent and to external callers.
TOOLS: list[BaseTool] = [govern_tool(tool) for tool in _RAW_TOOLS]
