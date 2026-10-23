"""GraphRAG queries over the industrial maintenance knowledge graph.

The graph intentionally models a small but realistic multi-hop reasoning path::

    Equipment -[PRÉSENTE]-> Symptom -[RECOMMANDE]-> Procedure
    MaintenanceHistory -[A_CONSTATÉ]-> Symptom
    Sensor -[MESURE]-> Equipment

``get_reasoning_path`` returns structured rows, ``format_reasoning_path``
renders a human readable trace and ``query_knowledge_graph`` turns a natural
language question into GraphRAG context for the LLM.
"""

from __future__ import annotations

import re
from typing import Any

from neo4j.graph import Node

from .connection import get_driver

EQUIPMENT_RE = re.compile(r"\b([A-Z]-\d{3})\b", re.IGNORECASE)

_SYMPTOM_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("vibr", "vibration"),
    ("surchauff", "surchauffe"),
    ("tempér", "surchauffe"),
    ("temper", "surchauffe"),
    ("bruit", "bruit"),
    ("fuite", "fuite"),
    ("pression", "pression"),
    ("débit", "débit"),
    ("debit", "débit"),
)

_REASONING_PATH_QUERY = """
MATCH (e:Equipment {id: $equipment_id})
OPTIONAL MATCH (e)-[:`PRÉSENTE`]->(s:Symptom)
WHERE $symptom IS NULL
   OR toLower(s.name) CONTAINS toLower($symptom)
   OR toLower(coalesce(s.description, '')) CONTAINS toLower($symptom)
OPTIONAL MATCH (s)-[:`RECOMMANDE`]->(p:Procedure)
WITH e, s, collect(DISTINCT p) AS procedures
OPTIONAL MATCH (h:MaintenanceHistory)-[:`A_CONSTATÉ`]->(s)
WHERE h.equipment_id = e.id
WITH e, s, procedures, collect(DISTINCT h) AS histories
OPTIONAL MATCH (c:Sensor)-[:`MESURE`]->(e)
RETURN e, s, procedures, histories, collect(DISTINCT c) AS sensors
ORDER BY s.id
"""

# Public copy used by the UI to display the exact Cypher query.
REASONING_PATH_CYPHER = _REASONING_PATH_QUERY.strip()

_EQUIPMENT_EXISTS_QUERY = """
MATCH (e:Equipment {id: $equipment_id})
RETURN count(e) > 0 AS exists
"""

# Domain ordering used to highlight the most direct corrective procedure first.
_PROCEDURE_PRIORITY = {
    "REMPLACER_ROULEMENT": 0,
    "REMPLACER_JOINT": 1,
    "GRAISSER_PALIER": 2,
    "REALIGNER_ARBRE": 3,
    "VIDANGER_HUILE": 4,
    "CALIBRER_CAPTEUR": 5,
}


def _node_dict(node: Node | None) -> dict[str, Any] | None:
    """Convert a Neo4j node to a plain dictionary, preserving its label.

    Args:
        node: Node returned by the driver, or ``None``.

    Returns:
        A dictionary of properties plus a ``label`` key, or ``None``.
    """
    if node is None:
        return None
    data = dict(node)
    labels = list(node.labels)
    data["label"] = labels[0] if labels else ""
    return data


def _relation_label(source: str, rel_type: str, target: str) -> str:
    """Render a compact, human readable relationship string."""
    return f"{source} -[{rel_type}]-> {target}"


def _dedupe(items: list[str]) -> list[str]:
    """Return items without duplicates while preserving their order."""
    return list(dict.fromkeys(items))


def equipment_exists(equipment_id: str) -> bool:
    """Return whether an equipment node exists in the graph.

    Args:
        equipment_id: Equipment identifier, e.g. ``P-204``.

    Returns:
        ``True`` when the equipment is present.
    """
    with get_driver().session() as session:
        row = session.run(_EQUIPMENT_EXISTS_QUERY, equipment_id=equipment_id).single()
    return bool(row and row["exists"])


def get_reasoning_path(
    equipment_id: str,
    symptom_description: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the multi-hop GraphRAG reasoning path for an equipment.

    The query walks ``Equipment -> Symptom -> Procedure`` and enriches each
    symptom with maintenance histories and the sensors attached to the
    equipment. One structured dictionary is returned per matching symptom.

    Args:
        equipment_id: Equipment identifier, e.g. ``P-204``.
        symptom_description: Optional symptom keyword filtered case-insensitively
            against symptom name and description.

    Returns:
        A list of dictionaries with the keys ``equipment``, ``symptom``,
        ``procedures``, ``histories``, ``sensors`` and ``relations``. The list is
        empty when the equipment does not exist or no symptom matches.
    """
    normalized_id = equipment_id.strip().upper()
    symptom = symptom_description.strip() if symptom_description else None

    with get_driver().session() as session:
        records = list(
            session.run(
                _REASONING_PATH_QUERY,
                equipment_id=normalized_id,
                symptom=symptom,
            )
        )

    paths: list[dict[str, Any]] = []
    for record in records:
        equipment = _node_dict(record["e"])
        symptom_node = _node_dict(record["s"])
        if equipment is None or symptom_node is None:
            continue

        procedures = sorted(
            (_node_dict(node) for node in record["procedures"] if node is not None),
            key=lambda item: (
                _PROCEDURE_PRIORITY.get(item.get("code", ""), 99),
                item.get("code", ""),
            ),
        )
        histories = sorted(
            (_node_dict(node) for node in record["histories"] if node is not None),
            key=lambda item: item.get("date", ""),
            reverse=True,
        )
        sensors = sorted(
            (_node_dict(node) for node in record["sensors"] if node is not None),
            key=lambda item: item.get("id", ""),
        )

        relations: list[str] = [
            _relation_label(equipment["id"], "PRÉSENTE", symptom_node["id"])
        ]
        relations.extend(
            _relation_label(symptom_node["id"], "RECOMMANDE", procedure["id"])
            for procedure in procedures
        )
        relations.extend(
            _relation_label(history["id"], "A_CONSTATÉ", symptom_node["id"])
            for history in histories
        )
        relations.extend(
            _relation_label(sensor["id"], "MESURE", equipment["id"]) for sensor in sensors
        )

        paths.append(
            {
                "equipment": equipment,
                "symptom": symptom_node,
                "procedures": procedures,
                "histories": histories,
                "sensors": sensors,
                "relations": _dedupe(relations),
            }
        )

    return paths


def format_reasoning_path(paths: list[dict[str, Any]]) -> str:
    """Format structured reasoning paths as a readable trace.

    Example:
        ``P-204 (Pompe centrifuge) → vibration anormale → historique 2024-03-12
        → procédure REMPLACER_ROULEMENT``

    Args:
        paths: Output of :func:`get_reasoning_path`.

    Returns:
        One line per symptom, or an explanatory message when the list is empty.
    """
    if not paths:
        return "Aucun chemin de raisonnement trouvé dans le graphe."

    lines: list[str] = []
    for path in paths:
        equipment = path.get("equipment") or {}
        symptom = path.get("symptom") or {}
        if not equipment or not symptom:
            continue

        segments = [f"{equipment.get('id', '?')} ({equipment.get('name', '?')})"]
        segments.append(symptom.get("name", "symptôme inconnu"))

        histories = path.get("histories") or []
        if histories:
            segments.append(f"historique {histories[0].get('date', 'date inconnue')}")

        procedures = path.get("procedures") or []
        if procedures:
            codes = [procedure.get("code", "?") for procedure in procedures]
            if len(codes) == 1:
                segments.append(f"procédure {codes[0]}")
            else:
                segments.append(f"procédures {', '.join(codes)}")

        lines.append(" → ".join(segments))

    return "\n".join(lines) if lines else "Aucun chemin de raisonnement exploitable."


def collect_sources(paths: list[dict[str, Any]]) -> list[str]:
    """Collect the identifiers of every node consulted for a reasoning path.

    Args:
        paths: Output of :func:`get_reasoning_path`.

    Returns:
        Ordered, de-duplicated source identifiers.
    """
    sources: list[str] = []
    for path in paths:
        equipment = path.get("equipment") or {}
        symptom = path.get("symptom") or {}
        if equipment.get("id"):
            sources.append(str(equipment["id"]))
        if symptom.get("id"):
            sources.append(str(symptom["id"]))
        sources.extend(
            str(procedure["id"]) for procedure in path.get("procedures", []) if procedure.get("id")
        )
        sources.extend(
            str(history["id"]) for history in path.get("histories", []) if history.get("id")
        )
        sources.extend(
            str(sensor["id"]) for sensor in path.get("sensors", []) if sensor.get("id")
        )
    return _dedupe(sources)


def extract_equipment_id(question: str) -> str | None:
    """Extract an equipment identifier such as ``P-204`` from a question.

    Args:
        question: Natural language question.

    Returns:
        The upper-cased identifier, or ``None``.
    """
    match = EQUIPMENT_RE.search(question)
    return match.group(1).upper() if match else None


def extract_symptom(question: str) -> str | None:
    """Extract a symptom keyword from a French natural language question.

    Args:
        question: Natural language question.

    Returns:
        A canonical symptom keyword, or ``None`` when nothing matches.
    """
    lowered = question.lower()
    for keyword, canonical in _SYMPTOM_KEYWORDS:
        if keyword in lowered:
            return canonical
    return None


def _format_context(equipment_id: str, paths: list[dict[str, Any]]) -> str:
    """Build the textual GraphRAG context injected into the agent conversation."""
    equipment = paths[0]["equipment"]
    lines = [
        f"Contexte GraphRAG — Équipement {equipment['id']} ({equipment.get('name', '?')})",
        f"Site : {equipment.get('site', 'inconnu')} | Criticité : {equipment.get('criticality', 'inconnue')}",
        "",
        "Chemin de raisonnement :",
        format_reasoning_path(paths),
        "",
        "Procédures recommandées :",
    ]

    seen_procedures: set[str] = set()
    for path in paths:
        for procedure in path.get("procedures", []):
            code = procedure.get("code", "?")
            if code in seen_procedures:
                continue
            seen_procedures.add(code)
            lines.append(
                f"- {code} — {procedure.get('name', '?')} "
                f"(durée {procedure.get('duration_min', '?')} min)"
            )
    if not seen_procedures:
        lines.append("- Aucune procédure recommandée.")

    lines.extend(["", "Historiques de maintenance :"])
    seen_histories: set[str] = set()
    for path in paths:
        for history in path.get("histories", []):
            history_id = history.get("id", "?")
            if history_id in seen_histories:
                continue
            seen_histories.add(history_id)
            lines.append(
                f"- {history.get('date', '?')} ({history_id}) : "
                f"{history.get('description', '?')} — technicien {history.get('technician', '?')}"
            )
    if not seen_histories:
        lines.append("- Aucun historique pour ce symptôme.")

    lines.extend(["", "Capteurs associés :"])
    seen_sensors: set[str] = set()
    for path in paths:
        for sensor in path.get("sensors", []):
            sensor_id = sensor.get("id", "?")
            if sensor_id in seen_sensors:
                continue
            seen_sensors.add(sensor_id)
            lines.append(
                f"- {sensor_id} — {sensor.get('name', '?')} "
                f"({sensor.get('unit', '?')}, dernière valeur {sensor.get('last_value', '?')})"
            )
    if not seen_sensors:
        lines.append("- Aucun capteur associé.")

    lines.extend(["", f"Sources : {', '.join(collect_sources(paths))}"])
    return "\n".join(lines)


def query_knowledge_graph_with_paths(question: str) -> tuple[str, list[dict[str, Any]], str | None]:
    """Query the graph and return both formatted context and structured paths.

    Args:
        question: Natural language maintenance question.

    Returns:
        A tuple ``(context, paths, equipment_id)``. ``context`` is always a
        human readable string suitable for the LLM; ``paths`` is empty when no
        reasoning path was found.
    """
    equipment_id = extract_equipment_id(question)
    if not equipment_id:
        return (
            "Aucun identifiant d'équipement (format P-###) n'a été trouvé dans la question. "
            "Précisez l'équipement concerné, par exemple P-204.",
            [],
            None,
        )

    symptom = extract_symptom(question)
    paths = get_reasoning_path(equipment_id, symptom)
    if not paths:
        if equipment_exists(equipment_id):
            suffix = f" pour le symptôme « {symptom} »" if symptom else ""
            return (
                f"L'équipement {equipment_id} existe, mais aucun symptôme correspondant{suffix} "
                "n'est présent dans le graphe.",
                [],
                equipment_id,
            )
        return (
            f"L'équipement {equipment_id} est introuvable dans le graphe de connaissance.",
            [],
            equipment_id,
        )

    return _format_context(equipment_id, paths), paths, equipment_id


def get_reasoning_subgraph(question: str) -> dict[str, Any]:
    """Build a structured subgraph payload for the reasoning visualization.

    The payload contains plain dictionaries only, so it can be cached and
    serialized by Streamlit. Nodes keep their business properties while edges
    carry a display label (``PRÉSENTE``, ``RECOMMANDE``, ``A_CONSTATÉ``,
    ``MESURE``) for the UI.

    Args:
        question: Natural language maintenance question.

    Returns:
        A dictionary with ``equipment``, ``paths``, ``nodes``, ``edges``,
        ``context``, ``equipment_id``, ``path_count`` and ``cypher``.
    """
    context, paths, equipment_id = query_knowledge_graph_with_paths(question)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node: dict[str, Any], node_type: str, subtitle: str) -> None:
        """Register a node once, keeping the richest display subtitle."""
        node_id = str(node.get("id") or "")
        if not node_id or node_id in nodes:
            return
        nodes[node_id] = {
            "id": node_id,
            "label": node_id,
            "type": node_type,
            "subtitle": subtitle,
            "properties": node,
        }

    def add_edge(source: Any, target: Any, relation: str) -> None:
        """Register a de-duplicated edge between two known node ids."""
        source_id = str(source or "")
        target_id = str(target or "")
        if not source_id or not target_id:
            return
        key = (source_id, target_id, relation)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": source_id, "target": target_id, "label": relation})

    equipment: dict[str, Any] | None = None
    if paths:
        equipment = paths[0].get("equipment") or {}
        add_node(
            equipment,
            "equipment",
            f"{equipment.get('name', '?')} · criticité {equipment.get('criticality', '?')}",
        )

        for path in paths:
            symptom = path.get("symptom") or {}
            add_node(
                symptom,
                "symptom",
                f"{symptom.get('name', '?')} · sévérité {symptom.get('severity', '?')}",
            )
            add_edge(equipment.get("id"), symptom.get("id"), "PRÉSENTE")

            for procedure in path.get("procedures", []):
                add_node(
                    procedure,
                    "procedure",
                    f"{procedure.get('code', '?')} · {procedure.get('duration_min', '?')} min",
                )
                add_edge(symptom.get("id"), procedure.get("id"), "RECOMMANDE")

            for history in path.get("histories", []):
                add_node(
                    history,
                    "history",
                    f"{history.get('date', '?')} · {history.get('technician', '?')}",
                )
                add_edge(history.get("id"), symptom.get("id"), "A_CONSTATÉ")

            for sensor in path.get("sensors", []):
                add_node(
                    sensor,
                    "sensor",
                    f"{sensor.get('name', '?')} · {sensor.get('last_value', '?')} "
                    f"{sensor.get('unit', '')}".strip(),
                )
                add_edge(sensor.get("id"), equipment.get("id"), "MESURE")

    return {
        "equipment": equipment,
        "equipment_id": equipment_id,
        "paths": paths,
        "nodes": list(nodes.values()),
        "edges": edges,
        "context": context,
        "path_count": len(paths),
        "cypher": REASONING_PATH_CYPHER,
    }


def query_knowledge_graph(question: str) -> str:
    """Return GraphRAG context for a natural language maintenance question.

    The function extracts an equipment identifier (``P-\\d+``), optionally a
    symptom keyword, then walks the graph and formats the result.

    Args:
        question: Natural language maintenance question.

    Returns:
        A formatted context string. Errors and empty results are returned as
        explicit messages so the agent can recover gracefully.
    """
    try:
        context, _paths, _equipment_id = query_knowledge_graph_with_paths(question)
        return context
    except Exception as exc:  # pragma: no cover - depends on external Neo4j
        return f"Erreur lors de l'interrogation du graphe de connaissance : {exc}"
