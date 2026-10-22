"""Tests for the GraphRAG knowledge layer."""

from __future__ import annotations

from typing import Any

import pytest

from graph.queries import (
    collect_sources,
    extract_equipment_id,
    extract_symptom,
    format_reasoning_path,
    get_reasoning_path,
    query_knowledge_graph,
)
from graph.seed_graph import seed


def _synthetic_path() -> dict[str, Any]:
    """Return a representative structured reasoning path."""
    return {
        "equipment": {"id": "P-204", "name": "Pompe centrifuge"},
        "symptom": {"id": "S-VIB-01", "name": "vibration anormale"},
        "procedures": [
            {"id": "PR-ROUL-01", "code": "REMPLACER_ROULEMENT"},
            {"id": "PR-ALIGN-01", "code": "REALIGNER_ARBRE"},
        ],
        "histories": [{"id": "H-2024-03-12", "date": "2024-03-12"}],
        "sensors": [{"id": "CAP-VIB-01", "name": "Capteur de vibration"}],
        "relations": [
            "P-204 -[PRÉSENTE]-> S-VIB-01",
            "S-VIB-01 -[RECOMMANDE]-> PR-ROUL-01",
        ],
    }


def test_format_reasoning_path_is_readable() -> None:
    """The formatted path matches the expected diagnostic narrative."""
    formatted = format_reasoning_path([_synthetic_path()])

    assert formatted.startswith("P-204 (Pompe centrifuge) → vibration anormale")
    assert "historique 2024-03-12" in formatted
    assert "REMPLACER_ROULEMENT" in formatted
    assert "REALIGNER_ARBRE" in formatted


def test_format_reasoning_path_empty() -> None:
    """An empty result is explicit rather than silently blank."""
    assert "Aucun chemin" in format_reasoning_path([])


def test_extract_equipment_and_symptom() -> None:
    """Natural language extraction finds the equipment and symptom."""
    question = "L'équipement P-204 présente une vibration anormale. Que faire ?"
    assert extract_equipment_id(question) == "P-204"
    assert extract_symptom(question) == "vibration"
    assert extract_equipment_id("Le compresseur C-101 surchauffe") == "C-101"
    assert extract_equipment_id("Pas d'identifiant ici") is None


def test_collect_sources_deduplicates() -> None:
    """Sources are ordered and de-duplicated."""
    sources = collect_sources([_synthetic_path(), _synthetic_path()])

    assert sources == [
        "P-204",
        "S-VIB-01",
        "PR-ROUL-01",
        "PR-ALIGN-01",
        "H-2024-03-12",
        "CAP-VIB-01",
    ]


def test_seed_counts_and_labels(seeded_neo4j: bool) -> None:
    """The seeded graph contains exactly 30 nodes and 27 relationships."""
    from graph.connection import get_driver

    with get_driver().session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        labels = {
            record["label"]: record["count"]
            for record in session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count"
            )
        }
        relationships = {
            record["type"]: record["count"]
            for record in session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count"
            )
        }

    assert node_count == 30
    assert rel_count == 27
    assert labels == {
        "Equipment": 6,
        "Symptom": 6,
        "Procedure": 6,
        "MaintenanceHistory": 6,
        "Sensor": 6,
    }
    assert relationships == {
        "PRÉSENTE": 7,
        "RECOMMANDE": 8,
        "A_CONSTATÉ": 6,
        "MESURE": 6,
    }


def test_seed_is_idempotent(seeded_neo4j: bool) -> None:
    """Running the seed again keeps the graph at 30 nodes and 27 relations."""
    counts = seed(reset=False)

    assert counts == {"nodes": 30, "relationships": 27}


def test_reasoning_path_for_p204(seeded_neo4j: bool) -> None:
    """P-204 follows Equipment → Symptom → Procedure + History + Sensor."""
    paths = get_reasoning_path("P-204", "vibration")

    assert len(paths) == 1
    path = paths[0]
    assert path["equipment"]["id"] == "P-204"
    assert path["symptom"]["name"] == "vibration anormale"
    assert {procedure["code"] for procedure in path["procedures"]} == {
        "REMPLACER_ROULEMENT",
        "REALIGNER_ARBRE",
    }
    assert path["histories"][0]["date"] == "2024-03-12"
    assert path["sensors"][0]["id"] == "CAP-VIB-01"
    assert any("PRÉSENTE" in relation for relation in path["relations"])
    assert any("A_CONSTATÉ" in relation for relation in path["relations"])


def test_query_knowledge_graph_context(seeded_neo4j: bool) -> None:
    """The GraphRAG context contains the expected reasoning blocks."""
    context = query_knowledge_graph(
        "L'équipement P-204 présente une vibration anormale. Que faire ?"
    )

    assert "Contexte GraphRAG" in context
    assert "P-204" in context
    assert "REMPLACER_ROULEMENT" in context
    assert "2024-03-12" in context
    assert "CAP-VIB-01" in context


def test_query_knowledge_graph_without_equipment() -> None:
    """A question without an equipment id returns a clear message."""
    context = query_knowledge_graph("Que faire en cas de vibration ?")

    assert "Aucun identifiant d'équipement" in context


def test_get_reasoning_path_unknown_equipment(seeded_neo4j: bool) -> None:
    """Unknown equipment yields no path."""
    assert get_reasoning_path("P-999") == []


@pytest.mark.neo4j
def test_connection(seeded_neo4j: bool) -> None:
    """The Neo4j driver can verify connectivity."""
    from graph.connection import check_connection

    assert check_connection() is True
