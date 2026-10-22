"""Seed the 30-node industrial maintenance knowledge graph.

Run with::

    python -m graph.seed_graph --reset

The script is idempotent: ``MERGE`` is used for every node and relationship.
``--reset`` first removes all existing nodes so the demo starts from a known
state. The graph contains exactly 6 equipments, 6 symptoms, 6 procedures,
6 maintenance histories and 6 sensors (30 nodes), linked by 27 relationships.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from .connection import close_driver, get_driver

logger = logging.getLogger(__name__)

EQUIPMENTS: list[dict[str, Any]] = [
    {
        "id": "P-204",
        "name": "Pompe centrifuge",
        "type": "Pompe",
        "site": "Lyon",
        "criticality": "haute",
    },
    {
        "id": "C-101",
        "name": "Compresseur d'air",
        "type": "Compresseur",
        "site": "Lyon",
        "criticality": "haute",
    },
    {
        "id": "M-310",
        "name": "Moteur asynchrone",
        "type": "Moteur",
        "site": "Grenoble",
        "criticality": "moyenne",
    },
    {
        "id": "V-450",
        "name": "Vanne de régulation",
        "type": "Vanne",
        "site": "Grenoble",
        "criticality": "moyenne",
    },
    {
        "id": "E-120",
        "name": "Échangeur thermique",
        "type": "Échangeur",
        "site": "Lyon",
        "criticality": "haute",
    },
    {
        "id": "R-077",
        "name": "Réducteur mécanique",
        "type": "Réducteur",
        "site": "Saint-Étienne",
        "criticality": "basse",
    },
]

SYMPTOMS: list[dict[str, Any]] = [
    {
        "id": "S-VIB-01",
        "name": "vibration anormale",
        "description": "Vibration excessive du bloc pompe, mesurée au-delà du seuil constructeur.",
        "severity": "haute",
    },
    {
        "id": "S-TEMP-01",
        "name": "surchauffe",
        "description": "Température de palier supérieure à la plage nominale.",
        "severity": "haute",
    },
    {
        "id": "S-BRUIT-01",
        "name": "bruit métallique",
        "description": "Bruit de frottement métallique intermittent côté accouplement.",
        "severity": "moyenne",
    },
    {
        "id": "S-FUITE-01",
        "name": "fuite d'huile",
        "description": "Présence d'huile sous le presse-étoupe et baisse du niveau.",
        "severity": "moyenne",
    },
    {
        "id": "S-PRESS-01",
        "name": "perte de pression",
        "description": "Pression de refoulement instable avec chute progressive.",
        "severity": "haute",
    },
    {
        "id": "S-DEBIT-01",
        "name": "débit insuffisant",
        "description": "Débit mesuré inférieur à la consigne de production.",
        "severity": "moyenne",
    },
]

PROCEDURES: list[dict[str, Any]] = [
    {
        "id": "PR-ROUL-01",
        "code": "REMPLACER_ROULEMENT",
        "name": "Remplacer le roulement",
        "description": "Démonter le bloc, remplacer le roulement usé et contrôler l'alignement.",
        "duration_min": 120,
    },
    {
        "id": "PR-ALIGN-01",
        "code": "REALIGNER_ARBRE",
        "name": "Réaligner l'arbre",
        "description": "Mesurer le faux-rond puis réaligner l'arbre et l'accouplement.",
        "duration_min": 90,
    },
    {
        "id": "PR-GRAISS-01",
        "code": "GRAISSER_PALIER",
        "name": "Graisser le palier",
        "description": "Nettoyer et regraisser le palier avec la graisse préconisée.",
        "duration_min": 30,
    },
    {
        "id": "PR-VIDANGE-01",
        "code": "VIDANGER_HUILE",
        "name": "Vidanger l'huile",
        "description": "Vidanger, rincer et remplir le carter avec l'huile spécifiée.",
        "duration_min": 60,
    },
    {
        "id": "PR-JOINT-01",
        "code": "REMPLACER_JOINT",
        "name": "Remplacer le joint",
        "description": "Remplacer le joint d'étanchéité et contrôler la portée.",
        "duration_min": 75,
    },
    {
        "id": "PR-CALIB-01",
        "code": "CALIBRER_CAPTEUR",
        "name": "Calibrer le capteur",
        "description": "Vérifier et recalibrer la chaîne de mesure de débit.",
        "duration_min": 45,
    },
]

HISTORIES: list[dict[str, Any]] = [
    {
        "id": "H-2024-03-12",
        "date": "2024-03-12",
        "description": "Vibration mesurée à 7,8 mm/s ; roulement remplacé en preventive.",
        "technician": "A. Martin",
        "equipment_id": "P-204",
    },
    {
        "id": "H-2023-11-08",
        "date": "2023-11-08",
        "description": "Surchauffe du palier liée à un défaut de graissage.",
        "technician": "S. Bernard",
        "equipment_id": "C-101",
    },
    {
        "id": "H-2024-01-22",
        "date": "2024-01-22",
        "description": "Bruit métallique à l'accouplement, réalignement réalisé.",
        "technician": "N. Petit",
        "equipment_id": "M-310",
    },
    {
        "id": "H-2024-02-15",
        "date": "2024-02-15",
        "description": "Fuite d'huile au presse-étoupe, joint remplacé.",
        "technician": "A. Martin",
        "equipment_id": "V-450",
    },
    {
        "id": "H-2023-09-30",
        "date": "2023-09-30",
        "description": "Perte de pression après encrassement du circuit d'huile.",
        "technician": "L. Dubois",
        "equipment_id": "V-450",
    },
    {
        "id": "H-2024-04-05",
        "date": "2024-04-05",
        "description": "Débit insuffisant, dérive du capteur de débit constatée.",
        "technician": "S. Bernard",
        "equipment_id": "E-120",
    },
]

SENSORS: list[dict[str, Any]] = [
    {
        "id": "CAP-VIB-01",
        "name": "Capteur de vibration",
        "unit": "mm/s",
        "last_value": 7.8,
        "status": "alerte",
    },
    {
        "id": "CAP-TEMP-01",
        "name": "Capteur de température",
        "unit": "°C",
        "last_value": 92.0,
        "status": "alerte",
    },
    {
        "id": "CAP-PRESS-01",
        "name": "Capteur de pression",
        "unit": "bar",
        "last_value": 3.2,
        "status": "normal",
    },
    {
        "id": "CAP-DEBIT-01",
        "name": "Capteur de débit",
        "unit": "m³/h",
        "last_value": 18.5,
        "status": "bas",
    },
    {
        "id": "CAP-COURANT-01",
        "name": "Capteur de courant",
        "unit": "A",
        "last_value": 24.1,
        "status": "normal",
    },
    {
        "id": "CAP-NIVEAU-01",
        "name": "Capteur de niveau d'huile",
        "unit": "%",
        "last_value": 48,
        "status": "normal",
    },
]

PRESENTE: list[dict[str, str]] = [
    {"equipment": "P-204", "symptom": "S-VIB-01"},
    {"equipment": "C-101", "symptom": "S-TEMP-01"},
    {"equipment": "M-310", "symptom": "S-BRUIT-01"},
    {"equipment": "V-450", "symptom": "S-FUITE-01"},
    {"equipment": "V-450", "symptom": "S-PRESS-01"},
    {"equipment": "E-120", "symptom": "S-DEBIT-01"},
    {"equipment": "R-077", "symptom": "S-VIB-01"},
]

RECOMMANDE: list[dict[str, str]] = [
    {"symptom": "S-VIB-01", "procedure": "PR-ROUL-01"},
    {"symptom": "S-VIB-01", "procedure": "PR-ALIGN-01"},
    {"symptom": "S-TEMP-01", "procedure": "PR-GRAISS-01"},
    {"symptom": "S-TEMP-01", "procedure": "PR-VIDANGE-01"},
    {"symptom": "S-BRUIT-01", "procedure": "PR-ALIGN-01"},
    {"symptom": "S-FUITE-01", "procedure": "PR-JOINT-01"},
    {"symptom": "S-PRESS-01", "procedure": "PR-VIDANGE-01"},
    {"symptom": "S-DEBIT-01", "procedure": "PR-CALIB-01"},
]

A_CONSTATE: list[dict[str, str]] = [
    {"history": "H-2024-03-12", "symptom": "S-VIB-01"},
    {"history": "H-2023-11-08", "symptom": "S-TEMP-01"},
    {"history": "H-2024-01-22", "symptom": "S-BRUIT-01"},
    {"history": "H-2024-02-15", "symptom": "S-FUITE-01"},
    {"history": "H-2023-09-30", "symptom": "S-PRESS-01"},
    {"history": "H-2024-04-05", "symptom": "S-DEBIT-01"},
]

MESURE: list[dict[str, str]] = [
    {"sensor": "CAP-VIB-01", "equipment": "P-204"},
    {"sensor": "CAP-TEMP-01", "equipment": "C-101"},
    {"sensor": "CAP-PRESS-01", "equipment": "V-450"},
    {"sensor": "CAP-DEBIT-01", "equipment": "E-120"},
    {"sensor": "CAP-COURANT-01", "equipment": "M-310"},
    {"sensor": "CAP-NIVEAU-01", "equipment": "R-077"},
]

_CONSTRAINTS = [
    "CREATE CONSTRAINT equipment_id IF NOT EXISTS FOR (n:Equipment) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (n:Symptom) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT procedure_id IF NOT EXISTS FOR (n:Procedure) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT history_id IF NOT EXISTS FOR (n:MaintenanceHistory) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT sensor_id IF NOT EXISTS FOR (n:Sensor) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX equipment_name IF NOT EXISTS FOR (n:Equipment) ON (n.name)",
    "CREATE INDEX equipment_type IF NOT EXISTS FOR (n:Equipment) ON (n.type)",
    "CREATE INDEX symptom_name IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
    "CREATE INDEX symptom_description IF NOT EXISTS FOR (n:Symptom) ON (n.description)",
    "CREATE INDEX procedure_code IF NOT EXISTS FOR (n:Procedure) ON (n.code)",
    "CREATE INDEX procedure_name IF NOT EXISTS FOR (n:Procedure) ON (n.name)",
    "CREATE INDEX history_date IF NOT EXISTS FOR (n:MaintenanceHistory) ON (n.date)",
    "CREATE INDEX sensor_name IF NOT EXISTS FOR (n:Sensor) ON (n.name)",
]


def _create_schema(session: Any) -> None:
    """Create uniqueness constraints and lookup indexes."""
    for statement in _CONSTRAINTS:
        session.run(statement).consume()


def _create_nodes(session: Any) -> None:
    """Merge every graph node from the declarative seed data."""
    session.run(
        "UNWIND $rows AS row MERGE (n:Equipment {id: row.id}) SET n += row",
        rows=EQUIPMENTS,
    ).consume()
    session.run(
        "UNWIND $rows AS row MERGE (n:Symptom {id: row.id}) SET n += row",
        rows=SYMPTOMS,
    ).consume()
    session.run(
        "UNWIND $rows AS row MERGE (n:Procedure {id: row.id}) SET n += row",
        rows=PROCEDURES,
    ).consume()
    session.run(
        "UNWIND $rows AS row MERGE (n:MaintenanceHistory {id: row.id}) SET n += row",
        rows=HISTORIES,
    ).consume()
    session.run(
        "UNWIND $rows AS row MERGE (n:Sensor {id: row.id}) SET n += row",
        rows=SENSORS,
    ).consume()


def _create_relationships(session: Any) -> None:
    """Merge the 27 relationships connecting the seeded nodes."""
    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:Equipment {id: row.equipment})
        MATCH (s:Symptom {id: row.symptom})
        MERGE (e)-[:`PRÉSENTE`]->(s)
        """,
        rows=PRESENTE,
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        MATCH (s:Symptom {id: row.symptom})
        MATCH (p:Procedure {id: row.procedure})
        MERGE (s)-[:`RECOMMANDE`]->(p)
        """,
        rows=RECOMMANDE,
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        MATCH (h:MaintenanceHistory {id: row.history})
        MATCH (s:Symptom {id: row.symptom})
        MERGE (h)-[:`A_CONSTATÉ`]->(s)
        """,
        rows=A_CONSTATE,
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        MATCH (c:Sensor {id: row.sensor})
        MATCH (e:Equipment {id: row.equipment})
        MERGE (c)-[:`MESURE`]->(e)
        """,
        rows=MESURE,
    ).consume()


def seed(reset: bool = False) -> dict[str, int]:
    """Create the demo graph.

    Args:
        reset: When ``True``, delete all existing nodes and relationships first.

    Returns:
        A dictionary with the final node and relationship counts.
    """
    driver = get_driver()
    with driver.session() as session:
        if reset:
            logger.info("Resetting the graph")
            session.run("MATCH (n) DETACH DELETE n").consume()

        _create_schema(session)
        _create_nodes(session)
        _create_relationships(session)

        node_row = session.run("MATCH (n) RETURN count(n) AS count").single()
        rel_row = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
        counts = {
            "nodes": int(node_row["count"]) if node_row else 0,
            "relationships": int(rel_row["count"]) if rel_row else 0,
        }

    logger.info("Graph seeded: %s nodes, %s relationships", counts["nodes"], counts["relationships"])
    return counts


def main() -> None:
    """CLI entry point for ``python -m graph.seed_graph``."""
    parser = argparse.ArgumentParser(description="Seed the industrial maintenance graph.")
    parser.add_argument("--reset", action="store_true", help="Delete existing data before seeding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    driver = get_driver()
    try:
        counts = seed(reset=args.reset)
        print(f"Seed terminé : {counts['nodes']} nœuds, {counts['relationships']} relations.")
    finally:
        close_driver(driver)


if __name__ == "__main__":
    main()
