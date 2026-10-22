"""Neo4j connection helpers for the industrial knowledge graph.

The module centralises driver creation so that the graph queries, the seed
script and the tests all share the same connection settings. Configuration is
read from ``NEO4J_URI``, ``NEO4J_USERNAME`` and ``NEO4J_PASSWORD``.
"""

from __future__ import annotations

import logging
import os
import threading

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

logger = logging.getLogger(__name__)

_DRIVER: Driver | None = None
_LOCK = threading.Lock()


def get_driver() -> Driver:
    """Return a lazily created, process-wide Neo4j driver.

    Returns:
        The shared :class:`neo4j.Driver` instance.

    Raises:
        ValueError: If ``NEO4J_PASSWORD`` is not configured.
    """
    global _DRIVER
    with _LOCK:
        if _DRIVER is not None:
            return _DRIVER

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError(
                "NEO4J_PASSWORD is not set. Copy .env.example to .env or export the variable."
            )

        logger.debug("Creating Neo4j driver for %s", uri)
        _DRIVER = GraphDatabase.driver(uri, auth=(username, password))
        return _DRIVER


def close_driver(driver: Driver | None = None) -> None:
    """Close a Neo4j driver and clear the module cache when relevant.

    Args:
        driver: Driver to close. When omitted, the cached driver is closed.
    """
    global _DRIVER
    with _LOCK:
        target = driver if driver is not None else _DRIVER
        if target is not None:
            target.close()
        if target is _DRIVER:
            _DRIVER = None


def check_connection() -> bool:
    """Check that the configured Neo4j instance is reachable.

    Returns:
        ``True`` when the driver can verify connectivity.

    Raises:
        Exception: Any driver-level connectivity error is propagated so callers
            can decide whether to skip or fail the current operation.
    """
    get_driver().verify_connectivity()
    return True
