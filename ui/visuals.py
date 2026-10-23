"""Self-contained HTML/SVG renderers for the Streamlit demonstration.

The visualizations are generated as plain HTML so they work offline inside the
Docker image, without loading a JavaScript library from a CDN. The reasoning
graph uses CSS animations and SVG ``animateMotion`` to reveal the GraphRAG path
in the same order as the agent's reasoning.
"""

from __future__ import annotations

import html
from textwrap import dedent
from typing import Any

_NODE_COLORS = {
    "equipment": "#38bdf8",
    "symptom": "#fbbf24",
    "procedure": "#34d399",
    "history": "#a78bfa",
    "sensor": "#94a3b8",
}

_NODE_LABELS = {
    "equipment": "Équipement",
    "symptom": "Symptôme",
    "procedure": "Procédure",
    "history": "Historique",
    "sensor": "Capteur",
}

_BASE_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #070d17;
  color: #e2e8f0;
  font-family: "Segoe UI", Inter, system-ui, -apple-system, sans-serif;
}
.graph-shell, .audit-shell {
  border: 1px solid #1e293b;
  border-radius: 16px;
  padding: 16px 18px 8px;
  background:
    radial-gradient(circle at 12% 10%, rgba(56, 189, 248, 0.13), transparent 34%),
    radial-gradient(circle at 88% 84%, rgba(52, 211, 153, 0.09), transparent 34%),
    #0b1220;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
  overflow: hidden;
}
.graph-head, .audit-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.eyebrow {
  color: #38bdf8;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.graph-title, .audit-title {
  color: #f8fafc;
  font-size: 17px;
  font-weight: 650;
  line-height: 1.25;
}
.chip {
  border: 1px solid #334155;
  border-radius: 999px;
  color: #cbd5e1;
  font-size: 11px;
  padding: 5px 10px;
  background: rgba(15, 23, 42, 0.72);
  white-space: nowrap;
}
.chip.ok { border-color: rgba(52, 211, 153, 0.5); color: #6ee7b7; }
.chip.ko { border-color: rgba(248, 113, 113, 0.55); color: #fca5a5; }
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 7px 14px;
  margin: 2px 0 4px;
}
.legend-item {
  color: #94a3b8;
  font-size: 10px;
  letter-spacing: 0.02em;
}
.legend-item i {
  display: inline-block;
  width: 7px;
  height: 7px;
  margin-right: 5px;
  border-radius: 50%;
  vertical-align: 1px;
}
svg { display: block; width: 100%; height: auto; }
.node {
  opacity: 0;
  animation: nodeIn 0.55s cubic-bezier(0.2, 0.7, 0.2, 1) forwards;
}
@keyframes nodeIn {
  from { opacity: 0; transform: translateY(7px); }
  to { opacity: 1; transform: translateY(0); }
}
.node-title {
  fill: #f8fafc;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.node-sub {
  fill: #b6c2d2;
  font-size: 10.5px;
  font-weight: 400;
}
.edge {
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-dasharray: 1;
  stroke-dashoffset: 1;
  animation: drawEdge 0.85s ease-out forwards;
}
@keyframes drawEdge { to { stroke-dashoffset: 0; } }
.edge-label {
  fill: #a8b5c6;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.10em;
  paint-order: stroke;
  stroke: #0b1220;
  stroke-width: 4px;
  stroke-linejoin: round;
  opacity: 0;
  animation: fadeIn 0.4s ease-out forwards;
}
@keyframes fadeIn { to { opacity: 1; } }
.traveler { filter: url(#softGlow); }
.empty {
  color: #cbd5e1;
  font-size: 13px;
  line-height: 1.55;
  padding: 18px 4px 22px;
  white-space: pre-wrap;
}
@media (prefers-reduced-motion: reduce) {
  .node, .edge, .edge-label { animation: none; opacity: 1; stroke-dashoffset: 0; }
}
"""

_AUDIT_CSS = """
.audit-shell { padding-bottom: 14px; }
.audit-rail {
  display: flex;
  align-items: stretch;
  gap: 8px;
  overflow-x: auto;
  padding: 8px 2px 4px;
}
.audit-card {
  flex: 0 0 176px;
  min-width: 176px;
  border: 1px solid #334155;
  border-radius: 11px;
  padding: 10px 11px;
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.88), rgba(15, 23, 42, 0.9));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.audit-card.ko { border-color: rgba(248, 113, 113, 0.55); }
.audit-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 6px;
}
.seq {
  color: #38bdf8;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
}
.event {
  color: #f8fafc;
  font-size: 11px;
  font-weight: 650;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.actor {
  color: #94a3b8;
  font-size: 9.5px;
  margin-bottom: 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.hash, .prev {
  color: #64748b;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 8.5px;
  line-height: 1.45;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.hash { color: #6ee7b7; }
.arrow {
  align-self: center;
  color: #334155;
  font-size: 15px;
}
.audit-empty { color: #94a3b8; font-size: 12px; padding: 12px 2px; }
"""

_ALERT_CSS = """
<style>
@keyframes approvalPulse {
  0%, 100% {
    box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.20), 0 0 24px rgba(239, 68, 68, 0.20);
  }
  50% {
    box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.55), 0 0 44px rgba(239, 68, 68, 0.42);
  }
}
@keyframes approvalPing {
  0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.65); }
  70%, 100% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
}
.iah-approval-alert {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 6px 0 16px;
  padding: 17px 20px;
  border: 1px solid #ef4444;
  border-radius: 14px;
  background: linear-gradient(100deg, rgba(127, 29, 29, 0.88), rgba(69, 10, 10, 0.94));
  animation: approvalPulse 1.8s ease-in-out infinite;
}
.iah-approval-dot {
  flex: 0 0 auto;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #ef4444;
  animation: approvalPing 1.4s ease-out infinite;
}
.iah-approval-kicker {
  color: #fecaca;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.iah-approval-title {
  color: #ffffff;
  font-size: 20px;
  font-weight: 750;
  line-height: 1.2;
  margin: 2px 0 3px;
}
.iah-approval-detail { color: #fee2e2; font-size: 12.5px; line-height: 1.4; }
.iah-approval-count {
  margin-left: auto;
  border: 1px solid rgba(254, 202, 202, 0.45);
  border-radius: 999px;
  color: #fecaca;
  font-size: 11px;
  font-weight: 700;
  padding: 6px 11px;
  white-space: nowrap;
}
</style>
"""


def _escape(value: Any, limit: int = 40) -> str:
    """Truncate and HTML-escape a value for safe inline rendering."""
    text = str(value if value is not None else "")
    if len(text) > limit:
        text = text[: max(1, limit - 1)] + "…"
    return html.escape(text, quote=True)


def _wrap(body: str, height: int) -> tuple[str, int]:
    """Wrap an HTML body in a standalone document for ``components.html``."""
    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_BASE_CSS}{_AUDIT_CSS}</style></head><body>{body}</body></html>"
    )
    return document, height


def _node_svg(node: dict[str, Any], cx: float, cy: float, delay: float) -> str:
    """Render one colored node card as an SVG group."""
    node_type = str(node.get("type", "sensor"))
    color = _NODE_COLORS.get(node_type, _NODE_COLORS["sensor"])
    title = _escape(node.get("label", "?"), 23)
    subtitle = _escape(node.get("subtitle", ""), 30)
    return f"""
    <g class="node" style="animation-delay:{delay:.2f}s">
      <rect x="{cx - 82:.1f}" y="{cy - 30:.1f}" width="164" height="60" rx="13"
            fill="{color}" fill-opacity="0.13" stroke="{color}" stroke-width="1.35"/>
      <circle cx="{cx - 66:.1f}" cy="{cy - 11:.1f}" r="4.2" fill="{color}"/>
      <text x="{cx - 55:.1f}" y="{cy - 7:.1f}" class="node-title">{title}</text>
      <text x="{cx - 66:.1f}" y="{cy + 14:.1f}" class="node-sub">{subtitle}</text>
    </g>
    """


def _edge_svg(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    color: str,
    delay: float,
) -> str:
    """Render a curved animated edge with a small relationship label."""
    dx = x2 - x1
    dy = y2 - y1
    c1x, c1y = x1 + dx * 0.42, y1 + dy * 0.04
    c2x, c2y = x1 + dx * 0.58, y2 - dy * 0.04
    mid_x, mid_y = x1 + dx * 0.5, y1 + dy * 0.5

    if abs(dx) >= abs(dy):
        label_x, label_y, anchor = mid_x, mid_y - 8, "middle"
    else:
        label_x, label_y, anchor = mid_x + 11, mid_y + 3, "start"

    return f"""
    <path class="edge" pathLength="1"
          d="M {x1:.1f} {y1:.1f} C {c1x:.1f} {c1y:.1f}, {c2x:.1f} {c2y:.1f}, {x2:.1f} {y2:.1f}"
          stroke="{color}" style="animation-delay:{delay:.2f}s"/>
    <text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" class="edge-label"
          style="animation-delay:{delay + 0.38:.2f}s">{_escape(label, 20)}</text>
    """


def build_reasoning_view(payload: dict[str, Any]) -> tuple[str, int]:
    """Build the animated GraphRAG reasoning view.

    Args:
        payload: Output of :func:`graph.queries.get_reasoning_subgraph`.

    Returns:
        A tuple ``(html, height)`` ready for ``streamlit.components.v1.html``.
    """
    paths = payload.get("paths") or []
    equipment_id = payload.get("equipment_id") or "?"

    if not paths:
        message = _escape(payload.get("context", "Aucun chemin de raisonnement."), 500)
        body = f"""
        <div class="graph-shell">
          <div class="graph-head">
            <div>
              <div class="eyebrow">GRAPHRAG · NEO4J</div>
              <div class="graph-title">Chemin de raisonnement</div>
            </div>
            <div class="chip">{_escape(equipment_id, 18)}</div>
          </div>
          <div class="empty">{message}</div>
        </div>
        """
        return _wrap(body, 190)

    path = paths[0]
    equipment = path.get("equipment") or {}
    symptom = path.get("symptom") or {}
    procedures = path.get("procedures") or []
    histories = path.get("histories") or []
    sensors = path.get("sensors") or []

    side_span = max(
        (len(procedures) - 1) * 88,
        (len(histories) - 1) * 76,
        0,
    )
    height = max(470, 320 + 2 * side_span)
    center_y = height / 2

    equipment_x = 205.0
    symptom_x = 505.0
    procedure_x = 805.0
    sensor_x = 205.0
    history_x = 505.0

    sensor_positions = [(sensor_x, 95.0 + index * 76.0) for index in range(len(sensors))]
    history_positions = [
        (history_x, center_y + 120.0 + index * 76.0) for index in range(len(histories))
    ]
    procedure_positions = [
        (procedure_x, center_y + (index - (len(procedures) - 1) / 2) * 88.0)
        for index in range(len(procedures))
    ]

    edges: list[str] = []
    for index, sensor_y in enumerate(position[1] for position in sensor_positions):
        edges.append(
            _edge_svg(
                sensor_x,
                sensor_y + 30,
                equipment_x,
                center_y - 30,
                "MESURE",
                _NODE_COLORS["sensor"],
                0.45 + index * 0.08,
            )
        )

    edges.append(
        _edge_svg(
            equipment_x + 82,
            center_y,
            symptom_x - 82,
            center_y,
            "PRÉSENTE",
            _NODE_COLORS["symptom"],
            1.0,
        )
    )

    for index, (_, history_y) in enumerate(history_positions):
        edges.append(
            _edge_svg(
                history_x,
                history_y - 30,
                symptom_x,
                center_y + 30,
                "A_CONSTATÉ",
                _NODE_COLORS["history"],
                0.66 + index * 0.08,
            )
        )

    for index, (_, procedure_y) in enumerate(procedure_positions):
        edges.append(
            _edge_svg(
                symptom_x + 82,
                center_y,
                procedure_x - 82,
                procedure_y,
                "RECOMMANDE",
                _NODE_COLORS["procedure"],
                1.4 + index * 0.14,
            )
        )

    nodes: list[str] = [
        _node_svg(
            {
                "type": "equipment",
                "label": equipment.get("id", equipment_id),
                "subtitle": f"{equipment.get('name', '?')} · {equipment.get('criticality', '?')}",
            },
            equipment_x,
            center_y,
            0.15,
        )
    ]
    nodes.append(
        _node_svg(
            {
                "type": "symptom",
                "label": symptom.get("id", "symptôme"),
                "subtitle": f"{symptom.get('name', '?')} · {symptom.get('severity', '?')}",
            },
            symptom_x,
            center_y,
            1.15,
        )
    )
    for index, (sensor, (node_x, node_y)) in enumerate(zip(sensors, sensor_positions, strict=False)):
        nodes.append(
            _node_svg(
                {
                    "type": "sensor",
                    "label": sensor.get("id", "capteur"),
                    "subtitle": (
                        f"{sensor.get('name', '?')} · {sensor.get('last_value', '?')} "
                        f"{sensor.get('unit', '')}"
                    ).strip(),
                },
                node_x,
                node_y,
                0.55 + index * 0.08,
            )
        )
    for index, (history, (node_x, node_y)) in enumerate(zip(histories, history_positions, strict=False)):
        nodes.append(
            _node_svg(
                {
                    "type": "history",
                    "label": history.get("id", "historique"),
                    "subtitle": f"{history.get('date', '?')} · {history.get('technician', '?')}",
                },
                node_x,
                node_y,
                0.75 + index * 0.08,
            )
        )
    for index, (procedure, (node_x, node_y)) in enumerate(zip(procedures, procedure_positions, strict=False)):
        nodes.append(
            _node_svg(
                {
                    "type": "procedure",
                    "label": procedure.get("id", "procédure"),
                    "subtitle": (
                        f"{procedure.get('code', '?')} · {procedure.get('duration_min', '?')} min"
                    ),
                },
                node_x,
                node_y,
                1.55 + index * 0.14,
            )
        )

    flow_points = [
        (equipment_x + 82, center_y),
        (symptom_x - 82, center_y),
    ]
    if procedure_positions:
        flow_points.append((procedure_x - 82, procedure_positions[0][1]))
    flow_path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in flow_points)

    traveler = f"""
    <path id="flow-path" d="{flow_path}" fill="none" stroke="none"/>
    <circle class="traveler" r="4.5" fill="#f8fafc">
      <animateMotion dur="3.2s" begin="2.3s" repeatCount="indefinite">
        <mpath href="#flow-path"/>
      </animateMotion>
    </circle>
    """

    legend = "".join(
        f'<span class="legend-item"><i style="background:{color}"></i>'
        f"{_NODE_LABELS[node_type]}</span>"
        for node_type, color in _NODE_COLORS.items()
    )
    path_note = f"{len(paths)} chemins · 1 affiché" if len(paths) > 1 else "1 chemin"
    equipment_name = _escape(equipment.get("name", "?"), 28)

    body = f"""
    <div class="graph-shell">
      <div class="graph-head">
        <div>
          <div class="eyebrow">GRAPHRAG · NEO4J · {path_note}</div>
          <div class="graph-title">Chemin de raisonnement</div>
        </div>
        <div class="chip">{_escape(equipment_id, 16)} · {equipment_name}</div>
      </div>
      <div class="legend">{legend}</div>
      <svg viewBox="0 0 1000 {height:.0f}" role="img"
           aria-label="Sous-graphe de raisonnement pour {_escape(equipment_id, 16)}">
        <defs>
          <pattern id="dot-grid" width="28" height="28" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" fill="#1e293b"/>
          </pattern>
          <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="3.5" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <rect x="0" y="0" width="1000" height="{height:.0f}" fill="url(#dot-grid)" opacity="0.55"/>
        {''.join(edges)}
        {traveler}
        {''.join(nodes)}
      </svg>
    </div>
    """
    return _wrap(body, int(height) + 54)


def build_audit_view(events: list[dict[str, Any]], valid: bool) -> tuple[str, int]:
    """Build a compact horizontal HMAC chain visualization.

    Args:
        events: Audit records, oldest first.
        valid: Result of the live chain verification.

    Returns:
        A tuple ``(html, height)`` ready for ``streamlit.components.v1.html``.
    """
    cards: list[str] = []
    for index, event in enumerate(events):
        if index:
            cards.append('<div class="arrow">→</div>')
        status = "" if valid else " ko"
        cards.append(
            f"""
            <div class="audit-card{status}">
              <div class="audit-top">
                <span class="seq">#{_escape(event.get('seq', '?'), 8)}</span>
                <span class="event">{_escape(event.get('event_type', '?'), 22)}</span>
              </div>
              <div class="actor">{_escape(event.get('actor', '?'), 20)} ·
                {_escape(event.get('tool') or '—', 20)}</div>
              <div class="hash">h {_escape(str(event.get('hash', ''))[:12], 12)}…</div>
              <div class="prev">‹ {_escape(str(event.get('prev_hash', ''))[:12], 12)}…</div>
            </div>
            """
        )

    if not cards:
        cards.append('<div class="audit-empty">Aucun événement dans le registre.</div>')

    status_label = "chaîne vérifiée" if valid else "chaîne invalide"
    status_class = "ok" if valid else "ko"
    body = f"""
    <div class="audit-shell">
      <div class="audit-head">
        <div>
          <div class="eyebrow">REGISTRE APPEND-ONLY · HMAC-SHA256</div>
          <div class="audit-title">Chaîne d'intégrité</div>
        </div>
        <div class="chip {status_class}">{status_label}</div>
      </div>
      <div class="audit-rail">{''.join(cards)}</div>
    </div>
    """
    return _wrap(body, 224)


def render_approval_alert(pending: list[dict[str, Any]]) -> str:
    """Render the pulsing red approval banner shown above the decision panel."""
    if not pending:
        return ""

    summary = " · ".join(
        f"{request.get('equipment_id', '?')} — {request.get('action', '?')}"
        for request in pending[:3]
    )
    count = len(pending)
    count_label = "1 demande bloquée" if count == 1 else f"{count} demandes bloquées"
    markup = dedent(
        f"""
        <div class="iah-approval-alert" role="alert">
          <span class="iah-approval-dot"></span>
          <div>
            <div class="iah-approval-kicker">Politique require_approval</div>
            <div class="iah-approval-title">Approbation humaine requise</div>
            <div class="iah-approval-detail">
              L'agent est bloqué. Aucune action ne sera exécutée avant votre décision.
              {_escape(summary, 90)}
            </div>
          </div>
          <div class="iah-approval-count">{count_label}</div>
        </div>
        """
    ).strip()
    return _ALERT_CSS + markup


