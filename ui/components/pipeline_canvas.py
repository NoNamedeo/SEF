"""
Interactive pipeline canvas renderer for Streamlit.

The component is intentionally UI-only: it receives a typed graph payload and
renders a large pannable/zoomable workspace with draggable nodes, typed ports,
directional edges and contract validation affordances.
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from ui.components.pipeline_canvas_models import PipelineCanvasModel


def render_pipeline_canvas(model: PipelineCanvasModel, *, height: int = 620) -> None:
    """Render the typed pipeline canvas inside the Streamlit app."""
    theme = {
        "base": st.get_option("theme.base") or "light",
        "primary": st.get_option("theme.primaryColor") or "#ff4b4b",
        "background": st.get_option("theme.backgroundColor") or "#ffffff",
        "secondary": st.get_option("theme.secondaryBackgroundColor") or "#f0f2f6",
        "text": st.get_option("theme.textColor") or "#31333f",
    }
    payload = model.to_payload()
    payload["theme"] = theme
    components.html(
        _canvas_html(payload),
        height=height,
    )


def _canvas_html(payload: dict) -> str:
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {{
          --primary: {payload["theme"]["primary"]};
          --bg: {payload["theme"]["background"]};
          --panel: {payload["theme"]["secondary"]};
          --text: {payload["theme"]["text"]};
          --border: color-mix(in srgb, var(--text) 12%, transparent);
          --muted: color-mix(in srgb, var(--text) 58%, transparent);
          --shadow: color-mix(in srgb, black 12%, transparent);
          --good: #22a06b;
          --warn: #d99a1f;
          --bad: #d14343;
          --cat-source: #3b82f6;
          --cat-transform: #8b5cf6;
          --cat-signal: #06b6d4;
          --cat-analytics: #f59e0b;
          --cat-presentation: #10b981;
          --cat-event: #ef4444;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: Inter, ui-sans-serif, system-ui, sans-serif;
          color: var(--text);
          background: transparent;
        }}
        .canvas-shell {{
          border: 1px solid var(--border);
          border-radius: 22px;
          overflow: hidden;
          background:
            radial-gradient(circle at 15% 0%, color-mix(in srgb, var(--primary) 10%, transparent), transparent 28%),
            linear-gradient(180deg, color-mix(in srgb, var(--panel) 72%, transparent), color-mix(in srgb, var(--bg) 90%, transparent));
        }}
        .canvas-toolbar {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          padding: 14px 18px;
          border-bottom: 1px solid var(--border);
          background: color-mix(in srgb, var(--bg) 84%, transparent);
        }}
        .toolbar-title {{
          display: flex;
          flex-direction: column;
          gap: 4px;
        }}
        .toolbar-title strong {{
          font-size: 14px;
        }}
        .toolbar-title span,
        .toolbar-meta {{
          color: var(--muted);
          font-size: 12px;
        }}
        .toolbar-meta {{
          display: flex;
          align-items: center;
          gap: 14px;
          flex-wrap: wrap;
        }}
        .viewport {{
          position: relative;
          height: 720px;
          overflow: hidden;
          cursor: grab;
          touch-action: none;
        }}
        .viewport.dragging {{
          cursor: grabbing;
        }}
        .grid {{
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(color-mix(in srgb, var(--text) 5%, transparent) 1px, transparent 1px),
            linear-gradient(90deg, color-mix(in srgb, var(--text) 5%, transparent) 1px, transparent 1px);
          background-size: 32px 32px;
          pointer-events: none;
        }}
        .surface {{
          position: absolute;
          transform-origin: 0 0;
          will-change: transform;
        }}
        .surface svg {{
          position: absolute;
          inset: 0;
          overflow: visible;
          pointer-events: none;
        }}
        .node {{
          position: absolute;
          width: 268px;
          min-height: 180px;
          z-index: 2;
          border-radius: 20px;
          background: color-mix(in srgb, var(--bg) 78%, transparent);
          border: 1px solid var(--border);
          box-shadow: 0 16px 38px var(--shadow);
          backdrop-filter: blur(8px);
          padding: 14px 16px 16px 16px;
          user-select: none;
          cursor: grab;
          transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease;
        }}
        .node.dragging {{
          z-index: 8;
          cursor: grabbing;
        }}
        .node:hover {{
          transform: translateY(-2px);
          box-shadow: 0 22px 44px color-mix(in srgb, black 14%, transparent);
        }}
        .node.selected {{
          box-shadow: 0 0 0 2px color-mix(in srgb, var(--primary) 72%, transparent), 0 24px 48px color-mix(in srgb, black 18%, transparent);
        }}
        .node.state-idle {{ border-color: color-mix(in srgb, var(--text) 22%, transparent); }}
        .node.state-configured {{ border-color: color-mix(in srgb, var(--primary) 40%, transparent); }}
        .node.state-running {{ border-color: color-mix(in srgb, #3b82f6 70%, transparent); }}
        .node.state-completed {{ border-color: color-mix(in srgb, var(--good) 72%, transparent); }}
        .node.state-error {{ border-color: color-mix(in srgb, var(--bad) 85%, transparent); }}
        .node.warning {{
          box-shadow: 0 0 0 2px color-mix(in srgb, var(--warn) 16%, transparent), 0 18px 36px color-mix(in srgb, black 14%, transparent);
        }}
        .node-header {{
          display: flex;
          justify-content: space-between;
          gap: 10px;
          align-items: flex-start;
          margin-bottom: 10px;
        }}
        .node-title {{
          display: flex;
          flex-direction: column;
          gap: 4px;
          min-width: 0;
          cursor: pointer;
        }}
        .node-type {{
          font-size: 11px;
          line-height: 1;
          letter-spacing: .06em;
          text-transform: uppercase;
          color: var(--muted);
          font-weight: 700;
        }}
        .node-title strong {{
          font-size: 15px;
          line-height: 1.2;
        }}
        .node-meta {{
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
          margin-bottom: 8px;
        }}
        .pill {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          min-height: 24px;
          padding: 0 10px;
          border-radius: 999px;
          border: 1px solid color-mix(in srgb, var(--text) 14%, transparent);
          background: color-mix(in srgb, var(--panel) 72%, transparent);
          font-size: 11px;
        }}
        .pill.count {{
          font-weight: 700;
        }}
        .pill.category-source {{ border-color: color-mix(in srgb, var(--cat-source) 38%, transparent); }}
        .pill.category-transform {{ border-color: color-mix(in srgb, var(--cat-transform) 38%, transparent); }}
        .pill.category-signal {{ border-color: color-mix(in srgb, var(--cat-signal) 38%, transparent); }}
        .pill.category-analytics {{ border-color: color-mix(in srgb, var(--cat-analytics) 38%, transparent); }}
        .pill.category-presentation {{ border-color: color-mix(in srgb, var(--cat-presentation) 38%, transparent); }}
        .pill.category-event {{ border-color: color-mix(in srgb, var(--cat-event) 38%, transparent); }}
        .state-dot {{
          width: 9px;
          height: 9px;
          border-radius: 999px;
          background: color-mix(in srgb, var(--text) 40%, transparent);
        }}
        .state-running .state-dot {{ background: #3b82f6; }}
        .state-completed .state-dot {{ background: var(--good); }}
        .state-error .state-dot {{ background: var(--bad); }}
        .state-configured .state-dot {{ background: var(--primary); }}
        .output {{
          font-size: 12px;
          color: var(--muted);
          margin-bottom: 10px;
        }}
        .components {{
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-bottom: 10px;
        }}
        .component-chip {{
          display: inline-flex;
          align-items: center;
          min-height: 24px;
          padding: 0 8px;
          border-radius: 999px;
          background: color-mix(in srgb, var(--primary) 10%, var(--panel));
          border: 1px solid color-mix(in srgb, var(--primary) 24%, transparent);
          font-size: 11px;
          max-width: 100%;
        }}
        .component-chip span {{
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }}
        .preview {{
          font-size: 12px;
          line-height: 1.45;
          color: var(--text);
          background: color-mix(in srgb, var(--panel) 58%, transparent);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 10px 12px;
          margin-bottom: 10px;
        }}
        .warning-box {{
          font-size: 12px;
          line-height: 1.4;
          color: color-mix(in srgb, var(--warn) 80%, var(--text));
          border: 1px solid color-mix(in srgb, var(--warn) 24%, transparent);
          background: color-mix(in srgb, var(--warn) 8%, transparent);
          border-radius: 12px;
          padding: 8px 10px;
          margin-bottom: 10px;
        }}
        .node-actions {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
        }}
        .action-button,
        .details-toggle {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 24px;
          padding: 0 10px;
          border-radius: 999px;
          border: 1px solid var(--border);
          background: color-mix(in srgb, var(--bg) 88%, transparent);
          color: var(--text);
          font-size: 11px;
          text-decoration: none;
          cursor: pointer;
        }}
        .details {{
          display: none;
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid var(--border);
          font-size: 12px;
        }}
        .details.open {{
          display: block;
        }}
        .detail-group {{
          margin-bottom: 10px;
        }}
        .detail-group strong {{
          display: block;
          margin-bottom: 4px;
          font-size: 11px;
          letter-spacing: .04em;
          text-transform: uppercase;
          color: var(--muted);
        }}
        .detail-list {{
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }}
        .detail-token {{
          padding: 4px 8px;
          border-radius: 9px;
          border: 1px solid var(--border);
          background: color-mix(in srgb, var(--panel) 72%, transparent);
        }}
        pre.config {{
          margin: 0;
          white-space: pre-wrap;
          word-break: break-word;
          font-size: 11px;
          line-height: 1.45;
          background: color-mix(in srgb, var(--panel) 58%, transparent);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 10px;
          max-height: 170px;
          overflow: auto;
        }}
        .port {{
          position: absolute;
          width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--bg);
          box-shadow: 0 0 0 4px color-mix(in srgb, var(--bg) 76%, transparent);
          cursor: crosshair;
        }}
        .port.input {{ left: -9px; }}
        .port.output {{ right: -9px; }}
        .port-shape {{
          width: 18px;
          height: 18px;
          background: color-mix(in srgb, var(--text) 80%, var(--primary));
          border: 1px solid color-mix(in srgb, var(--bg) 36%, transparent);
          transition: transform .14s ease, box-shadow .14s ease, background .14s ease;
        }}
        .port[data-type="frame"] .port-shape,
        .port[data-type="video"] .port-shape {{ border-radius: 4px; }}
        .port[data-type="signal"] .port-shape {{ border-radius: 999px; }}
        .port[data-type="event"] .port-shape {{ transform: rotate(45deg); border-radius: 2px; }}
        .port[data-type="analysis"] .port-shape {{
          clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
        }}
        .port[data-type="view"] .port-shape {{
          clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
        }}
        .port-label {{
          position: absolute;
          top: -8px;
          font-size: 10px;
          line-height: 1;
          color: var(--muted);
          pointer-events: none;
          white-space: nowrap;
        }}
        .port.input .port-label {{ left: 22px; }}
        .port.output .port-label {{ right: 22px; }}
        .port.compatible .port-shape {{
          background: color-mix(in srgb, var(--good) 84%, var(--bg));
          box-shadow: 0 0 0 3px color-mix(in srgb, var(--good) 18%, transparent);
        }}
        .port.incompatible .port-shape {{
          background: color-mix(in srgb, var(--bad) 84%, var(--bg));
          box-shadow: 0 0 0 3px color-mix(in srgb, var(--bad) 18%, transparent);
        }}
        .edge-main {{
          stroke: color-mix(in srgb, var(--primary) 44%, var(--text));
          stroke-width: 3.2;
        }}
        .edge-secondary {{
          stroke: color-mix(in srgb, var(--cat-event) 42%, var(--text));
          stroke-width: 2.8;
          stroke-dasharray: 10 8;
        }}
        .edge-event {{
          stroke: color-mix(in srgb, var(--cat-event) 75%, transparent);
          stroke-width: 2.6;
          stroke-dasharray: 3 8;
        }}
        .edge {{
          fill: none;
          stroke-linecap: round;
          filter: drop-shadow(0 2px 7px color-mix(in srgb, var(--text) 10%, transparent));
          pointer-events: stroke;
          cursor: pointer;
        }}
        .edge-label-bg {{
          fill: color-mix(in srgb, var(--bg) 95%, transparent);
          stroke: var(--border);
          rx: 9;
          ry: 9;
          pointer-events: all;
          cursor: pointer;
        }}
        .edge-label {{
          font-size: 11px;
          fill: var(--text);
          text-anchor: middle;
          dominant-baseline: middle;
          pointer-events: none;
        }}
        .edge-remove {{
          font-size: 11px;
          fill: color-mix(in srgb, var(--bad) 84%, var(--text));
          text-anchor: middle;
          dominant-baseline: middle;
          pointer-events: none;
        }}
        .tooltip {{
          position: absolute;
          z-index: 20;
          display: none;
          max-width: 260px;
          padding: 8px 10px;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: color-mix(in srgb, var(--bg) 96%, transparent);
          box-shadow: 0 14px 28px color-mix(in srgb, black 14%, transparent);
          font-size: 12px;
        }}
        .tooltip.show {{
          display: block;
        }}
      </style>
    </head>
    <body>
      <div class="canvas-shell">
        <div class="canvas-toolbar">
          <div class="toolbar-title">
            <strong>Pipeline canvas</strong>
            <span>Core-driven workspace: stage contracts, runtime state, events and secondary branches.</span>
          </div>
          <div class="toolbar-meta">
            <span>drag nodes</span>
            <span>drag background to pan</span>
            <span>wheel to zoom</span>
            <span id="zoomValue">100%</span>
          </div>
        </div>
        <div class="viewport" id="viewport">
          <div class="grid"></div>
          <div class="surface" id="surface" style="width:{payload["surface_width"]}px;height:{payload["surface_height"]}px;">
            <svg id="edges" width="{payload["surface_width"]}" height="{payload["surface_height"]}" viewBox="0 0 {payload["surface_width"]} {payload["surface_height"]}">
              <defs>
                <!-- troppo grandi  <marker id="arrowhead-main" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                  <polygon points="0 0, 120 3, 0 4" fill="color-mix(in srgb, var(--primary) 44%, var(--text))"></polygon>
                </marker>
                <marker id="arrowhead-secondary" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                  <polygon points="0 0, 120 3, 0 4" fill="color-mix(in srgb, var(--cat-event) 42%, var(--text))"></polygon>
                </marker>
                <marker id="arrowhead-event" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                  <polygon points="0 0, 120 3, 0 4" fill="color-mix(in srgb, var(--cat-event) 75%, transparent)"></polygon>
                </marker> -->
              </defs>
            </svg> 
          </div>
          <div class="tooltip" id="tooltip"></div>
        </div>
      </div>

      <script>
        const payload = {json.dumps(payload)};
        const viewport = document.getElementById("viewport");
        const surface = document.getElementById("surface");
        const svg = document.getElementById("edges");
        const tooltip = document.getElementById("tooltip");
        const zoomValue = document.getElementById("zoomValue");

        const state = {{
          panX: payload.initial_pan_x,
          panY: payload.initial_pan_y,
          zoom: payload.initial_zoom,
          edges: payload.edges.map(edge => ({{ ...edge }})),
          draggingCanvas: false,
          draggedNode: null,
          dragStartX: 0,
          dragStartY: 0,
          nodeStartX: 0,
          nodeStartY: 0,
          surfaceStartX: 0,
          surfaceStartY: 0,
          tempConnection: null,
        }};

        const nodeMap = new Map();
        const portMap = new Map();
        let viewportPersistTimer = null;

        function persistNodeLayout() {{
          const layout = {{}};
          nodeMap.forEach((element, nodeId) => {{
            layout[nodeId] = {{
              x: Math.round(element.offsetLeft),
              y: Math.round(element.offsetTop),
            }};
          }});
          try {{
            const url = new URL(window.parent.location.href);
            const encoded = btoa(JSON.stringify(layout))
              .replace(/\\+/g, "-")
              .replace(/\\//g, "_")
              .replace(/=+$/g, "");
            url.searchParams.set("canvas_layout", encoded);
            window.parent.history.replaceState({{}}, "", url.toString());
          }} catch (error) {{
            console.warn("Unable to persist canvas layout", error);
          }}
        }}

        function persistViewport() {{
          try {{
            const url = new URL((window.top || window.parent || window).location.href);
            const encoded = btoa(JSON.stringify({{
              pan_x: state.panX,
              pan_y: state.panY,
              zoom: state.zoom,
            }}))
              .replace(/\\+/g, "-")
              .replace(/\\//g, "_")
              .replace(/=+$/g, "");
            url.searchParams.set("canvas_viewport", encoded);
            (window.top || window.parent || window).history.replaceState({{}}, "", url.toString());
          }} catch (error) {{
            console.warn("Unable to persist canvas viewport", error);
          }}
        }}

        function scheduleViewportPersist() {{
          window.clearTimeout(viewportPersistTimer);
          viewportPersistTimer = window.setTimeout(persistViewport, 180);
        }}

        function updateTransform() {{
          surface.style.transform = `translate(${{state.panX}}px, ${{state.panY}}px) scale(${{state.zoom}})`;
          zoomValue.textContent = `${{Math.round(state.zoom * 100)}}%`;
        }}

        function portOffset(index, total) {{
          const base = 82;
          const gap = 34;
          const spread = Math.max(total - 1, 0);
          return base + (spread ? index * gap : 0);
        }}

        function shapeLabel(dataType) {{
          if (dataType === "frame" || dataType === "video") return "square";
          if (dataType === "signal") return "circle";
          if (dataType === "event") return "diamond";
          if (dataType === "analysis") return "hexagon";
          return "pentagon";
        }}

        function createPortMarkup(nodeId, ports, direction) {{
          const filtered = ports.filter(port => port.direction === direction);
          return filtered.map((port, index) => `
            <button
              class="port ${{direction}}"
              data-port-id="${{port.port_id}}"
              data-node-id="${{nodeId}}"
              data-direction="${{port.direction}}"
              data-type="${{port.data_type}}"
              data-required="${{port.required}}"
              style="${{direction === "input" ? "left" : "right"}}:-9px; top:${{portOffset(index, filtered.length)}}px;"
              title="${{port.label}} · ${{shapeLabel(port.data_type)}}"
              type="button"
            >
              <span class="port-shape"></span>
              <span class="port-label" style="margin-top: 10px">${{port.label}}</span>
            </button>
          `).join("");
        }}

        function createNode(node, index) {{
          const element = document.createElement("div");
          element.className = `node state-${{node.state}} ${{node.selected ? "selected" : ""}} ${{node.warnings.length ? "warning" : ""}}`;
          element.dataset.nodeId = node.node_id;
          element.dataset.stageKey = node.stage_key;
          const position = Array.isArray(node.position) && node.position.length === 2
            ? node.position
            : [120 + index * 310, 180];
          element.style.left = `${{Number(position[0]) || (120 + index * 310)}}px`;
          element.style.top = `${{Number(position[1]) || 180}}px`;

          const configJson = JSON.stringify(node.details.configuration, null, 2);
          const executionJson = JSON.stringify(node.details.execution || {}, null, 2);
          const hasExecution = node.details.execution && Object.keys(node.details.execution).length > 0;
          element.innerHTML = `
            ${{createPortMarkup(node.node_id, node.ports, "input")}}
            ${{createPortMarkup(node.node_id, node.ports, "output")}}
            <div class="node-header">
              <div class="node-title">
                <strong>${{node.title}}</strong>
              </div>
              <span class="pill count state-${{node.state}}">
                <span class="state-dot"></span>
                ${{node.state}}
              </span>
            </div>
            <div class="node-meta" style="padding-bottom: 32px;">
              <span class="pill category-${{node.category}}">${{node.category}}</span>
              <span class="pill count">${{node.components.filter(item => item !== "none").length}}</span>
            </div>
            <div class="components">
              ${{node.components.map(component => `<span class="component-chip"><span>${{component}}</span></span>`).join("")}}
            </div>
            ${{node.preview ? `<div class="preview">${{node.preview}}</div>` : ""}}
            ${{node.warnings.length ? `<div class="warning-box">${{node.warnings[0]}}</div>` : ""}}
            <div class="node-actions">
              <button type="button" class="details-toggle">Details</button>
              <button type="button" class="action-button open-editor">Open editor</button>
            </div>
            <div class="details">
              <div class="detail-group">
                <strong>Input type(s)</strong>
                <div class="detail-list">${{node.details.input_types.map(item => `<span class="detail-token">${{item}}</span>`).join("") || "<span class='detail-token'>none</span>"}}</div>
              </div>
              <div class="detail-group">
                <strong>Output type(s)</strong>
                <div class="detail-list">${{node.details.output_types.map(item => `<span class="detail-token">${{item}}</span>`).join("") || "<span class='detail-token'>none</span>"}}</div>
              </div>
              <div class="detail-group">
                <strong>Emitted events</strong>
                <div class="detail-list">${{node.details.emitted_events.map(item => `<span class="detail-token">${{item}}</span>`).join("") || "<span class='detail-token'>none observed</span>"}}</div>
              </div>
              ${{hasExecution ? `
                <div class="detail-group">
                  <strong>Execution</strong>
                  <pre class="config">${{executionJson}}</pre>
                </div>
              ` : ""}}
              <div class="detail-group">
                <strong>Configuration parameters</strong>
                <pre class="config">${{configJson}}</pre>
              </div>
            </div>
          `;
          surface.appendChild(element);
          nodeMap.set(node.node_id, element);
          element.querySelectorAll(".port").forEach(port => portMap.set(port.dataset.portId, port));
          wireNode(element, node);
        }}

        function wireNode(element, node) {{
          const details = element.querySelector(".details");
          const toggle = element.querySelector(".details-toggle");
          toggle.addEventListener("click", event => {{
            event.stopPropagation();
            details.classList.toggle("open");
            toggle.textContent = details.classList.contains("open") ? "Hide details" : "Details";
            drawEdges();
          }});

          const headerClickTargets = [
            element.querySelector(".node-title"),
            element.querySelector(".preview"),
            element.querySelector(".open-editor"),
          ].filter(Boolean);
          headerClickTargets.forEach(target => target.addEventListener("click", () => openStage(node.stage_key)));

          let startPointerX = 0;
          let startPointerY = 0;
          let startNodeX = 0;
          let startNodeY = 0;
          let moved = false;

          element.addEventListener("pointerdown", event => {{
            if (event.target.closest(".port") || event.target.closest(".details-toggle") || event.target.closest(".open-editor")) {{
              return;
            }}
            state.draggedNode = element;
            startPointerX = event.clientX;
            startPointerY = event.clientY;
            startNodeX = element.offsetLeft;
            startNodeY = element.offsetTop;
            moved = false;
            element.setPointerCapture(event.pointerId);
            element.classList.add("dragging");
          }});

          element.addEventListener("pointermove", event => {{
            if (state.draggedNode !== element) return;
            const deltaX = (event.clientX - startPointerX) / state.zoom;
            const deltaY = (event.clientY - startPointerY) / state.zoom;
            moved = moved || Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3;
            const nextX = Math.max(48, Math.min(payload.surface_width - element.offsetWidth - 48, startNodeX + deltaX));
            const nextY = Math.max(64, Math.min(payload.surface_height - element.offsetHeight - 48, startNodeY + deltaY));
            element.style.left = `${{nextX}}px`;
            element.style.top = `${{nextY}}px`;
            drawEdges();
          }});

          function endNodeDrag(event) {{
            if (state.draggedNode !== element) return;
            state.draggedNode = null;
            element.classList.remove("dragging");
            if (moved) {{
              persistNodeLayout();
            }}
            if (!moved) {{
              openStage(node.stage_key);
            }}
            try {{ element.releasePointerCapture(event.pointerId); }} catch (error) {{}}
          }}

          element.addEventListener("pointerup", endNodeDrag);
          element.addEventListener("pointercancel", endNodeDrag);
        }}

        function openStage(stageKey) {{
          try {{
            const topWindow = window.top || window.parent || window;
            const url = new URL(topWindow.location.href);
            url.searchParams.set("stage", stageKey);
            topWindow.location.href = url.toString();
          }} catch (error) {{
            console.warn("Unable to open stage editor", error);
          }}
        }}

        function portCenter(port) {{
          const node = port.closest(".node");
          if (!node) {{
            return [port.offsetLeft + port.offsetWidth / 2, port.offsetTop + port.offsetHeight / 2];
          }}
          const x = node.offsetLeft + port.offsetLeft + port.offsetWidth / 2;
          const y = node.offsetTop + port.offsetTop + port.offsetHeight / 2;
          return [x, y];
        }}

        function edgeClass(kind) {{
          if (kind === "secondary") return "edge edge-secondary";
          if (kind === "event") return "edge edge-event";
          return "edge edge-main";
        }}

        function edgeMarker(kind) {{
          if (kind === "secondary") return "url(#arrowhead-secondary)";
          if (kind === "event") return "url(#arrowhead-event)";
          return "url(#arrowhead-main)";
        }}

        function cubicPath(x1, y1, x2, y2) {{
          const distance = Math.max(120, Math.abs(x2 - x1) * 0.44);
          return `M ${{x1}} ${{y1}} C ${{x1 + distance}} ${{y1}}, ${{x2 - distance}} ${{y2}}, ${{x2}} ${{y2}}`;
        }}

        function clearEdges() {{
          svg.querySelectorAll(".edge, .edge-label, .edge-label-bg, .edge-remove, .edge.temp").forEach(element => element.remove());
        }}

        function drawEdges() {{
          clearEdges();
          state.edges.forEach(edge => {{
            const sourcePort = portMap.get(edge.source_port_id);
            const targetPort = portMap.get(edge.target_port_id);
            if (!sourcePort || !targetPort) return;
            const [x1, y1] = portCenter(sourcePort);
            const [x2, y2] = portCenter(targetPort);
            const pathString = cubicPath(x1, y1, x2, y2);

            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("class", edgeClass(edge.kind));
            path.setAttribute("marker-end", edgeMarker(edge.kind));
            path.setAttribute("d", pathString);
            path.addEventListener("click", event => {{
              event.stopPropagation();
              removeEdge(edge.edge_id);
            }});
            svg.appendChild(path);

            const labelX = (x1 + x2) / 2;
            const labelY = (y1 + y2) / 2 - 16;

            const labelBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            labelBg.setAttribute("class", "edge-label-bg");
            labelBg.setAttribute("x", String(labelX - 52));
            labelBg.setAttribute("y", String(labelY - 12));
            labelBg.setAttribute("width", "104");
            labelBg.setAttribute("height", "24");
            labelBg.addEventListener("click", event => {{
              event.stopPropagation();
              removeEdge(edge.edge_id);
            }});
            svg.appendChild(labelBg);

            const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
            label.setAttribute("class", "edge-label");
            label.setAttribute("x", String(labelX));
            label.setAttribute("y", String(labelY));
            label.textContent = edge.label;
            svg.appendChild(label);

            // const remove = document.createElementNS("http://www.w3.org/2000/svg", "text");
            // remove.setAttribute("class", "edge-remove");
            // remove.setAttribute("x", String(labelX + 30));
            // remove.setAttribute("y", String(labelY));
            // remove.textContent = "×";
            // svg.appendChild(remove);
          }});
        }}

        function removeEdge(edgeId) {{
          state.edges = state.edges.filter(edge => edge.edge_id !== edgeId);
          drawEdges();
          showTooltip(24, 20, "Connection removed from the workspace.");
        }}

        function clearPortValidation() {{
          portMap.forEach(port => port.classList.remove("compatible", "incompatible"));
        }}

        function showTooltip(x, y, message) {{
          tooltip.textContent = message;
          tooltip.style.left = `${{x}}px`;
          tooltip.style.top = `${{y}}px`;
          tooltip.classList.add("show");
          window.clearTimeout(tooltip._timer);
          tooltip._timer = window.setTimeout(() => tooltip.classList.remove("show"), 1500);
        }}

        function viewportToSurface(event) {{
          const rect = viewport.getBoundingClientRect();
          const x = (event.clientX - rect.left - state.panX) / state.zoom;
          const y = (event.clientY - rect.top - state.panY) / state.zoom;
          return [x, y];
        }}

        function bindPorts() {{
          portMap.forEach(port => {{
            if (port.dataset.direction !== "output") return;
            port.addEventListener("pointerdown", event => {{
              event.stopPropagation();
              const [x, y] = portCenter(port);
              state.tempConnection = {{
                sourcePort: port,
                sourceType: port.dataset.type,
              }};
              clearPortValidation();
              portMap.forEach(candidate => {{
                if (candidate.dataset.direction !== "input") return;
                if (candidate.dataset.type === port.dataset.type) {{
                  candidate.classList.add("compatible");
                }} else {{
                  candidate.classList.add("incompatible");
                }}
              }});

              const temp = document.createElementNS("http://www.w3.org/2000/svg", "path");
              temp.setAttribute("class", "edge temp edge-main");
              temp.setAttribute("marker-end", "url(#arrowhead-main)");
              temp.setAttribute("id", "temp-connection");
              temp.setAttribute("d", cubicPath(x, y, x, y));
              svg.appendChild(temp);
              port.setPointerCapture(event.pointerId);
            }});

            port.addEventListener("pointermove", event => {{
              if (!state.tempConnection || state.tempConnection.sourcePort !== port) return;
              const temp = document.getElementById("temp-connection");
              if (!temp) return;
              const [x1, y1] = portCenter(port);
              const [x2, y2] = viewportToSurface(event);
              temp.setAttribute("d", cubicPath(x1, y1, x2, y2));
            }});

            function endPortDrag(event) {{
              if (!state.tempConnection || state.tempConnection.sourcePort !== port) return;
              const dropTarget = document.elementFromPoint(event.clientX, event.clientY)?.closest(".port.input");
              const temp = document.getElementById("temp-connection");
              if (temp) temp.remove();
              if (!dropTarget) {{
                showTooltip(event.clientX - 120, event.clientY - 18, "Drop on an input port to validate the contract.");
              }} else if (dropTarget.dataset.type !== state.tempConnection.sourceType) {{
                showTooltip(event.clientX - 120, event.clientY - 18, `${{state.tempConnection.sourceType}} cannot connect to ${{dropTarget.dataset.type}}.`);
              }} else {{
                createLocalEdge(port, dropTarget);
                showTooltip(event.clientX - 120, event.clientY - 18, "Connection added to the workspace.");
              }}
              clearPortValidation();
              state.tempConnection = null;
              try {{ port.releasePointerCapture(event.pointerId); }} catch (error) {{}}
            }}

            port.addEventListener("pointerup", endPortDrag);
            port.addEventListener("pointercancel", endPortDrag);
          }});
        }}

        function bindViewport() {{
          viewport.addEventListener("wheel", event => {{
            event.preventDefault();
            const direction = event.deltaY > 0 ? -0.02 : 0.02;
            state.zoom = Math.max(0.45, Math.min(1.55, state.zoom + direction));
            updateTransform();
            scheduleViewportPersist();
          }}, {{ passive: false }});

          viewport.addEventListener("pointerdown", event => {{
            if (event.target.closest(".node")) return;
            state.draggingCanvas = true;
            state.dragStartX = event.clientX;
            state.dragStartY = event.clientY;
            state.surfaceStartX = state.panX;
            state.surfaceStartY = state.panY;
            viewport.classList.add("dragging");
            viewport.setPointerCapture(event.pointerId);
          }});

          viewport.addEventListener("pointermove", event => {{
            if (!state.draggingCanvas) return;
            state.panX = state.surfaceStartX + (event.clientX - state.dragStartX);
            state.panY = state.surfaceStartY + (event.clientY - state.dragStartY);
            updateTransform();
            scheduleViewportPersist();
          }});

          function endCanvasDrag(event) {{
            if (!state.draggingCanvas) return;
            state.draggingCanvas = false;
            viewport.classList.remove("dragging");
            persistViewport();
            try {{ viewport.releasePointerCapture(event.pointerId); }} catch (error) {{}}
          }}

          viewport.addEventListener("pointerup", endCanvasDrag);
          viewport.addEventListener("pointercancel", endCanvasDrag);
        }}

        function inferEdgeKind(sourcePort, targetPort) {{
          if (sourcePort.dataset.type === "event") {{
            return sourcePort.dataset.nodeId === "branch_trigger" ? "secondary" : "event";
          }}
          return "main";
        }}

        function createLocalEdge(sourcePort, targetPort) {{
          if (sourcePort.dataset.nodeId === targetPort.dataset.nodeId) {{
            return;
          }}
          state.edges = state.edges.filter(
            edge =>
              edge.target_port_id !== targetPort.dataset.portId &&
              !(
                edge.source_port_id === sourcePort.dataset.portId &&
                edge.target_port_id === targetPort.dataset.portId
              ),
          );

          state.edges.push({{
            edge_id: `local-${{sourcePort.dataset.portId}}-${{targetPort.dataset.portId}}-${{Date.now()}}`,
            source_node_id: sourcePort.dataset.nodeId,
            source_port_id: sourcePort.dataset.portId,
            target_node_id: targetPort.dataset.nodeId,
            target_port_id: targetPort.dataset.portId,
            label: sourcePort.querySelector(".port-label")?.textContent || sourcePort.dataset.type,
            kind: inferEdgeKind(sourcePort, targetPort),
          }});
          drawEdges();
        }}

        payload.nodes.forEach(createNode);
        bindPorts();
        bindViewport();
        updateTransform();
        drawEdges();
      </script>
    </body>
    </html>
    """
