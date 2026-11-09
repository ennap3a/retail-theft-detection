from __future__ import annotations

import html
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import (
    EVALUATION_LOG_PATH,
    PROJECTS_DIR,
    SETTINGS,
    TMP_DIR,
)
from .multipart import MultipartError, parse_multipart
from .project import (
    list_projects,
    load_project,
    paths_for_project,
    read_json,
    update_project,
)
from .review import (
    append_evaluation_row,
    build_review_package,
    create_project,
    ensure_local_review_workspace,
)
from .video import extract_frame, format_duration, is_allowed_video, safe_filename
from .zones import load_zones, save_zones, validate_zone_document, zone_counts


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def js_json(value: object) -> str:
    return json.dumps(value).replace("</", "<\\/")


def page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #08111f;
      --panel: #0f1b2c;
      --panel-2: #111f32;
      --line: #24344d;
      --text: #eef4ff;
      --muted: #9fb1ca;
      --accent: #2dd4bf;
      --accent-2: #60a5fa;
      --danger: #fb7185;
      --warn: #fbbf24;
      --ok: #34d399;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .layout {{ max-width: 1220px; margin: 0 auto; padding: 26px; }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 22px;
    }}
    .brand h1 {{ margin: 0; font-size: 25px; letter-spacing: 0; }}
    .brand p {{ margin: 6px 0 0; color: var(--muted); }}
    .badge {{
      border: 1px solid var(--line);
      color: var(--muted);
      padding: 8px 11px;
      border-radius: 8px;
      white-space: nowrap;
    }}
    .topbar-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .grid {{ display: grid; gap: 18px; }}
    .grid.two {{ grid-template-columns: minmax(0, 1fr) 360px; align-items: start; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .panel h2, .panel h3 {{ margin: 0 0 12px; letter-spacing: 0; }}
    .panel p {{ color: var(--muted); line-height: 1.55; }}
    .upload-box {{
      border: 1px dashed #3f587d;
      background: #0b1728;
      border-radius: 8px;
      padding: 22px;
    }}
    .field {{ display: grid; gap: 8px; margin: 14px 0; }}
    label {{ color: var(--muted); font-size: 14px; }}
    input[type=file], input[type=number], input[type=text], select, textarea {{
      width: 100%;
      color: var(--text);
      background: #07101e;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
    }}
    textarea {{ min-height: 92px; resize: vertical; }}
    button, .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 10px 13px;
      color: #04111f;
      background: var(--accent);
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
    }}
    button.secondary, .button.secondary {{
      background: #17263a;
      color: var(--text);
      border-color: var(--line);
    }}
    button.compact, .button.compact {{
      padding: 8px 11px;
      font-size: 13px;
    }}
    button.warn {{ background: var(--warn); color: #1a1300; }}
    button.danger {{ background: var(--danger); color: #fff; }}
    button:disabled {{ opacity: .45; cursor: not-allowed; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 13px; }}
    .project-list {{ display: grid; gap: 10px; }}
    .project-item {{
      display: grid;
      gap: 7px;
      padding: 13px;
      background: #0b1728;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .project-item strong {{ color: #fff; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{
      border: 1px solid var(--line);
      background: #0a1424;
      color: var(--muted);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
    }}
    .notice {{
      border-left: 4px solid var(--accent-2);
      background: #0a1729;
      padding: 12px 14px;
      border-radius: 8px;
      color: var(--muted);
    }}
    .choice-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 9px;
      margin: 10px 0 14px;
    }}
    .choice {{
      display: flex;
      gap: 8px;
      align-items: flex-start;
      border: 1px solid var(--line);
      background: #0a1424;
      border-radius: 8px;
      padding: 10px;
      color: var(--text);
      line-height: 1.35;
    }}
    .choice input {{ margin-top: 3px; }}
    .canvas-wrap {{
      position: relative;
      width: 100%;
      overflow: auto;
      background: #020711;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    canvas {{
      display: block;
      max-width: 100%;
      height: auto;
      cursor: crosshair;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
      margin: 12px 0;
    }}
    .mode-active {{
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }}
    .zone-list {{ display: grid; gap: 9px; margin-top: 12px; }}
    .zone-card {{
      background: #091422;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .zone-card b {{ display: block; margin-bottom: 5px; }}
    ol.instructions, ul.instructions {{ padding-left: 21px; color: var(--muted); line-height: 1.5; }}
    .status {{
      min-height: 20px;
      margin-top: 10px;
      color: var(--muted);
    }}
    .status.ok {{ color: var(--ok); }}
    .status.err {{ color: var(--danger); }}
    video {{
      width: 100%;
      max-height: 68vh;
      background: #000;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: block;
    }}
    .video-compare {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .video-panel {{
      display: grid;
      gap: 9px;
    }}
    .video-panel h4 {{
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .video-panel video {{
      aspect-ratio: 16 / 9;
      max-height: 52vh;
      object-fit: contain;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      overflow: hidden;
      border-radius: 8px;
    }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ color: var(--accent); font-size: 12px; text-transform: uppercase; }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .thumb-grid img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #020711;
    }}
    @media (max-width: 900px) {{
      .grid.two {{ grid-template-columns: 1fr; }}
      .video-compare {{ grid-template-columns: 1fr; }}
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .topbar-actions {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main class="layout">
    <div class="topbar">
      <div class="brand">
        <h1>Retail Theft Review Assistant</h1>
        <p>Single-video setup, evidence workflow, and human-review safety by design.</p>
      </div>
      <div class="topbar-actions">
        <a class="button secondary compact" href="/">New video</a>
        <span class="badge">Free local demo</span>
      </div>
    </div>
    {body}
  </main>
</body>
</html>"""


def home_page() -> str:
    projects = list_projects()
    items = []
    for project in projects:
        video = project.get("video", {})
        info = video.get("info", {})
        project_id = project.get("project_id", "")
        zones_saved = "yes" if project.get("zones_saved") else "no"
        result = project.get("last_results") or {}
        result_pill = ""
        result_link = ""
        if result:
            result_pill = f"""<span class="pill">alerts: {e(result.get("total_alerts", 0))}</span>"""
            result_link = f"""<a class="button secondary" href="/results/{e(project_id)}">Results</a>"""
        items.append(
            f"""<div class="project-item">
  <strong>{e(video.get("original_filename", "Uploaded video"))}</strong>
  <div class="meta">
    <span class="pill">{e(project_id)}</span>
    <span class="pill">{e(info.get("width", "?"))}x{e(info.get("height", "?"))}</span>
    <span class="pill">{e(format_duration(float(info.get("duration_seconds") or 0)))}</span>
    <span class="pill">zones: {zones_saved}</span>
    {result_pill}
  </div>
  <div class="row">
    <a class="button secondary" href="/setup/{e(project_id)}">Open setup</a>
    <a class="button secondary" href="/project/{e(project_id)}">Project summary</a>
    {result_link}
  </div>
</div>"""
        )
    project_list = "\n".join(items) or "<p class='muted'>No uploaded videos yet.</p>"

    body = f"""<div class="grid two">
  <section class="panel">
    <h2>Start a New Test</h2>
    <p>Upload one CCTV or surveillance video. After upload, you will choose the product zones and customer zone on a setup frame before running any model logic.</p>
    <form class="upload-box" method="post" action="/upload" enctype="multipart/form-data">
      <div class="field">
        <label for="video">Video file</label>
        <input id="video" name="video" type="file" accept="video/*" required>
      </div>
      <button type="submit">Upload and configure zones</button>
      <p class="small muted">Supported extensions: MP4, MOV, AVI, MKV, WEBM. For the free local demo, keep files reasonably short while testing.</p>
    </form>
  </section>

  <aside class="panel">
    <h3>Professional Setup Flow</h3>
    <ol class="instructions">
      <li>Upload one video from one camera.</li>
      <li>Pick a clear setup frame where shelves and walking area are visible.</li>
      <li>Draw one or more product zones tightly around displays.</li>
      <li>Draw exactly one customer zone where browsing should happen.</li>
      <li>Save the setup, then the detector can analyze the sequence.</li>
    </ol>
    <div class="notice">This first stage prepares clean inputs. The next stage connects the single-video detector and evidence generator.</div>
  </aside>
</div>

<section class="panel" style="margin-top:18px">
  <h2>Existing Local Projects</h2>
  <div class="project-list">{project_list}</div>
</section>"""
    return page_shell("Retail Theft Review Assistant", body)


def setup_page(project_id: str, project: dict, zones: dict | None) -> str:
    video = project.get("video", {})
    info = video.get("info", {})
    frame_seconds = float(project.get("setup_frame_seconds") or 0)
    initial_zones = zones or {
        "schema_version": 1,
        "project_id": project_id,
        "frame": {},
        "zones": [],
    }

    body = f"""<div class="grid two">
  <section class="panel">
    <div class="row" style="justify-content:space-between">
      <div>
        <h2>Zone Setup</h2>
        <p class="muted">{e(video.get("original_filename", "Uploaded video"))} - {e(info.get("width", "?"))}x{e(info.get("height", "?"))}, {e(format_duration(float(info.get("duration_seconds") or 0)))}</p>
      </div>
      <a class="button secondary" href="/">Back</a>
    </div>

    <div class="row" style="margin: 12px 0">
      <label for="frameSeconds">Setup frame time</label>
      <input id="frameSeconds" type="number" min="0" step="0.1" value="{frame_seconds:.1f}" style="max-width:140px">
      <button id="loadFrameBtn" class="secondary" type="button">Load frame</button>
      <span class="small muted">Use a clear frame before the suspicious action if possible.</span>
    </div>

    <div class="canvas-wrap">
      <canvas id="zoneCanvas"></canvas>
    </div>

    <div class="toolbar">
      <button id="productModeBtn" type="button">Product zone</button>
      <button id="customerModeBtn" class="secondary" type="button">Customer zone</button>
      <button id="finishBtn" class="secondary" type="button">Finish polygon</button>
      <button id="undoBtn" class="secondary" type="button">Undo point</button>
      <button id="cancelBtn" class="secondary" type="button">Cancel drawing</button>
      <button id="clearBtn" class="danger" type="button">Clear zones</button>
      <button id="saveBtn" class="warn" type="button">Save setup</button>
    </div>
    <div id="status" class="status">Click on the frame to draw the selected zone.</div>
  </section>

  <aside class="panel">
    <h3>How To Draw Zones</h3>
    <ul class="instructions">
      <li><b>Product zone:</b> draw tightly around shelves, racks, tables, counters, or display areas where items are taken from.</li>
      <li><b>Customer zone:</b> draw the browsing area where a person is expected to stand while interacting with products.</li>
      <li>Keep exits, hallways, cashier areas, and unrelated floor space outside the customer zone.</li>
      <li>Use at least 3 points per polygon. More points are better for angled shelves.</li>
      <li>Draw exactly one customer zone and at least one product zone.</li>
    </ul>
    <div class="notice">Good zones reduce false alerts. The model should only alert after a suspicious sequence, not just because a person is holding an item.</div>

    <h3 style="margin-top:18px">Saved Zones</h3>
    <div id="zoneList" class="zone-list"></div>
    <div class="row" style="margin-top:14px">
      <a class="button secondary" href="/project/{e(project_id)}">Project summary</a>
    </div>
  </aside>
</div>

<script>
const projectId = {js_json(project_id)};
let zones = {js_json(initial_zones.get("zones", []))};
let mode = "product";
let currentPoints = [];
let productCounter = zones.filter(z => z.type === "product").length + 1;
let frameUrl = "/media/" + projectId + "/setup-frame.jpg";
let imageVersion = Date.now();

const canvas = document.getElementById("zoneCanvas");
const ctx = canvas.getContext("2d");
const img = new Image();
const statusEl = document.getElementById("status");
const zoneListEl = document.getElementById("zoneList");
const productModeBtn = document.getElementById("productModeBtn");
const customerModeBtn = document.getElementById("customerModeBtn");

const colors = {{
  product: "#2dd4bf",
  customer: "#60a5fa"
}};

function setStatus(text, kind = "") {{
  statusEl.textContent = text;
  statusEl.className = "status " + kind;
}}

function setMode(nextMode) {{
  mode = nextMode;
  productModeBtn.className = mode === "product" ? "mode-active" : "secondary";
  customerModeBtn.className = mode === "customer" ? "mode-active" : "secondary";
  setStatus("Drawing " + (mode === "product" ? "product zone" : "customer zone") + ". Click points on the frame.");
}}

function loadImage() {{
  img.onload = () => {{
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    draw();
  }};
  img.onerror = () => setStatus("Could not load setup frame.", "err");
  img.src = frameUrl + "?v=" + imageVersion;
}}

function canvasPoint(event) {{
  const rect = canvas.getBoundingClientRect();
  return {{
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height)
  }};
}}

function drawPolygon(points, color, fill, label) {{
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  if (fill && points.length >= 3) ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();
  if (fill && points.length >= 3) {{
    ctx.fillStyle = color + "33";
    ctx.fill();
  }}
  for (const point of points) {{
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }}
  if (label && points[0]) {{
    ctx.fillStyle = "#020711";
    ctx.fillRect(points[0].x + 8, points[0].y - 22, Math.max(120, label.length * 8), 24);
    ctx.fillStyle = "#ffffff";
    ctx.font = "15px system-ui";
    ctx.fillText(label, points[0].x + 14, points[0].y - 5);
  }}
}}

function draw() {{
  if (!img.complete) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  for (const zone of zones) {{
    drawPolygon(zone.points, colors[zone.type] || "#ffffff", true, zone.label);
  }}
  drawPolygon(currentPoints, colors[mode] || "#ffffff", false, "drawing");
  renderZoneList();
}}

function renderZoneList() {{
  if (!zones.length) {{
    zoneListEl.innerHTML = "<p class='muted small'>No zones saved in this setup yet.</p>";
    return;
  }}
  zoneListEl.innerHTML = zones.map((zone, index) => `
    <div class="zone-card">
      <b>${{escapeHtml(zone.label)}}</b>
      <div class="small muted">${{zone.type}} - ${{zone.points.length}} points</div>
      <button class="secondary" type="button" onclick="deleteZone(${{index}})" style="margin-top:8px">Delete</button>
    </div>
  `).join("");
}}

function escapeHtml(text) {{
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}}

function finishPolygon() {{
  if (currentPoints.length < 3) {{
    setStatus("Draw at least 3 points before finishing a zone.", "err");
    return;
  }}
  if (mode === "customer") {{
    zones = zones.filter(zone => zone.type !== "customer");
  }}
  const id = mode === "customer" ? "customer_zone" : "product_zone_" + productCounter++;
  const label = mode === "customer" ? "Customer Zone" : "Product Zone " + (productCounter - 1);
  zones.push({{
    id,
    type: mode,
    label,
    points: currentPoints.map(point => ({{
      x: Math.round(point.x * 10) / 10,
      y: Math.round(point.y * 10) / 10
    }}))
  }});
  currentPoints = [];
  setStatus(label + " added. Save setup when all zones are ready.", "ok");
  draw();
}}

function deleteZone(index) {{
  zones.splice(index, 1);
  draw();
}}

function buildZoneDocument() {{
  return {{
    schema_version: 1,
    project_id: projectId,
    frame: {{
      width: canvas.width,
      height: canvas.height,
      source: "setup_frame.jpg",
      seconds: Number(document.getElementById("frameSeconds").value || 0)
    }},
    zones,
    instructions_version: "single_video_stage_1"
  }};
}}

function validateBeforeSave() {{
  const productCount = zones.filter(zone => zone.type === "product").length;
  const customerCount = zones.filter(zone => zone.type === "customer").length;
  if (productCount < 1) return "Draw at least one product zone.";
  if (customerCount !== 1) return "Draw exactly one customer zone.";
  return "";
}}

async function saveSetup() {{
  const error = validateBeforeSave();
  if (error) {{
    setStatus(error, "err");
    return;
  }}
  const response = await fetch("/api/projects/" + projectId + "/zones", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify(buildZoneDocument())
  }});
  const data = await response.json();
  if (!response.ok) {{
    setStatus(data.error || "Could not save zones.", "err");
    return;
  }}
  setStatus("Setup saved. This video is ready for the detector stage.", "ok");
}}

async function loadSetupFrame() {{
  const seconds = Number(document.getElementById("frameSeconds").value || 0);
  const response = await fetch("/api/projects/" + projectId + "/setup-frame", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ seconds }})
  }});
  const data = await response.json();
  if (!response.ok) {{
    setStatus(data.error || "Could not load setup frame.", "err");
    return;
  }}
  imageVersion = Date.now();
  loadImage();
  setStatus("Setup frame updated. Existing zones remain, but review their alignment.", "ok");
}}

canvas.addEventListener("click", event => {{
  currentPoints.push(canvasPoint(event));
  draw();
}});
productModeBtn.addEventListener("click", () => setMode("product"));
customerModeBtn.addEventListener("click", () => setMode("customer"));
document.getElementById("finishBtn").addEventListener("click", finishPolygon);
document.getElementById("undoBtn").addEventListener("click", () => {{
  currentPoints.pop();
  draw();
}});
document.getElementById("cancelBtn").addEventListener("click", () => {{
  currentPoints = [];
  draw();
}});
document.getElementById("clearBtn").addEventListener("click", () => {{
  if (confirm("Clear all zones for this setup?")) {{
    zones = [];
    currentPoints = [];
    productCounter = 1;
    draw();
  }}
}});
document.getElementById("saveBtn").addEventListener("click", saveSetup);
document.getElementById("loadFrameBtn").addEventListener("click", loadSetupFrame);

setMode("product");
loadImage();
</script>"""
    return page_shell("Zone Setup", body)


def project_page(project_id: str, project: dict, zones: dict | None) -> str:
    video = project.get("video", {})
    info = video.get("info", {})
    counts = zone_counts(zones)
    ready = counts["product"] >= 1 and counts["customer"] == 1
    result = project.get("last_results") or {}
    stage = project.get("stage", "setup")
    if stage == "detector_running":
        status = "Detector running"
    elif result:
        status = f"Detector complete - {result.get('total_alerts', 0)} alerts"
    elif ready:
        status = "Ready for detector stage"
    else:
        status = "Zone setup incomplete"
    zone_rows = []
    for zone in (zones or {}).get("zones", []):
        zone_rows.append(
            f"""<div class="zone-card">
  <b>{e(zone.get("label", zone.get("id", "Zone")))}</b>
  <div class="small muted">{e(zone.get("type", ""))} - {len(zone.get("points", []))} points</div>
</div>"""
        )
    zone_html = "\n".join(zone_rows) or "<p class='muted'>No zones saved yet.</p>"
    if ready:
        run_control = f"""<form method="post" action="/run/{e(project_id)}">
        <button type="submit">Run detector</button>
      </form>"""
    else:
        run_control = """<button disabled title="Complete zone setup first">Run detector</button>"""
    results_link = ""
    if result:
        results_link = f"""<a class="button secondary" href="/results/{e(project_id)}">View results</a>"""

    body = f"""<div class="grid two">
  <section class="panel">
    <div class="row" style="justify-content:space-between">
      <div>
        <h2>Project Summary</h2>
        <p class="muted">{e(project_id)}</p>
      </div>
      <a class="button secondary" href="/">Home</a>
    </div>
    <div class="meta" style="margin:14px 0">
      <span class="pill">{e(status)}</span>
      <span class="pill">{e(info.get("width", "?"))}x{e(info.get("height", "?"))}</span>
      <span class="pill">{e(format_duration(float(info.get("duration_seconds") or 0)))}</span>
      <span class="pill">{e(round(float(info.get("fps") or 0), 2))} FPS</span>
    </div>
    <h3>Video</h3>
    <p>{e(video.get("original_filename", ""))}</p>
    <div class="row">
      <a class="button" href="/setup/{e(project_id)}">Edit setup zones</a>
      {run_control}
      {results_link}
    </div>
    <div class="notice" style="margin-top:14px">Stage 2 runs a first single-video YOLO/tracking/pose pass. It produces suspected-event evidence for human review, not an automatic accusation.</div>
  </section>

  <aside class="panel">
    <h3>Zone Readiness</h3>
    <div class="meta">
      <span class="pill">product zones: {counts["product"]}</span>
      <span class="pill">customer zones: {counts["customer"]}</span>
    </div>
    <div class="zone-list">{zone_html}</div>
  </aside>
</div>"""
    return page_shell("Project Summary", body)


def result_media_url(project_id: str, relative_path: str) -> str:
    return f"/media/{e(project_id)}/result/{e(relative_path)}"


def project_media_url(project_id: str, relative_path: str) -> str:
    return f"/media/{e(project_id)}/file/{e(relative_path)}"


def results_page(project_id: str, project: dict, evaluation_saved: bool = False) -> str:
    paths = paths_for_project(project_id)
    summary_path = paths.results_dir / "summary.json"
    alerts_path = paths.results_dir / "alerts.json"
    if not summary_path.exists() or not alerts_path.exists():
        body = f"""<section class="panel">
  <h2>No Results Yet</h2>
  <p class="muted">Run the detector from the project summary page first.</p>
  <a class="button secondary" href="/project/{e(project_id)}">Back to project</a>
</section>"""
        return page_shell("Results", body)

    summary = read_json(summary_path)
    alerts_doc = read_json(alerts_path)
    alerts = alerts_doc.get("alerts", [])
    results = summary.get("results", {})
    total_alerts = int(results.get("total_alerts", 0) or 0)
    result_message = (
        f"{total_alerts} human-review alert{'s' if total_alerts != 1 else ''} generated."
        if total_alerts
        else "No human-review alerts were generated for this video."
    )
    video_rel = results.get("annotated_video", "")
    csv_rel = results.get("alerts_csv", "")
    summary_csv_rel = results.get("summary_csv", "")
    annotated_video_html = ""
    if video_rel:
        annotated_video_html = f"""<video controls preload="metadata">
  <source src="{result_media_url(project_id, video_rel)}" type="video/mp4">
</video>"""
    original_video_html = ""
    stored_filename = project.get("video", {}).get("stored_filename", "")
    if stored_filename:
        original_rel = f"input/{stored_filename}"
        original_mime = mimetypes.guess_type(stored_filename)[0] or "video/mp4"
        original_video_html = f"""<video controls preload="metadata">
  <source src="{project_media_url(project_id, original_rel)}" type="{e(original_mime)}">
</video>
<a class="button secondary" href="{project_media_url(project_id, original_rel)}" target="_blank">Open original video</a>"""

    rows = []
    cards = []
    for alert in alerts:
        alert_id = alert.get("alert_id", "Alert")
        rows.append(
            f"""<tr>
  <td>{e(alert_id)}</td>
  <td>{e(alert.get("person_id", ""))}</td>
  <td>{e(alert.get("time_seconds", ""))}s</td>
  <td>{e(alert.get("suspicion_score", ""))}</td>
  <td>{e(alert.get("product_zone_label", alert.get("product_zone", "")))}</td>
  <td>{e(alert.get("notes", ""))}</td>
</tr>"""
        )
        thumbs = []
        for label, key in (
            ("Full frame", "full_frame_image"),
            ("Person crop", "person_crop_image"),
            ("Suspected item", "suspected_item_crop_image"),
            ("Changed-zone debug", "debug_image"),
        ):
            rel = alert.get(key, "")
            if rel:
                thumbs.append(
                    f"""<div>
  <p class="small muted">{e(label)}</p>
  <img src="{result_media_url(project_id, rel)}" alt="{e(label)}">
</div>"""
                )
        clip_link = ""
        if alert.get("evidence_clip"):
            clip_link = f"""<a class="button secondary" href="{result_media_url(project_id, alert["evidence_clip"])}" target="_blank">Open evidence clip</a>"""
        cards.append(
            f"""<section class="panel">
  <h3>{e(alert_id)}</h3>
  <div class="meta">
    <span class="pill">person: {e(alert.get("person_id", ""))}</span>
    <span class="pill">time: {e(alert.get("time_seconds", ""))}s</span>
    <span class="pill">score: {e(alert.get("suspicion_score", ""))}</span>
    <span class="pill">{e(alert.get("review_priority", "REVIEW"))}</span>
  </div>
  <div class="thumb-grid">{''.join(thumbs) or "<p class='muted'>No images saved for this alert.</p>"}</div>
  <div class="row" style="margin-top:12px">{clip_link}</div>
</section>"""
        )

    table = "<p class='muted'>No human-review alerts were generated. Review the annotated video for detector behavior.</p>"
    if rows:
        table = f"""<table>
  <thead><tr><th>Alert</th><th>Person</th><th>Time</th><th>Score</th><th>Zone</th><th>Notes</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""

    csv_link = ""
    if csv_rel:
        csv_link = f"""<a class="button secondary" href="{result_media_url(project_id, csv_rel)}" download>Download alerts CSV</a>"""
    summary_csv_link = ""
    if summary_csv_rel:
        summary_csv_link = f"""<a class="button secondary" href="{result_media_url(project_id, summary_csv_rel)}" download>Download summary CSV</a>"""
    package_link = f"""<a class="button" href="/export/{e(project_id)}">Download review package</a>"""
    evaluation_notice = ""
    if evaluation_saved:
        evaluation_notice = """<div class="notice" style="margin-top:12px">Evaluation saved to the local CSV log.</div>"""

    evaluation_form = f"""<section class="panel">
    <h3>Evaluation Log</h3>
    <p class="muted">After reviewing this result, answer these questions once. The app will append the row to the local evaluation CSV.</p>
    {evaluation_notice}
    <form method="post" action="/evaluate/{e(project_id)}">
      <div class="field">
        <label>What really happens in this video?</label>
        <div class="choice-grid">
          <label class="choice"><input type="radio" name="real_case" value="normal" required> Normal shopping / no theft</label>
          <label class="choice"><input type="radio" name="real_case" value="suspicious" required> Suspicious or shoplifting behavior</label>
          <label class="choice"><input type="radio" name="real_case" value="unsure" required> Unsure / needs another review</label>
        </div>
      </div>

      <div class="field">
        <label>How did the model perform?</label>
        <div class="choice-grid">
          <label class="choice"><input type="radio" name="model_review" value="correct_no_alert" required> Correct: no alert needed</label>
          <label class="choice"><input type="radio" name="model_review" value="correct_alert" required> Correct: alert matched the suspicious action</label>
          <label class="choice"><input type="radio" name="model_review" value="false_alert" required> False alert: normal behavior was flagged</label>
          <label class="choice"><input type="radio" name="model_review" value="missed_event" required> Missed event: suspicious action was not caught</label>
          <label class="choice"><input type="radio" name="model_review" value="wrong_timestamp" required> Alert happened at the wrong time</label>
          <label class="choice"><input type="radio" name="model_review" value="unsure" required> Unsure / needs another review</label>
        </div>
      </div>

      <div class="field">
        <label>What caused the mistake? Choose all that apply.</label>
        <div class="choice-grid">
          <label class="choice"><input type="checkbox" name="mistake_reason" value="item_visible_in_hand"> Item was visible in hand</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="item_returned_to_shelf"> Item was returned to shelf</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="bag_false_alert"> Bag caused false alert</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="clothing_body_false_alert"> Clothing/body false alert</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="missed_small_product"> Missed small product</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="bad_zones"> Zone drawing needs improvement</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="lighting_occlusion_angle"> Lighting, occlusion, or camera angle</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="tracking_issue"> Tracking ID issue</label>
          <label class="choice"><input type="checkbox" name="mistake_reason" value="other"> Other</label>
        </div>
      </div>

      <div class="field">
        <label for="badTimestamp">Problem timestamp, if any</label>
        <input id="badTimestamp" name="bad_timestamp" type="text" placeholder="Example: 00:14 or 14.2s">
      </div>

      <div class="field">
        <label for="reviewerNotes">Notes</label>
        <textarea id="reviewerNotes" name="reviewer_notes" placeholder="Short note about what happened and what should improve."></textarea>
      </div>

      <div class="row">
        <button type="submit">Save evaluation row</button>
        <a class="button secondary" href="/evaluation-log" download>Download evaluation CSV</a>
      </div>
    </form>
  </section>"""

    body = f"""<div class="grid">
  <section class="panel">
    <div class="row" style="justify-content:space-between">
      <div>
        <h2>Detector Results</h2>
        <p class="muted">{e(project.get("video", {}).get("original_filename", ""))}</p>
      </div>
      <div class="row">
        <a class="button secondary" href="/project/{e(project_id)}">Project summary</a>
        {package_link}
        {csv_link}
        {summary_csv_link}
      </div>
    </div>
    <div class="meta" style="margin:14px 0">
      <span class="pill">alerts: {e(results.get("total_alerts", 0))}</span>
      <span class="pill">processed frames: {e(summary.get("video", {}).get("processed_frames", 0))}</span>
      <span class="pill">analysis FPS: {e(summary.get("video", {}).get("analysis_fps", 0))}</span>
      <span class="pill">runtime: {e(results.get("runtime_seconds", 0))}s</span>
    </div>
    <div class="status" style="font-weight:700">{e(result_message)}</div>
    <div class="notice">Responsible AI: these are suspected events for trained human review. They are not automatic accusations.</div>
  </section>

  <section class="panel">
    <h3>Video Comparison</h3>
    <div class="video-compare">
      <div class="video-panel">
        <h4>Original Camera Video</h4>
        {original_video_html or "<p class='muted'>Original video is not available.</p>"}
      </div>
      <div class="video-panel">
        <h4>Annotated Model Video</h4>
        {annotated_video_html or "<p class='muted'>No annotated video was generated.</p>"}
      </div>
    </div>
  </section>

  <section class="panel">
    <h3>Alert Summary</h3>
    {table}
  </section>
  {evaluation_form}
  {''.join(cards)}
</div>"""
    return page_shell("Detector Results", body)


class RetailTheftHandler(BaseHTTPRequestHandler):
    server_version = "RetailTheftReview/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_html(self, html_text: str, status: int = 200) -> None:
        data = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def read_form_body(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length)
        return parse_qs(raw.decode("utf-8"), keep_blank_values=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        parts = [part for part in path.split("/") if part]

        try:
            if path == "/":
                self.send_html(home_page())
                return
            if path == "/evaluation-log":
                ensure_local_review_workspace()
                self.send_file(EVALUATION_LOG_PATH, download_name="evaluation_log.csv")
                return
            if len(parts) == 2 and parts[0] == "setup":
                project_id = parts[1]
                project = load_project(project_id)
                zones = load_zones(paths_for_project(project_id).zones_path)
                self.send_html(setup_page(project_id, project, zones))
                return
            if len(parts) == 2 and parts[0] == "project":
                project_id = parts[1]
                project = load_project(project_id)
                zones = load_zones(paths_for_project(project_id).zones_path)
                self.send_html(project_page(project_id, project, zones))
                return
            if len(parts) == 2 and parts[0] == "results":
                project_id = parts[1]
                project = load_project(project_id)
                query = parse_qs(parsed.query)
                self.send_html(results_page(project_id, project, evaluation_saved=query.get("evaluation_saved") == ["1"]))
                return
            if len(parts) == 2 and parts[0] == "export":
                project_id = parts[1]
                project = load_project(project_id)
                package_path = build_review_package(project_id, project)
                self.send_file(package_path, download_name=package_path.name)
                return
            if len(parts) == 3 and parts[0] == "media" and parts[2] == "setup-frame.jpg":
                self.send_file(paths_for_project(parts[1]).first_frame_path)
                return
            if len(parts) >= 4 and parts[0] == "media" and parts[2] == "result":
                self.send_project_file(parts[1], Path(*parts[3:]))
                return
            if len(parts) >= 4 and parts[0] == "media" and parts[2] == "file":
                self.send_project_file(parts[1], Path(*parts[3:]))
                return
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "projects":
                project_id = parts[2]
                project = load_project(project_id)
                zones = load_zones(paths_for_project(project_id).zones_path)
                self.send_json({"project": project, "zones": zones})
                return
        except FileNotFoundError:
            self.send_html(page_shell("Not Found", "<section class='panel'><h2>Project not found</h2><p class='muted'>The requested local project does not exist.</p><a class='button secondary' href='/'>Home</a></section>"), status=404)
            return
        except Exception as exc:
            self.send_error_page(str(exc), status=500)
            return

        self.send_error_page("Page not found.", status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        parts = [part for part in path.split("/") if part]

        try:
            if path == "/upload":
                self.handle_upload()
                return
            if len(parts) == 2 and parts[0] == "run":
                self.handle_run_detector(parts[1])
                return
            if len(parts) == 2 and parts[0] == "evaluate":
                self.handle_save_evaluation(parts[1])
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "projects" and parts[3] == "zones":
                self.handle_save_zones(parts[2])
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "projects" and parts[3] == "setup-frame":
                self.handle_setup_frame(parts[2])
                return
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self.send_json({"ok": False, "error": "Endpoint not found."}, status=404)

    def send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error_page("File not found.", status=404)
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{safe_filename(download_name)}"')
        self.end_headers()
        self.wfile.write(data)

    def send_project_file(self, project_id: str, relative_path: Path) -> None:
        paths = paths_for_project(project_id)
        base = paths.base_dir.resolve()
        target = (paths.base_dir / relative_path).resolve()
        if base not in target.parents and target != base:
            self.send_error_page("Invalid file path.", status=400)
            return
        self.send_file(target)

    def send_error_page(self, message: str, status: int) -> None:
        self.send_html(
            page_shell(
                "Error",
                f"<section class='panel'><h2>Error</h2><p class='muted'>{e(message)}</p><a class='button secondary' href='/'>Home</a></section>",
            ),
            status=status,
        )

    def handle_upload(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        max_bytes = SETTINGS.max_upload_mb * 1024 * 1024
        if content_length > max_bytes:
            self.send_error_page(f"Upload is larger than {SETTINGS.max_upload_mb} MB.", status=413)
            return

        try:
            form = parse_multipart(
                self.rfile,
                self.headers.get("Content-Type", ""),
                content_length,
                tmp_dir=TMP_DIR,
            )
        except MultipartError as exc:
            self.send_error_page(str(exc), status=400)
            return

        try:
            upload = form.first_file("video")
            if upload is None:
                self.send_error_page("Upload form did not include a video file.", status=400)
                return
            if not upload.filename or upload.size == 0:
                self.send_error_page("Choose a video file before uploading.", status=400)
                return
            if not is_allowed_video(upload.filename):
                self.send_error_page("Unsupported video type.", status=400)
                return
            project = create_project(upload.path, upload.filename)
        except RuntimeError as exc:
            self.send_error_page(str(exc), status=400)
            return
        finally:
            form.cleanup()

        self.redirect(f"/setup/{project['project_id']}")

    def handle_save_zones(self, project_id: str) -> None:
        load_project(project_id)
        doc = self.read_json_body()
        errors = validate_zone_document(doc)
        if errors:
            self.send_json({"ok": False, "error": " ".join(errors)}, status=400)
            return
        save_zones(paths_for_project(project_id).zones_path, doc)
        update_project(project_id, {"zones_saved": True, "stage": "ready_for_detector"})
        self.send_json({"ok": True, "message": "Zones saved."})

    def handle_run_detector(self, project_id: str) -> None:
        load_project(project_id)
        paths = paths_for_project(project_id)
        if not paths.zones_path.exists():
            self.send_error_page("Save setup zones before running the detector.", status=400)
            return
        zones = load_zones(paths.zones_path)
        errors = validate_zone_document(zones or {})
        if errors:
            self.send_error_page(" ".join(errors), status=400)
            return
        # Imported lazily: loading Ultralytics/PyTorch takes several seconds.
        from .detector import DetectorSettings, analyze_project

        analyze_project(project_id, DetectorSettings())
        self.redirect(f"/results/{project_id}")

    def handle_save_evaluation(self, project_id: str) -> None:
        project = load_project(project_id)
        form = self.read_form_body()
        row = append_evaluation_row(project_id, project, form)
        update_project(
            project_id,
            {
                "last_evaluation": {
                    "reviewed_at": row["reviewed_at"],
                    "result_type": row["result_type"],
                    "real_case": row["real_case"],
                    "model_review": row["model_review"],
                }
            },
        )
        self.redirect(f"/results/{project_id}?evaluation_saved=1")

    def handle_setup_frame(self, project_id: str) -> None:
        project = load_project(project_id)
        body = self.read_json_body()
        seconds = float(body.get("seconds") or 0.0)
        duration = float(project.get("video", {}).get("info", {}).get("duration_seconds") or 0)
        if seconds < 0:
            seconds = 0.0
        if duration and seconds > duration:
            seconds = duration
        video_path = Path(project["video"]["path"])
        paths = paths_for_project(project_id)
        extract_frame(video_path, paths.first_frame_path, at_seconds=seconds)
        update_project(project_id, {"setup_frame_seconds": round(seconds, 3)})
        self.send_json({"ok": True, "seconds": round(seconds, 3)})


def main() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_local_review_workspace()
    server = ThreadingHTTPServer((SETTINGS.host, SETTINGS.port), RetailTheftHandler)
    url = f"http://{SETTINGS.host}:{SETTINGS.port}"
    print(f"Retail Theft Review Assistant running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
