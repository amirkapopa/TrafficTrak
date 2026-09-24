"""Scene-geometry calibration helpers.

    # 1. interactive, browser-based (no GUI toolkit needed): click shapes, copy YAML
    python scripts/calibrate_geometry.py html   --image outputs/eda/reference_median.jpg
    # 2. review the current config/camera_geometry.yaml on a frame
    python scripts/calibrate_geometry.py render --image outputs/eda/reference_median.jpg
    # 3. frame with a normalised-coordinate grid (to read coordinates by eye)
    python scripts/calibrate_geometry.py grid   --image outputs/eda/reference_median.jpg
    # grab a frame from a video instead of an image:
    python scripts/calibrate_geometry.py grid   --video samples/clip.mp4 --at 5

Outputs go to outputs/calibration/.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import yaml
from _bootstrap import ROOT

from src.config import geometry_config_path
from src.geometry import load_geometry
from src.visualize import draw_geometry

OUT = ROOT / "outputs" / "calibration"


def load_frame(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"cannot read image {args.image}")
        return img
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read a frame from {args.video}")
    return img


def cmd_render(args, img):
    g = load_geometry(args.geometry, img.shape[1], img.shape[0])
    out = OUT / "geometry_overlay.jpg"
    cv2.imwrite(str(out), draw_geometry(img, g))
    print(json.dumps(g.summary(), indent=2))
    for w in g.warnings:
        print("WARNING:", w)
    print(f"-> {out}")


def cmd_grid(args, img):
    h, w = img.shape[:2]
    out = img.copy()
    for k in range(1, 20):
        x, y = int(w * k / 20), int(h * k / 20)
        col = (0, 255, 255) if k % 5 == 0 else (160, 160, 160)
        cv2.line(out, (x, 0), (x, h), col, 1)
        cv2.line(out, (0, y), (w, y), col, 1)
        cv2.putText(out, f"{k / 20:.2f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        cv2.putText(out, f"{k / 20:.2f}", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    path = OUT / "grid.jpg"
    cv2.imwrite(str(path), out)
    print(f"-> {path}")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Geometry calibration</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#fff; --muted:#c3c2b7; --line:#3a3a38; --accent:#3987e5; } }
body { margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--ink); }
main { display:grid; grid-template-columns: minmax(0,1fr) 340px; gap:16px; padding:16px; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
canvas { width:100%; height:auto; border:1px solid var(--line); cursor:crosshair; }
select, input, button, textarea { font:inherit; color:inherit; background:transparent; border:1px solid var(--line); border-radius:6px; padding:6px 8px; }
button { cursor:pointer; } button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
.row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
textarea { width:100%; height:320px; box-sizing:border-box; font-family: ui-monospace, monospace; font-size:12px; }
.hint { color:var(--muted); font-size:12px; }
ul { padding-left:18px; }
</style></head><body><main>
<div><canvas id="c"></canvas><p class="hint" id="pos">move over the image</p></div>
<div>
  <div class="row">
    <select id="kind">
      <option value="carriageway">carriageway polygon</option>
      <option value="lane_polygon">lane polygon</option>
      <option value="lane_direction">lane direction path (in driving order)</option>
      <option value="stop_line">stop line (2 points)</option>
      <option value="stop_approach">stop-line approach (2 points, travel direction)</option>
      <option value="crossing">pedestrian crossing</option>
      <option value="intersection">intersection box</option>
      <option value="signal">signal head ROI (2 corners)</option>
      <option value="solid_line">solid line (polyline)</option>
      <option value="turn_from">prohibited turn: from zone</option>
      <option value="turn_to">prohibited turn: to zone</option>
      <option value="exclusion_zones">exclusion zone (parking / bus stop)</option>
      <option value="sidewalks">sidewalk</option>
      <option value="u_turn_prohibited_zones">no-U-turn zone</option>
      <option value="obstacle_regions">obstacle search region</option>
    </select>
    <input id="name" placeholder="id (e.g. nb_1)" size="10">
  </div>
  <div class="row">
    <button class="primary" id="finish">Finish shape</button>
    <button id="undo">Undo point</button>
    <button id="clear">Discard shape</button>
  </div>
  <p class="hint">Click to add points. Lanes: draw the polygon, then its direction path with the same id; set <code>group</code> in the YAML.
  Stop lines: line + approach with the same id. Coordinates are normalised to the image size.</p>
  <ul id="shapes"></ul>
  <textarea id="yaml" spellcheck="false"></textarea>
  <div class="row"><button id="copy">Copy YAML</button></div>
</div></main>
<script>
const IMG = "data:image/jpeg;base64,__IMG__";
const EXISTING = __EXISTING__;
const img = new Image(); const c = document.getElementById('c'); const ctx = c.getContext('2d');
let shapes = [], cur = [];
const colors = {carriageway:'#1baf7a', lane_polygon:'#eda100', lane_direction:'#eda100', stop_line:'#e34948', stop_approach:'#e34948',
  crossing:'#ffffff', intersection:'#00c8f0', signal:'#e87ba4', solid_line:'#ffe000', turn_from:'#eb6834', turn_to:'#eb6834',
  exclusion_zones:'#999999', sidewalks:'#999999', u_turn_prohibited_zones:'#eb6834', obstacle_regions:'#4a3aa7'};
const r4 = v => Math.round(v * 10000) / 10000;
function norm(e) { const b = c.getBoundingClientRect(); return [r4((e.clientX - b.left) / b.width), r4((e.clientY - b.top) / b.height)]; }
function drawShape(s, dashed) {
  const pts = s.points.map(p => [p[0] * c.width, p[1] * c.height]); if (!pts.length) return;
  ctx.strokeStyle = colors[s.kind] || '#fff'; ctx.lineWidth = 3; ctx.setLineDash(dashed ? [8, 6] : []);
  ctx.beginPath();
  if (s.kind === 'signal' && pts.length === 2) { ctx.rect(pts[0][0], pts[0][1], pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]); }
  else { pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    if (/carriageway|polygon|crossing|intersection|zone|turn_|sidewalk|regions/.test(s.kind) && !dashed) ctx.closePath(); }
  ctx.stroke(); pts.forEach(p => { ctx.fillStyle = ctx.strokeStyle; ctx.fillRect(p[0] - 3, p[1] - 3, 6, 6); });
  if (s.name) { ctx.fillStyle = '#000'; ctx.fillRect(pts[0][0], pts[0][1] - 18, ctx.measureText(s.name).width + 8, 18);
    ctx.fillStyle = '#fff'; ctx.fillText(s.name, pts[0][0] + 4, pts[0][1] - 5); }
}
function redraw() { ctx.setLineDash([]); ctx.drawImage(img, 0, 0); ctx.font = '14px system-ui';
  shapes.forEach(s => drawShape(s, false)); drawShape({kind: document.getElementById('kind').value, points: cur}, true); render(); }
function yamlPts(p) { return '[' + p.map(q => '[' + q[0] + ', ' + q[1] + ']').join(', ') + ']'; }
function render() {
  const by = k => shapes.filter(s => s.kind === k); const L = [];
  const poly = (key, kind) => { const xs = by(kind); L.push(key + ':' + (xs.length ? '' : ' []')); xs.forEach(s => L.push('  - ' + yamlPts(s.points))); };
  L.push('calibrated: false   # set true after reviewing the overlay'); poly('carriageway', 'carriageway');
  poly('exclusion_zones', 'exclusion_zones'); poly('sidewalks', 'sidewalks');
  const lp = by('lane_polygon'); L.push('lanes:' + (lp.length ? '' : ' []'));
  lp.forEach(s => { const d = by('lane_direction').find(x => x.name === s.name);
    L.push('  - id: ' + s.name, '    group: CHANGE_ME', '    polygon: ' + yamlPts(s.points), '    direction: ' + (d ? yamlPts(d.points) : '[]  # draw a direction path with the same id')); });
  const sl = by('stop_line'); L.push('stop_lines:' + (sl.length ? '' : ' []'));
  sl.forEach(s => { const a = by('stop_approach').find(x => x.name === s.name);
    L.push('  - id: ' + s.name, '    line: ' + yamlPts(s.points.slice(0, 2)), '    signal: main', '    lanes: []');
    if (a) L.push('    approach: ' + yamlPts(a.points.slice(0, 2))); });
  const cw = by('crossing'); L.push('crossings:' + (cw.length ? '' : ' []'));
  cw.forEach(s => L.push('  - id: ' + s.name, '    polygon: ' + yamlPts(s.points)));
  const it = by('intersection'); L.push('intersection: ' + (it.length ? yamlPts(it[0].points) : 'null'));
  const sg = by('signal'); L.push('signals:' + (sg.length ? '' : ' []'));
  sg.forEach(s => { const p = s.points; if (p.length < 2) return;
    L.push('  - id: ' + (s.name || 'main'), '    roi: [' + [Math.min(p[0][0], p[1][0]), Math.min(p[0][1], p[1][1]), Math.max(p[0][0], p[1][0]), Math.max(p[0][1], p[1][1])].join(', ') + ']',
           '    layout: vertical', '    lamps: [red, amber, green]'); });
  const so = by('solid_line'); L.push('solid_lines:' + (so.length ? '' : ' []'));
  so.forEach(s => L.push('  - id: ' + s.name, '    points: ' + yamlPts(s.points)));
  const tf = by('turn_from'); L.push('prohibited_turns:' + (tf.length ? '' : ' []'));
  tf.forEach(s => { const t = by('turn_to').find(x => x.name === s.name);
    L.push('  - id: ' + s.name, '    label: illegal_turn', '    from: ' + yamlPts(s.points), '    to: ' + (t ? yamlPts(t.points) : '[]'), '    max_duration_sec: 12'); });
  L.push('u_turn_prohibited: false'); poly('u_turn_prohibited_zones', 'u_turn_prohibited_zones'); poly('obstacle_regions', 'obstacle_regions');
  document.getElementById('yaml').value = L.join('\n') + '\n';
  const ul = document.getElementById('shapes'); ul.innerHTML = '';
  shapes.forEach((s, i) => { const li = document.createElement('li'); li.textContent = s.kind + ' ' + (s.name || '') + ' (' + s.points.length + ' pts) ';
    const b = document.createElement('button'); b.textContent = 'remove'; b.onclick = () => { shapes.splice(i, 1); redraw(); }; li.appendChild(b); ul.appendChild(li); });
}
c.addEventListener('click', e => { cur.push(norm(e)); redraw(); });
c.addEventListener('mousemove', e => { const p = norm(e); document.getElementById('pos').textContent = 'x=' + p[0] + '  y=' + p[1]; });
document.getElementById('finish').onclick = () => { if (!cur.length) return;
  shapes.push({kind: document.getElementById('kind').value, name: document.getElementById('name').value.trim() || ('s' + shapes.length), points: cur}); cur = []; redraw(); };
document.getElementById('undo').onclick = () => { cur.pop(); redraw(); };
document.getElementById('clear').onclick = () => { cur = []; redraw(); };
document.getElementById('kind').onchange = redraw;
document.getElementById('copy').onclick = () => navigator.clipboard.writeText(document.getElementById('yaml').value);
img.onload = () => { c.width = img.naturalWidth; c.height = img.naturalHeight; shapes = EXISTING; redraw(); };
img.src = IMG;
</script></body></html>
"""


def existing_shapes(path) -> list:
    """Convert the current YAML into editable shapes for the HTML tool."""
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except Exception:  # noqa: BLE001
        return []
    out = []
    for k in ("carriageway", "exclusion_zones", "sidewalks", "u_turn_prohibited_zones", "obstacle_regions"):
        for i, p in enumerate(data.get(k) or []):
            out.append({"kind": k, "name": f"{k}{i}", "points": p})
    for ln in data.get("lanes") or []:
        out.append({"kind": "lane_polygon", "name": ln.get("id", ""), "points": ln.get("polygon", [])})
        out.append({"kind": "lane_direction", "name": ln.get("id", ""), "points": ln.get("direction", [])})
    for sl in data.get("stop_lines") or []:
        out.append({"kind": "stop_line", "name": sl.get("id", ""), "points": sl.get("line", [])})
        if sl.get("approach"):
            out.append({"kind": "stop_approach", "name": sl.get("id", ""), "points": sl["approach"]})
    for cw in data.get("crossings") or []:
        pts = cw["polygon"] if isinstance(cw, dict) else cw
        out.append({"kind": "crossing", "name": cw.get("id", "") if isinstance(cw, dict) else "", "points": pts})
    if data.get("intersection"):
        out.append({"kind": "intersection", "name": "intersection", "points": data["intersection"]})
    for s in data.get("signals") or []:
        r = s.get("roi", [0, 0, 0, 0])
        out.append({"kind": "signal", "name": s.get("id", ""), "points": [[r[0], r[1]], [r[2], r[3]]]})
    for s in data.get("solid_lines") or []:
        out.append({"kind": "solid_line", "name": s.get("id", "") if isinstance(s, dict) else "",
                    "points": s["points"] if isinstance(s, dict) else s})
    for t in data.get("prohibited_turns") or []:
        out.append({"kind": "turn_from", "name": t.get("id", ""), "points": t.get("from", [])})
        out.append({"kind": "turn_to", "name": t.get("id", ""), "points": t.get("to", [])})
    return out


def cmd_html(args, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    page = HTML.replace("__IMG__", base64.b64encode(buf.tobytes()).decode()).replace(
        "__EXISTING__", json.dumps(existing_shapes(args.geometry)))
    path = OUT / "calibrate.html"
    path.write_text(page)
    print(f"-> {path}  (open in a browser; paste the YAML into config/camera_geometry.yaml)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["html", "render", "grid"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--video")
    ap.add_argument("--at", type=float, default=0.0, help="seconds into --video")
    ap.add_argument("--geometry", default=str(geometry_config_path()))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    img = load_frame(args)
    {"html": cmd_html, "render": cmd_render, "grid": cmd_grid}[args.command](args, img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
