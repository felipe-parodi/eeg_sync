"""Blind A/B human-verify harness for pose models.

Because every automatic metric we have is graded against teacher labels (circular),
the only non-circular quality signal is a human looking at predictions on the real
footage. This builds a *blind* comparison: for a stratified sample of frames it
renders model A vs model B side by side with the left/right order randomized, so
the reviewer cannot tell which model is which. The reviewer marks L / R / T (tie)
in a CSV; ``summarize`` maps the blind sides back to model names and tallies.

Usage:
    # 1. Build the set (renders pairs + a verdicts.csv to fill in)
    python -m pose_training.verify build --weights-a <fused3.pt> --weights-b <new.pt> --n 150
    # 2. Review in the browser (click or press Left / Tie / Right; auto-saves)
    python -m pose_training.verify review
    #    -- or hand-edit the 'verdict' column of verdicts.csv with L / R / T.
    # 3. Tally
    python -m pose_training.verify summarize
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

VERDICT_FIELDS = ["id", "image", "frame_path", "left_is_a", "verdict"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "pose_training" / "data" / "frame_manifest.csv"
DEFAULT_OUT = REPO_ROOT / "pose_training" / "data" / "verify"


def stratified_sample(
    rows: List[Dict[str, str]], n: int, seed: int
) -> List[Dict[str, str]]:
    """Sample ``n`` frames spread evenly across (participant, angle) sessions."""
    groups: Dict[Any, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["participant"], row["angle"]), []).append(row)
    rng = random.Random(seed)
    for frames in groups.values():
        rng.shuffle(frames)
    order = sorted(groups)
    picked: List[Dict[str, str]] = []
    cursors = {key: 0 for key in order}
    while len(picked) < n and any(cursors[k] < len(groups[k]) for k in order):
        for key in order:
            if cursors[key] < len(groups[key]):
                picked.append(groups[key][cursors[key]])
                cursors[key] += 1
                if len(picked) >= n:
                    break
    return picked


def _write_records(csv_path: Path, records: List[Dict[str, Any]]) -> None:
    """Write the verdict records back to ``csv_path`` with the fixed schema."""
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VERDICT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in VERDICT_FIELDS})


def set_verdict(records: List[Dict[str, Any]], frame_id: Any, verdict: str) -> bool:
    """Set the verdict (L/R/T) on the record whose id matches ``frame_id``.

    Returns True if a matching record was updated, False otherwise. An empty
    verdict clears the cell (marks the pair unjudged again).
    """
    normalized = str(verdict).strip().upper()
    if normalized not in ("L", "R", "T", ""):
        return False
    for record in records:
        if str(record.get("id")) == str(frame_id):
            record["verdict"] = normalized
            return True
    return False


def tally(records: List[Dict[str, Any]], name_a: str, name_b: str) -> Dict[str, int]:
    """Map blind L/R verdicts back to model names and count wins/ties."""
    result = {name_a: 0, name_b: 0, "tie": 0, "unjudged": 0}
    for record in records:
        verdict = str(record.get("verdict", "")).strip().upper()
        left_is_a = str(record.get("left_is_a")).lower() in ("true", "1", "yes")
        if verdict == "L":
            result[name_a if left_is_a else name_b] += 1
        elif verdict == "R":
            result[name_b if left_is_a else name_a] += 1
        elif verdict == "T":
            result["tie"] += 1
        else:
            result["unjudged"] += 1
    return result


def _predict(model: Any, frame_path: str, imgsz: int, device: str, max_persons: int):
    from .evaluate import extract_persons
    from .pseudo_label import _result_to_arrays

    result = model(frame_path, imgsz=imgsz, conf=0.3, device=device, verbose=False)[0]
    boxes, confs, kxy, kconf = _result_to_arrays(result)
    return extract_persons(boxes, confs, kxy, kconf, max_persons)


def build_verify_set(
    weights_a: str,
    weights_b: str,
    rows: List[Dict[str, str]],
    out_dir: Path,
    n: int,
    seed: int,
    imgsz: int = 960,
    device: str = "0",
    max_persons: int = 4,
) -> Path:
    """Render blind side-by-side A/B pairs and a verdicts CSV. Returns CSV path."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from .evaluate import _draw_pose

    model_a, model_b = YOLO(weights_a), YOLO(weights_b)
    picked = stratified_sample(rows, n, seed)
    rng = random.Random(seed)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(picked):
        path = row["frame_path"]
        image = cv2.imread(path)
        if image is None:
            continue
        a = _draw_pose(image, _predict(model_a, path, imgsz, device, max_persons))
        b = _draw_pose(image, _predict(model_b, path, imgsz, device, max_persons))
        left_is_a = rng.random() < 0.5
        left, right = (a, b) if left_is_a else (b, a)
        cv2.putText(
            left, "LEFT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            right, "RIGHT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        gap = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
        name = f"{index:03d}_{Path(path).stem}.png"
        cv2.imwrite(str(pairs_dir / name), np.hstack([left, gap, right]))
        records.append(
            {
                "id": index,
                "image": name,
                "frame_path": path,
                "left_is_a": left_is_a,
                "verdict": "",
            }
        )
    csv_path = out_dir / "verdicts.csv"
    _write_records(csv_path, records)
    return csv_path


REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pose verify - blind A/B</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: system-ui, sans-serif; background:#111; color:#eee;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:8px 16px; display:flex; align-items:center; gap:16px;
           border-bottom:1px solid #333; }
  #bar { flex:1; height:12px; background:#333; border-radius:6px; overflow:visible;
         position:relative; }
  #fill { height:100%; width:0%; background:#4caf50; transition:width .2s;
          border-radius:6px 0 0 6px; }
  #pos { position:absolute; top:-3px; width:3px; height:18px; background:#fff;
         border-radius:2px; transform:translateX(-1px); transition:left .15s; }
  #count { font-variant-numeric: tabular-nums; white-space:nowrap; }
  #toast { position:fixed; left:50%; bottom:84px; transform:translateX(-50%);
           background:#4caf50; color:#063; font-weight:600; padding:8px 16px;
           border-radius:8px; opacity:0; transition:opacity .12s; pointer-events:none; }
  #toast.show { opacity:.96; }
  main { flex:1; display:flex; align-items:center; justify-content:center;
         overflow:hidden; padding:8px; position:relative; }
  img { max-width:100%; max-height:100%; object-fit:contain; border:1px solid #333; }
  footer { padding:12px 16px; display:flex; gap:12px; align-items:center;
           justify-content:center; border-top:1px solid #333; flex-wrap:wrap; }
  button { font-size:16px; padding:12px 20px; border:none; border-radius:8px;
           cursor:pointer; color:#fff; }
  #prev,#next { background:#444; }
  #left { background:#1565c0; }
  #tie  { background:#616161; }
  #right{ background:#6a1b9a; }
  .cur { outline:3px solid #fff; }
  kbd { background:#000; border:1px solid #555; border-radius:4px; padding:0 6px;
        font-size:12px; }
  #hint { opacity:.65; font-size:13px; min-width:120px; }
  #badge { display:none; gap:8px; align-items:center; color:#4caf50;
           font-size:13px; white-space:nowrap; }
  #badge code { background:#000; color:#eee; padding:3px 8px; border-radius:5px; }
</style>
</head>
<body>
<header>
  <strong>Blind verify</strong>
  <div id="bar"><div id="fill"></div><div id="pos"></div></div>
  <span id="count">0 / 0</span>
  <span id="badge">&#x2713; all judged &mdash; run <code id="cmd"></code></span>
</header>
<main>
  <img id="img" alt="pair">
</main>
<div id="toast"></div>
<footer>
  <button id="prev">&#x27F5; Prev <kbd>&larr;</kbd></button>
  <button id="left">Left better <kbd>1</kbd></button>
  <button id="tie">Tie <kbd>2</kbd></button>
  <button id="right">Right better <kbd>3</kbd></button>
  <button id="next">Next <kbd>&rarr;</kbd> &#x27F6;</button>
  <span id="hint"></span>
</footer>
<script>
let records = [], i = 0, summarizeCmd = '';
const $ = id => document.getElementById(id);
async function load() {
  const state = await (await fetch('/state')).json();
  records = state.records;
  summarizeCmd = state.summarize_cmd || summarizeCmd;
  $('cmd').textContent = summarizeCmd;
  const firstUn = records.findIndex(r => !r.verdict);
  i = firstUn === -1 ? 0 : firstUn;
  render();
}
function judgedCount() { return records.filter(r => r.verdict).length; }
function render() {
  if (!records.length) return;
  const r = records[i];
  $('img').src = '/pairs/' + encodeURIComponent(r.image) + '?v=' + i;
  const judged = judgedCount();
  const left = records.length - judged;
  $('count').textContent =
    'pair ' + (i + 1) + ' / ' + records.length +
    '  ·  ' + judged + ' judged  ·  ' + left + ' left';
  $('fill').style.width = (100 * judged / records.length) + '%';
  $('pos').style.left =
    (records.length > 1 ? (100 * i / (records.length - 1)) : 0) + '%';
  const map = { left:'L', tie:'T', right:'R' };
  for (const id of ['left','tie','right'])
    $(id).classList.toggle('cur', r.verdict === map[id]);
  const label = { L:'Left', R:'Right', T:'Tie' };
  $('hint').textContent = r.verdict ? ('marked: ' + label[r.verdict]) : 'unjudged';
  $('badge').style.display = records.every(x => x.verdict) ? 'flex' : 'none';
}
async function vote(v) {
  const r = records[i];
  r.verdict = v;
  fetch('/verdict', { method:'POST', headers:{'Content-Type':'application/json'},
                      body: JSON.stringify({ id: r.id, verdict: v }) });
  flash(v);  // confirm the vote registered even when the count does not change
  i = Math.min(i + 1, records.length - 1);  // advance to the next pair
  render();
}
let toastTimer = null;
function flash(v) {
  const label = { L:'Left better', R:'Right better', T:'Tie' };
  const t = $('toast');
  t.textContent = '\\u2713 saved: ' + label[v];
  t.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 700);
}
function move(d) { i = Math.max(0, Math.min(records.length - 1, i + d)); render(); }
$('left').onclick  = () => vote('L');
$('tie').onclick   = () => vote('T');
$('right').onclick = () => vote('R');
$('prev').onclick  = () => move(-1);
$('next').onclick  = () => move(1);
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') move(-1);        // navigate, do not vote
  else if (e.key === 'ArrowRight') move(1);
  else if (e.key === '1') vote('L');
  else if (e.key === '2') vote('T');
  else if (e.key === '3') vote('R');
});
load();
</script>
</body>
</html>
"""


class ReviewState:
    """Shared, lock-guarded state for the review server (records + CSV path)."""

    def __init__(
        self, csv_path: Path, pairs_dir: Path, records: List[Dict[str, Any]], cmd: str
    ) -> None:
        self.csv_path = csv_path
        self.pairs_dir = pairs_dir.resolve()
        self.records = records
        self.summarize_cmd = cmd
        self.lock = threading.Lock()

    def state_json(self) -> bytes:
        """Blind client payload: only id/image/verdict (never ``left_is_a``)."""
        records = [
            {
                "id": r.get("id"),
                "image": r.get("image"),
                "verdict": r.get("verdict", ""),
            }
            for r in self.records
        ]
        payload = {"records": records, "summarize_cmd": self.summarize_cmd}
        return json.dumps(payload).encode("utf-8")

    def save_verdict(self, frame_id: Any, verdict: str) -> bool:
        """Update one verdict and persist the whole CSV. Thread-safe."""
        with self.lock:
            if not set_verdict(self.records, frame_id, verdict):
                return False
            _write_records(self.csv_path, self.records)
            return True


class _ReviewHandler(BaseHTTPRequestHandler):
    """Serves the review UI, the pair images, and verdict writes."""

    def __init__(self, *args: Any, state: ReviewState, **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *args: Any) -> None:  # noqa: D102 -- silence stdlib logging
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- stdlib handler name
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send(200, REVIEW_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            self._send(200, self.state.state_json(), "application/json")
        elif path.startswith("/pairs/"):
            self._serve_image(Path(path).name)
        else:
            self._send(404, b"not found", "text/plain")

    def _serve_image(self, name: str) -> None:
        target = (self.state.pairs_dir / name).resolve()
        if self.state.pairs_dir not in target.parents or not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, target.read_bytes(), "image/png")

    def do_POST(self) -> None:  # noqa: N802 -- stdlib handler name
        if unquote(urlparse(self.path).path) != "/verdict":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            ok = self.state.save_verdict(payload.get("id"), payload.get("verdict", ""))
        except (ValueError, KeyError):
            ok = False
        self._send(
            200 if ok else 400,
            json.dumps({"ok": ok}).encode("utf-8"),
            "application/json",
        )


def run_review_server(
    out_dir: Path, name_a: str, name_b: str, port: int, open_browser: bool = True
) -> None:
    """Serve the blind review UI from ``out_dir`` until interrupted (Ctrl+C)."""
    csv_path = out_dir / "verdicts.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found -- run 'verify build' first to create the set."
        )
    with csv_path.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    summarize_cmd = (
        f"python -m pose_training.verify summarize "
        f"--name-a {name_a} --name-b {name_b}"
    )
    state = ReviewState(csv_path, out_dir / "pairs", records, summarize_cmd)
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(_ReviewHandler, state=state)
    )
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    judged = sum(1 for r in records if str(r.get("verdict", "")).strip())
    print(f"Review {len(records)} pairs at {url} ({judged} already judged).")
    print("Click or press 1/Left, 2/Tie, 3/Right. Verdicts auto-save. Ctrl+C to stop.")
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Verdicts saved to", csv_path)
    finally:
        server.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind A/B human-verify harness.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Render blind A/B pairs + verdicts.csv.")
    build.add_argument("--weights-a", required=True, type=str)
    build.add_argument("--weights-b", required=True, type=str)
    build.add_argument("--name-a", default="model_a", type=str)
    build.add_argument("--name-b", default="model_b", type=str)
    build.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=str)
    build.add_argument("--out-dir", default=str(DEFAULT_OUT), type=str)
    build.add_argument("--n", default=150, type=int)
    build.add_argument("--seed", default=0, type=int)
    build.add_argument("--device", default="0", type=str)
    review = sub.add_parser("review", help="Click through pairs in the browser.")
    review.add_argument("--out-dir", default=str(DEFAULT_OUT), type=str)
    review.add_argument("--name-a", default="model_a", type=str)
    review.add_argument("--name-b", default="model_b", type=str)
    review.add_argument("--port", default=8000, type=int)
    review.add_argument(
        "--no-open", action="store_true", help="Do not auto-open browser."
    )
    summ = sub.add_parser("summarize", help="Tally a filled-in verdicts.csv.")
    summ.add_argument("--csv", default=str(DEFAULT_OUT / "verdicts.csv"), type=str)
    summ.add_argument("--name-a", default="model_a", type=str)
    summ.add_argument("--name-b", default="model_b", type=str)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "build":
        with open(args.manifest, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        csv_path = build_verify_set(
            args.weights_a,
            args.weights_b,
            rows,
            Path(args.out_dir),
            args.n,
            args.seed,
            device=args.device,
        )
        print(f"Wrote {csv_path}. Open {Path(args.out_dir) / 'pairs'} and fill the")
        print("'verdict' column with L (left better), R (right better), or T (tie).")
        print(
            f"Then: python -m pose_training.verify summarize --csv {csv_path} "
            f"--name-a {args.name_a} --name-b {args.name_b}"
        )
    elif args.command == "review":
        run_review_server(
            Path(args.out_dir),
            args.name_a,
            args.name_b,
            args.port,
            open_browser=not args.no_open,
        )
    else:
        with open(args.csv, encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        result = tally(records, args.name_a, args.name_b)
        print("Verify tally:")
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
