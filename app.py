from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects.*", category=UserWarning)

import os
import sys
import json
import time
import queue
import shutil
import zipfile
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from flask import Flask, request, abort, send_file, render_template_string, jsonify, Response, make_response

from family_photo_detector import load_db, classify_image  # must exist in your project

# ---- Optional TurboJPEG (fast JPEG decode for thumbnails) ----
TURBOJPEG_AVAILABLE = False
TJ = None
try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE  # type: ignore
    TJ = TurboJPEG()
    TURBOJPEG_AVAILABLE = True
except Exception as e:
    TURBOJPEG_AVAILABLE = False
    TJ = None
    print(f"[INFO] TurboJPEG not available: {e}", file=sys.stderr)

app = Flask(__name__)

# -------------------- Paths / Defaults --------------------
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", tempfile.gettempdir())) / "family_photo_organizer_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

THUMB_DIRNAME = "_thumbs"
THUMB_MAX_SIDE = int(os.environ.get("THUMB_MAX_SIDE", "320"))  # grid thumbnails only
THUMB_QUALITY = int(os.environ.get("THUMB_QUALITY", "72"))     # jpeg quality (0..100)

DEFAULT_TOLERANCE = float(os.environ.get("TOLERANCE", "0.45"))
DEFAULT_DECISION = os.environ.get("DECISION", "any_known")
DEFAULT_MODEL = os.environ.get("MODEL", "cnn")

# Workers - increased for better parallelism
CLASSIFY_WORKERS = int(os.environ.get("CLASSIFY_WORKERS", "6"))
THUMB_WORKERS = int(os.environ.get("THUMB_WORKERS", "8"))  # ✅ Increased from 4 to 8

# Cache
DB_CACHE = None

# Thumb background queue/dedupe
THUMB_EXECUTOR = ThreadPoolExecutor(max_workers=THUMB_WORKERS)
_THUMB_INFLIGHT: Set[str] = set()
_THUMB_INFLIGHT_LOCK = threading.Lock()

# Placeholder PNG (1x1 light gray) - served instantly while thumb generates
PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xf8\xf8"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\x18\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _get_db_cached():
    global DB_CACHE
    if DB_CACHE is None:
        DB_CACHE = load_db()
    return DB_CACHE


def _new_job_id() -> str:
    return f"job_{os.getpid()}_{os.urandom(6).hex()}"


def _save_upload(file_storage, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(file_storage.filename).name
    out_path = out_dir / name
    file_storage.save(out_path)
    return out_path


def _safe_resolve_run_file(run_name: str, rel_path: Path) -> Path:
    """Prevent path traversal. rel_path is treated as relative."""
    run_dir = (UPLOAD_DIR / run_name).resolve()
    base = UPLOAD_DIR.resolve()

    if base not in run_dir.parents and run_dir != base:
        abort(403)

    p = (run_dir / rel_path).resolve()
    if run_dir not in p.parents and p != run_dir:
        abort(403)
    return p


def _thumb_key(run: str, name: str) -> str:
    return f"{run}::{name}"


def _thumb_paths(run: str, original_name: str) -> Tuple[Path, Path]:
    """Returns (orig_path, thumb_path) where thumb_path is run/_thumbs/<stem>.jpg"""
    orig_path = _safe_resolve_run_file(run, Path(original_name))
    thumb_name = f"{orig_path.stem}.jpg"
    thumb_rel = Path(THUMB_DIRNAME) / thumb_name
    thumb_path = _safe_resolve_run_file(run, thumb_rel)
    return orig_path, thumb_path


def _pick_turbo_scaling(max_side: int, w: int, h: int) -> Tuple[int, int]:
    """
    ✅ OPTIMIZED: Choose optimal TurboJPEG scaling factor.
    
    TurboJPEG supports: 1/1, 1/2, 1/4, 1/8
    We pick the smallest scale that keeps max(w,h) >= max_side
    so the final resize step is minimal.
    """
    if max_side <= 0:
        return (1, 1)
    
    m = max(w, h)
    
    # Try from smallest to largest output
    # Pick the largest downscale where scaled dimension >= target
    for den in (8, 4, 2):
        scaled = m // den
        if scaled >= max_side:
            return (1, den)
    
    return (1, 1)


def _encode_jpeg_turbo(bgr, quality: int) -> Optional[bytes]:
    """Encode BGR array to JPEG bytes using TurboJPEG."""
    if not TURBOJPEG_AVAILABLE or TJ is None:
        return None
    try:
        q = max(1, min(100, int(quality)))
        return TJ.encode(
            bgr,
            quality=q,
            pixel_format=TJPF_BGR,
            flags=TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE,
        )
    except Exception:
        return None


def _make_thumb_turbojpeg(orig_path: Path, thumb_path: Path, max_side: int, quality: int) -> bool:
    """
    ✅ OPTIMIZED TurboJPEG thumbnail generation.
    
    1. Read JPEG header to get dimensions (no full decode)
    2. Calculate optimal scaling factor based on actual dimensions
    3. Decode with scaling (much faster than full decode + resize)
    4. Final resize if needed
    5. Encode with TurboJPEG
    """
    if not TURBOJPEG_AVAILABLE or TJ is None:
        return False
    
    try:
        data = orig_path.read_bytes()
        
        # ✅ FIX: Read header to get actual dimensions WITHOUT decoding
        width, height, jpeg_subsample, jpeg_colorspace = TJ.decode_header(data)
        
        # ✅ FIX: Use smart scaling based on actual image dimensions
        scaling = _pick_turbo_scaling(max_side, width, height)
        
        # Decode with optimal scaling factor
        bgr = TJ.decode(
            data,
            pixel_format=TJPF_BGR,
            scaling_factor=scaling,
            flags=TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE,
        )
        
        # Final resize if still larger than max_side
        h, w = bgr.shape[:2]
        m = max(w, h)
        if max_side and m > max_side:
            scale = max_side / float(m)
            new_w = int(w * scale)
            new_h = int(h * scale)
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # ✅ Encode with TurboJPEG (faster than cv2.imwrite)
        out_bytes = TJ.encode(
            bgr,
            quality=quality,
            pixel_format=TJPF_BGR,
            flags=TJFLAG_FASTDCT,
        )
        
        thumb_path.write_bytes(out_bytes)
        return True
        
    except Exception as e:
        print(f"[DEBUG] TurboJPEG failed for {orig_path.name}: {e}", file=sys.stderr)
        return False


def _make_thumb_opencv(orig_path: Path, thumb_path: Path, max_side: int, quality: int) -> bool:
    """Fallback thumbnail generation using OpenCV (for PNG, HEIC, etc.)"""
    try:
        bgr = cv2.imread(str(orig_path))
        if bgr is None:
            return False
        
        h, w = bgr.shape[:2]
        m = max(h, w)
        if max_side and m > max_side:
            scale = max_side / float(m)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        ok = cv2.imwrite(str(thumb_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return bool(ok)
    except Exception as e:
        print(f"[DEBUG] OpenCV thumb failed for {orig_path.name}: {e}", file=sys.stderr)
        return False


def _make_thumb(orig_path: Path, thumb_path: Path) -> bool:
    """
    ✅ OPTIMIZED: Main thumbnail generation function.
    
    - JPEG files: Use TurboJPEG with optimal scaling
    - Other formats: Fallback to OpenCV
    """
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = orig_path.suffix.lower()
    is_jpeg = ext in (".jpg", ".jpeg")
    max_side = THUMB_MAX_SIDE
    quality = THUMB_QUALITY
    
    # Try TurboJPEG first for JPEG files
    if is_jpeg and TURBOJPEG_AVAILABLE:
        if _make_thumb_turbojpeg(orig_path, thumb_path, max_side, quality):
            return True
        # Fall through to OpenCV if TurboJPEG fails
    
    # Fallback to OpenCV (handles PNG, TIFF, HEIC if supported, etc.)
    return _make_thumb_opencv(orig_path, thumb_path, max_side, quality)


def copy_to_bucket(src: Path, dst_dir: Path):
    """COPY (không move). Tự tạo folder. Tránh overwrite bằng suffix _1, _2..."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(str(src), str(dst))
        return

    i = 1
    while True:
        candidate = dst_dir / f"{src.stem}_{i}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(str(src), str(candidate))
            return
        i += 1


def make_results_zip(run_dir: Path) -> Optional[Path]:
    """Create run_dir/results.zip containing family/ and non_family/ folders."""
    zip_path = run_dir / "results.zip"
    try:
        if zip_path.exists():
            zip_path.unlink()
    except Exception:
        pass

    fam = run_dir / "family"
    non = run_dir / "non_family"
    if not fam.exists() and not non.exists():
        return None

    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for sub in ("family", "non_family"):
            folder = run_dir / sub
            if not folder.exists():
                continue
            for p in folder.rglob("*"):
                if p.is_file():
                    arcname = str(p.relative_to(run_dir))
                    z.write(str(p), arcname)

    return zip_path if zip_path.exists() else None


# -------------------- Batch job state --------------------
@dataclass
class BatchJob:
    job_id: str
    run_dir: Path
    items: List[dict] = field(default_factory=list)
    originals: List[str] = field(default_factory=list)
    total: int = 0

    started_at: float = 0.0
    started: bool = False
    finished: bool = False

    done: int = 0
    yes: int = 0
    no: int = 0

    error: Optional[str] = None
    download_url: str = ""
    q: "queue.Queue[str]" = field(default_factory=queue.Queue)


JOBS: Dict[str, BatchJob] = {}
JOBS_LOCK = threading.Lock()


def _sse_send(job: BatchJob, event: str, data: dict):
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    job.q.put(msg)


def _thumb_notify_if_job(run: str, name: str):
    """If run is a BatchJob id, emit thumb_ready SSE event."""
    with JOBS_LOCK:
        job = JOBS.get(run)
    if not job:
        return
    _sse_send(job, "thumb_ready", {"name": name})


def _thumb_task(run: str, name: str):
    """
    ✅ OPTIMIZED: Background thumb generator.
    
    - Uses TurboJPEG with optimal scaling for JPEGs
    - Emits SSE thumb_ready when done
    - Deduped via _THUMB_INFLIGHT
    """
    key = _thumb_key(run, name)
    try:
        orig_path, thumb_path = _thumb_paths(run, name)
        if not orig_path.exists() or not orig_path.is_file():
            return
        
        # Already exists? Just notify
        if thumb_path.exists():
            _thumb_notify_if_job(run, name)
            return

        # Generate thumbnail
        t0 = time.perf_counter()
        ok = _make_thumb(orig_path, thumb_path)
        dt = (time.perf_counter() - t0) * 1000  # ms
        
        if ok and thumb_path.exists():
            _thumb_notify_if_job(run, name)
            # Uncomment for debugging:
            # print(f"[THUMB] {name}: {dt:.1f}ms", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] thumb task failed: run={run} name={name} ({e})", file=sys.stderr)
    finally:
        with _THUMB_INFLIGHT_LOCK:
            _THUMB_INFLIGHT.discard(key)


def _enqueue_thumb(run: str, name: str):
    """Dedupe + enqueue background thumb generation."""
    key = _thumb_key(run, name)
    with _THUMB_INFLIGHT_LOCK:
        if key in _THUMB_INFLIGHT:
            return
        _THUMB_INFLIGHT.add(key)
    THUMB_EXECUTOR.submit(_thumb_task, run, name)


def _classify_one(run_dir: Path, name: str, db, tolerance: float, decision: str, model: str):
    p = run_dir / name
    t0 = time.time()
    res = classify_image(Path(p), db, tolerance=tolerance, decision=decision, face_model=model)
    dt = time.time() - t0
    return {
        "name": name,
        "yes": bool(res.get("family", False)),
        "faces": int(res.get("faces", 0)),
        "recognized": int(res.get("recognized", 0)),
        "time_s": round(dt, 3),
    }


def _batch_worker(job_id: str, tolerance: float, decision: str, model: str):
    """
    Parallel classification with CLASSIFY_WORKERS.
    Emits SSE per completed item. Copies into buckets, then builds ZIP.
    """
    try:
        db = _get_db_cached()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            return

        _sse_send(job, "meta", {
            "total": job.total,
            "classify_workers": CLASSIFY_WORKERS,
            "thumb_workers": THUMB_WORKERS,
            "turbojpeg": TURBOJPEG_AVAILABLE,
        })

        with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
            future_to_name = {
                ex.submit(_classify_one, job.run_dir, name, db, tolerance, decision, model): name
                for name in job.originals
            }

            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    r = fut.result()
                    is_yes = bool(r["yes"])
                    faces = int(r["faces"])
                    recognized = int(r["recognized"])
                    time_s = r["time_s"]
                except Exception as e:
                    is_yes = False
                    faces = 0
                    recognized = 0
                    time_s = None
                    print(f"[WARN] classify failed: {name} ({e})", file=sys.stderr)

                try:
                    src = job.run_dir / name
                    if is_yes:
                        copy_to_bucket(src, job.run_dir / "family")
                    else:
                        copy_to_bucket(src, job.run_dir / "non_family")
                except Exception as e:
                    print(f"[WARN] bucket copy failed: {name} ({e})", file=sys.stderr)

                with JOBS_LOCK:
                    job.done += 1
                    if is_yes:
                        job.yes += 1
                    else:
                        job.no += 1
                    done, yes, no, total = job.done, job.yes, job.no, job.total

                _sse_send(job, "item", {
                    "name": name,
                    "yes": is_yes,
                    "done": done,
                    "total": total,
                    "yes_count": yes,
                    "no_count": no,
                    "faces": faces,
                    "recognized": recognized,
                    "time_s": time_s,
                })

        dl = ""
        try:
            zp = make_results_zip(job.run_dir)
            if zp is not None:
                dl = f"/download/{job.job_id}"
        except Exception as e:
            print(f"[WARN] zip build failed: {e}", file=sys.stderr)

        with JOBS_LOCK:
            job.finished = True
            job.download_url = dl

        _sse_send(job, "done", {
            "done": job.done,
            "total": job.total,
            "yes_count": job.yes,
            "no_count": job.no,
            "elapsed_s": round(time.time() - job.started_at, 2),
            "download_url": dl,
        })

    except Exception as e:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.error = str(e)
                job.finished = True
                _sse_send(job, "job_error", {"message": str(e)})


# -------------------- HTML --------------------
HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Family Photo Organizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>

  <style>
    body { font-family: -apple-system, Arial, sans-serif; margin: 16px; padding-bottom: 260px; }
    .card { border:1px solid #ddd; border-radius: 14px; padding: 14px; width: 100%; box-sizing: border-box; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:10px; }
    .btn { padding:9px 12px; border:1px solid #ccc; border-radius:12px; background:#fff; cursor:pointer; }
    .btn:hover { background:#f6f6f6; }
    input, select { padding:8px 10px; border-radius:12px; border:1px solid #ccc; }
    .muted { color:#666; font-size: 13px; }
    .small { font-size: 12px; }

    .gridWrap { margin-top: 14px; display:none; }
    .gridTitle { margin: 14px 0 10px 0; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .pill { display:inline-block; padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.18); font-size: 13px; font-weight: 600; background: rgba(0,0,0,0.04); }
    .linkbtn { padding:7px 10px; border-radius:12px; border:1px solid #2b6cb0; background:#ebf8ff; cursor:pointer; font-weight:600; text-decoration:none; }
    .linkbtn:hover { background:#dbefff; }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 10px;
      width: 100%;
    }

    .tile {
      border-radius: 14px;
      padding: 8px;
      overflow: hidden;
      background: rgba(0,0,0,0.04);
      border: 1px solid rgba(0,0,0,0.08);
      transition: background 160ms ease, border-color 160ms ease;
      min-height: 170px;
      box-sizing: border-box;
    }

    .tile.yes {
      background: rgba(60, 200, 120, 0.40);
      border-color: rgba(60, 200, 120, 0.75);
    }

    .tile.no {
      background: rgba(220, 80, 80, 0.40);
      border-color: rgba(220, 80, 80, 0.75);
    }

    .tile img {
      width: 100%;
      height: 130px;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.08);
      background: rgba(255,255,255,0.35);
      display: block;
    }
    .tileMeta { margin-top: 6px; font-size: 11px; color:#333; opacity: 0.75; word-break: break-all; }

    .console {
      position: fixed; left: 0; right: 0; bottom: 0; height: 210px;
      background: #0b0b0b; color: #eaeaea; border-top: 1px solid #333;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      padding: 10px 12px; overflow:auto; z-index: 999;
      box-sizing: border-box;
    }
    .console .title { color: #a6a6a6; margin-bottom: 6px; }
    .console pre { margin: 0; white-space: pre-wrap; word-break: break-word; }

    #statusLine { font-weight: 600; }
    
    .perf-badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: bold;
      margin-left: 6px;
    }
    .perf-badge.turbo { background: #10b981; color: white; }
    .perf-badge.opencv { background: #f59e0b; color: white; }
  </style>
</head>

<body>
  <div class="card">
    <h2 style="margin:0 0 6px 0;">Family Photo Organizer (localhost)
      {% if turbojpeg %}
      <span class="perf-badge turbo">TurboJPEG ON</span>
      {% else %}
      <span class="perf-badge opencv">OpenCV fallback</span>
      {% endif %}
    </h2>
    <div class="muted">
      Thumbs: placeholder instantly → SSE thumb_ready swaps to real JPEG.
      Workers: {{ thumb_workers }} thumb, {{ classify_workers }} classify.
    </div>

    <form id="mainForm" enctype="multipart/form-data">
      <div class="row">
        <label><b>Mode</b></label>
        <select name="mode" id="mode">
          <option value="check">check (single image)</option>
          <option value="batch">batch (folder)</option>
        </select>

        <label><b>Decision</b></label>
        <select name="decision" id="decision">
          <option value="any_known" selected>any_known</option>
          <option value="majority_known">majority_known</option>
          <option value="all_known">all_known</option>
        </select>

        <label><b>Tolerance</b></label>
        <input name="tolerance" id="tolerance" value="{{tolerance}}" size="6"/>

        <label><b>Model</b></label>
        <select name="model" id="model">
          <option value="cnn" {% if model == "cnn" %}selected{% endif %}>cnn</option>
          <option value="hog" {% if model == "hog" %}selected{% endif %}>hog</option>
        </select>

        <span class="muted small" id="statusLine">idle</span>
      </div>

      <div class="row">
        <button class="btn" type="button" id="browseBtn">Browse…</button>
        <input id="picker" name="files" type="file" style="display:none;" multiple />
        <button class="btn" type="submit">Run</button>
        <span class="muted small" id="pickedLabel">No selection</span>
        <a id="downloadLink" class="linkbtn" href="#" style="display:none;" download>Download ZIP</a>
      </div>

      <div class="muted small">
        Folder picking requires Chrome/Edge (<code>webkitdirectory</code>).
        For fastest thumbs: <code>pip install PyTurboJPEG</code> + <code>brew install jpeg-turbo</code> (macOS)
      </div>
    </form>

    <div class="gridWrap" id="gridWrap">
      <div class="gridTitle">
        <h3 style="margin:0;">Results</h3>
        <span class="pill" id="pillProgress">0/0</span>
        <span class="pill" id="pillYes">YES: 0</span>
        <span class="pill" id="pillNo">NO: 0</span>
        <span class="pill" id="pillThumbsReady" style="background:#e0f2fe;">Thumbs: 0</span>
      </div>
      <div class="grid" id="grid"></div>
    </div>
  </div>

  <div class="console" id="console">
    <div class="title">Debug console</div>
    <pre id="consoleText">Ready.\n</pre>
  </div>

<script>
  const modeEl = document.getElementById('mode');
  const picker = document.getElementById('picker');
  const browseBtn = document.getElementById('browseBtn');
  const pickedLabel = document.getElementById('pickedLabel');
  const form = document.getElementById('mainForm');

  const gridWrap = document.getElementById('gridWrap');
  const grid = document.getElementById('grid');

  const pillProgress = document.getElementById('pillProgress');
  const pillYes = document.getElementById('pillYes');
  const pillNo = document.getElementById('pillNo');
  const pillThumbsReady = document.getElementById('pillThumbsReady');

  const consoleText = document.getElementById('consoleText');
  const statusLine = document.getElementById('statusLine');
  const downloadLink = document.getElementById('downloadLink');

  let currentJobId = null;
  let currentEventSource = null;
  let thumbsReadyCount = 0;
  let totalImages = 0;

  function logLine(s) {
    consoleText.textContent += s + "\\n";
    consoleText.parentElement.scrollTop = consoleText.parentElement.scrollHeight;
  }

  function setStatus(s) { statusLine.textContent = s; }

  function hideDownload() {
    downloadLink.style.display = "none";
    downloadLink.href = "#";
  }
  function showDownload(url) {
    downloadLink.href = url;
    downloadLink.style.display = "inline-block";
  }

  function configurePicker() {
    const mode = modeEl.value;
    picker.value = '';
    pickedLabel.textContent = 'No selection';
    hideDownload();

    picker.removeAttribute('webkitdirectory');
    picker.removeAttribute('directory');

    if (mode === 'batch') {
      picker.setAttribute('webkitdirectory', '');
      picker.setAttribute('directory', '');
      picker.multiple = true;
      picker.accept = '';
    } else {
      picker.multiple = false;
      picker.accept = 'image/*';
    }
  }

  modeEl.addEventListener('change', configurePicker);
  browseBtn.addEventListener('click', () => picker.click());

  picker.addEventListener('change', () => {
    hideDownload();
    if (!picker.files || picker.files.length === 0) {
      pickedLabel.textContent = 'No selection';
      return;
    }
    pickedLabel.textContent = (modeEl.value === 'batch')
      ? `Selected folder files: ${picker.files.length}`
      : `Selected: ${picker.files[0].name}`;
  });

  function buildGrid(items) {
    gridWrap.style.display = 'block';
    grid.innerHTML = '';
    thumbsReadyCount = 0;
    totalImages = items.length;
    pillThumbsReady.textContent = `Thumbs: 0/${totalImages}`;

    for (const it of items) {
      const tile = document.createElement('div');
      tile.className = 'tile';
      tile.dataset.name = it.name;

      const img = document.createElement('img');
      img.loading = 'lazy';
      img.dataset.thumbUrl = it.url || "";
      if (it.url) img.src = it.url + "&v=" + (Date.now() + Math.random()).toString(16);

      tile.appendChild(img);

      const meta = document.createElement('div');
      meta.className = 'tileMeta';
      meta.textContent = it.name;
      tile.appendChild(meta);

      grid.appendChild(tile);
    }
  }

  function updateTile(name, isYes) {
    const tile = [...document.querySelectorAll('.tile')].find(t => t.dataset.name === name);
    if (!tile) return;
    tile.classList.remove('yes', 'no');
    tile.classList.add(isYes ? 'yes' : 'no');
  }

  function setCounters(done, total, yes, no) {
    pillProgress.textContent = `${done}/${total}`;
    pillYes.textContent = `YES: ${yes}`;
    pillNo.textContent = `NO: ${no}`;
  }

  function swapThumb(name) {
    const tile = [...document.querySelectorAll('.tile')].find(t => t.dataset.name === name);
    if (!tile) return;
    const img = tile.querySelector("img");
    if (!img) return;
    const baseUrl = img.dataset.thumbUrl;
    if (!baseUrl) return;
    
    // Check if already loaded (avoid double counting)
    if (img.dataset.loaded === "1") return;
    img.dataset.loaded = "1";
    
    const bust = (Date.now() + Math.random()).toString(16);
    img.src = baseUrl + "&v=" + bust;
    
    thumbsReadyCount++;
    pillThumbsReady.textContent = `Thumbs: ${thumbsReadyCount}/${totalImages}`;
  }

  async function startBatchPhase1(formData) {
    setStatus("uploading…");
    hideDownload();
    if (currentEventSource) { try { currentEventSource.close(); } catch(e) {} }

    const t0 = performance.now();
    const resp = await fetch('/batch_start', { method: 'POST', body: formData });
    if (!resp.ok) {
      setStatus("idle");
      alert(await resp.text());
      return null;
    }

    const data = await resp.json();
    const t1 = performance.now();

    currentJobId = data.job_id;
    logLine(`[batch_start] ${(t1 - t0).toFixed(0)}ms total=${data.total} job=${data.job_id}`);

    buildGrid(data.items);
    setCounters(0, data.total, 0, 0);
    setStatus(`processing… 0/${data.total}`);

    return data;
  }

  async function startBatchPhase2Run(job_id, decision, tolerance, model) {
    const es = new EventSource(`/batch_events/${job_id}`);
    currentEventSource = es;

    es.addEventListener('thumb_ready', (ev) => {
      const m = JSON.parse(ev.data);
      swapThumb(m.name);
    });

    es.addEventListener('item', (ev) => {
      const m = JSON.parse(ev.data);
      updateTile(m.name, m.yes);
      setCounters(m.done, m.total, m.yes_count, m.no_count);
      setStatus(`processing… ${m.done}/${m.total}  YES=${m.yes_count}  NO=${m.no_count}`);
    });

    es.addEventListener('done', (ev) => {
      const m = JSON.parse(ev.data);
      setCounters(m.done, m.total, m.yes_count, m.no_count);
      setStatus(`done ✓  ${m.done}/${m.total}  YES=${m.yes_count}  NO=${m.no_count}`);
      logLine(`[done] elapsed=${m.elapsed_s}s YES=${m.yes_count} NO=${m.no_count}`);

      if (m.download_url) {
        showDownload(m.download_url);
        logLine(`[zip] ready: ${m.download_url}`);
      }
      es.close();
    });

    es.addEventListener('job_error', (ev) => {
      const m = JSON.parse(ev.data);
      setStatus("idle");
      logLine(`[ERROR] ${m.message}`);
      es.close();
      alert(m.message);
    });

    const fd = new FormData();
    fd.append("job_id", job_id);
    fd.append("decision", decision);
    fd.append("tolerance", tolerance);
    fd.append("model", model);

    const r = await fetch("/batch_run", { method: "POST", body: fd });
    if (!r.ok) {
      setStatus("idle");
      const msg = await r.text();
      logLine(`[batch_run] ERROR: ${msg}`);
      alert(msg);
      es.close();
      return;
    }

    logLine(`[batch_run] started job=${job_id}`);
  }

  async function startCheck(formData) {
    setStatus("uploading…");
    hideDownload();
    if (currentEventSource) { try { currentEventSource.close(); } catch(e) {} }

    const t0 = performance.now();
    const resp = await fetch('/check', { method: 'POST', body: formData });
    if (!resp.ok) {
      setStatus("idle");
      alert(await resp.text());
      return;
    }
    const data = await resp.json();
    const t1 = performance.now();

    buildGrid([{ name: data.name, url: data.thumb_url }]);
    updateTile(data.name, data.yes);
    setCounters(1, 1, data.yes ? 1 : 0, data.yes ? 0 : 1);
    thumbsReadyCount = 1;
    pillThumbsReady.textContent = `Thumbs: 1/1`;

    setStatus(`done ✓ 1/1  YES=${data.yes ? 1 : 0}  NO=${data.yes ? 0 : 1}`);
    logLine(`[check] ${(t1 - t0).toFixed(0)}ms ${data.name} YES=${data.yes} faces=${data.faces} rec=${data.recognized} time=${data.time_s}s`);
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!picker.files || picker.files.length === 0) {
      alert('Please click Browse and select an image (check) or a folder (batch).');
      return;
    }

    hideDownload();
    const mode = modeEl.value;

    const decision = document.getElementById('decision').value;
    const tolerance = document.getElementById('tolerance').value;
    const model = document.getElementById('model').value;

    const fd = new FormData();
    fd.append('decision', decision);
    fd.append('tolerance', tolerance);
    fd.append('model', model);
    for (const f of picker.files) fd.append('files', f, f.name);

    logLine(`--- run mode=${mode} files=${picker.files.length} ---`);
    setStatus("starting…");

    if (mode === 'batch') {
      const data = await startBatchPhase1(fd);
      if (!data) return;
      await startBatchPhase2Run(data.job_id, decision, tolerance, model);
    } else {
      await startCheck(fd);
    }
  });

  configurePicker();
</script>

</body>
</html>
"""


# -------------------- Routes --------------------
@app.get("/")
def home():
    return render_template_string(
        HTML,
        tolerance=DEFAULT_TOLERANCE,
        model=DEFAULT_MODEL,
        turbojpeg=TURBOJPEG_AVAILABLE,
        thumb_workers=THUMB_WORKERS,
        classify_workers=CLASSIFY_WORKERS,
    )


@app.post("/batch_start")
def batch_start():
    """
    Phase 1: upload only. Return grid items.
    Thumbs are generated asynchronously and announced via SSE thumb_ready.
    """
    files = request.files.getlist("files")
    if not files:
        abort(400, "No folder selected (or browser does not support folder picking).")

    job_id = _new_job_id()
    run_dir = UPLOAD_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    items: List[dict] = []
    originals: List[str] = []

    for fs in files:
        if not fs or not fs.filename:
            continue
        name = Path(fs.filename).name
        try:
            _save_upload(fs, run_dir)
            originals.append(name)
            items.append({"name": name, "url": f"/thumb?run={job_id}&name={name}"})
        except Exception as e:
            print(f"[WARN] upload failed: {name} ({e})", file=sys.stderr)
            items.append({"name": name, "url": ""})

    if not originals:
        abort(400, "No valid image files uploaded from folder.")

    job = BatchJob(job_id=job_id, run_dir=run_dir, items=items, originals=originals, total=len(originals))
    with JOBS_LOCK:
        JOBS[job_id] = job

    # ✅ Pre-enqueue all thumbs immediately for parallel generation
    for n in originals:
        _enqueue_thumb(job_id, n)

    return jsonify({"job_id": job_id, "total": job.total, "items": job.items})


@app.post("/batch_run")
def batch_run():
    """Phase 2: start classification after UI grid exists."""
    job_id = (request.form.get("job_id") or "").strip()
    if not job_id:
        abort(400, "Missing job_id")

    decision = request.form.get("decision", DEFAULT_DECISION)
    tolerance = float(request.form.get("tolerance", str(DEFAULT_TOLERANCE)))
    model = request.form.get("model", DEFAULT_MODEL)

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            abort(404, "Unknown job id")
        if job.started:
            return jsonify({"ok": True, "already_started": True})
        job.started = True
        job.started_at = time.time()

    th = threading.Thread(target=_batch_worker, args=(job_id, tolerance, decision, model), daemon=True)
    th.start()
    return jsonify({"ok": True})


@app.get("/batch_events/<job_id>")
def batch_events(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404, "Unknown job id")

    def gen():
        yield "event: meta\ndata: {}\n\n"
        while True:
            try:
                msg = job.q.get(timeout=10.0)
                yield msg
                if job.finished and job.q.empty():
                    break
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                if job.finished:
                    break

    return Response(gen(), mimetype="text/event-stream")


@app.post("/check")
def check_one():
    """Single image: upload -> generate thumb synchronously -> classify -> return json."""
    decision = request.form.get("decision", DEFAULT_DECISION)
    tolerance = float(request.form.get("tolerance", str(DEFAULT_TOLERANCE)))
    model = request.form.get("model", DEFAULT_MODEL)

    files = request.files.getlist("files")
    if not files or not files[0].filename:
        abort(400, "No image selected.")

    db = _get_db_cached()
    run_id = f"run_{os.getpid()}_{os.urandom(4).hex()}"
    run_dir = UPLOAD_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_path = _save_upload(files[0], run_dir)
    except Exception as e:
        abort(400, f"Upload failed: {e}")

    # Make thumb synchronously for single-image UX
    try:
        _, thumb_path = _thumb_paths(run_id, img_path.name)
        if not thumb_path.exists():
            _make_thumb(img_path, thumb_path)
    except Exception as e:
        print(f"[WARN] check thumb failed: {img_path.name} ({e})", file=sys.stderr)

    t0 = time.time()
    try:
        res = classify_image(img_path, db, tolerance=tolerance, decision=decision, face_model=model)
        is_yes = bool(res.get("family", False))
        faces = int(res.get("faces", 0))
        recognized = int(res.get("recognized", 0))
    except Exception as e:
        is_yes = False
        faces = 0
        recognized = 0
        print(f"[WARN] check classify failed: {img_path.name} ({e})", file=sys.stderr)

    dt = time.time() - t0

    return jsonify({
        "name": img_path.name,
        "yes": is_yes,
        "faces": faces,
        "recognized": recognized,
        "time_s": round(dt, 3),
        "thumb_url": f"/thumb?run={run_id}&name={img_path.name}",
        "file_url": f"/file?run={run_id}&name={img_path.name}",
    })


@app.get("/thumb")
def thumb():
    """
    GET /thumb?run=<id>&name=<original_filename>

    - If thumb exists: serve JPEG (200)
    - Else: enqueue background generation and serve placeholder PNG (200)
    """
    run_name = request.args.get("run", "")
    name = request.args.get("name", "")
    if not run_name or not name:
        abort(400)

    orig_path, thumb_path = _thumb_paths(run_name, name)
    if not orig_path.exists() or not orig_path.is_file():
        abort(404)

    if thumb_path.exists():
        return send_file(str(thumb_path), mimetype="image/jpeg", max_age=3600)

    # Not ready: enqueue and return placeholder immediately
    _enqueue_thumb(run_name, name)
    resp = make_response(PLACEHOLDER_PNG)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/file")
def file():
    run_name = request.args.get("run", "")
    name = request.args.get("name", "")
    if not run_name or not name:
        abort(400)

    p = _safe_resolve_run_file(run_name, Path(name))
    if not p.exists() or not p.is_file():
        abort(404)

    return send_file(str(p))


@app.get("/download/<job_id>")
def download(job_id: str):
    run_dir = (UPLOAD_DIR / job_id).resolve()
    base = UPLOAD_DIR.resolve()
    if base not in run_dir.parents and run_dir != base:
        abort(403)

    zip_path = run_dir / "results.zip"
    if not zip_path.exists():
        abort(404, "ZIP not ready (run batch first)")

    return send_file(
        str(zip_path),
        as_attachment=True,
        download_name=f"{job_id}_results.zip",
        mimetype="application/zip",
        max_age=0,
    )


if __name__ == "__main__":
    print(f"[INFO] TurboJPEG: {'ENABLED' if TURBOJPEG_AVAILABLE else 'DISABLED (fallback to OpenCV)'}")
    print(f"[INFO] Workers: {THUMB_WORKERS} thumb, {CLASSIFY_WORKERS} classify")
    print(f"[INFO] Thumb settings: max_side={THUMB_MAX_SIDE}, quality={THUMB_QUALITY}")
    
    # 5050 avoids macOS AirPlay Receiver conflicts on 5000
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False, threaded=True)