#!/usr/bin/env python3
"""
Family Photo Organizer v2 - High Performance Web UI with Two-Tier Detection

Two-Tier Detection:
  - FAST pass: HOG model, smaller image, 1 jitter → ~5-10x faster
  - FULL pass: CNN model, larger image, 2 jitters → more accurate
  - Logic: If FAST pass detects family → done. If FAIL → retry with FULL pass.

Run:
  python family_photo_organizer_v2.py
  PROCESS_WORKERS=8 python family_photo_organizer_v2.py
  TWO_TIER=1 python family_photo_organizer_v2.py  # Enable two-tier by default

URL: http://127.0.0.1:5050
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects.*", category=UserWarning)

import os, sys, json, time, queue, shutil, zipfile, tempfile, threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from flask import Flask, request, abort, send_file, render_template_string, jsonify, Response, make_response

from family_photo_detector import (
    load_db, load_edge_db, classify_image, classify_image_from_array,
    classify_image_from_array_two_tier, classify_image_two_tier,
    classify_image_from_array_three_tier, train_edge,
    DetectionConfig, ClassifyResult, TRAIN_EDGE_DIR, EDGE_DB_PATH, TRAIN_FAMILY_DIR,
    DEFAULT_TOLERANCE, DEFAULT_DECISION, DEFAULT_FACE_MODEL_CHECK,
    DEFAULT_MAX_SIDE, DEFAULT_JITTERS_CHECK,
)

# ---- TurboJPEG (Thread-Safe with Lock) ----
TURBOJPEG_AVAILABLE = False
TURBOJPEG_DISABLED = os.environ.get("TURBOJPEG_DISABLE", "0").lower() in ("1", "true", "yes")
_tj_lock = threading.Lock()  # Serialize TurboJPEG calls to prevent segfault
_tj_instance = None
_TurboJPEG = None
_TJPF_BGR = None
_TJFLAG_FASTDCT = None
_TJFLAG_FASTUPSAMPLE = None

if not TURBOJPEG_DISABLED:
    try:
        from turbojpeg import TurboJPEG as _TurboJPEG_cls, TJPF_BGR, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE
        _TurboJPEG = _TurboJPEG_cls
        _TJPF_BGR = TJPF_BGR
        _TJFLAG_FASTDCT = TJFLAG_FASTDCT
        _TJFLAG_FASTUPSAMPLE = TJFLAG_FASTUPSAMPLE
        _tj_instance = _TurboJPEG()  # Single instance, protected by lock
        TURBOJPEG_AVAILABLE = True
    except Exception as e:
        print(f"[INFO] TurboJPEG not available: {e}", file=sys.stderr)
else:
    print("[INFO] TurboJPEG disabled via TURBOJPEG_DISABLE=1", file=sys.stderr)


# ---- File Processing State Management ----
from enum import Enum

class FileState(Enum):
    PENDING = "pending"
    DECODING = "decoding"
    THUMBNAILING = "thumbnailing"  
    CLASSIFYING_FAST = "classifying_fast"
    CLASSIFYING_FULL = "classifying_full"
    SAVING = "saving"
    DONE = "done"
    ERROR = "error"


@dataclass
class FileProcessingContext:
    """Thread-safe state management for a single file"""
    name: str
    state: FileState = FileState.PENDING
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Data holders (isolated per file)
    raw_bytes: Optional[bytes] = None
    bgr_full: Optional[np.ndarray] = None
    rgb_full: Optional[np.ndarray] = None
    thumb_bytes: Optional[bytes] = None
    
    # Result
    result: Optional['ProcessResult'] = None
    error: Optional[str] = None
    
    def set_state(self, new_state: FileState):
        with self.lock:
            self.state = new_state
    
    def get_state(self) -> FileState:
        with self.lock:
            return self.state


class FileStateManager:
    """Manages processing state for all files in a batch job"""
    def __init__(self):
        self._files: Dict[str, FileProcessingContext] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, raw_bytes: bytes) -> FileProcessingContext:
        ctx = FileProcessingContext(name=name, raw_bytes=raw_bytes)
        with self._lock:
            self._files[name] = ctx
        return ctx
    
    def get(self, name: str) -> Optional[FileProcessingContext]:
        with self._lock:
            return self._files.get(name)
    
    def all_contexts(self) -> List[FileProcessingContext]:
        with self._lock:
            return list(self._files.values())
    
    def count_by_state(self, state: FileState) -> int:
        with self._lock:
            return sum(1 for ctx in self._files.values() if ctx.state == state)
    
    def is_all_done(self) -> bool:
        with self._lock:
            return all(ctx.state in (FileState.DONE, FileState.ERROR) for ctx in self._files.values())

app = Flask(__name__)

# ---- Config ----
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", tempfile.gettempdir())) / "family_photo_organizer_v2"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

THUMB_DIRNAME = "_thumbs"
THUMB_MAX_SIDE = int(os.environ.get("THUMB_MAX_SIDE", "320"))
THUMB_QUALITY = int(os.environ.get("THUMB_QUALITY", "72"))
PROCESS_WORKERS = int(os.environ.get("PROCESS_WORKERS", "6"))
CLASSIFY_MAX_SIDE = int(os.environ.get("CLASSIFY_MAX_SIDE", str(DEFAULT_MAX_SIDE)))
CLASSIFY_JITTERS = int(os.environ.get("CLASSIFY_JITTERS", str(DEFAULT_JITTERS_CHECK)))

# Two-tier detection settings
TWO_TIER_DEFAULT = os.environ.get("TWO_TIER", "1").lower() in ("1", "true", "yes")

DB_CACHE = None
EDGE_DB_CACHE = None
DB_LOCK = threading.Lock()

PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xf8\xf8"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\x18\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _get_db_cached():
    global DB_CACHE
    with DB_LOCK:
        if DB_CACHE is None:
            DB_CACHE = load_db()
        return DB_CACHE

def _get_edge_db_cached():
    global EDGE_DB_CACHE
    with DB_LOCK:
        if EDGE_DB_CACHE is None:
            EDGE_DB_CACHE = load_edge_db()
        return EDGE_DB_CACHE

def _reload_edge_db():
    global EDGE_DB_CACHE
    with DB_LOCK:
        EDGE_DB_CACHE = load_edge_db()
    return EDGE_DB_CACHE is not None

def _new_job_id() -> str:
    return f"job_{os.getpid()}_{os.urandom(6).hex()}"

# ---- Image Utils ----
def _pick_turbo_scaling(max_side: int, w: int, h: int) -> Tuple[int, int]:
    if max_side <= 0: return (1, 1)
    m = max(w, h)
    for den in (8, 4, 2):
        if (m // den) >= max_side: return (1, den)
    return (1, 1)

def _decode_jpeg_turbo(data: bytes, for_thumb: bool = False, thumb_max_side: int = 320):
    """Try TurboJPEG decode with lock, return None on any failure (caller should fallback to OpenCV)"""
    if not TURBOJPEG_AVAILABLE or _tj_instance is None:
        return None
    try:
        with _tj_lock:
            if for_thumb:
                w, h, _, _ = _tj_instance.decode_header(data)
                scaling = _pick_turbo_scaling(thumb_max_side, w, h)
                return _tj_instance.decode(data, pixel_format=_TJPF_BGR, scaling_factor=scaling, flags=_TJFLAG_FASTDCT | _TJFLAG_FASTUPSAMPLE)
            return _tj_instance.decode(data, pixel_format=_TJPF_BGR, flags=_TJFLAG_FASTDCT | _TJFLAG_FASTUPSAMPLE)
    except Exception:
        return None  # Fallback to OpenCV

def _decode_opencv(data: bytes):
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except: return None

def _encode_jpeg(bgr: np.ndarray, quality: int = 72) -> Optional[bytes]:
    """Try TurboJPEG encode with lock, fallback to OpenCV on failure"""
    if TURBOJPEG_AVAILABLE and _tj_instance is not None:
        try:
            with _tj_lock:
                return _tj_instance.encode(bgr, quality=quality, pixel_format=_TJPF_BGR, flags=_TJFLAG_FASTDCT)
        except Exception:
            pass  # Fallback to OpenCV
    try:
        ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None
    except Exception:
        return None

def _resize_if_needed(bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0: return bgr
    h, w = bgr.shape[:2]
    m = max(w, h)
    if m <= max_side: return bgr
    scale = max_side / float(m)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def _make_thumb_from_jpeg(jpeg_data: bytes, max_side: int = 320, quality: int = 72) -> Optional[bytes]:
    """Try TurboJPEG for thumbnail with lock, fallback to OpenCV on failure"""
    if TURBOJPEG_AVAILABLE and _tj_instance is not None:
        try:
            with _tj_lock:
                w, h, _, _ = _tj_instance.decode_header(jpeg_data)
                scaling = _pick_turbo_scaling(max_side, w, h)
                bgr = _tj_instance.decode(jpeg_data, pixel_format=_TJPF_BGR, scaling_factor=scaling, flags=_TJFLAG_FASTDCT | _TJFLAG_FASTUPSAMPLE)
                bgr = _resize_if_needed(bgr, max_side)
                return _tj_instance.encode(bgr, quality=quality, pixel_format=_TJPF_BGR, flags=_TJFLAG_FASTDCT)
        except Exception:
            pass  # Fallback to OpenCV
    
    # OpenCV fallback
    bgr = _decode_opencv(jpeg_data)
    if bgr is None: 
        return None
    bgr = _resize_if_needed(bgr, max_side)
    return _encode_jpeg(bgr, quality)

def _make_thumb_from_bgr(bgr: np.ndarray, max_side: int = 320, quality: int = 72) -> Optional[bytes]:
    thumb_bgr = _resize_if_needed(bgr, max_side)
    return _encode_jpeg(thumb_bgr, quality)

# ---- Processing ----
@dataclass
class ProcessResult:
    name: str
    success: bool = False
    is_family: bool = False
    faces: int = 0
    recognized: int = 0
    thumb_ready: bool = False
    process_time_ms: float = 0
    error: Optional[str] = None
    # Two-tier specific
    mode_used: str = "single"  # "fast", "full", or "single"
    fast_passed: bool = False


def _process_single_image_with_context(ctx: FileProcessingContext, run_dir: Path, db: Any, edge_db: Any,
                                        tolerance: float, decision: str, model: str,
                                        use_two_tier: bool = False, use_edge: bool = False) -> ProcessResult:
    """Process a single image with proper state management"""
    t0 = time.perf_counter()
    result = ProcessResult(name=ctx.name)
    
    try:
        ext = Path(ctx.name).suffix.lower()
        is_jpeg = ext in ('.jpg', '.jpeg')
        
        # STEP 1: Save original
        ctx.set_state(FileState.SAVING)
        orig_path = run_dir / ctx.name
        orig_path.write_bytes(ctx.raw_bytes)
        
        # STEP 2: Decode
        ctx.set_state(FileState.DECODING)
        ctx.bgr_full = _decode_jpeg_turbo(ctx.raw_bytes) if is_jpeg else None
        if ctx.bgr_full is None:
            ctx.bgr_full = _decode_opencv(ctx.raw_bytes)
        if ctx.bgr_full is None:
            ctx.set_state(FileState.ERROR)
            result.error = "Failed to decode"
            return result
        
        # STEP 3: Thumbnail
        ctx.set_state(FileState.THUMBNAILING)
        ctx.thumb_bytes = _make_thumb_from_jpeg(ctx.raw_bytes, THUMB_MAX_SIDE, THUMB_QUALITY) if is_jpeg else _make_thumb_from_bgr(ctx.bgr_full, THUMB_MAX_SIDE, THUMB_QUALITY)
        if ctx.thumb_bytes:
            thumb_dir = run_dir / THUMB_DIRNAME
            thumb_dir.mkdir(parents=True, exist_ok=True)
            (thumb_dir / f"{Path(ctx.name).stem}.jpg").write_bytes(ctx.thumb_bytes)
            result.thumb_ready = True
        
        # STEP 4: Classify
        ctx.rgb_full = cv2.cvtColor(ctx.bgr_full, cv2.COLOR_BGR2RGB)
        
        if use_two_tier:
            ctx.set_state(FileState.CLASSIFYING_FAST)
            if use_edge and edge_db is not None:
                # Three-tier: fast → full → edge
                classify_result = classify_image_from_array_three_tier(
                    ctx.rgb_full, db, edge_db, decision=decision
                )
            else:
                # Two-tier: fast → full
                classify_result = classify_image_from_array_two_tier(
                    ctx.rgb_full, db, decision=decision
                )
            if classify_result.mode_used == "full":
                ctx.set_state(FileState.CLASSIFYING_FULL)
            result.is_family = classify_result.family
            result.faces = classify_result.faces
            result.recognized = classify_result.recognized
            result.mode_used = classify_result.mode_used
            result.fast_passed = classify_result.fast_passed
        else:
            ctx.set_state(FileState.CLASSIFYING_FAST)
            classify_result = classify_image_from_array(
                ctx.rgb_full, db, tolerance=tolerance, decision=decision,
                face_model=model, max_side=CLASSIFY_MAX_SIDE, num_jitters=CLASSIFY_JITTERS
            )
            result.is_family = bool(classify_result.get("family", False))
            result.faces = int(classify_result.get("faces", 0))
            result.recognized = int(classify_result.get("recognized", 0))
            result.mode_used = "single"
        
        # STEP 5: Copy to bucket
        ctx.set_state(FileState.SAVING)
        bucket = "family" if result.is_family else "non_family"
        bucket_dir = run_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        dst = bucket_dir / ctx.name
        if not dst.exists():
            shutil.copy2(str(orig_path), str(dst))
        else:
            i = 1
            while (bucket_dir / f"{orig_path.stem}_{i}{orig_path.suffix}").exists(): i += 1
            shutil.copy2(str(orig_path), str(bucket_dir / f"{orig_path.stem}_{i}{orig_path.suffix}"))
        
        result.success = True
        ctx.set_state(FileState.DONE)
        
    except Exception as e:
        result.error = str(e)
        ctx.error = str(e)
        ctx.set_state(FileState.ERROR)
        print(f"[ERROR] {ctx.name}: {e}", file=sys.stderr)
    finally:
        # Clear large data to free memory
        ctx.raw_bytes = None
        ctx.bgr_full = None
        ctx.rgb_full = None
        ctx.thumb_bytes = None
    
    result.process_time_ms = (time.perf_counter() - t0) * 1000
    ctx.result = result
    return result


def _process_single_image(name: str, image_bytes: bytes, run_dir: Path, db: Any, edge_db: Any,
                          tolerance: float, decision: str, model: str,
                          use_two_tier: bool = False, use_edge: bool = False) -> ProcessResult:
    """Legacy wrapper - creates context and processes"""
    ctx = FileProcessingContext(name=name, raw_bytes=image_bytes)
    return _process_single_image_with_context(ctx, run_dir, db, edge_db, tolerance, decision, model, use_two_tier, use_edge)

# ---- Batch Job ----
@dataclass
class BatchJob:
    job_id: str
    run_dir: Path
    tolerance: float = DEFAULT_TOLERANCE
    decision: str = DEFAULT_DECISION
    model: str = DEFAULT_FACE_MODEL_CHECK
    use_two_tier: bool = False
    use_edge: bool = False
    total: int = 0
    done: int = 0
    yes: int = 0
    no: int = 0
    fast_passed: int = 0
    edge_passed: int = 0
    started: bool = False
    finished: bool = False
    started_at: float = 0.0
    pending_queue: "queue.Queue[FileProcessingContext]" = field(default_factory=queue.Queue)
    upload_finished: bool = False
    sse_queue: "queue.Queue[str]" = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    state_manager: FileStateManager = field(default_factory=FileStateManager)

JOBS: Dict[str, BatchJob] = {}
JOBS_LOCK = threading.Lock()

def _sse_send(job: BatchJob, event: str, data: dict):
    job.sse_queue.put(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

def _batch_processor(job: BatchJob):
    db = _get_db_cached()
    edge_db = _get_edge_db_cached() if job.use_edge else None
    executor = ThreadPoolExecutor(max_workers=PROCESS_WORKERS)
    futures = {}
    
    def submit(ctx: FileProcessingContext):
        futures[executor.submit(
            _process_single_image_with_context, ctx, job.run_dir, db, edge_db,
            job.tolerance, job.decision, job.model, job.use_two_tier, job.use_edge
        )] = ctx
    
    def handle_done():
        for fut in [f for f in futures if f.done()]:
            ctx = futures.pop(fut)
            try:
                r = fut.result()
                with job.lock:
                    job.done += 1
                    if r.is_family: 
                        job.yes += 1
                        if r.fast_passed:
                            job.fast_passed += 1
                        elif r.mode_used == "edge":
                            job.edge_passed += 1
                    else: 
                        job.no += 1
                
                _sse_send(job, "item", {
                    "name": ctx.name, 
                    "yes": r.is_family, 
                    "done": job.done, 
                    "total": job.total,
                    "yes_count": job.yes, 
                    "no_count": job.no, 
                    "faces": r.faces,
                    "recognized": r.recognized, 
                    "time_ms": round(r.process_time_ms, 1), 
                    "thumb_ready": r.thumb_ready,
                    "mode_used": r.mode_used,
                    "fast_passed": r.fast_passed,
                    "fast_passed_total": job.fast_passed,
                    "edge_passed_total": job.edge_passed,
                    "state": ctx.get_state().value,
                })
                if r.thumb_ready: 
                    _sse_send(job, "thumb_ready", {"name": ctx.name})
            except Exception as e:
                print(f"[ERROR] {ctx.name}: {e}", file=sys.stderr)
                ctx.set_state(FileState.ERROR)
                with job.lock: 
                    job.done += 1
                    job.no += 1
    
    while True:
        try:
            while True:
                ctx = job.pending_queue.get_nowait()
                submit(ctx)
        except queue.Empty: pass
        handle_done()
        with job.lock:
            if job.upload_finished and job.pending_queue.empty() and not futures: break
        time.sleep(0.01)
    
    executor.shutdown(wait=True)
    
    # ZIP
    dl = ""
    try:
        zip_path = job.run_dir / "results.zip"
        if zip_path.exists(): zip_path.unlink()
        fam, non = job.run_dir / "family", job.run_dir / "non_family"
        if fam.exists() or non.exists():
            with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
                for sub in ("family", "non_family"):
                    folder = job.run_dir / sub
                    if folder.exists():
                        for p in folder.rglob("*"):
                            if p.is_file(): z.write(str(p), str(p.relative_to(job.run_dir)))
            if zip_path.exists(): dl = f"/download/{job.job_id}"
    except Exception as e:
        print(f"[WARN] zip: {e}", file=sys.stderr)
    
    with job.lock: job.finished = True
    
    # Calculate fast pass rate
    fast_rate = (job.fast_passed / max(job.yes, 1)) * 100 if job.yes > 0 else 0
    
    _sse_send(job, "done", {
        "done": job.done, 
        "total": job.total, 
        "yes_count": job.yes, 
        "no_count": job.no,
        "fast_passed_total": job.fast_passed,
        "edge_passed_total": job.edge_passed,
        "fast_pass_rate": round(fast_rate, 1),
        "elapsed_s": round(time.time() - job.started_at, 2), 
        "download_url": dl,
        "two_tier": job.use_two_tier,
        "use_edge": job.use_edge,
    })

def _safe_resolve(run_name: str, rel_path: Path) -> Path:
    run_dir = (UPLOAD_DIR / run_name).resolve()
    base = UPLOAD_DIR.resolve()
    if base not in run_dir.parents and run_dir != base: abort(403)
    p = (run_dir / rel_path).resolve()
    if run_dir not in p.parents and p != run_dir: abort(403)
    return p

# ---- HTML ----
HTML = r'''<!doctype html>
<html>
<head>
  <meta charset="utf-8"/><title>Family Photo Organizer v2</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    *{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:16px;padding-bottom:240px;background:#f5f5f7}
    .card{background:#fff;border-radius:16px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}
    h2{margin:0 0 8px 0;font-size:24px;font-weight:600}h3{margin:0 0 16px 0;font-size:18px}
    .btn{padding:10px 18px;border:none;border-radius:10px;background:#e5e5ea;cursor:pointer;font-weight:500;font-size:14px;transition:all 0.15s}
    .btn:hover{background:#d1d1d6}.btn-primary{background:#007aff;color:#fff}.btn-primary:hover{background:#0066d6}
    .btn-rescue{background:#f59e0b;color:#fff}.btn-rescue:hover{background:#d97706}
    .btn-danger{background:#ef4444;color:#fff}.btn-danger:hover{background:#dc2626}
    .btn-sm{padding:6px 12px;font-size:12px}
    input,select{padding:10px 14px;border-radius:10px;border:1px solid #d1d1d6;font-size:14px;background:#fff}
    .muted{color:#86868b;font-size:13px}
    .badge{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600}
    .badge-green{background:#d1f2d9;color:#1d7d3f}.badge-yellow{background:#fff3cd;color:#856404}.badge-blue{background:#d1e7ff;color:#0056b3}
    .badge-purple{background:#e8daff;color:#6b21a8}.badge-orange{background:#ffedd5;color:#c2410c}
    .stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
    .stat{padding:10px 16px;border-radius:12px;font-size:14px;font-weight:600;background:#f5f5f7;border:1px solid #e5e5ea}
    .stat.yes{background:#d1f2d9;border-color:#a3e4b1;color:#1d7d3f}.stat.no{background:#ffd9d9;border-color:#ffb3b3;color:#c0392b}
    .stat.status{background:#d1e7ff;border-color:#a3c9ff;color:#0056b3}
    .stat.fast{background:#e8daff;border-color:#d4b3ff;color:#6b21a8}
    .stat.rescue{background:#ffedd5;border-color:#fed7aa;color:#c2410c}
    .linkbtn{padding:10px 18px;border-radius:10px;border:none;background:#007aff;color:#fff;font-weight:600;text-decoration:none;display:inline-block;font-size:14px}
    .grid-wrap{margin-top:16px;display:none}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
    .tile{border-radius:14px;padding:10px;background:#f5f5f7;border:2px solid transparent;transition:all 0.2s;position:relative;cursor:pointer}
    .tile.yes{background:rgba(34,197,94,0.35);border-color:rgba(34,197,94,0.7)}.tile.no{background:rgba(239,68,68,0.35);border-color:rgba(239,68,68,0.7)}
    .tile.selected{border-color:#f59e0b !important;box-shadow:0 0 0 3px rgba(245,158,11,0.3)}
    .tile img{width:100%;height:110px;object-fit:cover;border-radius:10px;background:#e5e5ea;display:block}
    .tile-meta{margin-top:8px;font-size:11px;color:#86868b;word-break:break-all;line-height:1.4}.tile-time{font-size:10px;color:#aeaeb2;margin-top:4px}
    .tile-mode{position:absolute;top:6px;right:6px;font-size:9px;padding:2px 6px;border-radius:4px;font-weight:600;text-transform:uppercase}
    .tile-mode.fast{background:rgba(107,33,168,0.8);color:#fff}.tile-mode.full{background:rgba(59,130,246,0.8);color:#fff}.tile-mode.edge{background:rgba(245,158,11,0.8);color:#fff}
    .tile-check{position:absolute;top:6px;left:6px;width:20px;height:20px;background:#f59e0b;border-radius:50%;display:none;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:bold}
    .tile.selected .tile-check{display:flex}
    .rescue-panel{display:none;margin-top:16px;padding:16px;background:#fffbeb;border:1px solid #fcd34d;border-radius:12px}
    .rescue-panel.active{display:block}
    .rescue-mode-toggle{display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap}
    .mode-option{display:flex;flex-direction:column;padding:12px;border:2px solid #e5e5ea;border-radius:10px;cursor:pointer;flex:1;min-width:200px;transition:all 0.15s}
    .mode-option:has(input:checked){border-color:#007aff;background:#f0f7ff}
    .mode-option input{display:none}
    .mode-label{font-weight:600;font-size:14px;color:#1d1d1f}
    .mode-hint{font-size:11px;color:#86868b;margin-top:4px}
    .retrain-panel{display:none;margin-top:12px;padding:16px;background:#d1f2d9;border:1px solid #a3e4b1;border-radius:12px}
    .retrain-panel.active{display:block}
    .finalize-panel{display:none;margin-top:12px;padding:16px;background:#d1e7ff;border:1px solid #a3c9ff;border-radius:12px}
    .finalize-panel.active{display:block}
    .rescue-info{font-size:13px;color:#92400e;margin-bottom:12px}
    .retrain-panel .rescue-info{color:#1d7d3f}
    .finalize-panel .rescue-info{color:#0056b3}
    .train-console{display:none;margin-top:12px;background:#1c1c1e;border-radius:8px;padding:10px;max-height:200px;overflow:auto}
    .train-console.active{display:block}
    .train-console-title{color:#86868b;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
    .train-console pre{margin:0;font-family:'SF Mono',ui-monospace,monospace;font-size:11px;color:#f5f5f7;white-space:pre-wrap;line-height:1.5}
    .train-console .log-progress{color:#64d2ff}
    .train-console .log-done{color:#30d158}
    .train-console .log-error{color:#ff453a}
    .train-console .log-info{color:#ffd60a}
    .console{position:fixed;left:0;right:0;bottom:0;height:200px;background:#1c1c1e;color:#f5f5f7;border-top:1px solid #38383a;font-family:'SF Mono',ui-monospace,monospace;font-size:12px;padding:12px 16px;overflow:auto;z-index:1000}
    .console-title{color:#86868b;margin-bottom:10px;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.5px}
    .console pre{margin:0;white-space:pre-wrap;word-break:break-word;line-height:1.6}
    .log-info{color:#64d2ff}.log-success{color:#30d158}.log-warn{color:#ffd60a}.log-error{color:#ff453a}.log-fast{color:#c084fc}.log-rescue{color:#fbbf24}
    .toggle-label{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:14px;font-weight:500}
    .toggle-label input{width:18px;height:18px}
    .tt-hint{font-size:11px;color:#86868b;font-style:italic}
    .filter-btns{display:flex;gap:8px;margin-bottom:12px}
    .filter-btn{padding:6px 14px;border-radius:8px;border:1px solid #d1d1d6;background:#fff;cursor:pointer;font-size:12px;font-weight:500}
    .filter-btn.active{background:#007aff;color:#fff;border-color:#007aff}
  </style>
</head>
<body>
<div class="card">
  <h2>Family Photo Organizer <span class="badge badge-blue">v2</span>
    {% if turbojpeg %}<span class="badge badge-green">TurboJPEG</span>{% else %}<span class="badge badge-yellow">OpenCV</span>{% endif %}
    <span class="badge badge-purple">Two-Tier</span>
  </h2>
  <div class="muted">In-memory pipeline • Shared decode • {{ workers }} workers • Two-tier detection (fast→full)</div>
  <form id="mainForm" enctype="multipart/form-data">
    <div class="row">
      <select name="mode" id="mode"><option value="batch" selected>Batch</option><option value="check">Single</option></select>
      <select name="decision" id="decision"><option value="any_known" selected>any_known</option><option value="majority_known">majority_known</option><option value="all_known">all_known</option></select>
      <input name="tolerance" id="tolerance" value="{{ tolerance }}" size="5" placeholder="Tolerance"/>
      <select name="model" id="model"><option value="cnn" {% if model == 'cnn' %}selected{% endif %}>CNN</option><option value="hog" {% if model == 'hog' %}selected{% endif %}>HOG</option></select>
      <label class="toggle-label"><input type="checkbox" name="two_tier" id="two_tier" {% if two_tier_default %}checked{% endif %}/> Two-Tier</label>
      <span class="muted tt-hint">(Fast: HOG → Slow: CNN)</span>
      <label class="toggle-label"><input type="checkbox" name="use_edge" id="use_edge" {% if edge_available %}{% endif %}/> Edge Model</label>
    </div>
    <div class="row">
      <button type="button" class="btn" id="browseBtn">Browse…</button>
      <input type="file" id="picker" name="files" style="display:none;" multiple/>
      <button type="submit" class="btn btn-primary">Run</button>
      <span class="muted" id="pickedLabel">No selection</span>
      <a id="downloadLink" class="linkbtn" href="#" style="display:none;">Download ZIP</a>
    </div>
  </form>
  <div class="stats" id="statsRow" style="display:none;">
    <span class="stat" id="statProgress">0/0</span>
    <span class="stat yes" id="statYes">YES: 0</span>
    <span class="stat no" id="statNo">NO: 0</span>
    <span class="stat" id="statThumbs">Thumbs: 0</span>
    <span class="stat fast" id="statFast" style="display:none;">⚡ Fast: 0</span>
    <span class="stat status" id="statStatus">Idle</span>
  </div>
</div>
<div class="card grid-wrap" id="gridWrap">
  <div class="filter-btns">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="yes">✓ Family</button>
    <button class="filter-btn" data-filter="no">✗ Not Family</button>
  </div>
  <h3>Results <span class="muted" id="selectHint" style="display:none;font-weight:400;">(Click NO photos to select → Move to Family or Train Edge)</span></h3>
  <div class="grid" id="grid"></div>
  <div class="rescue-panel" id="rescuePanel">
    <div class="rescue-mode-toggle">
      <label class="mode-option">
        <input type="radio" name="rescue_mode" value="move" checked/> 
        <span class="mode-label">📁 Move to Family</span>
        <span class="mode-hint">Override: move selected to family folder</span>
      </label>
      <label class="mode-option">
        <input type="radio" name="rescue_mode" value="train"/> 
        <span class="mode-label">🧠 Train Edge Model</span>
        <span class="mode-hint">Copy to train/edge/ for model training</span>
      </label>
    </div>
    <div class="row">
      <span class="stat rescue" id="statSelected">Selected: 0</span>
      <button class="btn btn-rescue" id="rescueBtn">📁 Move to Family</button>
      <button class="btn btn-sm" id="clearSelectBtn">Clear Selection</button>
    </div>
  </div>
  <div class="retrain-panel" id="retrainPanel">
    <div class="rescue-info">✅ Photos copied to train/edge/. Ready to train edge model.</div>
    <div class="row">
      <button class="btn btn-primary" id="retrainBtn">🔄 Train Edge Model</button>
      <span class="muted" id="retrainStatus"></span>
    </div>
    <div class="train-console" id="trainConsole">
      <div class="train-console-title">Training Log</div>
      <pre id="trainLog"></pre>
    </div>
  </div>
  <div class="finalize-panel" id="finalizePanel">
    <div class="rescue-info">✅ Review complete! Click to regenerate ZIP with updated family folder.</div>
    <div class="row">
      <button class="btn btn-primary" id="regenerateZipBtn">📦 Generate Download ZIP</button>
      <a id="downloadLinkFinal" class="linkbtn" href="#" style="display:none;">Download ZIP</a>
    </div>
  </div>
</div>
<div class="console"><div class="console-title">Console</div><pre id="consoleText"></pre></div>
<script>
const $=id=>document.getElementById(id),modeEl=$('mode'),picker=$('picker'),pickedLabel=$('pickedLabel'),grid=$('grid'),gridWrap=$('gridWrap'),statsRow=$('statsRow'),statProgress=$('statProgress'),statYes=$('statYes'),statNo=$('statNo'),statThumbs=$('statThumbs'),statFast=$('statFast'),statStatus=$('statStatus'),downloadLink=$('downloadLink'),consoleText=$('consoleText'),twoTierEl=$('two_tier'),useEdgeEl=$('use_edge'),rescuePanel=$('rescuePanel'),retrainPanel=$('retrainPanel'),statSelected=$('statSelected'),selectHint=$('selectHint'),retrainStatus=$('retrainStatus');
let currentJobId=null,currentES=null,thumbsReady=0,totalFiles=0,isTwoTier=false,useEdge=false,selectedFiles=new Set(),isJobDone=false,currentFilter='all';

function log(msg,type='info'){const ts=new Date().toLocaleTimeString(),cls={info:'log-info',success:'log-success',warn:'log-warn',error:'log-error',fast:'log-fast',rescue:'log-rescue'}[type]||'';consoleText.innerHTML+=`<span class="${cls}">[${ts}] ${msg}</span>\n`;consoleText.parentElement.scrollTop=consoleText.parentElement.scrollHeight}

function configurePicker(){picker.value='';pickedLabel.textContent='No selection';picker.removeAttribute('webkitdirectory');if(modeEl.value==='batch'){picker.setAttribute('webkitdirectory','');picker.multiple=true;picker.accept=''}else{picker.multiple=false;picker.accept='image/*'}}
modeEl.addEventListener('change',configurePicker);$('browseBtn').addEventListener('click',()=>picker.click());
picker.addEventListener('change',()=>{downloadLink.style.display='none';if(!picker.files?.length){pickedLabel.textContent='No selection';return}pickedLabel.textContent=modeEl.value==='batch'?`${picker.files.length} files`:picker.files[0].name});

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter=btn.dataset.filter;
    applyFilter();
  });
});

function applyFilter(){
  document.querySelectorAll('.tile').forEach(tile=>{
    const isYes=tile.classList.contains('yes');
    const isNo=tile.classList.contains('no');
    if(currentFilter==='all')tile.style.display='';
    else if(currentFilter==='yes')tile.style.display=isYes?'':'none';
    else if(currentFilter==='no')tile.style.display=isNo?'':'none';
  });
}

function buildGrid(names){
  gridWrap.style.display='block';statsRow.style.display='flex';grid.innerHTML='';thumbsReady=0;totalFiles=names.length;selectedFiles.clear();isJobDone=false;movedToFamily=false;
  rescuePanel.classList.remove('active');retrainPanel.classList.remove('active');$('finalizePanel').classList.remove('active');selectHint.style.display='none';updateSelectedCount();
  $('downloadLinkFinal').style.display='none';
  for(const name of names){
    const tile=document.createElement('div');tile.className='tile';tile.dataset.name=name;
    const check=document.createElement('div');check.className='tile-check';check.textContent='✓';tile.appendChild(check);
    const img=document.createElement('img');img.loading='lazy';tile.appendChild(img);
    const meta=document.createElement('div');meta.className='tile-meta';meta.textContent=name.length>25?name.slice(0,22)+'...':name;meta.title=name;tile.appendChild(meta);
    const timeDiv=document.createElement('div');timeDiv.className='tile-time';tile.appendChild(timeDiv);
    tile.addEventListener('click',()=>toggleSelect(tile));
    grid.appendChild(tile);
  }
  updateStats(0,names.length,0,0,0,0);
}

function toggleSelect(tile){
  if(!isJobDone)return;
  if(!tile.classList.contains('no'))return;
  const name=tile.dataset.name;
  if(selectedFiles.has(name)){
    selectedFiles.delete(name);
    tile.classList.remove('selected');
  }else{
    selectedFiles.add(name);
    tile.classList.add('selected');
  }
  updateSelectedCount();
}

function updateSelectedCount(){
  statSelected.textContent=`Selected: ${selectedFiles.size}`;
  if(selectedFiles.size>0){
    rescuePanel.classList.add('active');
  }else{
    rescuePanel.classList.remove('active');
  }
}

$('clearSelectBtn').addEventListener('click',()=>{
  selectedFiles.clear();
  document.querySelectorAll('.tile.selected').forEach(t=>t.classList.remove('selected'));
  updateSelectedCount();
  log('Selection cleared','info');
});

// Rescue mode toggle
document.querySelectorAll('input[name="rescue_mode"]').forEach(radio=>{
  radio.addEventListener('change',()=>{
    const mode=document.querySelector('input[name="rescue_mode"]:checked').value;
    const btn=$('rescueBtn');
    if(mode==='move'){
      btn.textContent='📁 Move to Family';
      btn.classList.remove('btn-primary');
      btn.classList.add('btn-rescue');
    }else{
      btn.textContent='🧠 Copy to Edge Training';
      btn.classList.remove('btn-rescue');
      btn.classList.add('btn-primary');
    }
  });
});

// Track if any photos were moved to family (for regenerate ZIP)
let movedToFamily=false;

$('rescueBtn').addEventListener('click',async()=>{
  if(selectedFiles.size===0){alert('No photos selected');return}
  if(!currentJobId){alert('No job found');return}
  
  const mode=document.querySelector('input[name="rescue_mode"]:checked').value;
  const names=[...selectedFiles];
  
  if(mode==='move'){
    // Move to Family mode
    log(`📁 Moving ${names.length} photos to family folder...`,'rescue');
    try{
      const resp=await fetch('/move_to_family',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({job_id:currentJobId,names})
      });
      const data=await resp.json();
      if(resp.ok){
        log(`✓ Moved ${data.moved} photos to family/`,'success');
        names.forEach(name=>{
          const tile=[...document.querySelectorAll('.tile')].find(t=>t.dataset.name===name);
          if(tile){
            tile.classList.remove('no','selected');
            tile.classList.add('yes');
            selectedFiles.delete(name);
          }
        });
        // Update stats
        const yesCount=document.querySelectorAll('.tile.yes').length;
        const noCount=document.querySelectorAll('.tile.no').length;
        statYes.textContent=`YES: ${yesCount}`;
        statNo.textContent=`NO: ${noCount}`;
        updateSelectedCount();
        movedToFamily=true;
        // Show finalize panel
        $('finalizePanel').classList.add('active');
      }else{
        log(`Error: ${data.error}`,'error');
      }
    }catch(e){
      log(`Error: ${e}`,'error');
    }
  }else{
    // Train Edge mode
    log(`🧠 Copying ${names.length} photos to edge training folder...`,'rescue');
    try{
      const resp=await fetch('/rescue_copy',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({job_id:currentJobId,names})
      });
      const data=await resp.json();
      if(resp.ok){
        log(`✓ Copied ${data.copied} photos to train/edge/`,'success');
        names.forEach(name=>{
          const tile=[...document.querySelectorAll('.tile')].find(t=>t.dataset.name===name);
          if(tile){
            tile.classList.remove('selected');
            selectedFiles.delete(name);
          }
        });
        updateSelectedCount();
        retrainPanel.classList.add('active');
        retrainStatus.textContent='';
      }else{
        log(`Error: ${data.error}`,'error');
      }
    }catch(e){
      log(`Error: ${e}`,'error');
    }
  }
});

// Regenerate ZIP button
$('regenerateZipBtn').addEventListener('click',async()=>{
  log('📦 Regenerating ZIP with updated family folder...','info');
  $('regenerateZipBtn').disabled=true;
  $('regenerateZipBtn').textContent='Generating...';
  
  try{
    const resp=await fetch(`/regenerate_zip/${currentJobId}`,{method:'POST'});
    const data=await resp.json();
    if(resp.ok&&data.download_url){
      log(`✓ ZIP regenerated!`,'success');
      $('downloadLinkFinal').href=data.download_url;
      $('downloadLinkFinal').style.display='inline-block';
      downloadLink.href=data.download_url;
      downloadLink.style.display='inline-block';
    }else{
      log(`Error: ${data.error||'Failed to generate ZIP'}`,'error');
    }
  }catch(e){
    log(`Error: ${e}`,'error');
  }finally{
    $('regenerateZipBtn').disabled=false;
    $('regenerateZipBtn').textContent='📦 Generate Download ZIP';
  }
});

$('retrainBtn').addEventListener('click',async()=>{
  log('🔄 Starting edge model training...','rescue');
  retrainStatus.textContent='Training...';
  $('retrainBtn').disabled=true;
  
  // Show and clear training console
  const trainConsole=$('trainConsole');
  const trainLog=$('trainLog');
  trainConsole.classList.add('active');
  trainLog.innerHTML='';
  
  function trainLogAdd(msg, cls=''){
    const line=document.createElement('span');
    line.className=cls;
    line.textContent=msg+'\n';
    trainLog.appendChild(line);
    trainConsole.scrollTop=trainConsole.scrollHeight;
  }
  
  trainLogAdd('[START] Edge model training initiated...', 'log-info');
  const startTime=Date.now();
  
  try{
    const es=new EventSource('/train_edge_stream');
    es.addEventListener('log',e=>{
      const m=JSON.parse(e.data);
      log(m.message,m.type||'info');
      trainLogAdd(`[LOG] ${m.message}`, 'log-info');
    });
    es.addEventListener('progress',e=>{
      const m=JSON.parse(e.data);
      const eta=m.eta>0?` (ETA: ${Math.round(m.eta)}s)`:'';
      retrainStatus.textContent=`${m.current}/${m.total} images${eta}`;
      trainLogAdd(`[PROGRESS] ${m.current}/${m.total} images${eta}`, 'log-progress');
    });
    es.addEventListener('done',e=>{
      const m=JSON.parse(e.data);
      const elapsed=m.elapsed||((Date.now()-startTime)/1000).toFixed(1);
      log(`✅ Edge model trained! ${m.clusters} clusters in ${elapsed}s`,'success');
      trainLogAdd(`[DONE] ✅ Training complete!`, 'log-done');
      trainLogAdd(`[DONE] Created ${m.clusters} clusters in ${elapsed}s`, 'log-done');
      trainLogAdd(`[DONE] Edge model saved to db/edge_db.pkl`, 'log-done');
      trainLogAdd(`[INFO] "Edge Model" checkbox enabled automatically`, 'log-done');
      retrainStatus.textContent=`✓ Done! ${m.clusters} clusters`;
      $('retrainBtn').disabled=false;
      useEdgeEl.checked=true;
      es.close();
      fetch('/reload_edge_db',{method:'POST'});
    });
    es.addEventListener('error',e=>{
      let errMsg='Training failed';
      try{if(e.data){const m=JSON.parse(e.data);errMsg=m.error||errMsg}}catch{}
      log(`❌ ${errMsg}`,'error');
      trainLogAdd(`[ERROR] ❌ ${errMsg}`, 'log-error');
      retrainStatus.textContent='Failed';
      $('retrainBtn').disabled=false;
      es.close();
    });
  }catch(e){
    log(`Error: ${e}`,'error');
    trainLogAdd(`[ERROR] ❌ ${e}`, 'log-error');
    retrainStatus.textContent='Failed';
    $('retrainBtn').disabled=false;
  }
});

function updateStats(done,total,yes,no,fastPassed,edgePassed){
  statProgress.textContent=`${done}/${total}`;statYes.textContent=`YES: ${yes}`;statNo.textContent=`NO: ${no}`;
  statThumbs.textContent=`Thumbs: ${thumbsReady}/${totalFiles}`;
  if(isTwoTier){statFast.style.display='inline';let txt=`⚡ Fast: ${fastPassed}`;if(useEdge&&edgePassed>0)txt+=` | Edge: ${edgePassed}`;statFast.textContent=txt}else{statFast.style.display='none'}
}

function updateTile(name,isYes,timeMs,thumbReady,modeUsed,fastPassed){
  const tile=[...document.querySelectorAll('.tile')].find(t=>t.dataset.name===name);if(!tile)return;
  tile.classList.add(isYes?'yes':'no');
  const timeDiv=tile.querySelector('.tile-time');if(timeDiv&&timeMs)timeDiv.textContent=`${timeMs.toFixed(0)}ms`;
  if(isTwoTier&&isYes&&modeUsed){let modeBadge=tile.querySelector('.tile-mode');if(!modeBadge){modeBadge=document.createElement('div');modeBadge.className='tile-mode '+modeUsed;tile.appendChild(modeBadge)}modeBadge.textContent=modeUsed}
  if(thumbReady)loadThumb(name);
  applyFilter();
}

function loadThumb(name){const tile=[...document.querySelectorAll('.tile')].find(t=>t.dataset.name===name);if(!tile)return;const img=tile.querySelector('img');if(!img||img.dataset.loaded)return;img.dataset.loaded='1';img.src=`/thumb?run=${currentJobId}&name=${encodeURIComponent(name)}&v=${Date.now()}`;thumbsReady++;statThumbs.textContent=`Thumbs: ${thumbsReady}/${totalFiles}`}

function onJobDone(){
  isJobDone=true;
  selectHint.style.display='inline';
  log('💡 Click on NO photos to select them, then choose: Move to Family or Train Edge Model','info');
}

async function runBatch(fd){
  if(currentES){currentES.close();currentES=null}downloadLink.style.display='none';statStatus.textContent='Uploading...';
  isTwoTier=twoTierEl.checked;useEdge=useEdgeEl.checked;
  let modeStr=isTwoTier?(useEdge?'Three-Tier (fast→full→edge)':'Two-Tier (fast→full)'):'Single';
  log(`Starting... (${modeStr})`,'info');
  const t0=performance.now();
  const resp=await fetch('/batch_stream',{method:'POST',body:fd});
  if(!resp.ok){log(`Failed: ${await resp.text()}`,'error');statStatus.textContent='Error';return}
  const data=await resp.json();currentJobId=data.job_id;
  log(`Uploaded ${data.total} files in ${(performance.now()-t0).toFixed(0)}ms`,'success');
  buildGrid(data.names);statStatus.textContent='Processing...';
  const es=new EventSource(`/batch_events/${data.job_id}`);currentES=es;
  es.addEventListener('item',e=>{const m=JSON.parse(e.data);updateTile(m.name,m.yes,m.time_ms,m.thumb_ready,m.mode_used,m.fast_passed);updateStats(m.done,m.total,m.yes_count,m.no_count,m.fast_passed_total||0,m.edge_passed_total||0);statStatus.textContent=`${m.done}/${m.total}`;if(m.mode_used==='fast'&&isTwoTier)log(`⚡ ${m.name}: FAST pass (${m.time_ms.toFixed(0)}ms)`,'fast');if(m.mode_used==='edge')log(`🔸 ${m.name}: EDGE pass (${m.time_ms.toFixed(0)}ms)`,'rescue')});
  es.addEventListener('thumb_ready',e=>{loadThumb(JSON.parse(e.data).name)});
  es.addEventListener('done',e=>{const m=JSON.parse(e.data);updateStats(m.done,m.total,m.yes_count,m.no_count,m.fast_passed_total||0,m.edge_passed_total||0);let summary=`Complete: YES=${m.yes_count} NO=${m.no_count}`;if(m.two_tier)summary+=` | Fast: ${m.fast_passed_total}`;if(m.edge_passed_total>0)summary+=` | Edge: ${m.edge_passed_total}`;statStatus.textContent=`Done ✓ ${m.elapsed_s}s`;log(summary,'success');if(m.download_url){downloadLink.href=m.download_url;downloadLink.style.display='inline-block'}onJobDone();es.close()});
  es.addEventListener('error',()=>{log('Connection lost','error');statStatus.textContent='Disconnected'});
}

async function runCheck(fd){statStatus.textContent='Processing...';isTwoTier=twoTierEl.checked;useEdge=useEdgeEl.checked;log(`Checking...`,'info');const resp=await fetch('/check',{method:'POST',body:fd});if(!resp.ok){log(`Failed: ${await resp.text()}`,'error');statStatus.textContent='Error';return}const data=await resp.json();currentJobId=data.run_id;buildGrid([data.name]);updateTile(data.name,data.yes,data.time_ms,true,data.mode_used,data.fast_passed);updateStats(1,1,data.yes?1:0,data.yes?0:1,data.fast_passed?1:0,data.mode_used==='edge'?1:0);thumbsReady=1;statThumbs.textContent='Thumbs: 1/1';statStatus.textContent='Done ✓';let result=`${data.name} → ${data.yes?'FAMILY':'NOT FAMILY'} (faces=${data.faces})`;if(data.mode_used)result+=` [${data.mode_used}]`;log(`Result: ${result}`,'success');onJobDone()}

$('mainForm').addEventListener('submit',async e=>{e.preventDefault();if(!picker.files?.length){alert('Select files');return}const fd=new FormData();fd.append('decision',$('decision').value);fd.append('tolerance',$('tolerance').value);fd.append('model',$('model').value);fd.append('two_tier',twoTierEl.checked?'1':'0');fd.append('use_edge',useEdgeEl.checked?'1':'0');for(const f of picker.files)fd.append('files',f,f.name);log(`Mode: ${modeEl.value}, Files: ${picker.files.length}`,'info');if(modeEl.value==='batch')await runBatch(fd);else await runCheck(fd)});
configurePicker();log('Ready (Two-Tier + Edge Model support)','info');
</script>
</body>
</html>'''

# ---- Routes ----
@app.get("/")
def home():
    edge_available = EDGE_DB_PATH.exists()
    return render_template_string(HTML, tolerance=DEFAULT_TOLERANCE, model=DEFAULT_FACE_MODEL_CHECK,
                                  turbojpeg=TURBOJPEG_AVAILABLE, workers=PROCESS_WORKERS,
                                  two_tier_default=TWO_TIER_DEFAULT, edge_available=edge_available)

@app.post("/batch_stream")
def batch_stream():
    files = request.files.getlist("files")
    if not files: abort(400, "No files")
    
    decision = request.form.get("decision", DEFAULT_DECISION)
    tolerance = float(request.form.get("tolerance", DEFAULT_TOLERANCE))
    model = request.form.get("model", DEFAULT_FACE_MODEL_CHECK)
    use_two_tier = request.form.get("two_tier", "0") == "1"
    use_edge = request.form.get("use_edge", "0") == "1"
    
    job_id = _new_job_id()
    run_dir = UPLOAD_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    job = BatchJob(job_id=job_id, run_dir=run_dir, tolerance=tolerance, 
                   decision=decision, model=model, use_two_tier=use_two_tier, use_edge=use_edge)
    
    names = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    for fs in files:
        if not fs or not fs.filename: continue
        name = Path(fs.filename).name
        if name.startswith('.') or Path(name).suffix.lower() not in valid_exts: continue
        try:
            data = fs.read()
            # Register file with state manager and create context
            ctx = job.state_manager.register(name, data)
            job.pending_queue.put(ctx)
            names.append(name)
        except: pass
    
    if not names: abort(400, "No valid images")
    
    job.total = len(names)
    job.started = True
    job.started_at = time.time()
    job.upload_finished = True
    
    with JOBS_LOCK: JOBS[job_id] = job
    threading.Thread(target=_batch_processor, args=(job,), daemon=True).start()
    
    return jsonify({"job_id": job_id, "total": len(names), "names": names, "two_tier": use_two_tier})

@app.get("/batch_events/<job_id>")
def batch_events(job_id: str):
    with JOBS_LOCK: job = JOBS.get(job_id)
    if not job: abort(404)
    
    def gen():
        yield "event: connected\ndata: {}\n\n"
        while True:
            try:
                msg = job.sse_queue.get(timeout=15)
                yield msg
                with job.lock:
                    if job.finished and job.sse_queue.empty(): break
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                with job.lock:
                    if job.finished: break
    return Response(gen(), mimetype="text/event-stream")

@app.post("/check")
def check_one():
    files = request.files.getlist("files")
    if not files or not files[0].filename: abort(400, "No file")
    
    decision = request.form.get("decision", DEFAULT_DECISION)
    tolerance = float(request.form.get("tolerance", DEFAULT_TOLERANCE))
    model = request.form.get("model", DEFAULT_FACE_MODEL_CHECK)
    use_two_tier = request.form.get("two_tier", "0") == "1"
    use_edge = request.form.get("use_edge", "0") == "1"
    
    fs = files[0]
    name = Path(fs.filename).name
    data = fs.read()
    
    run_id = _new_job_id()
    run_dir = UPLOAD_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    db = _get_db_cached()
    edge_db = _get_edge_db_cached() if use_edge else None
    result = _process_single_image(name, data, run_dir, db, edge_db, tolerance, decision, model, use_two_tier, use_edge)
    
    return jsonify({
        "run_id": run_id, 
        "name": name, 
        "yes": result.is_family, 
        "faces": result.faces,
        "recognized": result.recognized, 
        "time_ms": round(result.process_time_ms, 1), 
        "thumb_ready": result.thumb_ready,
        "mode_used": result.mode_used,
        "fast_passed": result.fast_passed,
        "two_tier": use_two_tier,
        "use_edge": use_edge,
    })

@app.get("/thumb")
def thumb():
    run, name = request.args.get("run", ""), request.args.get("name", "")
    if not run or not name: abort(400)
    thumb_path = _safe_resolve(run, Path(THUMB_DIRNAME) / f"{Path(name).stem}.jpg")
    if thumb_path.exists(): return send_file(str(thumb_path), mimetype="image/jpeg", max_age=3600)
    resp = make_response(PLACEHOLDER_PNG)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/file")
def file_serve():
    run, name = request.args.get("run", ""), request.args.get("name", "")
    if not run or not name: abort(400)
    p = _safe_resolve(run, Path(name))
    if not p.exists(): abort(404)
    return send_file(str(p))

@app.get("/download/<job_id>")
def download(job_id: str):
    zip_path = _safe_resolve(job_id, Path("results.zip"))
    if not zip_path.exists(): abort(404, "ZIP not ready")
    return send_file(str(zip_path), as_attachment=True, download_name=f"{job_id}_results.zip", mimetype="application/zip")


# ---- Rescue Training (Edge Model) ----
@app.post("/rescue_copy")
def rescue_copy():
    """Copy selected NO photos to train/edge/ for edge model training"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    job_id = data.get("job_id")
    names = data.get("names", [])
    
    if not job_id or not names:
        return jsonify({"error": "Missing job_id or names"}), 400
    
    # Ensure train/edge folder exists
    TRAIN_EDGE_DIR.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    errors = []
    
    for name in names:
        try:
            # Look in non_family folder first
            src = _safe_resolve(job_id, Path("non_family") / name)
            if not src.exists():
                # Try root folder
                src = _safe_resolve(job_id, Path(name))
            
            if not src.exists():
                errors.append(f"{name}: not found")
                continue
            
            # Copy to train/edge with unique name if needed
            dst = TRAIN_EDGE_DIR / name
            if dst.exists():
                i = 1
                stem = Path(name).stem
                suffix = Path(name).suffix
                while (TRAIN_EDGE_DIR / f"{stem}_{i}{suffix}").exists():
                    i += 1
                dst = TRAIN_EDGE_DIR / f"{stem}_{i}{suffix}"
            
            shutil.copy2(str(src), str(dst))
            copied += 1
            print(f"[RESCUE] Copied {name} -> {dst}")
            
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"[RESCUE ERROR] {name}: {e}", file=sys.stderr)
    
    return jsonify({
        "copied": copied,
        "errors": errors,
        "train_dir": str(TRAIN_EDGE_DIR),
    })


@app.post("/move_to_family")
def move_to_family():
    """Move selected NO photos to family folder (manual override)"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    job_id = data.get("job_id")
    names = data.get("names", [])
    
    if not job_id or not names:
        return jsonify({"error": "Missing job_id or names"}), 400
    
    moved = 0
    errors = []
    
    run_dir = UPLOAD_DIR / job_id
    family_dir = run_dir / "family"
    non_family_dir = run_dir / "non_family"
    family_dir.mkdir(parents=True, exist_ok=True)
    
    for name in names:
        try:
            # Look in non_family folder
            src = non_family_dir / name
            if not src.exists():
                # Try root folder
                src = run_dir / name
            
            if not src.exists():
                errors.append(f"{name}: not found")
                continue
            
            # Move to family folder
            dst = family_dir / name
            if dst.exists():
                i = 1
                stem = Path(name).stem
                suffix = Path(name).suffix
                while (family_dir / f"{stem}_{i}{suffix}").exists():
                    i += 1
                dst = family_dir / f"{stem}_{i}{suffix}"
            
            shutil.move(str(src), str(dst))
            moved += 1
            print(f"[MOVE] {name} -> family/")
            
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"[MOVE ERROR] {name}: {e}", file=sys.stderr)
    
    return jsonify({
        "moved": moved,
        "errors": errors,
    })


@app.post("/regenerate_zip/<job_id>")
def regenerate_zip(job_id: str):
    """Regenerate ZIP file after manual moves"""
    try:
        run_dir = UPLOAD_DIR / job_id
        if not run_dir.exists():
            return jsonify({"error": "Job not found"}), 404
        
        zip_path = run_dir / "results.zip"
        if zip_path.exists():
            zip_path.unlink()
        
        fam = run_dir / "family"
        non = run_dir / "non_family"
        
        if not fam.exists() and not non.exists():
            return jsonify({"error": "No folders to zip"}), 400
        
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
            for sub in ("family", "non_family"):
                folder = run_dir / sub
                if folder.exists():
                    for p in folder.rglob("*"):
                        if p.is_file():
                            z.write(str(p), str(p.relative_to(run_dir)))
        
        if zip_path.exists():
            return jsonify({"success": True, "download_url": f"/download/{job_id}"})
        else:
            return jsonify({"error": "Failed to create ZIP"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/train_edge_stream")
def train_edge_stream():
    """Stream edge model training progress via SSE - runs in subprocess to avoid memory leaks"""
    import subprocess
    
    def generate():
        try:
            yield f"event: log\ndata: {json.dumps({'message': 'Starting edge model training...', 'type': 'info'})}\n\n"
            
            # Run training in subprocess with unbuffered output
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c", """
import sys
import json
import time

sys.path.insert(0, '.')
from family_photo_detector import train_edge, TRAIN_EDGE_DIR, EDGE_DB_PATH

start_time = time.time()

# Custom progress callback that prints JSON
def progress_cb(current, total, msg):
    elapsed = time.time() - start_time
    eta = (elapsed / current) * (total - current) if current > 0 else 0
    print(json.dumps({
        "type": "progress", 
        "current": current, 
        "total": total,
        "elapsed": round(elapsed, 1),
        "eta": round(eta, 1)
    }), flush=True)

try:
    print(json.dumps({"type": "log", "message": f"Found {len(list(TRAIN_EDGE_DIR.glob('*')))} files in train/edge/"}), flush=True)
    print(json.dumps({"type": "log", "message": "Extracting face encodings..."}), flush=True)
    
    clusters = train_edge(progress_callback=progress_cb, quiet=True)
    
    elapsed = time.time() - start_time
    print(json.dumps({
        "type": "done", 
        "clusters": clusters,
        "elapsed": round(elapsed, 1),
        "db_path": str(EDGE_DB_PATH)
    }), flush=True)
except Exception as e:
    import traceback
    print(json.dumps({"type": "error", "error": str(e), "traceback": traceback.format_exc()}), flush=True)
    sys.exit(1)
"""],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            
            # Stream output line by line
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    msg_type = data.get("type", "log")
                    
                    if msg_type == "progress":
                        eta_str = f" | ETA: {data.get('eta', 0):.0f}s" if data.get('eta', 0) > 0 else ""
                        yield f"event: progress\ndata: {json.dumps({'current': data['current'], 'total': data['total'], 'eta': data.get('eta', 0)})}\n\n"
                    elif msg_type == "done":
                        yield f"event: done\ndata: {json.dumps({'clusters': data['clusters'], 'elapsed': data.get('elapsed', 0)})}\n\n"
                    elif msg_type == "error":
                        yield f"event: error\ndata: {json.dumps({'error': data['error']})}\n\n"
                    elif msg_type == "log":
                        yield f"event: log\ndata: {json.dumps({'message': data.get('message', line), 'type': 'info'})}\n\n"
                except json.JSONDecodeError:
                    # Regular log line (non-JSON)
                    if line and not line.startswith('\r'):
                        yield f"event: log\ndata: {json.dumps({'message': line, 'type': 'info'})}\n\n"
            
            proc.wait()
            
            # Check for errors
            if proc.returncode != 0:
                yield f"event: error\ndata: {json.dumps({'error': f'Process exited with code {proc.returncode}'})}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


@app.post("/reload_edge_db")
def reload_edge_db():
    """Reload edge DB cache after training"""
    success = _reload_edge_db()
    return jsonify({"success": success, "available": success})

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Family Photo Organizer v2 - Three-Tier + Edge Model")
    print("="*60)
    print(f"  TurboJPEG:  {'✓ ON' if TURBOJPEG_AVAILABLE else '✗ OFF'}")
    print(f"  Workers:    {PROCESS_WORKERS}")
    print(f"  Thumbnail:  {THUMB_MAX_SIDE}px @ Q{THUMB_QUALITY}")
    print(f"  Two-Tier:   {'✓ Default ON' if TWO_TIER_DEFAULT else '○ Default OFF'}")
    print(f"  Edge Model: {'✓ Available' if EDGE_DB_PATH.exists() else '○ Not trained'}")
    print(f"  Upload dir: {UPLOAD_DIR}")
    print(f"  Edge dir:   {TRAIN_EDGE_DIR.absolute()}")
    print("="*60)
    print("  Three-Tier Detection:")
    print("    FAST: HOG @ 800px (main DB)")
    print("    FULL: CNN @ 1600px (main DB)")
    print("    EDGE: CNN @ 1600px (edge DB)")
    print("="*60)
    print("  Edge Model Training:")
    print("    1. Select NO photos that have family members")
    print("    2. Click 'Copy to Edge Training'")
    print("    3. Click 'Train Edge Model'")
    print("    4. Enable 'Edge Model' checkbox")
    print("="*60)
    print("  URL: http://127.0.0.1:5050")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False, threaded=True)