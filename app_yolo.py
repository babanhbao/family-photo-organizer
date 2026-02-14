#!/usr/bin/env python3
"""
Family Photo Organizer - YOLOv8 Version
=======================================
Web UI với YOLOv8 face detection + Side Panel chi tiết.

Run:
  python app_yolo.py
  
URL: http://127.0.0.1:5050
"""

import os, sys, json, time, queue, shutil, zipfile, tempfile, threading
import re
from urllib.parse import quote
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import face_recognition
from sklearn.cluster import DBSCAN
from flask import Flask, request, abort, send_file, render_template_string, jsonify, Response, make_response, after_this_request

from family_photo_detector_yolo import (
    load_db, load_edge_db, classify_image_from_array,
    classify_image_from_array_two_tier, get_yolo_model, train_auto, detect_faces_yolo,
    ClassifyResult, EDGE_DB_PATH, DB_PATH,
    DEFAULT_TOLERANCE, DEFAULT_CONF,
)

app = Flask(__name__)

# ---- Config ----
APP_DIR = Path(os.environ.get("FPO_APP_DATA_DIR", Path(__file__).resolve().parent)).resolve()
APP_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = APP_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = RUNTIME_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_FACES_DIR = APP_DIR / "trained_faces"
RESULTS_DIR = APP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLOUD_MODELS_DIR = APP_DIR / "cloud_models"
CLOUD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_UPLOAD_DIRS = [
    Path(tempfile.gettempdir()) / "family_organizer_yolo",
    Path(tempfile.gettempdir()) / "family_photo_organizer_yolo",
]
APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "5050"))

THUMB_MAX_SIDE = 320
THUMB_QUALITY = 72
PROCESS_WORKERS = int(os.environ.get("PROCESS_WORKERS", "4"))
RUN_CACHE_TTL_S = int(os.environ.get("RUN_CACHE_TTL_S", "300"))
RESULTS_RETENTION_S = int(os.environ.get("RESULTS_RETENTION_S", str(7 * 24 * 3600)))
RESULTS_KEEP_MAX = int(os.environ.get("RESULTS_KEEP_MAX", "40"))

DB_CACHE = None
EDGE_DB_CACHE = None
CLOUD_DB_CACHE: Dict[str, Dict[str, Any]] = {}
DB_LOCK = threading.Lock()

def _get_db():
    global DB_CACHE
    with DB_LOCK:
        if DB_CACHE is None:
            DB_CACHE = load_db()
        return DB_CACHE

def _get_edge_db():
    global EDGE_DB_CACHE
    with DB_LOCK:
        if EDGE_DB_CACHE is None:
            EDGE_DB_CACHE = load_edge_db()
        return EDGE_DB_CACHE

def _new_job_id():
    return f"job_{os.urandom(6).hex()}"

def _sanitize_model_name(name: str) -> str:
    model = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip())
    return model[:64].strip("_")

def _cloud_model_paths(model_name: str):
    safe = _sanitize_model_name(model_name)
    if not safe:
        raise ValueError("Invalid model name")
    model_dir = CLOUD_MODELS_DIR / safe
    return safe, model_dir, model_dir / "family_db.pkl", model_dir / "trained_faces"

def _list_cloud_models():
    models = []
    if not CLOUD_MODELS_DIR.exists():
        return models
    for d in sorted(CLOUD_MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        db_path = d / "family_db.pkl"
        if db_path.exists():
            models.append(d.name)
    return models

def _get_cloud_db(model_name: str):
    safe, _, db_path, faces_dir = _cloud_model_paths(model_name)
    if not db_path.exists():
        raise FileNotFoundError(f"Cloud model not found: {safe}")
    mtime = db_path.stat().st_mtime
    with DB_LOCK:
        cached = CLOUD_DB_CACHE.get(safe)
        if cached and cached.get("mtime") == mtime:
            return cached["db"]
        db = load_db(db_path=db_path, trained_faces_dir=faces_dir)
        CLOUD_DB_CACHE[safe] = {"mtime": mtime, "db": db}
        return db

def _get_family_thumb(family_name, trained_faces_root: Path = TRAINED_FACES_DIR):
    folder = Path(trained_faces_root) / family_name
    if folder.exists():
        for f in folder.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                return str(f)
    return None


def _get_train_face_path(family_name: str, face_file: str,
                         trained_faces_root: Path = TRAINED_FACES_DIR) -> Optional[Path]:
    safe_family = Path(family_name).name
    safe_file = Path(face_file).name
    root = Path(trained_faces_root).resolve()
    folder = (root / safe_family).resolve()
    if not folder.exists() or not folder.is_dir():
        return None
    if folder.parent != root:
        return None
    file_path = (folder / safe_file).resolve()
    if file_path.parent != folder or not file_path.exists() or not file_path.is_file():
        return None
    return file_path

# ---- Image Utils ----
def _decode(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _encode_jpeg(bgr, quality=72):
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None

def _resize(bgr, max_side):
    if max_side <= 0: return bgr
    h, w = bgr.shape[:2]
    if max(w, h) <= max_side: return bgr
    scale = max_side / max(w, h)
    return cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def _make_thumb(bgr):
    return _encode_jpeg(_resize(bgr, THUMB_MAX_SIDE), THUMB_QUALITY)

# ---- Processing ----
@dataclass
class ProcessResult:
    name: str
    is_family: bool = False
    faces: int = 0
    recognized: int = 0
    time_ms: float = 0
    matches: List[Dict] = field(default_factory=list)
    raw_match_scores: List[float] = field(default_factory=list)
    match_scores: List[float] = field(default_factory=list)
    face_locations: List[Any] = field(default_factory=list)
    resized_shape: List[int] = field(default_factory=list)
    yolo_conf: float = 0.0
    sim_threshold: float = 0.55
    error: Optional[str] = None

def _save_query_face_thumbs(name: str, bgr, run_dir: Path, face_locations: List[Any], resized_shape: List[int]):
    if not face_locations:
        return
    try:
        rh, rw = int(resized_shape[0]), int(resized_shape[1])
    except Exception:
        rh, rw = bgr.shape[:2]
    if rh <= 0 or rw <= 0:
        rh, rw = bgr.shape[:2]

    oh, ow = bgr.shape[:2]
    sx = ow / float(rw)
    sy = oh / float(rh)
    out_dir = run_dir / "_query_faces"
    out_dir.mkdir(exist_ok=True)

    for i, loc in enumerate(face_locations):
        if not isinstance(loc, (list, tuple)) or len(loc) != 4:
            continue
        top, right, bottom, left = [int(v) for v in loc]
        top = max(0, min(oh, int(top * sy)))
        bottom = max(0, min(oh, int(bottom * sy)))
        left = max(0, min(ow, int(left * sx)))
        right = max(0, min(ow, int(right * sx)))
        if bottom <= top or right <= left:
            continue

        h, w = bottom - top, right - left
        pad_h, pad_w = int(h * 0.2), int(w * 0.2)
        t = max(0, top - pad_h)
        b = min(oh, bottom + pad_h)
        l = max(0, left - pad_w)
        r = min(ow, right + pad_w)
        crop = bgr[t:b, l:r]
        if crop.size == 0:
            continue
        out_path = out_dir / f"{Path(name).stem}_qface{i}.jpg"
        cv2.imwrite(str(out_path), crop)

def _apply_duplicate_face_bonus(matches: List[Dict], raw_scores: List[float], sim_threshold: float,
                                bonus_step: float = 0.15):
    effective_scores = list(raw_scores)
    grouped: Dict[str, List[int]] = {}

    for i, m in enumerate(matches):
        key = str(m.get("best") or "").strip()
        if not key:
            continue
        grouped.setdefault(key, []).append(i)

    for key, indices in grouped.items():
        group_size = len(indices)
        if group_size <= 1:
            continue
        best_idx = max(indices, key=lambda idx: raw_scores[idx] if idx < len(raw_scores) else 0.0)
        best_raw = raw_scores[best_idx] if best_idx < len(raw_scores) else 0.0
        bonus = bonus_step * (group_size - 1)
        boosted = min(1.0, best_raw + bonus)
        effective_scores[best_idx] = boosted

        for idx in indices:
            matches[idx]["dup_group_size"] = group_size
        matches[best_idx]["dup_bonus"] = round(bonus, 4)
        matches[best_idx]["boosted_from"] = round(best_raw, 4)
        matches[best_idx]["effective_score"] = round(boosted, 4)
        matches[best_idx]["dup_group_key"] = key

    recognized = 0
    for i, m in enumerate(matches):
        sim = effective_scores[i] if i < len(effective_scores) else 0.0
        m["known"] = sim >= sim_threshold
        if m["known"]:
            recognized += 1

    return recognized, effective_scores

def _process_image(name: str, data: bytes, run_dir: Path, db, edge_db,
                   conf: float, sim_threshold: float, use_edge: bool) -> ProcessResult:
    t0 = time.perf_counter()
    result = ProcessResult(name=name, yolo_conf=conf, sim_threshold=sim_threshold)
    
    try:
        bgr = _decode(data)
        if bgr is None:
            result.error = "Decode failed"
            return result
        
        (run_dir / name).write_bytes(data)
        
        thumb = _make_thumb(bgr)
        if thumb:
            thumb_dir = run_dir / "_thumbs"
            thumb_dir.mkdir(exist_ok=True)
            (thumb_dir / f"{Path(name).stem}.jpg").write_bytes(thumb)
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        if use_edge and edge_db:
            from family_photo_detector_yolo import classify_image_from_array_three_tier
            res = classify_image_from_array_three_tier(rgb, db, edge_db, decision="any_known", conf_threshold=conf)
        else:
            res = classify_image_from_array_two_tier(rgb, db, decision="any_known", conf_threshold=conf)
        
        result.faces = res.faces
        result.matches = res.matches
        result.raw_match_scores = list(res.match_scores)
        result.face_locations = list(getattr(res, "face_locations", []) or [])
        result.resized_shape = list(getattr(res, "resized_shape", []) or [])
        _save_query_face_thumbs(name, bgr, run_dir, result.face_locations, result.resized_shape)
        recognized, boosted_scores = _apply_duplicate_face_bonus(
            result.matches, result.raw_match_scores, sim_threshold, bonus_step=0.15
        )
        result.match_scores = boosted_scores
        
        result.recognized = recognized
        result.is_family = recognized > 0
        
        bucket = "family" if result.is_family else "non_family"
        bucket_dir = run_dir / bucket
        bucket_dir.mkdir(exist_ok=True)
        shutil.copy2(str(run_dir / name), str(bucket_dir / name))
        
    except Exception as e:
        result.error = str(e)
    
    result.time_ms = (time.perf_counter() - t0) * 1000
    return result

# ---- Batch Job ----
@dataclass
class BatchJob:
    job_id: str
    run_dir: Path
    conf: float = DEFAULT_CONF
    sim_threshold: float = 0.55
    use_edge: bool = False
    mode: str = "local"
    model_name: str = "local"
    trained_faces_root: str = str(TRAINED_FACES_DIR)
    db: Any = None
    edge_db: Any = None
    total: int = 0
    done: int = 0
    yes: int = 0
    no: int = 0
    finished: bool = False
    started_at: float = 0
    finished_at: float = 0
    pending: "queue.Queue" = field(default_factory=queue.Queue)
    events: "queue.Queue" = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: Dict[str, ProcessResult] = field(default_factory=dict)
    export_zip_path: Optional[str] = None

@dataclass
class CloudTrainJob:
    train_job_id: str
    model_name: str
    temp_dir: Path
    created_at: float = 0
    started_at: float = 0
    done: bool = False
    ok: bool = False
    message: str = "queued"
    error: Optional[str] = None

@dataclass
class FaceExtractSession:
    session_id: str
    temp_dir: Path
    created_at: float = 0
    faces: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    person_ids: List[str] = field(default_factory=list)
    include_unidentified: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

JOBS: Dict[str, BatchJob] = {}
CLOUD_TRAIN_JOBS: Dict[str, CloudTrainJob] = {}
EXTRACT_SESSIONS: Dict[str, FaceExtractSession] = {}
PWA_ICON_CACHE: Dict[int, bytes] = {}

def _safe_rmtree(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _safe_unlink(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

def _cleanup_legacy_tmp_dirs():
    current_upload = UPLOAD_DIR.resolve()
    for legacy in LEGACY_UPLOAD_DIRS:
        try:
            legacy_resolved = legacy.resolve()
        except Exception:
            legacy_resolved = legacy
        if legacy_resolved == current_upload:
            continue
        if not legacy.exists():
            continue
        _safe_rmtree(legacy)

def _cleanup_job_runtime(job_id: str, drop_job: bool = False):
    job = JOBS.get(job_id)
    run_dir = job.run_dir if job else (UPLOAD_DIR / job_id)
    _safe_rmtree(run_dir)
    if drop_job:
        JOBS.pop(job_id, None)

def _cleanup_stale_runtime_data():
    now = time.time()

    # Clean finished batch run data after short retention.
    stale_jobs = []
    for job_id, job in list(JOBS.items()):
        if not job.finished:
            continue
        finished_at = job.finished_at or job.started_at or now
        if now - finished_at >= RUN_CACHE_TTL_S:
            _safe_rmtree(job.run_dir)
            stale_jobs.append(job_id)
    for job_id in stale_jobs:
        JOBS.pop(job_id, None)

    # Clean orphan run dirs (defensive; handles interrupted sessions).
    if UPLOAD_DIR.exists():
        for d in UPLOAD_DIR.iterdir():
            if not d.is_dir():
                continue
            if d.name in ("_cloud_train", "_extract_sessions"):
                continue
            if d.name.startswith("job_"):
                try:
                    age = now - d.stat().st_mtime
                except Exception:
                    age = RUN_CACHE_TTL_S + 1
                if age >= RUN_CACHE_TTL_S:
                    _safe_rmtree(d)

    # Clean stale extract sessions.
    stale_sessions = []
    for sid, sess in list(EXTRACT_SESSIONS.items()):
        age = now - (sess.created_at or now)
        if age >= RUN_CACHE_TTL_S:
            _safe_rmtree(sess.temp_dir)
            stale_sessions.append(sid)
    for sid in stale_sessions:
        EXTRACT_SESSIONS.pop(sid, None)

    # Drop old finished train job metadata.
    stale_train_jobs = []
    for tid, tjob in list(CLOUD_TRAIN_JOBS.items()):
        if not tjob.done:
            continue
        age = now - (tjob.started_at or tjob.created_at or now)
        if age >= RUN_CACHE_TTL_S:
            stale_train_jobs.append(tid)
    for tid in stale_train_jobs:
        CLOUD_TRAIN_JOBS.pop(tid, None)

    # Keep result zips bounded by age + max count.
    zip_files = []
    for z in RESULTS_DIR.glob("results_*.zip"):
        try:
            mtime = z.stat().st_mtime
        except Exception:
            mtime = 0
        zip_files.append((mtime, z))
    zip_files.sort(key=lambda x: x[0], reverse=True)
    for idx, (mtime, z) in enumerate(zip_files):
        too_old = (now - mtime) >= RESULTS_RETENTION_S
        over_limit = idx >= RESULTS_KEEP_MAX
        if too_old or over_limit:
            _safe_unlink(z)

    _cleanup_legacy_tmp_dirs()

_cleanup_stale_runtime_data()

def _send(job, event, data):
    job.events.put(f"event: {event}\ndata: {json.dumps(data)}\n\n")

def _batch_worker(job: BatchJob):
    db = job.db if job.db is not None else _get_db()
    edge_db = job.edge_db if job.edge_db is not None else (_get_edge_db() if job.use_edge else None)
    executor = ThreadPoolExecutor(max_workers=PROCESS_WORKERS)
    futures = {}
    
    def submit(item):
        name, data = item
        futures[executor.submit(_process_image, name, data, job.run_dir, db, edge_db,
                                job.conf, job.sim_threshold, job.use_edge)] = name
    
    def handle_done():
        for fut in [f for f in futures if f.done()]:
            name = futures.pop(fut)
            try:
                r = fut.result()
                with job.lock:
                    job.done += 1
                    if r.is_family: job.yes += 1
                    else: job.no += 1
                    job.results[name] = r
                
                _send(job, "item", {
                    "name": name, 
                    "yes": r.is_family, 
                    "faces": r.faces,
                    "recognized": r.recognized, 
                    "time_ms": round(r.time_ms, 1),
                    "done": job.done, 
                    "total": job.total,
                    "yes_count": job.yes, 
                    "no_count": job.no,
                    "match_score": r.match_scores[0] if r.match_scores else 0,
                })
            except Exception as e:
                with job.lock:
                    job.done += 1
                    job.no += 1
    
    upload_done = False
    while True:
        try:
            while True:
                item = job.pending.get_nowait()
                if item is None:
                    upload_done = True
                else:
                    submit(item)
        except queue.Empty:
            pass
        handle_done()
        if upload_done and not futures:
            break
        time.sleep(0.01)
    
    executor.shutdown(wait=True)
    
    dl = ""
    try:
        zip_path = job.run_dir / "results.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("family", "non_family"):
                folder = job.run_dir / sub
                if folder.exists():
                    for p in folder.rglob("*"):
                        if p.is_file():
                            z.write(str(p), str(p.relative_to(job.run_dir)))
        export_name = f"results_{job.job_id}.zip"
        export_zip_path = RESULTS_DIR / export_name
        shutil.copy2(str(zip_path), str(export_zip_path))
        zip_path.unlink(missing_ok=True)
        with job.lock:
            job.export_zip_path = str(export_zip_path)
        dl = f"/download/{job.job_id}"
    except: pass
    
    with job.lock:
        job.finished = True
        job.finished_at = time.time()

    _cleanup_stale_runtime_data()
    
    _send(job, "done", {
        "done": job.done, "total": job.total,
        "yes_count": job.yes, "no_count": job.no,
        "elapsed_s": round(time.time() - job.started_at, 2),
        "download_url": dl,
        "saved_zip": job.export_zip_path or "",
        "mode": job.mode,
        "model_name": job.model_name,
    })

def _cloud_train_worker(train_job: CloudTrainJob):
    train_job.started_at = time.time()
    train_job.message = "training"
    try:
        safe_model, model_dir, db_path, faces_dir = _cloud_model_paths(train_job.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        train_auto(
            conf_threshold=DEFAULT_CONF,
            save_faces=True,
            train_family_dir=train_job.temp_dir,
            db_path=db_path,
            trained_faces_dir=faces_dir,
        )

        with DB_LOCK:
            CLOUD_DB_CACHE.pop(safe_model, None)

        train_job.ok = True
        train_job.message = "done"
    except Exception as e:
        train_job.ok = False
        train_job.error = str(e)
        train_job.message = "failed"
    finally:
        train_job.done = True
        try:
            if train_job.temp_dir.exists():
                shutil.rmtree(train_job.temp_dir)
        except Exception:
            pass

def _resize_for_extract(rgb, max_side: int = 1280):
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    s = max_side / float(m)
    return cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def _extract_groups_payload(session: FaceExtractSession):
    groups = []
    person_label_map = {pid: f"Person {idx+1}" for idx, pid in enumerate(session.person_ids)}
    all_ids = session.person_ids[:]
    if session.include_unidentified or any(f.get("person_id") == "unidentified" for f in session.faces.values()):
        all_ids.append("unidentified")
    all_ids.append("reject")
    for pid in all_ids:
        if pid == "reject":
            label = "Not a Face"
        elif pid == "unidentified":
            label = "Unidentified Person"
        else:
            label = person_label_map.get(pid, pid)
        items = []
        for face in session.faces.values():
            if face.get("person_id") != pid:
                continue
            items.append({
                "face_id": face["face_id"],
                "thumb_url": f"/cloud/extract_thumb/{session.session_id}/{face['face_id']}",
                "source_image": face.get("source_image", ""),
                "conf": round(float(face.get("conf", 0.0)) * 100),
            })
        items.sort(key=lambda x: x["face_id"])
        groups.append({
            "person_id": pid,
            "label": label,
            "is_reject": pid == "reject",
            "is_unidentified": pid == "unidentified",
            "count": len(items),
            "faces": items,
        })
    return {"session_id": session.session_id, "groups": groups}

def _assign_person_groups(encodings: List[np.ndarray], person_limit: Optional[int]):
    if not encodings:
        return [], []
    X = np.array(encodings, dtype=np.float32)
    if len(X) == 1:
        labels = np.array([0], dtype=np.int32)
    else:
        labels = DBSCAN(eps=0.42, min_samples=1, metric="euclidean", n_jobs=1).fit(X).labels_

    counts: Dict[int, int] = {}
    for lb in labels.tolist():
        counts[int(lb)] = counts.get(int(lb), 0) + 1
    label_order = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    if not label_order:
        label_order = [0]

    if person_limit is None or person_limit <= 0:
        person_count = len(label_order)
    else:
        person_count = max(1, min(int(person_limit), 50))

    person_ids = [f"person_{i+1:02d}" for i in range(person_count)]
    selected_labels = label_order[:max(1, min(person_count, len(label_order)))]
    label_to_person = {}
    for i, lb in enumerate(selected_labels):
        label_to_person[lb] = person_ids[i]

    if len(selected_labels) < person_count:
        # Keep extra empty rows if user requested more persons than extracted clusters.
        pass

    has_unidentified = len(label_order) > len(selected_labels)
    assigned_person_ids = []
    for idx, lb in enumerate(labels.tolist()):
        if lb in label_to_person:
            assigned_person_ids.append(label_to_person[lb])
            continue
        assigned_person_ids.append("unidentified")

    return person_ids, assigned_person_ids, has_unidentified

def _next_person_label(existing_labels: List[str], start_idx: int = 1) -> str:
    used = set(existing_labels)
    idx = max(1, int(start_idx))
    while True:
        label = f"PERSON_{idx:02d}"
        if label not in used:
            return label
        idx += 1

def _label_seed_from_existing(existing_labels: List[str]) -> int:
    max_idx = 0
    for label in existing_labels:
        m = re.fullmatch(r"PERSON_(\d+)", str(label).strip())
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

def _copy_face_file_unique(src: Path, dst_dir: Path, stem_prefix: str = "") -> Optional[Path]:
    if not src.exists() or not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower() or ".jpg"
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", src.stem).strip("_")[:80] or "face"
    if stem_prefix:
        pref = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem_prefix).strip("_")[:40]
        stem = f"{pref}_{stem}" if pref else stem
    dst = dst_dir / f"{stem}{ext}"
    i = 1
    while dst.exists():
        dst = dst_dir / f"{stem}_{i:03d}{ext}"
        i += 1
    shutil.copy2(str(src), str(dst))
    return dst

def _build_cloud_db_from_extract(session: FaceExtractSession, model_name: str, base_model: str = ""):
    safe_model, model_dir, db_path, faces_dir = _cloud_model_paths(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    build_faces_dir = model_dir / f"_faces_build_{os.urandom(4).hex()}"
    _safe_rmtree(build_faces_dir)
    build_faces_dir.mkdir(parents=True, exist_ok=True)

    base_safe = ""
    base_names: List[str] = []
    label_order: List[str] = []
    label_encodings: Dict[str, List[np.ndarray]] = {}
    label_sources: Dict[str, set] = {}

    def ensure_label(label: str):
        key = str(label).strip()
        if not key:
            return None
        if key not in label_order:
            label_order.append(key)
        label_encodings.setdefault(key, [])
        label_sources.setdefault(key, set())
        return key

    try:
        if base_model:
            base_safe, _, base_db_path, base_faces_dir = _cloud_model_paths(base_model)
            if not base_db_path.exists():
                raise RuntimeError(f"Base model not found: {base_model}")

            base_db = _get_cloud_db(base_safe)
            base_names = [str(x) for x in (base_db.get("names") or []) if str(x).strip()]
            base_train_faces = base_db.get("train_faces") or {}

            for label in base_names:
                key = ensure_label(label)
                if not key:
                    continue

                train_cluster = base_train_faces.get(key) or {}
                encs = train_cluster.get("encodings")
                if encs is not None:
                    for enc in np.array(encs, dtype=np.float32):
                        if getattr(enc, "size", 0) > 0:
                            label_encodings[key].append(np.array(enc, dtype=np.float32))

                src_dir = Path(base_faces_dir) / key
                if src_dir.exists() and src_dir.is_dir():
                    dst_dir = build_faces_dir / key
                    for p in sorted(src_dir.iterdir()):
                        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                            continue
                        copied = _copy_face_file_unique(p, dst_dir, "base")
                        if copied:
                            label_sources[key].add(copied.stem)

        seed = _label_seed_from_existing(label_order)
        person_to_label: Dict[str, str] = {}

        for idx, pid in enumerate(session.person_ids):
            if idx < len(base_names):
                mapped = ensure_label(base_names[idx])
                if mapped:
                    person_to_label[pid] = mapped
                continue
            new_label = _next_person_label(label_order, seed)
            m = re.fullmatch(r"PERSON_(\d+)", new_label)
            if m:
                seed = max(seed, int(m.group(1)) + 1)
            mapped = ensure_label(new_label)
            if mapped:
                person_to_label[pid] = mapped

        for pid in session.person_ids:
            label = person_to_label.get(pid)
            if not label:
                continue
            members = [f for f in session.faces.values() if f.get("person_id") == pid]
            if not members:
                continue

            folder = build_faces_dir / label
            folder.mkdir(parents=True, exist_ok=True)

            for i, m in enumerate(members, start=1):
                enc = m.get("encoding")
                if enc is not None and getattr(enc, "size", 0) > 0:
                    label_encodings[label].append(np.array(enc, dtype=np.float32))

                src = Path(m["thumb_path"])
                source_stem = Path(m.get("source_image", "img")).stem
                copied = _copy_face_file_unique(src, folder, f"new_{source_stem}_{i:03d}")
                if copied:
                    label_sources[label].add(m.get("source_image", copied.stem))

        centroids, thresholds, names, sizes, unique_images = [], [], [], [], []
        for label in label_order:
            enc_list = label_encodings.get(label) or []
            if not enc_list:
                continue
            encs = np.array(enc_list, dtype=np.float32)
            centroid = encs.mean(axis=0)
            if len(encs) > 1:
                dists = face_recognition.face_distance(encs, centroid)
                thr = min(float(np.quantile(dists, 0.90) + 0.02), 0.50)
            else:
                thr = 0.50

            centroids.append(centroid)
            thresholds.append(thr)
            names.append(label)
            sizes.append(int(len(encs)))
            unique_images.append(int(len(label_sources.get(label) or [])))

        if not centroids:
            raise RuntimeError("No valid person groups to train. Move at least one face into a person column.")

        if faces_dir.exists():
            shutil.rmtree(faces_dir)
        build_faces_dir.rename(faces_dir)

        with open(db_path, "wb") as f:
            pickle_obj = {
                "centroids": np.array(centroids, dtype=np.float32),
                "thresholds": np.array(thresholds, dtype=np.float32),
                "names": names,
                "sizes": sizes,
                "unique_images": unique_images,
                "meta": {
                    "detector": "yolov8",
                    "source": "cloud_extract",
                    "continued_from": base_safe or "",
                    "trained_faces_dir": str(faces_dir.resolve()),
                    "total_faces_encoded": int(sum(sizes)),
                },
            }
            import pickle
            pickle.dump(pickle_obj, f)

        with DB_LOCK:
            CLOUD_DB_CACHE.pop(safe_model, None)
            if base_safe:
                CLOUD_DB_CACHE.pop(base_safe, None)

        return safe_model, base_safe
    except Exception:
        _safe_rmtree(build_faces_dir)
        raise

def _build_pwa_icon(size: int) -> bytes:
    if size in PWA_ICON_CACHE:
        return PWA_ICON_CACHE[size]
    size = int(size)
    if size <= 0:
        size = 192
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (203, 107, 11)  # BGR
    cv2.rectangle(img, (int(size * 0.08), int(size * 0.08)), (int(size * 0.92), int(size * 0.92)), (255, 255, 255), thickness=max(2, size // 32))
    cv2.circle(img, (int(size * 0.5), int(size * 0.42)), int(size * 0.18), (255, 255, 255), thickness=max(2, size // 30))
    cv2.circle(img, (int(size * 0.44), int(size * 0.39)), int(size * 0.02), (203, 107, 11), thickness=-1)
    cv2.circle(img, (int(size * 0.56), int(size * 0.39)), int(size * 0.02), (203, 107, 11), thickness=-1)
    cv2.ellipse(img, (int(size * 0.5), int(size * 0.48)), (int(size * 0.07), int(size * 0.045)), 0, 0, 180, (203, 107, 11), thickness=max(2, size // 40))
    cv2.putText(img, "FPO", (int(size * 0.22), int(size * 0.78)), cv2.FONT_HERSHEY_SIMPLEX,
                size / 210.0, (255, 255, 255), thickness=max(2, size // 28), lineType=cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    data = buf.tobytes() if ok else b""
    PWA_ICON_CACHE[size] = data
    return data

# ---- HTML ----
HTML = '''<!doctype html>
<html>
<head>
  <meta charset="utf-8"><title>Family Photo - YOLO</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="theme-color" content="#0b6bcb">
  <link rel="manifest" href="/manifest.webmanifest">
  <link rel="apple-touch-icon" href="/pwa-icon-192.png">
  <style>
    *{box-sizing:border-box}
    body{font-family:system-ui,sans-serif;margin:0;padding:16px;padding-bottom:160px;background:#f5f5f7}
    .card{background:#fff;border-radius:16px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}
    h2{margin:0 0 8px;font-size:22px}
    .btn{padding:10px 18px;border:none;border-radius:10px;background:#e5e5ea;cursor:pointer;font-weight:500}
    .btn:hover{background:#d1d1d6}
    .btn-primary{background:#007aff;color:#fff}
    .btn-primary:hover{background:#0066d6}
    .btn-orange{background:#f59e0b;color:#fff}
    input,select{padding:10px 14px;border-radius:10px;border:1px solid #d1d1d6}
    .muted{color:#86868b;font-size:13px}
    .badge{padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;background:#e8daff;color:#6b21a8}
    .stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
    .stat{padding:10px 16px;border-radius:12px;font-size:14px;font-weight:600;background:#f5f5f7}
    .stat.yes{background:#d1f2d9;color:#1d7d3f}
    .stat.no{background:#ffd9d9;color:#c0392b}
    .linkbtn{padding:10px 18px;border-radius:10px;background:#007aff;color:#fff;font-weight:600;text-decoration:none;display:inline-block}
    
    .slider-group{margin-bottom:8px}
    .slider-row{display:flex;align-items:center;gap:10px}
    .slider-row input[type=range]{width:160px}
    .slider-row .slider-val{min-width:50px;font-weight:600;font-size:14px}
    .slider-row .slider-label{font-size:13px;font-weight:500;min-width:85px}
    .slider-hint{font-size:11px;color:#999;margin-top:3px;margin-left:95px}
    .mode-switch{display:flex;gap:10px;align-items:center}
    .mode-switch label{display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px}
    .section-box{margin-top:10px;padding:12px;border:1px solid #d1d1d6;border-radius:12px;background:#fafafa}
    .section-box h4{margin:0 0 8px;font-size:13px}
    .cloud-status{font-size:12px;color:#666}
    .extract-board{margin-top:10px}
    .extract-layout{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:10px;align-items:start}
    .extract-left{display:flex;flex-direction:column;gap:8px;min-width:0}
    .extract-right{min-width:0}
    .person-row{background:#fff;border:1px solid #e5e5ea;border-radius:10px;padding:8px}
    .person-row.unidentified{border-color:#f7cc6f;background:#fffaf0}
    .person-col{min-width:220px;background:#fff;border:1px solid #e5e5ea;border-radius:10px;padding:8px}
    .person-row.reject,
    .person-col.reject{border-color:#ffb4b4;background:#fff7f7}
    .person-row h5,
    .person-col h5{margin:0 0 8px;font-size:12px}
    .person-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(52px,1fr));gap:6px;min-height:72px}
    .extract-face{border:1px solid #e5e5ea;border-radius:8px;padding:3px;background:#fafafa;cursor:grab}
    .extract-face img{width:100%;height:48px;object-fit:cover;border-radius:6px;background:#e5e5ea;display:block}
    .extract-face .meta{font-size:9px;color:#666;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
    .face-menu{position:fixed;z-index:9999;background:#fff;border:1px solid #d1d1d6;border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,0.18);min-width:140px;padding:4px}
    .face-menu button{width:100%;text-align:left;background:#fff;border:none;border-radius:6px;padding:7px 8px;font-size:12px;cursor:pointer}
    .face-menu button:hover{background:#f3f4f6}
    @media (max-width: 900px){
      .extract-layout{grid-template-columns:1fr}
    }
    
    .main-content{display:flex;gap:16px}
    .grid-container{flex:1;min-width:0}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(115px,1fr));gap:10px;margin-top:16px}
    .tile{border-radius:12px;padding:8px;background:#f5f5f7;border:2px solid transparent;position:relative;cursor:pointer;transition:all 0.15s}
    .tile:hover{transform:scale(1.02)}
    .tile.yes{background:rgba(34,197,94,0.25);border-color:rgba(34,197,94,0.5)}
    .tile.no{background:rgba(239,68,68,0.25);border-color:rgba(239,68,68,0.5)}
    .tile.selected{border-color:#f59e0b;box-shadow:0 0 0 3px rgba(245,158,11,0.3)}
    .tile.active{border-color:#007aff;box-shadow:0 0 0 3px rgba(0,122,255,0.3)}
    .tile img{width:100%;height:80px;object-fit:cover;border-radius:8px;background:#e5e5ea}
    .tile-name{margin-top:5px;font-size:9px;color:#86868b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .tile-info{position:absolute;top:12px;right:12px;font-size:9px;padding:2px 6px;border-radius:4px;background:rgba(0,0,0,0.7);color:#fff}
    
    .filter-btns{display:flex;gap:8px;margin-bottom:12px}
    .filter-btn{padding:6px 14px;border-radius:8px;border:1px solid #d1d1d6;background:#fff;cursor:pointer;font-size:12px}
    .filter-btn.active{background:#007aff;color:#fff;border-color:#007aff}
    
    .side-panel{width:420px;min-width:420px;background:#fff;border-radius:16px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);display:none;max-height:calc(100vh - 180px);overflow-y:auto;position:sticky;top:16px}
    .side-panel.active{display:block}
    .side-panel h3{margin:0 0 16px;font-size:18px;display:flex;align-items:center;justify-content:space-between}
    .side-panel .close-btn{background:none;border:none;font-size:24px;cursor:pointer;color:#86868b;padding:0;line-height:1}
    .side-panel .close-btn:hover{color:#333}
    .preview-img{width:100%;max-height:260px;object-fit:contain;border-radius:12px;margin-bottom:16px;background:#f5f5f7}
    .result-badge{display:inline-block;padding:8px 16px;border-radius:8px;font-weight:600;font-size:15px;margin-bottom:12px}
    .result-badge.yes{background:#d1f2d9;color:#1d7d3f}
    .result-badge.no{background:#ffd9d9;color:#c0392b}
    
    .section-title{font-size:12px;font-weight:600;color:#666;margin:14px 0 8px;padding-top:10px;border-top:1px solid #eee;text-transform:uppercase;letter-spacing:0.5px}
    
    .step-box{background:#f9f9fb;border-radius:8px;padding:10px 12px;margin:6px 0;font-size:12px}
    .step-box.pass{border-left:3px solid #34c759}
    .step-box.fail{border-left:3px solid #ff3b30}
    .step-box .step-title{font-weight:600;margin-bottom:3px}
    .step-box .step-detail{color:#666;line-height:1.4}
    
    .face-item{display:flex;gap:12px;align-items:flex-start;padding:10px;background:#f9f9fb;border-radius:10px;margin-bottom:8px}
    .face-pair{display:flex;align-items:center;gap:6px;flex-shrink:0}
    .face-arrow{font-size:14px;color:#666}
    .face-thumb{width:50px;height:50px;border-radius:8px;object-fit:cover;background:#e5e5ea;flex-shrink:0}
    .face-info{flex:1;min-width:0}
    .face-info h4{margin:0 0 3px;font-size:13px}
    .face-info p{margin:0;font-size:11px;color:#666}
    .face-info .detail{font-size:10px;color:#999;margin-top:3px}
    .match-bar{height:6px;background:#e5e5ea;border-radius:3px;margin-top:5px;overflow:visible;position:relative}
    .match-bar-fill{height:100%;border-radius:3px}
    .match-bar-fill.high{background:#34c759}
    .match-bar-fill.med{background:#ff9500}
    .match-bar-fill.low{background:#ff3b30}
    .match-bar .threshold-line{position:absolute;top:-2px;bottom:-2px;width:2px;background:#333;border-radius:1px}
    
    .explain-box{background:#f0f7ff;border:1px solid #c7deff;border-radius:10px;padding:12px;margin-top:12px;font-size:12px;line-height:1.5}
    .explain-box.error{background:#fff5f5;border-color:#ffcdd2}
    .explain-box strong{color:#0056b3}
    .explain-box.error strong{color:#c0392b}
    .explain-box code{font-family:monospace;background:#e8f4ff;padding:1px 5px;border-radius:3px;font-size:11px}
    .explain-box.error code{background:#ffebee}
    .explain-box .tip{margin-top:8px;padding:8px;background:rgba(255,255,255,0.7);border-radius:6px;font-size:11px}
    .explain-box .verdict{margin-top:8px;padding:8px 10px;border-radius:6px;font-weight:500}
    .explain-box .verdict.yes{background:#d1f2d9;color:#1d7d3f}
    .explain-box .verdict.no{background:#ffd9d9;color:#c0392b}
    
    .console{position:fixed;left:0;right:0;bottom:0;height:130px;background:#1c1c1e;color:#f5f5f7;font-family:monospace;font-size:11px;padding:10px;overflow:auto}
    .console pre{margin:0;white-space:pre-wrap}
    .log-ok{color:#30d158}.log-err{color:#ff453a}
  </style>
</head>
<body>
<div class="card">
  <h2>Family Photo Organizer <span class="badge">YOLOv8</span></h2>
  <div class="muted">YOLO Face Detection • {{ workers }} workers • Mode: any_known</div>
  <form id="form">
    <div class="row">
      <div class="mode-switch">
        <label><input type="radio" name="runMode" value="local"> Local</label>
        <label><input type="radio" name="runMode" value="cloud" checked> Cloud</label>
      </div>
      <span class="muted" id="modeNote">Local mode: uses current local DB and optional edge model.</span>
    </div>
    <div class="section-box" id="cloudTrainBox" style="display:none">
      <h4>Cloud Model Training</h4>
      <div class="row">
        <select id="continueModel" title="Continue existing model">
          <option value="">Train as new model</option>
        </select>
        <input id="cloudModelName" placeholder="Model name (example: family_jan_2026)">
        <input id="personCount" type="number" min="1" max="20" value="3" title="Number of persons">
        <button type="button" class="btn" id="browseTrainBtn">Browse Train Folder…</button>
        <input type="file" id="trainPicker" style="display:none" webkitdirectory multiple>
        <button type="button" class="btn" id="extractFacesBtn">Extract Faces</button>
        <button type="button" class="btn" id="addPersonBtn">Add Person</button>
        <button type="button" class="btn btn-primary" id="trainModelBtn">Train Model</button>
        <button type="button" class="btn" id="resetCloudBtn">Reset + Cleanup</button>
      </div>
      <div class="row" style="margin-top:6px">
        <span class="muted" id="trainPickedLabel">No training folder selected</span>
      </div>
      <div class="row" style="margin-top:6px">
        <span class="cloud-status" id="cloudTrainStatus">Idle</span>
      </div>
      <div id="extractBoard" class="extract-board"></div>
      <div id="faceActionMenu" class="face-menu" style="display:none"></div>
    </div>
    <div class="section-box" id="detectBox">
      <h4>Use Trained Model to Detect Family Photos</h4>
      <div class="row" style="flex-direction:column;align-items:flex-start;gap:8px;margin-top:0">
        <div class="slider-group">
          <div class="slider-row">
            <span class="slider-label">YOLO Conf:</span>
            <input type="range" id="conf" min="5" max="95" value="20">
            <span class="slider-val" id="confVal">20%</span>
          </div>
          <div class="slider-hint">Minimum confidence to detect a face. Lower = detect more faces, may include false positives.</div>
        </div>
        <div class="slider-group">
          <div class="slider-row">
            <span class="slider-label">Similarity:</span>
            <input type="range" id="simThreshold" min="30" max="85" value="55">
            <span class="slider-val" id="simVal">55%</span>
          </div>
          <div class="slider-hint">Minimum similarity to count as family match. Higher = stricter, may miss some family.</div>
        </div>
      </div>
      <div class="row">
        <label id="useEdgeWrap" style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px">
          <input type="checkbox" id="useEdge"> Use Edge Model
        </label>
        <span id="cloudModelWrap" style="display:none">
          <select id="cloudModel"></select>
        </span>
        <button type="button" class="btn" id="refreshModelsBtn" style="display:none">Refresh Models</button>
        <button type="button" class="btn" id="browseBtn">Browse Folder…</button>
        <input type="file" id="picker" style="display:none" webkitdirectory multiple>
        <button type="submit" class="btn btn-primary">Run</button>
        <span class="muted" id="pickedLabel">No selection</span>
        <a id="downloadLink" class="linkbtn" href="#" style="display:none">Download ZIP</a>
      </div>
    </div>
  </form>
  <div class="stats" id="stats" style="display:none">
    <span class="stat" id="statProg">0/0</span>
    <span class="stat yes" id="statYes">YES: 0</span>
    <span class="stat no" id="statNo">NO: 0</span>
    <span class="stat" id="statSpeed">-</span>
  </div>
</div>

<div class="card" id="resultCard" style="display:none">
  <div class="filter-btns">
    <button class="filter-btn active" data-f="all">All</button>
    <button class="filter-btn" data-f="yes">✓ Family</button>
    <button class="filter-btn" data-f="no">✗ Not Family</button>
  </div>
  <div class="row" id="rescueRow" style="display:none">
    <span class="stat" id="statSel">Selected: 0</span>
    <button class="btn btn-orange" id="moveBtn">Move to Family</button>
    <button class="btn" id="clearBtn">Clear</button>
  </div>
  
  <div class="main-content">
    <div class="grid-container">
      <div class="grid" id="grid"></div>
    </div>
    
    <div class="side-panel" id="sidePanel">
      <h3>
        <span id="panelTitle">Details</span>
        <button class="close-btn" id="closePanel">&times;</button>
      </h3>
      <img class="preview-img" id="previewImg" src="">
      <div id="resultBadge" class="result-badge">-</div>
      
      <div class="section-title">Processing Steps</div>
      <div id="stepsList"></div>
      
      <div class="section-title">Detected Faces</div>
      <div id="faceList"></div>
      
      <div class="explain-box" id="explainBox"></div>
    </div>
  </div>
</div>

<div class="console"><pre id="log"></pre></div>

<script>
const $=s=>document.getElementById(s);
let jobId=null, es=null, selected=new Set(), filter='all', activeTile=null;
let currentConf=0.20, currentSimThreshold=0.55;
let currentMode='cloud', cloudTrainPoll=null;
let extractSessionId='', draggingFaceId='';
let extractGroupsState=[];

$('conf').oninput=e=>{currentConf=e.target.value/100;$('confVal').textContent=e.target.value+'%'};
$('simThreshold').oninput=e=>{currentSimThreshold=e.target.value/100;$('simVal').textContent=e.target.value+'%'};
$('browseBtn').onclick=()=>$('picker').click();
$('picker').onchange=e=>{$('pickedLabel').textContent=e.target.files.length?e.target.files.length+' files':'No selection';$('downloadLink').style.display='none'};
$('browseTrainBtn').onclick=()=>$('trainPicker').click();
$('trainPicker').onchange=e=>{
  $('trainPickedLabel').textContent=e.target.files.length?e.target.files.length+' files':'No training folder selected';
  extractSessionId='';
  $('extractBoard').innerHTML='';
};

function applyModeUI(){
  const cloud=currentMode==='cloud';
  $('useEdgeWrap').style.display=cloud?'none':'flex';
  $('cloudModelWrap').style.display=cloud?'inline-block':'none';
  $('refreshModelsBtn').style.display=cloud?'inline-block':'none';
  $('cloudTrainBox').style.display=cloud?'block':'none';
  $('modeNote').textContent=cloud
    ?'Cloud mode: choose a trained cloud model. Edge model is disabled.'
    :'Local mode: uses current local DB and optional edge model.';
  if(cloud){$('useEdge').checked=false}
  hideFaceActionMenu();
}

async function refreshCloudModels(preferred=''){
  try{
    const r=await fetch('/cloud/models');
    const d=await r.json();
    const models=(d.models||[]);

    const detectSel=$('cloudModel');
    detectSel.innerHTML='';
    if(!models.length){
      const op=document.createElement('option');
      op.value='';op.textContent='No cloud models yet';
      detectSel.appendChild(op);
    }else{
      models.forEach(name=>{
        const op=document.createElement('option');
        op.value=name;op.textContent=name;
        detectSel.appendChild(op);
      });
      if(preferred&&models.includes(preferred))detectSel.value=preferred;
    }

    const continueSel=$('continueModel');
    const keepContinue=continueSel.value;
    continueSel.innerHTML='';
    const newOpt=document.createElement('option');
    newOpt.value='';newOpt.textContent='Train as new model';
    continueSel.appendChild(newOpt);
    models.forEach(name=>{
      const op=document.createElement('option');
      op.value=name;op.textContent='Continue: '+name;
      continueSel.appendChild(op);
    });
    if(keepContinue&&models.includes(keepContinue)){
      continueSel.value=keepContinue;
    }
  }catch(e){
    log('Failed to load cloud models: '+e,false);
  }
}

document.querySelectorAll('input[name="runMode"]').forEach(r=>{
  r.onchange=async()=>{
    currentMode=r.value;
    applyModeUI();
    if(currentMode==='cloud')await refreshCloudModels();
  };
});

$('refreshModelsBtn').onclick=()=>refreshCloudModels($('cloudModel').value||'');

$('continueModel').onchange=()=>{
  const base=($('continueModel').value||'').trim();
  if(base&&!($('cloudModelName').value||'').trim()){
    $('cloudModelName').value=base;
  }
};

async function moveExtractFace(faceId,targetPerson){
  if(!faceId||!extractSessionId)return;
  const r=await fetch('/cloud/extract_move',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:extractSessionId,face_id:faceId,target_person:targetPerson})
  });
  const d=await r.json();
  if(!r.ok){$('cloudTrainStatus').textContent='Move failed: '+(d.error||'error');return}
  renderExtractBoard(d);
}

function hideFaceActionMenu(){
  const menu=$('faceActionMenu');
  menu.style.display='none';
  menu.innerHTML='';
}

function openFaceActionMenu(faceId,x,y){
  const menu=$('faceActionMenu');
  menu.innerHTML='';
  const groups=extractGroupsState||[];
  const opts=[{person_id:'reject',label:'No Face'}]
    .concat(groups.filter(g=>!g.is_reject).map(g=>({person_id:g.person_id,label:g.label})));
  opts.forEach(o=>{
    const btn=document.createElement('button');
    btn.type='button';
    btn.textContent=o.label;
    btn.onclick=async(ev)=>{
      ev.stopPropagation();
      hideFaceActionMenu();
      await moveExtractFace(faceId,o.person_id);
    };
    menu.appendChild(btn);
  });
  menu.style.left=Math.max(8,x)+'px';
  menu.style.top=Math.max(8,y)+'px';
  menu.style.display='block';
}

document.addEventListener('click',()=>hideFaceActionMenu());
document.addEventListener('keydown',e=>{if(e.key==='Escape')hideFaceActionMenu();});

function renderExtractBoard(payload){
  const board=$('extractBoard');
  board.innerHTML='';
  extractGroupsState=(payload&&payload.groups)?payload.groups:[];
  if(!payload||!payload.groups||!payload.groups.length){
    board.innerHTML='<div class="muted">No faces extracted yet.</div>';
    return;
  }
  const groups=payload.groups||[];
  const reject=groups.find(g=>g.is_reject)||{person_id:'reject',label:'Not a Face',faces:[],count:0,is_reject:true};
  const persons=groups.filter(g=>!g.is_reject);

  const makeGroupNode=(g,isRow)=>{
    const node=document.createElement('div');
    let cls=(isRow?'person-row':'person-col');
    if(g.is_reject)cls+=' reject';
    if(g.is_unidentified)cls+=' unidentified';
    node.className=cls;
    node.dataset.personId=g.person_id;
    node.innerHTML=`<h5>${g.label} (${g.count||0})</h5><div class="person-grid"></div>`;
    const grid=node.querySelector('.person-grid');
    grid.ondragover=e=>e.preventDefault();
    grid.ondrop=async e=>{
      e.preventDefault();
      if(!draggingFaceId||!extractSessionId)return;
      await moveExtractFace(draggingFaceId,g.person_id);
      draggingFaceId='';
    };
    (g.faces||[]).forEach(f=>{
      const item=document.createElement('div');
      item.className='extract-face';
      item.draggable=true;
      item.dataset.faceId=f.face_id;
      item.innerHTML=`<img src="${f.thumb_url}" onerror="this.style.opacity='0.3'"><div class="meta">${f.source_image}</div>`;
      item.ondragstart=()=>{draggingFaceId=f.face_id;};
      item.onclick=(ev)=>{
        ev.stopPropagation();
        openFaceActionMenu(f.face_id,ev.clientX,ev.clientY);
      };
      grid.appendChild(item);
    });
    return node;
  };

  const layout=document.createElement('div');
  layout.className='extract-layout';
  const left=document.createElement('div');
  left.className='extract-left';
  if(!persons.length){
    const empty=document.createElement('div');
    empty.className='muted';
    empty.textContent='No person rows yet. Click Add Person.';
    left.appendChild(empty);
  }else{
    persons.forEach(g=>left.appendChild(makeGroupNode(g,true)));
  }

  const right=document.createElement('div');
  right.className='extract-right';
  right.appendChild(makeGroupNode(reject,false));

  layout.appendChild(left);
  layout.appendChild(right);
  board.appendChild(layout);
}

$('extractFacesBtn').onclick=async()=>{
  const files=$('trainPicker').files;
  if(!files.length){alert('Select training folder');return}
  const personCountRaw=($('personCount').value||'').trim();
  const personCount=Number(personCountRaw);
  if(!personCountRaw||!Number.isInteger(personCount)||personCount<1||personCount>50){
    alert('Enter valid number of persons (1-50)');
    return;
  }
  const fd=new FormData();
  fd.append('person_count',String(personCount));
  for(const f of files)fd.append('files',f,f.name);
  $('cloudTrainStatus').textContent='Extracting faces...';
  try{
    const r=await fetch('/cloud/extract_faces',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok){$('cloudTrainStatus').textContent='Extract failed: '+(d.error||'error');return}
    extractSessionId=d.session_id;
    renderExtractBoard(d);
    $('cloudTrainStatus').textContent='Extracted with person limit '+personCount+'. Drag faces to adjust, move non-faces to Not a Face.';
  }catch(e){
    $('cloudTrainStatus').textContent='Extract request failed';
  }
};

$('addPersonBtn').onclick=async()=>{
  if(!extractSessionId){alert('Extract faces first');return}
  const r=await fetch('/cloud/extract_add_person',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:extractSessionId})
  });
  const d=await r.json();
  if(!r.ok){$('cloudTrainStatus').textContent='Add person failed: '+(d.error||'error');return}
  renderExtractBoard(d);
};

$('resetCloudBtn').onclick=async()=>{
  const ok=confirm('Reset models and cleanup all runtime data (cloud models, temp run files, extracted faces, and saved result zips)? This cannot be undone.');
  if(!ok)return;
  $('cloudTrainStatus').textContent='Resetting models and cleaning runtime data...';
  try{
    const r=await fetch('/cloud/reset_all',{method:'POST'});
    const d=await r.json();
    if(!r.ok){
      $('cloudTrainStatus').textContent='Reset failed: '+(d.error||'error');
      return;
    }
    extractSessionId='';
    $('cloudModelName').value='';
    $('trainPickedLabel').textContent='No training folder selected';
    $('extractBoard').innerHTML='';
    await refreshCloudModels();
    const removed=d.deleted_result_zips||0;
    $('cloudTrainStatus').textContent='Reset complete. Removed '+removed+' result zip(s) and cleaned temp data.';
    log('Reset + cleanup complete. Removed '+removed+' result zip(s).');
  }catch(e){
    $('cloudTrainStatus').textContent='Reset request failed';
  }
};

$('trainModelBtn').onclick=async()=>{
  const baseModel=($('continueModel').value||'').trim();
  const typedName=($('cloudModelName').value||'').trim();
  const modelName=typedName||baseModel;
  if(!modelName){alert('Enter model name or choose an existing model to continue');return}
  if(!extractSessionId){alert('Please click Extract Faces first');return}
  $('cloudTrainStatus').textContent=baseModel
    ?('Updating model "'+modelName+'" with new faces...')
    :'Training new model from extracted faces...';
  try{
    const r=await fetch('/cloud/extract_train',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({session_id:extractSessionId,model_name:modelName,base_model:baseModel})
    });
    const d=await r.json();
    if(!r.ok){$('cloudTrainStatus').textContent='Train failed: '+(d.error||'error');return}
    if(d.continued_from){
      $('cloudTrainStatus').textContent='Done: '+d.model_name+' (continued from '+d.continued_from+')';
    }else{
      $('cloudTrainStatus').textContent='Done: '+d.model_name;
    }
    await refreshCloudModels(d.model_name);
    $('cloudModel').value=d.model_name;
    log('Cloud training done: '+d.model_name+(d.continued_from?(' (continued from '+d.continued_from+')'):''));
  }catch(e){
    $('cloudTrainStatus').textContent='Train request failed';
  }
};

function log(m,ok=true){$('log').innerHTML+=`<span class="${ok?'log-ok':'log-err'}">[${new Date().toLocaleTimeString()}] ${m}</span>\\n`;$('log').parentElement.scrollTop=9999}

document.querySelectorAll('.filter-btn').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.filter-btn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');filter=b.dataset.f;
  document.querySelectorAll('.tile').forEach(t=>{t.style.display=(filter==='all'||(filter==='yes'&&t.classList.contains('yes'))||(filter==='no'&&t.classList.contains('no')))?'':'none'});
});

function toggleSel(tile,e){
  if(e.shiftKey&&tile.classList.contains('no')){
    const n=tile.dataset.name;
    if(selected.has(n)){selected.delete(n);tile.classList.remove('selected')}
    else{selected.add(n);tile.classList.add('selected')}
    $('statSel').textContent='Selected: '+selected.size;
    $('rescueRow').style.display=selected.size?'flex':'none';
  }
}

$('clearBtn').onclick=()=>{selected.clear();document.querySelectorAll('.tile.selected').forEach(t=>t.classList.remove('selected'));$('rescueRow').style.display='none'};

$('moveBtn').onclick=async()=>{
  if(!selected.size||!jobId)return;
  const names=[...selected];
  log('Moving '+names.length+' to family...');
  const r=await fetch('/move_to_family',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:jobId,names})});
  const d=await r.json();
  if(r.ok){
    log('Moved '+d.moved+' photos');
    names.forEach(n=>{const t=document.querySelector(`.tile[data-name="${n}"]`);if(t){t.classList.remove('no','selected');t.classList.add('yes')}});
    selected.clear();$('rescueRow').style.display='none';
    $('statYes').textContent='YES: '+document.querySelectorAll('.tile.yes').length;
    $('statNo').textContent='NO: '+document.querySelectorAll('.tile.no').length;
  }
};

$('closePanel').onclick=()=>{$('sidePanel').classList.remove('active');if(activeTile){activeTile.classList.remove('active');activeTile=null}};

async function showDetails(tile){
  const name=tile.dataset.name;
  if(activeTile)activeTile.classList.remove('active');
  tile.classList.add('active');activeTile=tile;
  
  $('sidePanel').classList.add('active');
  $('panelTitle').textContent=name.length>32?name.slice(0,30)+'...':name;
  $('previewImg').src='/thumb?run='+jobId+'&name='+encodeURIComponent(name)+'&full=1';
  
  try{
    const r=await fetch('/details/'+jobId+'/'+encodeURIComponent(name));
    const d=await r.json();
    
    const badge=$('resultBadge');
    badge.className='result-badge '+(d.is_family?'yes':'no');
    badge.textContent=d.is_family?'✓ FAMILY':'✗ NOT FAMILY';
    
    const yoloConf=d.yolo_conf||currentConf;
    const simThreshold=d.sim_threshold||currentSimThreshold;
    
    // Steps
    const stepsList=$('stepsList');stepsList.innerHTML='';
    
    const step1Pass=d.faces>0;
    const step1=document.createElement('div');
    step1.className='step-box '+(step1Pass?'pass':'fail');
    step1.innerHTML=`
      <div class="step-title">${step1Pass?'✓':'✗'} Step 1: Face Detection</div>
      <div class="step-detail">${d.faces>0
        ?`Found <strong>${d.faces}</strong> face(s) with YOLO confidence ≥ ${Math.round(yoloConf*100)}%`
        :`No faces detected with YOLO confidence ≥ ${Math.round(yoloConf*100)}%`
      }</div>
    `;
    stepsList.appendChild(step1);
    
    if(d.faces>0){
      const matched=d.matches.filter((m,i)=>(d.match_scores[i]||0)>=simThreshold).length;
      const step2Pass=matched>0;
      const step2=document.createElement('div');
      step2.className='step-box '+(step2Pass?'pass':'fail');
      step2.innerHTML=`
        <div class="step-title">${step2Pass?'✓':'✗'} Step 2: Similarity Matching</div>
        <div class="step-detail"><strong>${matched}</strong> of ${d.faces} face(s) have similarity ≥ ${Math.round(simThreshold*100)}%</div>
      `;
      stepsList.appendChild(step2);
    }
    
    // Face list
    const faceList=$('faceList');faceList.innerHTML='';
    if(d.faces===0){
      faceList.innerHTML='<p style="color:#999;font-size:12px;padding:10px;background:#f9f9fb;border-radius:8px">No faces passed the YOLO detection threshold.</p>';
    }else{
      d.matches.forEach((m,i)=>{
        const sim=d.match_scores[i]||0;
        const rawSim=(d.raw_match_scores&&d.raw_match_scores[i]!=null)?d.raw_match_scores[i]:sim;
        const pct=Math.round(sim*100);
        const rawPct=Math.round(rawSim*100);
        const isMatch=sim>=simThreshold;
        const barClass=pct>=65?'high':pct>=45?'med':'low';
        const trainThumb=m.train_thumb_url||m.thumb_url||'/placeholder';
        const queryThumb=m.query_thumb_url||'/placeholder';
        const trainTarget=isMatch
          ? (m.train_face_file?`${m.best}/${m.train_face_file}`:(m.best||'Unknown'))
          : '(none)';
        const sourceHint=(isMatch&&m.train_source_hint)?` • source: ${m.train_source_hint}`:'';
        const bonusPct=m.dup_bonus?Math.round(m.dup_bonus*100):0;
        const bonusText=bonusPct>0
          ? `<p class="detail">Boosted: ${rawPct}% + ${bonusPct}% = <strong>${pct}%</strong> (${m.dup_group_size||2} detections same person)</p>`
          : '';
        
        const item=document.createElement('div');
        item.className='face-item';
        item.innerHTML=`
          <div class="face-pair">
            <img class="face-thumb" src="${trainThumb}" onerror="this.style.opacity='0.3'" title="Train face">
            <span class="face-arrow">↔</span>
            <img class="face-thumb" src="${queryThumb}" onerror="this.style.opacity='0.3'" title="Detected face in current image">
          </div>
          <div class="face-info">
            <h4>Face ${i+1}</h4>
            <p>${isMatch?'✓ Matched':'✗ Not matched'} • <strong>S: ${pct}%</strong></p>
            <p class="detail">Face ${i+1} → ${trainTarget}</p>
            ${bonusText}
            <p class="detail">${pct}% ${isMatch?'≥':'<'} ${Math.round(simThreshold*100)}% threshold${sourceHint}</p>
            <div class="match-bar">
              <div class="match-bar-fill ${barClass}" style="width:${Math.min(pct,100)}%"></div>
              <div class="threshold-line" style="left:${simThreshold*100}%" title="Threshold"></div>
            </div>
          </div>
        `;
        faceList.appendChild(item);
      });
    }
    
    // Explanation
    const exp=$('explainBox');
    if(d.faces===0){
      exp.className='explain-box error';
      exp.innerHTML=`
        <strong>Why NOT FAMILY?</strong><br><br>
        ✗ <strong>Failed Step 1:</strong> No face detected<br><br>
        YOLO could not find any face with confidence ≥ <code>${Math.round(yoloConf*100)}%</code>
        <div class="tip">💡 <strong>Tip:</strong> Lower the YOLO Conf slider to detect faces with lower confidence.</div>
      `;
    }else{
      const matched=d.matches.filter((m,i)=>(d.match_scores[i]||0)>=simThreshold).length;
      const bestSim=d.match_scores.length?Math.max(...d.match_scores):0;
      
      if(matched>0){
        exp.className='explain-box';
        exp.innerHTML=`
          <strong>Why FAMILY?</strong><br><br>
          ✓ Step 1: ${d.faces} face(s) detected<br>
          ✓ Step 2: ${matched} face(s) matched (similarity ≥ ${Math.round(simThreshold*100)}%)
          <div class="verdict yes">any_known rule: At least 1 match → FAMILY</div>
        `;
      }else{
        exp.className='explain-box error';
        exp.innerHTML=`
          <strong>Why NOT FAMILY?</strong><br><br>
          ✓ Step 1: ${d.faces} face(s) detected<br>
          ✗ <strong>Failed Step 2:</strong> No face matched<br><br>
          Best similarity: <code>S: ${Math.round(bestSim*100)}%</code> < ${Math.round(simThreshold*100)}% threshold
          <div class="tip">💡 <strong>Tip:</strong> Lower the Similarity slider, or this person may not be in training data.</div>
          <div class="verdict no">any_known rule: No match → NOT FAMILY</div>
        `;
      }
    }
  }catch(e){log('Failed: '+e,false)}
}

function buildGrid(names){
  $('grid').innerHTML='';$('resultCard').style.display='block';$('stats').style.display='flex';
  $('sidePanel').classList.remove('active');selected.clear();activeTile=null;
  names.forEach(n=>{
    const t=document.createElement('div');t.className='tile';t.dataset.name=n;
    t.innerHTML=`<img loading="lazy"><div class="tile-name" title="${n}">${n.length>15?n.slice(0,13)+'...':n}</div>`;
    t.onclick=e=>{if(e.shiftKey)toggleSel(t,e);else showDetails(t)};
    $('grid').appendChild(t);
  });
}

function updateTile(name,yes,matchScore,faces){
  const t=document.querySelector(`.tile[data-name="${name}"]`);if(!t)return;
  t.classList.add(yes?'yes':'no');
  let info=faces+'f';
  if(faces>0)info+=' S:'+Math.round(matchScore*100)+'%';
  let c=t.querySelector('.tile-info');
  if(!c){c=document.createElement('div');c.className='tile-info';t.appendChild(c)}
  c.textContent=info;
  t.querySelector('img').src='/thumb?run='+jobId+'&name='+encodeURIComponent(name);
}

$('form').onsubmit=async e=>{
  e.preventDefault();
  const files=$('picker').files;
  if(!files.length){alert('Select files');return}
  if(currentMode==='cloud' && !($('cloudModel').value||'')){alert('Select cloud model');return}
  if(es){es.close();es=null}
  $('downloadLink').style.display='none';$('rescueRow').style.display='none';
  
  currentConf=$('conf').value/100;
  currentSimThreshold=$('simThreshold').value/100;
  
  const fd=new FormData();
  fd.append('conf',currentConf);
  fd.append('sim_threshold',currentSimThreshold);
  fd.append('mode',currentMode);
  if(currentMode==='local'){
    fd.append('use_edge',$('useEdge').checked?'1':'0');
  }else{
    fd.append('model_name',$('cloudModel').value);
  }
  for(const f of files)fd.append('files',f,f.name);
  
  const modeInfo=currentMode==='cloud' ? `Cloud:${$('cloudModel').value}` : 'Local';
  log(`Processing ${files.length} files [${modeInfo}] (YOLO≥${Math.round(currentConf*100)}%, Sim≥${Math.round(currentSimThreshold*100)}%)`);
  const r=await fetch('/batch',{method:'POST',body:fd});
  if(!r.ok){log('Upload failed',false);return}
  const d=await r.json();jobId=d.job_id;
  buildGrid(d.names);
  
  es=new EventSource('/events/'+jobId);
  es.addEventListener('item',e=>{
    const m=JSON.parse(e.data);
    updateTile(m.name,m.yes,m.match_score,m.faces);
    $('statProg').textContent=m.done+'/'+m.total;
    $('statYes').textContent='YES: '+m.yes_count;
    $('statNo').textContent='NO: '+m.no_count;
    $('statSpeed').textContent=Math.round(m.time_ms)+'ms';
  });
  es.addEventListener('done',e=>{
    const m=JSON.parse(e.data);
    log('Done! YES='+m.yes_count+' NO='+m.no_count+' ('+m.elapsed_s+'s)');
    if(m.saved_zip) log('Saved ZIP: '+m.saved_zip);
    if(m.download_url){$('downloadLink').href=m.download_url;$('downloadLink').style.display='inline-block'}
    es.close();
  });
  es.onerror=()=>log('Connection lost',false);
};

applyModeUI();
refreshCloudModels();
if('serviceWorker' in navigator){
  window.addEventListener('load',()=>{
    navigator.serviceWorker.register('/service-worker.js').catch(()=>{});
  });
}
log('Ready. Click thumbnail for details. Shift+Click RED tiles to select for rescue.');
</script>
</body>
</html>'''

# ---- Routes ----
@app.get("/")
def home():
    return render_template_string(HTML, workers=PROCESS_WORKERS)

@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "service": "family-photo-organizer",
        "port": APP_PORT,
    })

@app.get("/manifest.webmanifest")
def pwa_manifest():
    manifest = {
        "name": "Family Photo Organizer",
        "short_name": "Family Photo",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#f5f5f7",
        "theme_color": "#0b6bcb",
        "description": "Detect and organize family photos using YOLO face matching.",
        "icons": [
            {"src": "/pwa-icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/pwa-icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    }
    return Response(json.dumps(manifest), mimetype="application/manifest+json")

@app.get("/pwa-icon-<int:size>.png")
def pwa_icon(size: int):
    if size not in (192, 512):
        abort(404)
    icon_bytes = _build_pwa_icon(size)
    if not icon_bytes:
        abort(500)
    return Response(icon_bytes, mimetype="image/png")

@app.get("/service-worker.js")
def service_worker():
    sw = r"""
const CACHE_NAME = 'family-photo-pwa-v1';
const APP_SHELL = ['/', '/manifest.webmanifest', '/pwa-icon-192.png', '/pwa-icon-512.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/events')) return; // SSE

  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).catch(() => caches.match('/'))
    );
    return;
  }

  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, copy)).catch(() => {});
        return resp;
      });
    })
  );
});
"""
    resp = Response(sw, mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    return resp

@app.get("/cloud/models")
def cloud_models():
    return jsonify({"models": _list_cloud_models()})

@app.post("/cloud/extract_faces")
def cloud_extract_faces():
    _cleanup_stale_runtime_data()
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No training files"}), 400
    person_count_raw = (request.form.get("person_count", "") or "").strip()
    person_limit: Optional[int] = None
    if person_count_raw:
        try:
            person_limit = int(person_count_raw)
        except Exception:
            return jsonify({"error": "person_count must be an integer"}), 400
        if person_limit < 1 or person_limit > 50:
            return jsonify({"error": "person_count must be between 1 and 50"}), 400

    session_id = _new_job_id()
    session_dir = UPLOAD_DIR / "_extract_sessions" / session_id
    raw_dir = session_dir / "raw"
    crops_dir = session_dir / "crops"
    raw_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    valid_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    faces = {}
    encodings = []
    face_ids = []

    image_idx = 0
    face_idx = 0
    for f in files:
        if not f.filename:
            continue
        name = Path(f.filename).name
        if name.startswith('.') or Path(name).suffix.lower() not in valid_ext:
            continue
        image_idx += 1
        img_path = raw_dir / f"{image_idx:05d}_{name}"
        f.save(str(img_path))

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _resize_for_extract(rgb, 1280)
        locs, confs = detect_faces_yolo(rgb, conf_threshold=DEFAULT_CONF)
        if not locs:
            continue
        encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
        if not encs:
            continue

        for i, enc in enumerate(encs):
            if i >= len(locs):
                break
            top, right, bottom, left = locs[i]
            top = max(0, top)
            left = max(0, left)
            bottom = min(rgb.shape[0], bottom)
            right = min(rgb.shape[1], right)
            if bottom <= top or right <= left:
                continue
            h, w = bottom - top, right - left
            pad_h, pad_w = int(h * 0.2), int(w * 0.2)
            t = max(0, top - pad_h)
            b = min(rgb.shape[0], bottom + pad_h)
            l = max(0, left - pad_w)
            r = min(rgb.shape[1], right + pad_w)
            crop = rgb[t:b, l:r]
            if crop.size == 0:
                continue

            face_idx += 1
            face_id = f"face_{face_idx:05d}"
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            thumb_path = crops_dir / f"{face_id}.jpg"
            cv2.imwrite(str(thumb_path), crop_bgr)

            faces[face_id] = {
                "face_id": face_id,
                "thumb_path": str(thumb_path),
                "source_image": name,
                "conf": confs[i] if i < len(confs) else 0.0,
                "encoding": np.array(enc, dtype=np.float32),
                "person_id": "reject",
            }
            encodings.append(np.array(enc, dtype=np.float32))
            face_ids.append(face_id)

    if not faces:
        shutil.rmtree(session_dir, ignore_errors=True)
        return jsonify({"error": "No faces extracted"}), 400

    person_ids = []
    include_unidentified = False
    if encodings:
        person_ids, assigned_ids, include_unidentified = _assign_person_groups(encodings, person_limit)
        for idx, face_id in enumerate(face_ids):
            if idx < len(assigned_ids):
                faces[face_id]["person_id"] = assigned_ids[idx]

    session = FaceExtractSession(
        session_id=session_id,
        temp_dir=session_dir,
        created_at=time.time(),
        faces=faces,
        person_ids=person_ids,
        include_unidentified=include_unidentified,
    )
    EXTRACT_SESSIONS[session_id] = session
    return jsonify(_extract_groups_payload(session))

@app.get("/cloud/extract_thumb/<session_id>/<face_id>")
def cloud_extract_thumb(session_id, face_id):
    sess = EXTRACT_SESSIONS.get(session_id)
    if not sess:
        abort(404)
    face = sess.faces.get(face_id)
    if not face:
        abort(404)
    p = Path(face["thumb_path"])
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/jpeg")

@app.post("/cloud/extract_move")
def cloud_extract_move():
    data = request.get_json() or {}
    session_id = data.get("session_id", "")
    face_id = data.get("face_id", "")
    target = data.get("target_person", "")
    sess = EXTRACT_SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    with sess.lock:
        if face_id not in sess.faces:
            return jsonify({"error": "Face not found"}), 404
        if target not in ("reject", "unidentified") and target not in sess.person_ids:
            return jsonify({"error": "Invalid target person"}), 400
        if target == "unidentified":
            sess.include_unidentified = True
        sess.faces[face_id]["person_id"] = target
        payload = _extract_groups_payload(sess)
    return jsonify(payload)

@app.post("/cloud/extract_add_person")
def cloud_extract_add_person():
    data = request.get_json() or {}
    session_id = data.get("session_id", "")
    sess = EXTRACT_SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    with sess.lock:
        i = len(sess.person_ids) + 1
        new_id = f"person_{i:02d}"
        while new_id in sess.person_ids:
            i += 1
            new_id = f"person_{i:02d}"
        sess.person_ids.append(new_id)
        payload = _extract_groups_payload(sess)
    return jsonify(payload)

@app.post("/cloud/extract_train")
def cloud_extract_train():
    data = request.get_json() or {}
    session_id = data.get("session_id", "")
    model_name = (data.get("model_name", "") or "").strip()
    base_model = (data.get("base_model", "") or "").strip()
    if not model_name:
        return jsonify({"error": "Missing model_name"}), 400
    sess = EXTRACT_SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    try:
        with sess.lock:
            safe_model, continued_from = _build_cloud_db_from_extract(sess, model_name, base_model=base_model)
        return jsonify({"ok": True, "model_name": safe_model, "continued_from": continued_from or ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/cloud/reset_all")
def cloud_reset_all():
    active_train = any(not j.done for j in CLOUD_TRAIN_JOBS.values())
    active_batch = any(not j.finished for j in JOBS.values())
    if active_train or active_batch:
        return jsonify({"error": "Cannot reset while training/detection is running"}), 409

    try:
        if CLOUD_MODELS_DIR.exists():
            shutil.rmtree(CLOUD_MODELS_DIR)
        CLOUD_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        deleted_zips = 0
        for z in RESULTS_DIR.glob("results_*.zip"):
            if z.is_file():
                _safe_unlink(z)
                deleted_zips += 1

        with DB_LOCK:
            CLOUD_DB_CACHE.clear()
        for job in JOBS.values():
            _safe_rmtree(job.run_dir)
        JOBS.clear()
        CLOUD_TRAIN_JOBS.clear()
        EXTRACT_SESSIONS.clear()
        _cleanup_legacy_tmp_dirs()

        return jsonify({"ok": True, "deleted_result_zips": deleted_zips})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/cloud/train")
def cloud_train():
    _cleanup_stale_runtime_data()
    model_name = (request.form.get("model_name", "") or "").strip()
    files = request.files.getlist("files")
    if not model_name:
        return jsonify({"error": "Missing model_name"}), 400
    if not files:
        return jsonify({"error": "No training files"}), 400

    try:
        safe_model, _, _, _ = _cloud_model_paths(model_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    train_job_id = _new_job_id()
    train_dir = UPLOAD_DIR / "_cloud_train" / train_job_id / "family"
    train_dir.mkdir(parents=True, exist_ok=True)

    valid_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    saved = 0
    for f in files:
        if not f.filename:
            continue
        name = Path(f.filename).name
        if name.startswith('.') or Path(name).suffix.lower() not in valid_ext:
            continue
        dst = train_dir / name
        i = 1
        while dst.exists():
            dst = train_dir / f"{Path(name).stem}_{i}{Path(name).suffix}"
            i += 1
        f.save(str(dst))
        saved += 1

    if saved == 0:
        shutil.rmtree(train_dir.parent, ignore_errors=True)
        return jsonify({"error": "No valid training images"}), 400

    train_job = CloudTrainJob(
        train_job_id=train_job_id,
        model_name=safe_model,
        temp_dir=train_dir,
        created_at=time.time(),
        message=f"queued ({saved} files)",
    )
    CLOUD_TRAIN_JOBS[train_job_id] = train_job
    threading.Thread(target=_cloud_train_worker, args=(train_job,), daemon=True).start()
    return jsonify({"train_job_id": train_job_id, "model_name": safe_model, "files": saved})

@app.get("/cloud/train_status/<train_job_id>")
def cloud_train_status(train_job_id):
    job = CLOUD_TRAIN_JOBS.get(train_job_id)
    if not job:
        abort(404)
    return jsonify({
        "train_job_id": train_job_id,
        "model_name": job.model_name,
        "done": job.done,
        "ok": job.ok,
        "message": job.message,
        "error": job.error,
        "elapsed_s": round(time.time() - (job.started_at or job.created_at), 1),
    })

@app.post("/batch")
def batch():
    files = request.files.getlist("files")
    if not files:
        abort(400)
    _cleanup_stale_runtime_data()
    
    mode = (request.form.get("mode", "local") or "local").strip().lower()
    if mode not in ("local", "cloud"):
        return jsonify({"error": "Invalid mode"}), 400
    model_name = (request.form.get("model_name", "") or "").strip()
    conf = float(request.form.get("conf", DEFAULT_CONF))
    sim_threshold = float(request.form.get("sim_threshold", 0.55))
    use_edge = (request.form.get("use_edge") == "1") if mode == "local" else False

    try:
        if mode == "cloud":
            safe_model, _, _, faces_dir = _cloud_model_paths(model_name)
            db = _get_cloud_db(safe_model)
            edge_db = None
            trained_faces_root = str(faces_dir.resolve())
            model_name = safe_model
        else:
            db = _get_db()
            edge_db = _get_edge_db() if use_edge else None
            trained_faces_root = str(TRAINED_FACES_DIR.resolve())
            model_name = "local"
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    job_id = _new_job_id()
    run_dir = UPLOAD_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    job = BatchJob(
        job_id=job_id,
        run_dir=run_dir,
        conf=conf,
        sim_threshold=sim_threshold,
        use_edge=use_edge,
        mode=mode,
        model_name=model_name,
        trained_faces_root=trained_faces_root,
        db=db,
        edge_db=edge_db,
    )
    
    names = []
    valid_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for f in files:
        if not f.filename:
            continue
        name = Path(f.filename).name
        if name.startswith('.') or Path(name).suffix.lower() not in valid_ext:
            continue
        data = f.read()
        job.pending.put((name, data))
        names.append(name)
    
    if not names:
        abort(400, "No valid images")
    
    job.pending.put(None)
    job.total = len(names)
    job.started_at = time.time()
    
    JOBS[job_id] = job
    threading.Thread(target=_batch_worker, args=(job,), daemon=True).start()
    
    return jsonify({"job_id": job_id, "total": len(names), "names": names, "mode": mode, "model_name": model_name})

@app.get("/events/<job_id>")
def events(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    
    def gen():
        yield "event: connected\ndata: {}\n\n"
        while True:
            try:
                msg = job.events.get(timeout=30)
                yield msg
                with job.lock:
                    if job.finished and job.events.empty():
                        break
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                with job.lock:
                    if job.finished:
                        break
    
    return Response(gen(), mimetype="text/event-stream")

@app.get("/details/<job_id>/<path:name>")
def get_details(job_id, name):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    
    with job.lock:
        result = job.results.get(name)
    
    if not result:
        abort(404)

    trained_faces_root = Path(job.trained_faces_root) if job.trained_faces_root else TRAINED_FACES_DIR
    
    matches_with_thumbs = []
    for i, m in enumerate(result.matches):
        m_copy = dict(m)
        best_name = m.get("best")
        train_face_file = m.get("train_face_file")
        if best_name and train_face_file:
            train_face_path = _get_train_face_path(best_name, train_face_file, trained_faces_root)
            if train_face_path:
                m_copy["train_thumb_url"] = f"/train_face_thumb?job={job_id}&family={best_name}&file={train_face_file}"
        if best_name:
            thumb_path = _get_family_thumb(best_name, trained_faces_root)
            if thumb_path:
                m_copy["thumb_url"] = f"/family_thumb/{best_name}?job={job_id}"
        query_thumb = UPLOAD_DIR / job_id / "_query_faces" / f"{Path(name).stem}_qface{i}.jpg"
        if query_thumb.exists():
            m_copy["query_thumb_url"] = f"/query_face_thumb?run={quote(job_id)}&name={quote(name)}&idx={i}"
        matches_with_thumbs.append(m_copy)
    
    return jsonify({
        "name": name,
        "is_family": result.is_family,
        "faces": result.faces,
        "recognized": result.recognized,
        "matches": matches_with_thumbs,
        "raw_match_scores": result.raw_match_scores,
        "match_scores": result.match_scores,
        "yolo_conf": result.yolo_conf,
        "sim_threshold": result.sim_threshold,
        "time_ms": result.time_ms,
    })

@app.get("/family_thumb/<family_name>")
def family_thumb(family_name):
    job_id = request.args.get("job", "")
    trained_faces_root = TRAINED_FACES_DIR
    if job_id and job_id in JOBS:
        trained_faces_root = Path(JOBS[job_id].trained_faces_root)
    thumb_path = _get_family_thumb(family_name, trained_faces_root)
    if thumb_path and Path(thumb_path).exists():
        return send_file(thumb_path, mimetype="image/jpeg")
    abort(404)


@app.get("/train_face_thumb")
def train_face_thumb():
    job_id = request.args.get("job", "")
    family_name = request.args.get("family", "")
    face_file = request.args.get("file", "")
    if not family_name or not face_file:
        abort(400)
    trained_faces_root = TRAINED_FACES_DIR
    if job_id and job_id in JOBS:
        trained_faces_root = Path(JOBS[job_id].trained_faces_root)
    train_face_path = _get_train_face_path(family_name, face_file, trained_faces_root)
    if not train_face_path:
        abort(404)
    return send_file(str(train_face_path), mimetype="image/jpeg")

@app.get("/query_face_thumb")
def query_face_thumb():
    run = request.args.get("run", "")
    name = request.args.get("name", "")
    idx = request.args.get("idx", "")
    if not run or not name:
        abort(400)
    try:
        i = int(idx)
    except Exception:
        abort(400)
    if i < 0:
        abort(400)
    path = UPLOAD_DIR / run / "_query_faces" / f"{Path(name).stem}_qface{i}.jpg"
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="image/jpeg")

@app.get("/thumb")
def thumb():
    run = request.args.get("run", "")
    name = request.args.get("name", "")
    full = request.args.get("full", "")
    if not run or not name:
        abort(400)
    
    if full:
        orig_path = UPLOAD_DIR / run / name
        if orig_path.exists():
            return send_file(str(orig_path))
    
    thumb_path = UPLOAD_DIR / run / "_thumbs" / f"{Path(name).stem}.jpg"
    if thumb_path.exists():
        return send_file(str(thumb_path), mimetype="image/jpeg")
    
    abort(404)

@app.get("/placeholder")
def placeholder():
    return make_response(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xf8\xf8\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\x18\x00\x00\x00\x00IEND\xaeB`\x82'), 200, {'Content-Type': 'image/png'}

@app.get("/download/<job_id>")
def download(job_id):
    _cleanup_stale_runtime_data()

    job = JOBS.get(job_id)
    if job and job.export_zip_path:
        export_zip = Path(job.export_zip_path)
        if export_zip.exists():
            @after_this_request
            def _cleanup_after_download(response):
                _cleanup_job_runtime(job_id, drop_job=True)
                return response
            return send_file(str(export_zip), as_attachment=True, download_name=export_zip.name)
    export_zip = RESULTS_DIR / f"results_{job_id}.zip"
    if export_zip.exists():
        @after_this_request
        def _cleanup_after_download(response):
            _cleanup_job_runtime(job_id, drop_job=True)
            return response
        return send_file(str(export_zip), as_attachment=True, download_name=export_zip.name)
    abort(404)

@app.post("/move_to_family")
def move_to_family():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    job_id = data.get("job_id")
    names = data.get("names", [])
    if not job_id or not names:
        return jsonify({"error": "Missing data"}), 400
    
    run_dir = UPLOAD_DIR / job_id
    family_dir = run_dir / "family"
    non_family_dir = run_dir / "non_family"
    family_dir.mkdir(exist_ok=True)
    
    moved = 0
    for name in names:
        src = non_family_dir / name
        if not src.exists():
            continue
        dst = family_dir / name
        if dst.exists():
            i = 1
            while (family_dir / f"{Path(name).stem}_{i}{Path(name).suffix}").exists():
                i += 1
            dst = family_dir / f"{Path(name).stem}_{i}{Path(name).suffix}"
        shutil.move(str(src), str(dst))
        moved += 1
    
    try:
        zip_path = run_dir / "results.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("family", "non_family"):
                folder = run_dir / sub
                if folder.exists():
                    for p in folder.rglob("*"):
                        if p.is_file():
                            z.write(str(p), str(p.relative_to(run_dir)))
        export_zip = RESULTS_DIR / f"results_{job_id}.zip"
        shutil.copy2(str(zip_path), str(export_zip))
        zip_path.unlink(missing_ok=True)
        job = JOBS.get(job_id)
        if job:
            with job.lock:
                job.export_zip_path = str(export_zip)
    except:
        pass
    
    return jsonify({"moved": moved})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Family Photo Organizer - YOLOv8")
    print("="*50)
    print(f"  Workers:       {PROCESS_WORKERS}")
    print(f"  Mode:          any_known")
    print(f"  Edge Model:    {'✓' if EDGE_DB_PATH.exists() else '○'}")
    print(f"  Trained Faces: {'✓' if TRAINED_FACES_DIR.exists() else '○'}")
    print("="*50)
    print(f"  URL: http://{APP_HOST}:{APP_PORT}")
    print("="*50 + "\n")
    
    print("[INFO] Loading YOLO model...")
    try:
        get_yolo_model()
        print("[INFO] YOLO ready!\n")
    except Exception as e:
        print(f"[WARN] {e}\n")
    
    app.run(host=APP_HOST, port=APP_PORT, debug=False, threaded=True)
