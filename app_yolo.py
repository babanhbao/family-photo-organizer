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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from flask import Flask, request, abort, send_file, render_template_string, jsonify, Response, make_response

from family_photo_detector_yolo import (
    load_db, load_edge_db, classify_image_from_array,
    classify_image_from_array_two_tier, get_yolo_model,
    ClassifyResult, EDGE_DB_PATH, DB_PATH,
    DEFAULT_TOLERANCE, DEFAULT_CONF,
)

app = Flask(__name__)

# ---- Config ----
APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "family_organizer_yolo"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_FACES_DIR = APP_DIR / "trained_faces"
RESULTS_DIR = APP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THUMB_MAX_SIDE = 320
THUMB_QUALITY = 72
PROCESS_WORKERS = int(os.environ.get("PROCESS_WORKERS", "4"))

DB_CACHE = None
EDGE_DB_CACHE = None
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

def _get_family_thumb(family_name):
    folder = TRAINED_FACES_DIR / family_name
    if folder.exists():
        for f in folder.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                return str(f)
    return None


def _get_train_face_path(family_name: str, face_file: str) -> Optional[Path]:
    safe_family = Path(family_name).name
    safe_file = Path(face_file).name
    folder = (TRAINED_FACES_DIR / safe_family).resolve()
    if not folder.exists() or not folder.is_dir():
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
    yolo_conf: float = 0.0
    sim_threshold: float = 0.55
    error: Optional[str] = None

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
    total: int = 0
    done: int = 0
    yes: int = 0
    no: int = 0
    finished: bool = False
    started_at: float = 0
    pending: "queue.Queue" = field(default_factory=queue.Queue)
    events: "queue.Queue" = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: Dict[str, ProcessResult] = field(default_factory=dict)
    export_zip_path: Optional[str] = None

JOBS: Dict[str, BatchJob] = {}

def _send(job, event, data):
    job.events.put(f"event: {event}\ndata: {json.dumps(data)}\n\n")

def _batch_worker(job: BatchJob):
    db = _get_db()
    edge_db = _get_edge_db() if job.use_edge else None
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
    
    _send(job, "done", {
        "done": job.done, "total": job.total,
        "yes_count": job.yes, "no_count": job.no,
        "elapsed_s": round(time.time() - job.started_at, 2),
        "download_url": dl,
        "saved_zip": job.export_zip_path or "",
    })

# ---- HTML ----
HTML = '''<!doctype html>
<html>
<head>
  <meta charset="utf-8"><title>Family Photo - YOLO</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
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
    <div class="row" style="flex-direction:column;align-items:flex-start;gap:8px">
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
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px">
        <input type="checkbox" id="useEdge"> Use Edge Model
      </label>
      <button type="button" class="btn" id="browseBtn">Browse Folder…</button>
      <input type="file" id="picker" style="display:none" webkitdirectory multiple>
      <button type="submit" class="btn btn-primary">Run</button>
      <span class="muted" id="pickedLabel">No selection</span>
      <a id="downloadLink" class="linkbtn" href="#" style="display:none">Download ZIP</a>
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

$('conf').oninput=e=>{currentConf=e.target.value/100;$('confVal').textContent=e.target.value+'%'};
$('simThreshold').oninput=e=>{currentSimThreshold=e.target.value/100;$('simVal').textContent=e.target.value+'%'};
$('browseBtn').onclick=()=>$('picker').click();
$('picker').onchange=e=>{$('pickedLabel').textContent=e.target.files.length?e.target.files.length+' files':'No selection';$('downloadLink').style.display='none'};

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
          <img class="face-thumb" src="${trainThumb}" onerror="this.style.opacity='0.3'">
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
  if(es){es.close();es=null}
  $('downloadLink').style.display='none';$('rescueRow').style.display='none';
  
  currentConf=$('conf').value/100;
  currentSimThreshold=$('simThreshold').value/100;
  
  const fd=new FormData();
  fd.append('conf',currentConf);
  fd.append('sim_threshold',currentSimThreshold);
  fd.append('use_edge',$('useEdge').checked?'1':'0');
  for(const f of files)fd.append('files',f,f.name);
  
  log(`Processing ${files.length} files (YOLO≥${Math.round(currentConf*100)}%, Sim≥${Math.round(currentSimThreshold*100)}%)`);
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

log('Ready. Click thumbnail for details. Shift+Click RED tiles to select for rescue.');
</script>
</body>
</html>'''

# ---- Routes ----
@app.get("/")
def home():
    return render_template_string(HTML, workers=PROCESS_WORKERS)

@app.post("/batch")
def batch():
    files = request.files.getlist("files")
    if not files:
        abort(400)
    
    conf = float(request.form.get("conf", DEFAULT_CONF))
    sim_threshold = float(request.form.get("sim_threshold", 0.55))
    use_edge = request.form.get("use_edge") == "1"
    
    job_id = _new_job_id()
    run_dir = UPLOAD_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    job = BatchJob(job_id=job_id, run_dir=run_dir, conf=conf, sim_threshold=sim_threshold, use_edge=use_edge)
    
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
    
    return jsonify({"job_id": job_id, "total": len(names), "names": names})

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
    
    matches_with_thumbs = []
    for m in result.matches:
        m_copy = dict(m)
        best_name = m.get("best")
        train_face_file = m.get("train_face_file")
        if best_name and train_face_file:
            train_face_path = _get_train_face_path(best_name, train_face_file)
            if train_face_path:
                m_copy["train_thumb_url"] = f"/train_face_thumb?family={best_name}&file={train_face_file}"
        if best_name:
            thumb_path = _get_family_thumb(best_name)
            if thumb_path:
                m_copy["thumb_url"] = f"/family_thumb/{best_name}"
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
    thumb_path = _get_family_thumb(family_name)
    if thumb_path and Path(thumb_path).exists():
        return send_file(thumb_path, mimetype="image/jpeg")
    abort(404)


@app.get("/train_face_thumb")
def train_face_thumb():
    family_name = request.args.get("family", "")
    face_file = request.args.get("file", "")
    if not family_name or not face_file:
        abort(400)
    train_face_path = _get_train_face_path(family_name, face_file)
    if not train_face_path:
        abort(404)
    return send_file(str(train_face_path), mimetype="image/jpeg")

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
    job = JOBS.get(job_id)
    if job and job.export_zip_path:
        export_zip = Path(job.export_zip_path)
        if export_zip.exists():
            return send_file(str(export_zip), as_attachment=True, download_name=export_zip.name)
    export_zip = RESULTS_DIR / f"results_{job_id}.zip"
    if export_zip.exists():
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
    print("  URL: http://127.0.0.1:5050")
    print("="*50 + "\n")
    
    print("[INFO] Loading YOLO model...")
    try:
        get_yolo_model()
        print("[INFO] YOLO ready!\n")
    except Exception as e:
        print(f"[WARN] {e}\n")
    
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
