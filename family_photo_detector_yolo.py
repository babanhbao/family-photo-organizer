#!/usr/bin/env python3
"""
family_photo_detector_yolo.py

YOLOv8 Face Detection Version
=============================
Thay thế HOG/CNN của dlib bằng YOLOv8-face để detect khuôn mặt.
Vẫn dùng face_recognition (dlib) cho encoding/matching.

Install:
  pip install ultralytics face_recognition opencv-python numpy scikit-learn

CLI:
  python family_photo_detector_yolo.py train
  python family_photo_detector_yolo.py check --image "/path/to/photo.jpg"
  python family_photo_detector_yolo.py batch --folder "/path/to/folder"
"""

import argparse
import math
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

# ---------------- CONFIG ----------------
TRAIN_FAMILY_DIR = Path("train/family")
TRAIN_EDGE_DIR = Path("train/edge")
TRAINED_FACES_DIR = Path("trained_faces")
DB_DIR = Path("db")
DB_PATH = DB_DIR / "family_db.pkl"
EDGE_DB_PATH = DB_DIR / "edge_db.pkl"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_EPS = 0.42
DEFAULT_MIN_CLUSTER_SIZE = 12
DEFAULT_TOP_K = 8
DEFAULT_MIN_UNIQUE_IMAGES = 8
DEFAULT_THRESH_QUANTILE = 0.90
DEFAULT_THRESH_MARGIN = 0.02
DEFAULT_MAX_TOLERANCE_CAP = 0.50
DEFAULT_JITTERS_TRAIN = 1
DEFAULT_MAX_FACES_PER_IMAGE = 10

# YOLO config
YOLO_FACE_MODEL = "yolov8n-face.pt"
YOLO_FACE_MODEL_URL = "https://github.com/akanametov/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
DEFAULT_FAST_MAX_SIDE = 640
DEFAULT_FAST_JITTERS = 1
DEFAULT_FAST_TOLERANCE = 0.42
DEFAULT_FAST_CONF = 0.25
DEFAULT_FULL_MAX_SIDE = 1280
DEFAULT_FULL_JITTERS = 2
DEFAULT_FULL_TOLERANCE = 0.45
DEFAULT_FULL_CONF = 0.15
DEFAULT_JITTERS_CHECK = 2
DEFAULT_MAX_SIDE = 1280
DEFAULT_TOLERANCE = 0.45
DEFAULT_DECISION = "majority_known"
DEFAULT_CONF = 0.20

_yolo_model: Optional[YOLO] = None


def _download_yolo_face_model():
    """Download YOLOv8-face model nếu chưa có"""
    import urllib.request
    
    model_path = Path(YOLO_FACE_MODEL)
    if model_path.exists():
        return True
    
    print(f"[INFO] Downloading {YOLO_FACE_MODEL}...")
    print(f"[INFO] URL: {YOLO_FACE_MODEL_URL}")
    
    try:
        urllib.request.urlretrieve(YOLO_FACE_MODEL_URL, str(model_path))
        print(f"[INFO] Downloaded: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"[WARN] Failed to download: {e}")
        return False


def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        model_path = Path(YOLO_FACE_MODEL)
        
        # Tự động download nếu chưa có
        if not model_path.exists():
            _download_yolo_face_model()
        
        # Load model
        if model_path.exists():
            print(f"[INFO] Loading {YOLO_FACE_MODEL} (face-specific)...")
            _yolo_model = YOLO(str(model_path))
        else:
            print("[WARN] Face model not available. Using general YOLOv8n (may detect non-faces)...")
            _yolo_model = YOLO("yolov8n.pt")
        
        _yolo_model.fuse()
        print("[INFO] YOLO model ready!")
    return _yolo_model


class DetectionMode(Enum):
    FAST = "fast"
    FULL = "full"
    SINGLE = "single"


@dataclass
class DetectionConfig:
    max_side: int
    num_jitters: int
    tolerance: float
    conf_threshold: float
    
    @classmethod
    def fast(cls):
        return cls(DEFAULT_FAST_MAX_SIDE, DEFAULT_FAST_JITTERS, DEFAULT_FAST_TOLERANCE, DEFAULT_FAST_CONF)
    
    @classmethod
    def full(cls):
        return cls(DEFAULT_FULL_MAX_SIDE, DEFAULT_FULL_JITTERS, DEFAULT_FULL_TOLERANCE, DEFAULT_FULL_CONF)
    
    @classmethod
    def single(cls, max_side=DEFAULT_MAX_SIDE, num_jitters=DEFAULT_JITTERS_CHECK,
               tolerance=DEFAULT_TOLERANCE, conf_threshold=DEFAULT_CONF):
        return cls(max_side, num_jitters, tolerance, conf_threshold)


@dataclass
class ClassifyResult:
    family: bool
    faces: int
    recognized: int
    matches: List[Dict[str, Any]]
    mode_used: str = "single"
    fast_passed: bool = False
    confidences: List[float] = field(default_factory=list)  # YOLO conf (keep for debug)
    match_scores: List[float] = field(default_factory=list)  # Similarity với DB (0-1)
    
    def to_dict(self):
        return {
            "family": self.family, "faces": self.faces, "recognized": self.recognized,
            "matches": self.matches, "mode_used": self.mode_used,
            "fast_passed": self.fast_passed, "confidences": self.confidences,
            "match_scores": self.match_scores,
        }


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def safe_move(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return
    i = 1
    while True:
        candidate = dst_dir / f"{src.stem}_{i}{src.suffix}"
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return
        i += 1


def _format_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"


def _progress_bar(current, total, width=28):
    ratio = current / total if total else 0.0
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def load_image_rgb_fast(path: Path, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = bgr.shape[:2]
    m = max(h, w)
    if max_side and m > max_side:
        scale = max_side / float(m)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb_if_needed(rgb: np.ndarray, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    scale = max_side / float(m)
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def detect_faces_yolo(rgb: np.ndarray, conf_threshold: float = DEFAULT_CONF,
                      max_faces: int = DEFAULT_MAX_FACES_PER_IMAGE):
    model = get_yolo_model()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = model(bgr, verbose=False, conf=conf_threshold)
    
    face_locations = []
    confidences = []
    
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        indices = boxes.conf.argsort(descending=True)
        for idx in indices[:max_faces]:
            box = boxes.xyxy[idx].cpu().numpy().astype(int)
            conf = float(boxes.conf[idx].cpu().numpy())
            x1, y1, x2, y2 = box
            face_locations.append((y1, x2, y2, x1))
            confidences.append(conf)
    
    return face_locations, confidences


def load_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}. Run train first.")
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    for k in ("centroids", "names", "thresholds"):
        if k not in db:
            raise ValueError(f"Invalid DB: missing '{k}'")
    db["centroids"] = np.array(db["centroids"], dtype=np.float32)
    db["thresholds"] = np.array(db["thresholds"], dtype=np.float32)
    _attach_trained_face_index(db)
    return db


def load_edge_db():
    if not EDGE_DB_PATH.exists():
        return None
    with open(EDGE_DB_PATH, "rb") as f:
        db = pickle.load(f)
    for k in ("centroids", "names", "thresholds"):
        if k not in db:
            return None
    db["centroids"] = np.array(db["centroids"], dtype=np.float32)
    db["thresholds"] = np.array(db["thresholds"], dtype=np.float32)
    return db


def _extract_source_hint(face_filename: str) -> str:
    stem = Path(face_filename).stem
    pos = stem.rfind("_face")
    if pos == -1:
        return ""
    return stem[:pos]


def _encode_face_crop(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return None
    try:
        encs = face_recognition.face_encodings(rgb, known_face_locations=[(0, w, h, 0)], num_jitters=1)
        if not encs:
            encs = face_recognition.face_encodings(rgb, num_jitters=1)
        if not encs:
            return None
        return np.array(encs[0], dtype=np.float32)
    except Exception:
        return None


def _attach_trained_face_index(db):
    if "train_faces" in db:
        return
    names = db.get("names") or []
    if not names or not TRAINED_FACES_DIR.exists():
        db["train_faces"] = {}
        return

    train_faces = {}
    for cluster_name in names:
        cluster_dir = TRAINED_FACES_DIR / cluster_name
        if not cluster_dir.exists() or not cluster_dir.is_dir():
            continue

        encodings = []
        files = []
        source_hints = []
        for p in sorted(cluster_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            enc = _encode_face_crop(p)
            if enc is None:
                continue
            encodings.append(enc)
            files.append(p.name)
            source_hints.append(_extract_source_hint(p.name))

        if encodings:
            train_faces[cluster_name] = {
                "encodings": np.array(encodings, dtype=np.float32),
                "files": files,
                "source_hints": source_hints,
            }

    db["train_faces"] = train_faces


def train_auto(eps=DEFAULT_EPS, min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE, top_k=DEFAULT_TOP_K,
               min_unique_images=DEFAULT_MIN_UNIQUE_IMAGES, thresh_quantile=DEFAULT_THRESH_QUANTILE,
               thresh_margin=DEFAULT_THRESH_MARGIN, max_tolerance_cap=DEFAULT_MAX_TOLERANCE_CAP,
               num_jitters=DEFAULT_JITTERS_TRAIN, max_faces_per_image=DEFAULT_MAX_FACES_PER_IMAGE,
               conf_threshold=DEFAULT_CONF, save_faces=True):
    if not TRAIN_FAMILY_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {TRAIN_FAMILY_DIR}")

    images = list(iter_images(TRAIN_FAMILY_DIR))
    total_images = len(images)
    if total_images == 0:
        raise RuntimeError("No images found in train/family")

    print(f"[INFO] Training on {total_images} images (YOLO conf >= {conf_threshold})\n")
    _ = get_yolo_model()
    print("[INFO] YOLO model loaded\n")

    # Create trained_faces folder if save_faces=True
    if save_faces:
        if TRAINED_FACES_DIR.exists():
            shutil.rmtree(TRAINED_FACES_DIR)
        TRAINED_FACES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Will save detected faces to: {TRAINED_FACES_DIR}/\n")

    embeddings, src_image_ids = [], []
    face_metadata = []  # Store face info for later clustering assignment
    total_faces_detected, encoded_faces = 0, 0
    start_time = time.time()

    for idx, img in enumerate(images, start=1):
        elapsed = time.time() - start_time
        eta = (elapsed / idx) * (total_images - idx)
        bar = _progress_bar(idx, total_images)
        print(f"\r[{bar}] {idx}/{total_images} | det: {total_faces_detected} | enc: {encoded_faces} | ETA: {_format_eta(eta)}", end="", flush=True)

        try:
            rgb = load_image_rgb_fast(img, max_side=0)
            locs, confs = detect_faces_yolo(rgb, conf_threshold=conf_threshold, max_faces=max_faces_per_image)
            total_faces_detected += len(locs)
            if not locs:
                continue
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=num_jitters)
            if not encs:
                continue
            
            # Save cropped faces if enabled
            if save_faces and len(encs) == len(locs):
                for face_idx, (loc, conf) in enumerate(zip(locs[:len(encs)], confs[:len(encs)])):
                    top, right, bottom, left = loc
                    # Add padding (20%)
                    h, w = bottom - top, right - left
                    pad_h, pad_w = int(h * 0.2), int(w * 0.2)
                    top = max(0, top - pad_h)
                    bottom = min(rgb.shape[0], bottom + pad_h)
                    left = max(0, left - pad_w)
                    right = min(rgb.shape[1], right + pad_w)
                    
                    face_crop = rgb[top:bottom, left:right]
                    face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                    
                    # Save with format: imgname_faceX_confYY.jpg
                    face_filename = f"{img.stem}_face{face_idx}_conf{int(conf*100):02d}.jpg"
                    cv2.imwrite(str(TRAINED_FACES_DIR / face_filename), face_bgr)
                    
                    # Store metadata for cluster assignment later
                    face_metadata.append({
                        "filename": face_filename,
                        "source_image": img.name,
                        "confidence": conf,
                        "embedding_idx": len(embeddings) + face_idx,
                    })
            
            embeddings.extend(encs)
            src_image_ids.extend([idx] * len(encs))
            encoded_faces += len(encs)
        except Exception as e:
            print(f"\n[WARN] {img}: {e}", file=sys.stderr)

    print("\n[INFO] Face extraction done.")
    if not embeddings:
        raise RuntimeError("No face encodings extracted.")

    X = np.array(embeddings, dtype=np.float32)
    src_image_ids = np.array(src_image_ids, dtype=np.int32)

    print("[INFO] Clustering (DBSCAN)...")
    clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="euclidean", n_jobs=-1).fit(X)
    labels = clustering.labels_

    unique_labels = [int(l) for l in np.unique(labels) if int(l) != -1]
    if not unique_labels:
        raise RuntimeError("No clusters found. Try increasing --eps or decreasing --min-cluster-size.")

    cluster_infos = []
    for cid in unique_labels:
        mask = labels == cid
        size = int(mask.sum())
        unique_imgs = len(set(src_image_ids[mask].tolist()))
        cluster_infos.append((cid, size, unique_imgs))

    filtered = [info for info in cluster_infos if info[2] >= min_unique_images]
    if not filtered:
        filtered = cluster_infos
    filtered.sort(key=lambda t: t[1], reverse=True)
    chosen = filtered[:top_k]

    centroids, thresholds, names, sizes, unique_imgs_list = [], [], [], [], []
    cluster_face_map = {}  # Map cluster_id -> list of face indices
    
    for i, (cid, size, uniq_imgs) in enumerate(chosen, start=1):
        mask = labels == cid
        members = X[mask]
        centroid = members.mean(axis=0)
        dists = face_recognition.face_distance(members, centroid)
        thr = min(float(np.quantile(dists, thresh_quantile) + thresh_margin), max_tolerance_cap)
        centroids.append(centroid)
        thresholds.append(thr)
        cluster_name = f"FAMILY_{i:02d}"
        names.append(cluster_name)
        sizes.append(size)
        unique_imgs_list.append(uniq_imgs)
        
        # Get face indices for this cluster
        face_indices = np.where(mask)[0].tolist()
        cluster_face_map[cluster_name] = face_indices
    
    # Organize faces into cluster folders
    if save_faces and face_metadata:
        print("[INFO] Organizing faces by cluster...")
        
        # Create cluster subfolders
        for cluster_name in names:
            cluster_dir = TRAINED_FACES_DIR / cluster_name
            cluster_dir.mkdir(exist_ok=True)
        
        # Create NOISE folder for unclustered faces
        noise_dir = TRAINED_FACES_DIR / "NOISE"
        noise_dir.mkdir(exist_ok=True)
        
        # Move faces to their cluster folders
        moved_count = 0
        for face_info in face_metadata:
            emb_idx = face_info["embedding_idx"]
            src_file = TRAINED_FACES_DIR / face_info["filename"]
            
            if not src_file.exists():
                continue
            
            # Find which cluster this face belongs to
            assigned_cluster = None
            if emb_idx < len(labels):
                face_label = labels[emb_idx]
                for cluster_name, indices in cluster_face_map.items():
                    if emb_idx in indices:
                        assigned_cluster = cluster_name
                        break
            
            # Move to cluster folder or NOISE
            if assigned_cluster:
                dst_dir = TRAINED_FACES_DIR / assigned_cluster
            else:
                dst_dir = noise_dir
            
            dst_file = dst_dir / face_info["filename"]
            shutil.move(str(src_file), str(dst_file))
            moved_count += 1
        
        print(f"[INFO] Organized {moved_count} faces into {len(names)} clusters + NOISE")

    DB_DIR.mkdir(exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump({
            "centroids": np.array(centroids, dtype=np.float32),
            "thresholds": np.array(thresholds, dtype=np.float32),
            "names": names, "sizes": sizes, "unique_images": unique_imgs_list,
            "meta": {"detector": "yolov8", "total_images": total_images,
                     "total_faces_detected": total_faces_detected, "total_faces_encoded": encoded_faces}
        }, f)

    print(f"\n[OK] Training complete. Saved: {DB_PATH}")
    for nm, sz, uq, thr in zip(names, sizes, unique_imgs_list, thresholds):
        print(f"  {nm}: {sz} faces | {uq} images | thr: {thr:.3f}")


def train_edge(eps=0.48, min_cluster_size=3, top_k=10, min_unique_images=2,
               thresh_quantile=DEFAULT_THRESH_QUANTILE, thresh_margin=0.03,
               max_tolerance_cap=0.52, num_jitters=1, max_faces_per_image=DEFAULT_MAX_FACES_PER_IMAGE,
               conf_threshold=0.15, progress_callback=None, quiet=False):
    if not TRAIN_EDGE_DIR.exists():
        TRAIN_EDGE_DIR.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"Created {TRAIN_EDGE_DIR} - add edge case images")

    images = list(iter_images(TRAIN_EDGE_DIR))
    total_images = len(images)
    if total_images == 0:
        raise RuntimeError("No images in train/edge/")

    if not quiet:
        print(f"[INFO] Training EDGE on {total_images} images (YOLO)", flush=True)
    _ = get_yolo_model()

    embeddings, src_image_ids = [], []
    encoded_faces = 0
    start_time = time.time()

    for idx, img in enumerate(images, start=1):
        if progress_callback and (idx % 5 == 0 or idx == total_images):
            progress_callback(idx, total_images, f"Processing {idx}/{total_images}")
        elif not quiet:
            eta = ((time.time() - start_time) / idx) * (total_images - idx)
            print(f"\r[{_progress_bar(idx, total_images)}] {idx}/{total_images} | ETA: {_format_eta(eta)}", end="", flush=True)

        try:
            rgb = load_image_rgb_fast(img, max_side=0)
            locs, _ = detect_faces_yolo(rgb, conf_threshold=conf_threshold, max_faces=max_faces_per_image)
            if not locs:
                continue
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=num_jitters)
            if not encs:
                continue
            embeddings.extend(encs)
            src_image_ids.extend([idx] * len(encs))
            encoded_faces += len(encs)
        except Exception as e:
            if not quiet:
                print(f"\n[WARN] {img}: {e}", file=sys.stderr)

    if not quiet:
        print("\n[INFO] Edge extraction done.", flush=True)
    if not embeddings:
        raise RuntimeError("No encodings from edge images.")

    X = np.array(embeddings, dtype=np.float32)
    src_image_ids = np.array(src_image_ids, dtype=np.int32)

    if total_images < 5 or len(embeddings) < min_cluster_size:
        centroids, thresholds, names = [], [], []
        for i, img_id in enumerate(list(set(src_image_ids.tolist()))[:top_k], start=1):
            members = X[src_image_ids == img_id]
            centroid = members.mean(axis=0)
            thr = min(float(np.quantile(face_recognition.face_distance(members, centroid), 0.95) + thresh_margin) if len(members) > 1 else max_tolerance_cap, max_tolerance_cap)
            centroids.append(centroid)
            thresholds.append(thr)
            names.append(f"EDGE_{i:02d}")
    else:
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="euclidean", n_jobs=1).fit(X)
        labels = clustering.labels_
        unique_labels = [int(l) for l in np.unique(labels) if int(l) != -1]
        
        if not unique_labels:
            centroids = list(X[:top_k])
            thresholds = [max_tolerance_cap] * len(centroids)
            names = [f"EDGE_{i:02d}" for i in range(1, len(centroids) + 1)]
        else:
            cluster_infos = [(cid, int((labels == cid).sum()), len(set(src_image_ids[labels == cid].tolist()))) for cid in unique_labels]
            filtered = [c for c in cluster_infos if c[2] >= min_unique_images] or cluster_infos
            filtered.sort(key=lambda t: t[1], reverse=True)
            
            centroids, thresholds, names = [], [], []
            for i, (cid, _, _) in enumerate(filtered[:top_k], start=1):
                members = X[labels == cid]
                centroid = members.mean(axis=0)
                thr = min(float(np.quantile(face_recognition.face_distance(members, centroid), thresh_quantile) + thresh_margin), max_tolerance_cap)
                centroids.append(centroid)
                thresholds.append(thr)
                names.append(f"EDGE_{i:02d}")

    if not centroids:
        raise RuntimeError("No edge clusters created")

    DB_DIR.mkdir(exist_ok=True)
    with open(EDGE_DB_PATH, "wb") as f:
        pickle.dump({
            "centroids": np.array(centroids, dtype=np.float32),
            "thresholds": np.array(thresholds, dtype=np.float32),
            "names": names,
            "meta": {"detector": "yolov8", "total_images": total_images, "total_faces_encoded": encoded_faces}
        }, f)

    if not quiet:
        print(f"\n[OK] Edge model saved: {EDGE_DB_PATH} ({len(names)} clusters)", flush=True)
    return len(names)


def _match_faces_to_db(face_encodings, db, tolerance=DEFAULT_TOLERANCE):
    centroids, thresholds, names = db["centroids"], db["thresholds"], db["names"]
    train_faces = db.get("train_faces", {})
    recognized, matches = 0, []
    for enc in face_encodings:
        dists = face_recognition.face_distance(centroids, enc)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        effective_thr = min(float(thresholds[best_idx]), float(tolerance))
        is_known = best_dist <= effective_thr
        if is_known:
            recognized += 1
        best_name = names[best_idx]
        m = {
            "best": best_name,
            "dist": round(best_dist, 4),
            "thr": round(effective_thr, 4),
            "known": is_known,
        }

        train_cluster = train_faces.get(best_name)
        if train_cluster and len(train_cluster.get("encodings", [])) > 0:
            train_dists = face_recognition.face_distance(train_cluster["encodings"], enc)
            train_idx = int(np.argmin(train_dists))
            m["train_face_file"] = train_cluster["files"][train_idx]
            m["train_face_dist"] = round(float(train_dists[train_idx]), 4)
            source_hint = train_cluster["source_hints"][train_idx]
            if source_hint:
                m["train_source_hint"] = source_hint

        matches.append(m)
    return recognized, matches


def _decide_family(recognized, total, decision=DEFAULT_DECISION):
    if decision == "any_known":
        return recognized >= 1
    elif decision == "all_known":
        return recognized == total
    elif decision == "majority_known":
        return recognized > (total / 2.0)
    raise ValueError("Invalid decision")


def _detect_and_encode_yolo(rgb, config):
    rgb_resized = resize_rgb_if_needed(rgb, config.max_side)
    locs, confs = detect_faces_yolo(rgb_resized, conf_threshold=config.conf_threshold)
    if not locs:
        return [], [], []
    encs = face_recognition.face_encodings(rgb_resized, locs, num_jitters=config.num_jitters)
    return locs, encs, confs


def _classify_with_config(rgb, db, config, decision=DEFAULT_DECISION):
    locs, encs, confs = _detect_and_encode_yolo(rgb, config)
    if not locs:
        return ClassifyResult(False, 0, 0, [], confidences=[], match_scores=[])
    if not encs:
        return ClassifyResult(False, len(locs), 0, [], confidences=confs, match_scores=[])
    recognized, matches = _match_faces_to_db(encs, db, config.tolerance)
    
    # Tính match_scores: similarity = 1 - distance (capped at 0)
    match_scores = [max(0, 1 - m["dist"]) for m in matches]
    
    return ClassifyResult(
        _decide_family(recognized, len(encs), decision), 
        len(encs), recognized, matches, 
        confidences=confs[:len(encs)],
        match_scores=match_scores
    )


def classify_image_two_tier(image, db, decision=DEFAULT_DECISION, fast_config=None, full_config=None):
    fast_config = fast_config or DetectionConfig.fast()
    full_config = full_config or DetectionConfig.full()
    rgb = load_image_rgb_fast(image, max_side=0)
    result = _classify_with_config(rgb, db, fast_config, decision)
    if result.family:
        result.mode_used, result.fast_passed = "fast", True
        return result
    result = _classify_with_config(rgb, db, full_config, decision)
    result.mode_used, result.fast_passed = "full", False
    return result


def classify_image_from_array_two_tier(rgb_array, db, decision=DEFAULT_DECISION, fast_config=None, full_config=None, conf_threshold=None):
    """
    Two-tier classification. Nếu conf_threshold được set, sẽ override config mặc định.
    """
    if conf_threshold is not None:
        # User set conf → dùng conf đó cho cả FAST và FULL
        fast_config = DetectionConfig(
            max_side=DEFAULT_FAST_MAX_SIDE,
            num_jitters=DEFAULT_FAST_JITTERS,
            tolerance=DEFAULT_FAST_TOLERANCE,
            conf_threshold=conf_threshold,
        )
        full_config = DetectionConfig(
            max_side=DEFAULT_FULL_MAX_SIDE,
            num_jitters=DEFAULT_FULL_JITTERS,
            tolerance=DEFAULT_FULL_TOLERANCE,
            conf_threshold=conf_threshold,
        )
    else:
        fast_config = fast_config or DetectionConfig.fast()
        full_config = full_config or DetectionConfig.full()
    
    result = _classify_with_config(rgb_array, db, fast_config, decision)
    if result.family:
        result.mode_used, result.fast_passed = "fast", True
        return result
    result = _classify_with_config(rgb_array, db, full_config, decision)
    result.mode_used, result.fast_passed = "full", False
    return result


def classify_image_three_tier(image, db, edge_db=None, decision=DEFAULT_DECISION, fast_config=None, full_config=None):
    fast_config = fast_config or DetectionConfig.fast()
    full_config = full_config or DetectionConfig.full()
    rgb = load_image_rgb_fast(image, max_side=0)
    
    result = _classify_with_config(rgb, db, fast_config, decision)
    if result.family:
        result.mode_used, result.fast_passed = "fast", True
        return result
    
    result = _classify_with_config(rgb, db, full_config, decision)
    if result.family:
        result.mode_used, result.fast_passed = "full", False
        return result
    
    if edge_db:
        edge_result = _classify_with_config(rgb, edge_db, full_config, decision)
        if edge_result.family:
            edge_result.mode_used, edge_result.fast_passed = "edge", False
            return edge_result
    
    result.mode_used, result.fast_passed = "full", False
    return result


def classify_image_from_array_three_tier(rgb_array, db, edge_db=None, decision=DEFAULT_DECISION, fast_config=None, full_config=None, conf_threshold=None):
    """
    Three-tier classification. Nếu conf_threshold được set, sẽ override config mặc định.
    """
    if conf_threshold is not None:
        fast_config = DetectionConfig(
            max_side=DEFAULT_FAST_MAX_SIDE,
            num_jitters=DEFAULT_FAST_JITTERS,
            tolerance=DEFAULT_FAST_TOLERANCE,
            conf_threshold=conf_threshold,
        )
        full_config = DetectionConfig(
            max_side=DEFAULT_FULL_MAX_SIDE,
            num_jitters=DEFAULT_FULL_JITTERS,
            tolerance=DEFAULT_FULL_TOLERANCE,
            conf_threshold=conf_threshold,
        )
    else:
        fast_config = fast_config or DetectionConfig.fast()
        full_config = full_config or DetectionConfig.full()
    
    result = _classify_with_config(rgb_array, db, fast_config, decision)
    if result.family:
        result.mode_used, result.fast_passed = "fast", True
        return result
    
    result = _classify_with_config(rgb_array, db, full_config, decision)
    if result.family:
        result.mode_used, result.fast_passed = "full", False
        return result
    
    if edge_db:
        edge_result = _classify_with_config(rgb_array, edge_db, full_config, decision)
        if edge_result.family:
            edge_result.mode_used, edge_result.fast_passed = "edge", False
            return edge_result
    
    result.mode_used, result.fast_passed = "full", False
    return result


def classify_image(image, db, tolerance=DEFAULT_TOLERANCE, decision=DEFAULT_DECISION,
                   max_side=DEFAULT_MAX_SIDE, num_jitters=DEFAULT_JITTERS_CHECK, conf_threshold=DEFAULT_CONF):
    config = DetectionConfig.single(max_side, num_jitters, tolerance, conf_threshold)
    rgb = load_image_rgb_fast(image, max_side=0)
    result = _classify_with_config(rgb, db, config, decision)
    return {"family": result.family, "faces": result.faces, "recognized": result.recognized,
            "matches": result.matches, "confidences": result.confidences}


def classify_image_from_array(rgb_array, db, tolerance=DEFAULT_TOLERANCE, decision=DEFAULT_DECISION,
                              max_side=DEFAULT_MAX_SIDE, num_jitters=DEFAULT_JITTERS_CHECK, conf_threshold=DEFAULT_CONF):
    config = DetectionConfig.single(max_side, num_jitters, tolerance, conf_threshold)
    result = _classify_with_config(rgb_array, db, config, decision)
    return {"family": result.family, "faces": result.faces, "recognized": result.recognized,
            "matches": result.matches, "confidences": result.confidences}


def batch_organize(folder, decision, tolerance, use_two_tier=False, use_edge=False, conf_threshold=DEFAULT_CONF):
    db = load_db()
    edge_db = load_edge_db() if use_edge else None
    if use_edge and not edge_db:
        print("[WARN] Edge DB not found.")

    family_dir, nonfamily_dir = folder / "family", folder / "non-family"
    images = [p for p in iter_images(folder) if family_dir not in p.parents and nonfamily_dir not in p.parents]
    
    if not images:
        print("[INFO] No images found.")
        return

    print(f"[INFO] Batch: {len(images)} images (YOLO)")
    _ = get_yolo_model()

    fam, non, fast_passed, edge_passed = 0, 0, 0, 0
    t0 = time.time()

    for i, p in enumerate(images, start=1):
        try:
            if use_two_tier:
                result = classify_image_three_tier(p, db, edge_db, decision) if use_edge and edge_db else classify_image_two_tier(p, db, decision)
                is_family = result.family
                if result.fast_passed:
                    fast_passed += 1
                elif result.mode_used == "edge":
                    edge_passed += 1
            else:
                is_family = classify_image(p, db, decision=decision, tolerance=tolerance, conf_threshold=conf_threshold)["family"]
            
            if is_family:
                safe_move(p, family_dir)
                fam += 1
            else:
                safe_move(p, nonfamily_dir)
                non += 1
        except Exception as e:
            print(f"\n[WARN] {p}: {e}", file=sys.stderr)
            safe_move(p, nonfamily_dir)
            non += 1

        if i % 10 == 0 or i == len(images):
            speed = i / (time.time() - t0)
            print(f"\r[INFO] {i}/{len(images)} | YES={fam} NO={non} | {speed:.1f} img/s", end="", flush=True)

    print(f"\n[OK] Done. YES: {fam} -> {family_dir} | NO: {non} -> {nonfamily_dir}")


def main():
    p = argparse.ArgumentParser(prog="family_photo_detector_yolo")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--eps", type=float, default=DEFAULT_EPS)
    t.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    t.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    t.add_argument("--conf", type=float, default=DEFAULT_CONF)
    t.add_argument("--save-faces", action="store_true", default=True, help="Save cropped faces to trained_faces/ (default: True)")
    t.add_argument("--no-save-faces", action="store_true", help="Don't save cropped faces")

    te = sub.add_parser("train-edge")
    te.add_argument("--eps", type=float, default=0.48)
    te.add_argument("--min-cluster-size", type=int, default=3)
    te.add_argument("--conf", type=float, default=0.15)

    c = sub.add_parser("check")
    c.add_argument("--image", required=True)
    c.add_argument("--decision", default=DEFAULT_DECISION)
    c.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    c.add_argument("--conf", type=float, default=DEFAULT_CONF)
    c.add_argument("--two-tier", action="store_true")
    c.add_argument("--use-edge", action="store_true")

    b = sub.add_parser("batch")
    b.add_argument("--folder", required=True)
    b.add_argument("--decision", default=DEFAULT_DECISION)
    b.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    b.add_argument("--conf", type=float, default=DEFAULT_CONF)
    b.add_argument("--two-tier", action="store_true")
    b.add_argument("--use-edge", action="store_true")

    args = p.parse_args()

    if args.cmd == "train":
        save_faces = not args.no_save_faces
        train_auto(eps=args.eps, min_cluster_size=args.min_cluster_size, top_k=args.top_k, 
                   conf_threshold=args.conf, save_faces=save_faces)
    elif args.cmd == "train-edge":
        train_edge(eps=args.eps, min_cluster_size=args.min_cluster_size, conf_threshold=args.conf)
    elif args.cmd == "check":
        db = load_db()
        edge_db = load_edge_db() if args.use_edge else None
        if args.two_tier or args.use_edge:
            result = classify_image_three_tier(Path(args.image), db, edge_db, args.decision) if edge_db else classify_image_two_tier(Path(args.image), db, args.decision)
            print(result.to_dict())
        else:
            print(classify_image(Path(args.image), db, args.tolerance, args.decision, conf_threshold=args.conf))
    elif args.cmd == "batch":
        batch_organize(Path(args.folder), args.decision, args.tolerance, args.two_tier, args.use_edge, args.conf)


if __name__ == "__main__":
    main()
