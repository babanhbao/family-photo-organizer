#!/usr/bin/env python3
"""
family_photo_detector.py

Two-Tier Detection System:
  - FAST mode: HOG + low jitters + smaller image → quick screening
  - FULL mode: CNN + higher jitters + larger image → accurate fallback

Flow:
  1. Run FAST detection on all images
  2. Images that PASS → done (family)
  3. Images that FAIL → send to retry pool for FULL detection
  4. This significantly speeds up processing when most images contain family

Fixed structure:
  ./train/family/        -> training images
  ./db/family_db.pkl     -> auto-generated DB

CLI:
  python family_photo_detector.py train
  python family_photo_detector.py check --image "/path/to/photo.jpg"
  python family_photo_detector.py batch --folder "/path/to/folder"

Install:
  pip install face_recognition opencv-python numpy scikit-learn
"""

import argparse
import math
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN


# ---------------- CONFIG ----------------
TRAIN_FAMILY_DIR = Path("train/family")
TRAIN_EDGE_DIR = Path("train/edge")  # Edge cases folder
DB_DIR = Path("db")
DB_PATH = DB_DIR / "family_db.pkl"
EDGE_DB_PATH = DB_DIR / "edge_db.pkl"  # Separate DB for edge cases

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Train defaults
DEFAULT_EPS = 0.42
DEFAULT_MIN_CLUSTER_SIZE = 12
DEFAULT_TOP_K = 8

# Cluster filtering
DEFAULT_MIN_UNIQUE_IMAGES = 8

# Thresholding
DEFAULT_THRESH_QUANTILE = 0.90
DEFAULT_THRESH_MARGIN = 0.02
DEFAULT_MAX_TOLERANCE_CAP = 0.50

# Face extraction (Training)
DEFAULT_FACE_MODEL_TRAIN = "hog"
DEFAULT_JITTERS_TRAIN = 1
DEFAULT_MAX_FACES_PER_IMAGE = 10

# ============ TWO-TIER DETECTION CONFIG ============
# FAST mode (for quick screening)
DEFAULT_FAST_MODEL = "hog"          # HOG is ~5-10x faster than CNN
DEFAULT_FAST_MAX_SIDE = 800         # Smaller image = faster
DEFAULT_FAST_JITTERS = 1            # Minimal jitters
DEFAULT_FAST_TOLERANCE = 0.42       # Slightly stricter to avoid false positives

# FULL mode (accurate fallback for failed images)
DEFAULT_FULL_MODEL = "cnn"          # CNN is more accurate
DEFAULT_FULL_MAX_SIDE = 1600        # Larger image for better accuracy
DEFAULT_FULL_JITTERS = 2            # More jitters for better encoding
DEFAULT_FULL_TOLERANCE = 0.45       # Standard tolerance

# Legacy single-mode defaults (backward compatible)
DEFAULT_FACE_MODEL_CHECK = "cnn"
DEFAULT_JITTERS_CHECK = 2
DEFAULT_MAX_SIDE = 1600
DEFAULT_TOLERANCE = 0.45
DEFAULT_DECISION = "majority_known"
# ---------------------------------------


class DetectionMode(Enum):
    FAST = "fast"
    FULL = "full"
    SINGLE = "single"  # Legacy single-pass mode


@dataclass
class DetectionConfig:
    """Configuration for a detection pass"""
    model: str
    max_side: int
    num_jitters: int
    tolerance: float
    
    @classmethod
    def fast(cls) -> 'DetectionConfig':
        return cls(
            model=DEFAULT_FAST_MODEL,
            max_side=DEFAULT_FAST_MAX_SIDE,
            num_jitters=DEFAULT_FAST_JITTERS,
            tolerance=DEFAULT_FAST_TOLERANCE,
        )
    
    @classmethod
    def full(cls) -> 'DetectionConfig':
        return cls(
            model=DEFAULT_FULL_MODEL,
            max_side=DEFAULT_FULL_MAX_SIDE,
            num_jitters=DEFAULT_FULL_JITTERS,
            tolerance=DEFAULT_FULL_TOLERANCE,
        )
    
    @classmethod
    def single(cls, model: str = DEFAULT_FACE_MODEL_CHECK, 
               max_side: int = DEFAULT_MAX_SIDE,
               num_jitters: int = DEFAULT_JITTERS_CHECK,
               tolerance: float = DEFAULT_TOLERANCE) -> 'DetectionConfig':
        return cls(model=model, max_side=max_side, 
                   num_jitters=num_jitters, tolerance=tolerance)


@dataclass
class ClassifyResult:
    """Result of classification"""
    family: bool
    faces: int
    recognized: int
    matches: List[Dict[str, Any]]
    mode_used: str = "single"  # Which mode produced this result
    fast_passed: bool = False  # Did it pass on fast mode?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "faces": self.faces,
            "recognized": self.recognized,
            "matches": self.matches,
            "mode_used": self.mode_used,
            "fast_passed": self.fast_passed,
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
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _progress_bar(current, total, width=28):
    ratio = current / total if total else 0.0
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def load_image_rgb_fast(path: Path, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    """
    Read image + downscale so the longer side <= max_side.
    Huge speedup for face detection/encoding on high-res photos.
    
    Returns: RGB numpy array (H, W, 3)
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")

    h, w = bgr.shape[:2]
    m = max(h, w)
    if max_side and m > max_side:
        scale = max_side / float(m)
        bgr = cv2.resize(
            bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb_if_needed(rgb: np.ndarray, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    """
    Resize RGB array if larger than max_side.
    Used for in-memory classification.
    
    Returns: RGB numpy array (H, W, 3)
    """
    if max_side <= 0:
        return rgb
    
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    
    scale = max_side / float(m)
    resized = cv2.resize(
        rgb,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def load_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}. Run: python family_photo_detector.py train")
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    for k in ("centroids", "names", "thresholds"):
        if k not in db:
            raise ValueError(f"Invalid DB: missing key '{k}'")

    db["centroids"] = np.array(db["centroids"], dtype=np.float32)
    db["thresholds"] = np.array(db["thresholds"], dtype=np.float32)
    return db


def load_edge_db():
    """Load edge cases DB. Returns None if not exists."""
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


# ---------------- TRAIN ----------------
def train_auto(
    eps=DEFAULT_EPS,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    top_k=DEFAULT_TOP_K,
    min_unique_images=DEFAULT_MIN_UNIQUE_IMAGES,
    thresh_quantile=DEFAULT_THRESH_QUANTILE,
    thresh_margin=DEFAULT_THRESH_MARGIN,
    max_tolerance_cap=DEFAULT_MAX_TOLERANCE_CAP,
    face_model=DEFAULT_FACE_MODEL_TRAIN,
    num_jitters=DEFAULT_JITTERS_TRAIN,
    max_faces_per_image=DEFAULT_MAX_FACES_PER_IMAGE,
):
    if not TRAIN_FAMILY_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {TRAIN_FAMILY_DIR}")

    images = list(iter_images(TRAIN_FAMILY_DIR))
    total_images = len(images)
    if total_images == 0:
        raise RuntimeError("No images found in train/family")

    print(f"[INFO] Training on {total_images} images from {TRAIN_FAMILY_DIR}\n")

    embeddings = []
    src_image_ids = []
    total_faces_detected = 0
    encoded_faces = 0

    start_time = time.time()

    for idx, img in enumerate(images, start=1):
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        eta = avg_time * (total_images - idx)

        bar = _progress_bar(idx, total_images)
        print(
            f"\r[{bar}] {idx}/{total_images} | faces(det): {total_faces_detected} | faces(enc): {encoded_faces} | ETA: {_format_eta(eta)}",
            end="",
            flush=True,
        )

        try:
            # For training: keep full res for better clustering
            rgb = load_image_rgb_fast(img, max_side=0)

            locs = face_recognition.face_locations(rgb, model=face_model)
            total_faces_detected += len(locs)

            locs = locs[:max_faces_per_image]
            if not locs:
                continue

            encs = face_recognition.face_encodings(
                rgb,
                known_face_locations=locs,
                num_jitters=num_jitters,
            )
            if not encs:
                continue

            embeddings.extend(encs)
            src_image_ids.extend([idx] * len(encs))
            encoded_faces += len(encs)

        except Exception as e:
            print(f"\n[WARN] {img}: {e}", file=sys.stderr)

    print("\n[INFO] Face extraction completed.")

    if not embeddings:
        raise RuntimeError("No face encodings extracted.")

    X = np.array(embeddings, dtype=np.float32)
    src_image_ids = np.array(src_image_ids, dtype=np.int32)

    print("[INFO] Clustering faces (DBSCAN)...")
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_cluster_size,
        metric="euclidean",
        n_jobs=-1,
    ).fit(X)

    labels = clustering.labels_

    unique_labels = [int(l) for l in np.unique(labels) if int(l) != -1]
    if not unique_labels:
        raise RuntimeError(
            "No valid clusters found.\n"
            "Try: increase --eps (e.g. 0.48~0.55) or decrease --min-cluster-size (e.g. 8~12)."
        )

    cluster_infos = []
    for cid in unique_labels:
        mask = labels == cid
        size = int(mask.sum())
        unique_imgs = int(len(set(src_image_ids[mask].tolist())))
        cluster_infos.append((cid, size, unique_imgs))

    filtered = [info for info in cluster_infos if info[2] >= min_unique_images]
    if not filtered:
        print(
            f"[WARN] No clusters passed min_unique_images={min_unique_images}. Falling back to unfiltered clusters.",
            file=sys.stderr,
        )
        filtered = cluster_infos

    filtered.sort(key=lambda t: t[1], reverse=True)
    chosen = filtered[:top_k]

    centroids = []
    thresholds = []
    names = []
    sizes = []
    unique_imgs_list = []

    for i, (cid, size, uniq_imgs) in enumerate(chosen, start=1):
        mask = labels == cid
        members = X[mask]
        centroid = members.mean(axis=0)

        dists = face_recognition.face_distance(members, centroid)
        thr = float(np.quantile(dists, thresh_quantile) + thresh_margin)
        thr = min(thr, float(max_tolerance_cap))

        centroids.append(centroid)
        thresholds.append(thr)
        names.append(f"FAMILY_{i:02d}")
        sizes.append(size)
        unique_imgs_list.append(uniq_imgs)

    DB_DIR.mkdir(exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(
            {
                "centroids": np.array(centroids, dtype=np.float32),
                "thresholds": np.array(thresholds, dtype=np.float32),
                "names": names,
                "sizes": sizes,
                "unique_images": unique_imgs_list,
                "meta": {
                    "train_folder": str(TRAIN_FAMILY_DIR),
                    "eps": eps,
                    "min_cluster_size": min_cluster_size,
                    "top_k": top_k,
                    "min_unique_images": min_unique_images,
                    "thresh_quantile": thresh_quantile,
                    "thresh_margin": thresh_margin,
                    "max_tolerance_cap": max_tolerance_cap,
                    "face_model": face_model,
                    "num_jitters": num_jitters,
                    "max_faces_per_image": max_faces_per_image,
                    "total_images": total_images,
                    "total_faces_detected": int(total_faces_detected),
                    "total_faces_encoded": int(encoded_faces),
                },
            },
            f,
        )

    print("\n[OK] Training complete")
    print(f"     Saved: {DB_PATH}")
    print("     Clusters (faces = detections, not unique people):")
    for nm, sz, uq, thr in zip(names, sizes, unique_imgs_list, thresholds):
        print(f"       {nm}: {sz} faces | images: {uq} | thr: {thr:.3f}")


# ---------------- TRAIN EDGE MODEL ----------------
def train_edge(
    eps=0.48,  # More lenient for edge cases
    min_cluster_size=3,  # Smaller clusters OK for edge cases
    top_k=10,
    min_unique_images=2,  # Lower threshold for edge cases
    thresh_quantile=DEFAULT_THRESH_QUANTILE,
    thresh_margin=0.03,  # Slightly more margin
    max_tolerance_cap=0.52,  # More lenient
    face_model="hog",  # HOG is 5-10x faster than CNN
    num_jitters=1,  # 1 jitter for speed (2 is overkill for edge cases)
    max_faces_per_image=DEFAULT_MAX_FACES_PER_IMAGE,
    progress_callback=None,  # Optional callback for UI progress
    quiet=False,  # Suppress console output when using callback
):
    """
    Train edge model from train/edge/ folder.
    This model handles edge cases that the main model misses.
    Uses HOG for faster training since edge cases typically have clearer faces.
    """
    if not TRAIN_EDGE_DIR.exists():
        TRAIN_EDGE_DIR.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"Created folder {TRAIN_EDGE_DIR} - please add edge case images")

    images = list(iter_images(TRAIN_EDGE_DIR))
    total_images = len(images)
    if total_images == 0:
        raise RuntimeError("No images found in train/edge/")

    if not quiet:
        print(f"[INFO] Training EDGE model on {total_images} images from {TRAIN_EDGE_DIR}", flush=True)

    embeddings = []
    src_image_ids = []
    total_faces_detected = 0
    encoded_faces = 0

    start_time = time.time()

    for idx, img in enumerate(images, start=1):
        elapsed = time.time() - start_time
        avg_time = elapsed / idx if idx > 0 else 0
        eta = avg_time * (total_images - idx)

        # Progress callback only every 5 images or at end (to avoid spam)
        if progress_callback and (idx % 5 == 0 or idx == total_images):
            progress_callback(idx, total_images, f"Processing {idx}/{total_images}")
        elif not quiet:
            bar = _progress_bar(idx, total_images)
            msg = f"[{bar}] {idx}/{total_images} | faces: {encoded_faces} | ETA: {_format_eta(eta)}"
            print(f"\r{msg}", end="", flush=True)

        try:
            rgb = load_image_rgb_fast(img, max_side=0)

            locs = face_recognition.face_locations(rgb, model=face_model)
            total_faces_detected += len(locs)

            locs = locs[:max_faces_per_image]
            if not locs:
                continue

            encs = face_recognition.face_encodings(
                rgb,
                known_face_locations=locs,
                num_jitters=num_jitters,
            )
            if not encs:
                continue

            embeddings.extend(encs)
            src_image_ids.extend([idx] * len(encs))
            encoded_faces += len(encs)

        except Exception as e:
            if not quiet:
                print(f"\n[WARN] {img}: {e}", file=sys.stderr)

    if not quiet:
        print("\n[INFO] Face extraction completed.", flush=True)

    if not embeddings:
        raise RuntimeError("No face encodings extracted from edge images.")

    X = np.array(embeddings, dtype=np.float32)
    src_image_ids = np.array(src_image_ids, dtype=np.int32)

    # For edge cases with few images, skip clustering and use all embeddings
    if total_images < 5 or len(embeddings) < min_cluster_size:
        if not quiet:
            print("[INFO] Few edge images - using direct embedding without clustering", flush=True)
        # Create one cluster per image essentially
        centroids = []
        thresholds = []
        names = []
        
        # Group by image
        unique_img_ids = list(set(src_image_ids.tolist()))
        for i, img_id in enumerate(unique_img_ids[:top_k], start=1):
            mask = src_image_ids == img_id
            members = X[mask]
            centroid = members.mean(axis=0)
            
            if len(members) > 1:
                dists = face_recognition.face_distance(members, centroid)
                thr = float(np.quantile(dists, 0.95) + thresh_margin)
            else:
                thr = max_tolerance_cap
            
            thr = min(thr, float(max_tolerance_cap))
            centroids.append(centroid)
            thresholds.append(thr)
            names.append(f"EDGE_{i:02d}")
    else:
        if not quiet:
            print("[INFO] Clustering edge faces (DBSCAN)...", flush=True)
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_cluster_size,
            metric="euclidean",
            n_jobs=1,  # Avoid multiprocessing issues
        ).fit(X)

        labels = clustering.labels_

        unique_labels = [int(l) for l in np.unique(labels) if int(l) != -1]
        
        # For edge cases, also include noise points as individual clusters
        if not unique_labels:
            if not quiet:
                print("[INFO] No clusters found, using all embeddings as individual entries", flush=True)
            centroids = list(X[:top_k])
            thresholds = [max_tolerance_cap] * len(centroids)
            names = [f"EDGE_{i:02d}" for i in range(1, len(centroids) + 1)]
        else:
            cluster_infos = []
            for cid in unique_labels:
                mask = labels == cid
                size = int(mask.sum())
                unique_imgs = int(len(set(src_image_ids[mask].tolist())))
                cluster_infos.append((cid, size, unique_imgs))

            filtered = [info for info in cluster_infos if info[2] >= min_unique_images]
            if not filtered:
                filtered = cluster_infos

            filtered.sort(key=lambda t: t[1], reverse=True)
            chosen = filtered[:top_k]

            centroids = []
            thresholds = []
            names = []

            for i, (cid, size, uniq_imgs) in enumerate(chosen, start=1):
                mask = labels == cid
                members = X[mask]
                centroid = members.mean(axis=0)

                dists = face_recognition.face_distance(members, centroid)
                thr = float(np.quantile(dists, thresh_quantile) + thresh_margin)
                thr = min(thr, float(max_tolerance_cap))

                centroids.append(centroid)
                thresholds.append(thr)
                names.append(f"EDGE_{i:02d}")

    if not centroids:
        raise RuntimeError("No edge clusters created")

    DB_DIR.mkdir(exist_ok=True)
    with open(EDGE_DB_PATH, "wb") as f:
        pickle.dump(
            {
                "centroids": np.array(centroids, dtype=np.float32),
                "thresholds": np.array(thresholds, dtype=np.float32),
                "names": names,
                "meta": {
                    "train_folder": str(TRAIN_EDGE_DIR),
                    "eps": eps,
                    "min_cluster_size": min_cluster_size,
                    "top_k": top_k,
                    "face_model": face_model,
                    "num_jitters": num_jitters,
                    "total_images": total_images,
                    "total_faces_encoded": int(encoded_faces),
                },
            },
            f,
        )

    if not quiet:
        print("\n[OK] Edge model training complete", flush=True)
        print(f"     Saved: {EDGE_DB_PATH}", flush=True)
        print(f"     Clusters: {len(names)}", flush=True)
        for nm, thr in zip(names, thresholds):
            print(f"       {nm}: thr={thr:.3f}", flush=True)
    
    return len(names)


# ---------------- CORE MATCHING LOGIC ----------------
def _match_faces_to_db(
    face_encodings: List[np.ndarray],
    db: Dict[str, Any],
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple:
    """
    Core matching logic - shared between all classify functions.
    
    Returns: (recognized_count, matches_list)
    """
    centroids = db["centroids"]
    thresholds = db["thresholds"]
    names = db["names"]
    
    recognized = 0
    matches = []
    global_cap = float(tolerance)

    for enc in face_encodings:
        dists = face_recognition.face_distance(centroids, enc)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        per_thr = float(thresholds[best_idx])
        effective_thr = min(per_thr, global_cap)

        is_known = best_dist <= effective_thr
        if is_known:
            recognized += 1

        matches.append(
            {
                "best": names[best_idx] if is_known else "UNKNOWN",
                "dist": round(best_dist, 4),
                "thr": round(effective_thr, 4),
                "known": is_known,
            }
        )
    
    return recognized, matches


def _decide_family(
    recognized: int,
    total: int,
    decision: str = DEFAULT_DECISION,
) -> bool:
    """
    Decision logic - shared between all classify functions.
    """
    if decision == "any_known":
        return recognized >= 1
    elif decision == "all_known":
        return recognized == total
    elif decision == "majority_known":
        return recognized > (total / 2.0)
    else:
        raise ValueError("Invalid decision. Use: any_known | majority_known | all_known")


# ---------------- CORE DETECTION (Internal) ----------------
def _detect_and_encode(
    rgb: np.ndarray,
    config: DetectionConfig,
) -> Tuple[List, List[np.ndarray]]:
    """
    Internal: Run face detection and encoding with given config.
    Returns: (face_locations, face_encodings)
    """
    # Resize
    rgb_resized = resize_rgb_if_needed(rgb, max_side=config.max_side)
    
    # Detect
    upsample = 1 if config.model == "cnn" else 0
    locs = face_recognition.face_locations(
        rgb_resized, 
        model=config.model, 
        number_of_times_to_upsample=upsample
    )
    
    if not locs:
        return [], []
    
    # Encode
    encs = face_recognition.face_encodings(
        rgb_resized, 
        locs, 
        num_jitters=config.num_jitters
    )
    
    return locs, encs


def _classify_with_config(
    rgb: np.ndarray,
    db: Dict[str, Any],
    config: DetectionConfig,
    decision: str = DEFAULT_DECISION,
) -> ClassifyResult:
    """
    Internal: Classify image with specific detection config.
    """
    locs, encs = _detect_and_encode(rgb, config)
    
    if not locs:
        return ClassifyResult(
            family=False, faces=0, recognized=0, matches=[]
        )
    
    if not encs:
        return ClassifyResult(
            family=False, faces=len(locs), recognized=0, matches=[]
        )
    
    recognized, matches = _match_faces_to_db(encs, db, config.tolerance)
    total = len(encs)
    is_family = _decide_family(recognized, total, decision)
    
    return ClassifyResult(
        family=is_family,
        faces=total,
        recognized=recognized,
        matches=matches,
    )


# ============ TWO-TIER CLASSIFICATION API ============

def classify_image_two_tier(
    image: Path,
    db: Dict[str, Any],
    decision: str = DEFAULT_DECISION,
    fast_config: Optional[DetectionConfig] = None,
    full_config: Optional[DetectionConfig] = None,
) -> ClassifyResult:
    """
    Two-tier classification from file path.
    
    1. Run FAST detection first
    2. If PASS → return immediately (family photo confirmed)
    3. If FAIL → run FULL detection for second chance
    
    This is optimal when majority of photos are expected to be family photos.
    """
    fast_config = fast_config or DetectionConfig.fast()
    full_config = full_config or DetectionConfig.full()
    
    # Load image once
    rgb = load_image_rgb_fast(image, max_side=0)  # Load full, resize later per config
    
    # FAST pass
    result = _classify_with_config(rgb, db, fast_config, decision)
    
    if result.family:
        # Fast pass succeeded
        result.mode_used = "fast"
        result.fast_passed = True
        return result
    
    # Fast pass failed, try FULL
    result = _classify_with_config(rgb, db, full_config, decision)
    result.mode_used = "full"
    result.fast_passed = False
    return result


def classify_image_from_array_two_tier(
    rgb_array: np.ndarray,
    db: Dict[str, Any],
    decision: str = DEFAULT_DECISION,
    fast_config: Optional[DetectionConfig] = None,
    full_config: Optional[DetectionConfig] = None,
) -> ClassifyResult:
    """
    Two-tier classification from numpy array (RGB format).
    
    1. Run FAST detection first
    2. If PASS → return immediately (family photo confirmed)
    3. If FAIL → run FULL detection for second chance
    """
    fast_config = fast_config or DetectionConfig.fast()
    full_config = full_config or DetectionConfig.full()
    
    # FAST pass
    result = _classify_with_config(rgb_array, db, fast_config, decision)
    
    if result.family:
        result.mode_used = "fast"
        result.fast_passed = True
        return result
    
    # Fast pass failed, try FULL
    result = _classify_with_config(rgb_array, db, full_config, decision)
    result.mode_used = "full"
    result.fast_passed = False
    return result


# ============ THREE-TIER CLASSIFICATION (with Edge Model) ============

def classify_image_three_tier(
    image: Path,
    db: Dict[str, Any],
    edge_db: Optional[Dict[str, Any]] = None,
    decision: str = DEFAULT_DECISION,
    fast_config: Optional[DetectionConfig] = None,
    full_config: Optional[DetectionConfig] = None,
) -> ClassifyResult:
    """
    Three-tier classification from file path.
    
    1. Run FAST detection first
    2. If PASS → return immediately
    3. If FAIL → run FULL detection
    4. If FAIL + edge_db exists → run EDGE detection as final fallback
    """
    fast_config = fast_config or DetectionConfig.fast()
    full_config = full_config or DetectionConfig.full()
    
    # Load image once
    rgb = load_image_rgb_fast(image, max_side=0)
    
    # FAST pass
    result = _classify_with_config(rgb, db, fast_config, decision)
    if result.family:
        result.mode_used = "fast"
        result.fast_passed = True
        return result
    
    # FULL pass
    result = _classify_with_config(rgb, db, full_config, decision)
    if result.family:
        result.mode_used = "full"
        result.fast_passed = False
        return result
    
    # EDGE pass (if edge_db available)
    if edge_db is not None:
        edge_result = _classify_with_config(rgb, edge_db, full_config, decision)
        if edge_result.family:
            edge_result.mode_used = "edge"
            edge_result.fast_passed = False
            return edge_result
    
    # All passes failed
    result.mode_used = "full"
    result.fast_passed = False
    return result


def classify_image_from_array_three_tier(
    rgb_array: np.ndarray,
    db: Dict[str, Any],
    edge_db: Optional[Dict[str, Any]] = None,
    decision: str = DEFAULT_DECISION,
    fast_config: Optional[DetectionConfig] = None,
    full_config: Optional[DetectionConfig] = None,
) -> ClassifyResult:
    """
    Three-tier classification from numpy array (RGB format).
    
    1. Run FAST detection first (HOG, main DB)
    2. If PASS → return immediately
    3. If FAIL → run FULL detection (CNN, main DB)
    4. If FAIL + edge_db exists → run EDGE detection (CNN, edge DB)
    
    Args:
        rgb_array: RGB numpy array
        db: Main family database
        edge_db: Optional edge cases database (from train_edge)
        decision: "any_known" | "majority_known" | "all_known"
    """
    fast_config = fast_config or DetectionConfig.fast()
    full_config = full_config or DetectionConfig.full()
    
    # FAST pass (main DB)
    result = _classify_with_config(rgb_array, db, fast_config, decision)
    if result.family:
        result.mode_used = "fast"
        result.fast_passed = True
        return result
    
    # FULL pass (main DB)  
    result = _classify_with_config(rgb_array, db, full_config, decision)
    if result.family:
        result.mode_used = "full"
        result.fast_passed = False
        return result
    
    # EDGE pass (edge DB, if available)
    if edge_db is not None:
        edge_result = _classify_with_config(rgb_array, edge_db, full_config, decision)
        if edge_result.family:
            edge_result.mode_used = "edge"
            edge_result.fast_passed = False
            return edge_result
    
    # All passes failed
    result.mode_used = "full"
    result.fast_passed = False
    return result


# ============ LEGACY SINGLE-PASS API (Backward Compatible) ============

def classify_image(
    image: Path,
    db,
    tolerance: float = DEFAULT_TOLERANCE,
    decision: str = DEFAULT_DECISION,
    face_model: str = DEFAULT_FACE_MODEL_CHECK,
    max_side: int = DEFAULT_MAX_SIDE,
    num_jitters: int = DEFAULT_JITTERS_CHECK,
) -> Dict[str, Any]:
    """
    Classify image from file path (single-pass, legacy API).
    
    Returns:
        Dict with keys: family, faces, recognized, matches
    """
    config = DetectionConfig.single(
        model=face_model,
        max_side=max_side,
        num_jitters=num_jitters,
        tolerance=tolerance,
    )
    
    rgb = load_image_rgb_fast(image, max_side=0)
    result = _classify_with_config(rgb, db, config, decision)
    result.mode_used = "single"
    
    return {
        "family": result.family,
        "faces": result.faces,
        "recognized": result.recognized,
        "matches": result.matches,
    }


def classify_image_from_array(
    rgb_array: np.ndarray,
    db,
    tolerance: float = DEFAULT_TOLERANCE,
    decision: str = DEFAULT_DECISION,
    face_model: str = DEFAULT_FACE_MODEL_CHECK,
    max_side: int = DEFAULT_MAX_SIDE,
    num_jitters: int = DEFAULT_JITTERS_CHECK,
) -> Dict[str, Any]:
    """
    Classify image from numpy array (single-pass, legacy API).
    
    Returns:
        Dict with keys: family, faces, recognized, matches
    """
    config = DetectionConfig.single(
        model=face_model,
        max_side=max_side,
        num_jitters=num_jitters,
        tolerance=tolerance,
    )
    
    result = _classify_with_config(rgb_array, db, config, decision)
    result.mode_used = "single"
    
    return {
        "family": result.family,
        "faces": result.faces,
        "recognized": result.recognized,
        "matches": result.matches,
    }


# ---------------- BATCH ORGANIZE ----------------
def batch_organize(
    folder: Path, 
    decision: str, 
    tolerance: float, 
    face_model: str,
    use_two_tier: bool = False,
    use_edge: bool = False,
):
    """
    Scan `folder` recursively.
    Move:
      family -> folder/family/
      non-family -> folder/non-family/
    
    Args:
        use_two_tier: If True, use two-tier detection (fast + full fallback)
        use_edge: If True, also use edge model as final fallback
    """
    db = load_db()
    edge_db = load_edge_db() if use_edge else None
    
    if use_edge and edge_db is None:
        print("[WARN] Edge model requested but db/edge_db.pkl not found. Run: python family_photo_detector.py train-edge")

    family_dir = folder / "family"
    nonfamily_dir = folder / "non-family"

    images = list(iter_images(folder))
    # ignore already-sorted outputs
    images = [p for p in images if family_dir not in p.parents and nonfamily_dir not in p.parents]

    total = len(images)
    if total == 0:
        print("[INFO] No images found.")
        return

    mode_str = "three-tier (fast→full→edge)" if (use_two_tier and use_edge and edge_db) else \
               "two-tier (fast→full)" if use_two_tier else f"single ({face_model})"
    print(f"[INFO] Batch scanning {total} images in: {folder}")
    print(f"[INFO] Detection mode: {mode_str}")

    fam = 0
    non = 0
    fast_passed = 0
    edge_passed = 0
    t0 = time.time()

    for i, p in enumerate(images, start=1):
        try:
            if use_two_tier:
                if use_edge and edge_db:
                    result = classify_image_three_tier(p, db, edge_db, decision=decision)
                else:
                    result = classify_image_two_tier(p, db, decision=decision)
                is_family = result.family
                if result.fast_passed:
                    fast_passed += 1
                elif result.mode_used == "edge":
                    edge_passed += 1
            else:
                res = classify_image(p, db, decision=decision, tolerance=tolerance, face_model=face_model)
                is_family = res["family"]
            
            if is_family:
                safe_move(p, family_dir)
                fam += 1
            else:
                safe_move(p, nonfamily_dir)
                non += 1

        except Exception as e:
            # treat errors as non-family
            print(f"[WARN] {p}: {e}", file=sys.stderr)
            safe_move(p, nonfamily_dir)
            non += 1

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            if use_two_tier:
                extra = f" | edge_pass={edge_passed}" if use_edge and edge_db else ""
                print(f"\r[INFO] {i}/{total} done | YES={fam} NO={non} | fast_pass={fast_passed}{extra} | elapsed={elapsed:.1f}s", end="", flush=True)
            else:
                print(f"\r[INFO] {i}/{total} done | YES={fam} NO={non} | elapsed={elapsed:.1f}s", end="", flush=True)

    print("\n[OK] Batch complete")
    print(f"     YES (family): {fam} -> {family_dir}")
    print(f"     NO  (non-family): {non} -> {nonfamily_dir}")
    if use_two_tier:
        print(f"     Fast-pass rate: {fast_passed}/{fam} ({100*fast_passed/max(fam,1):.1f}% of family photos)")


# ---------------- CLI ----------------
def build_parser():
    p = argparse.ArgumentParser(prog="family_photo_detector", description="Family photo detector (train/check/batch).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Train DB from ./train/family (auto DBSCAN clustering).")
    t.add_argument("--eps", type=float, default=DEFAULT_EPS)
    t.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    t.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    t.add_argument("--min-unique-images", type=int, default=DEFAULT_MIN_UNIQUE_IMAGES)
    t.add_argument("--train-model", choices=["hog", "cnn"], default=DEFAULT_FACE_MODEL_TRAIN)
    t.add_argument("--train-jitters", type=int, default=DEFAULT_JITTERS_TRAIN)
    t.add_argument("--max-faces-per-image", type=int, default=DEFAULT_MAX_FACES_PER_IMAGE)
    t.add_argument("--thresh-quantile", type=float, default=DEFAULT_THRESH_QUANTILE)
    t.add_argument("--thresh-margin", type=float, default=DEFAULT_THRESH_MARGIN)
    t.add_argument("--max-tolerance-cap", type=float, default=DEFAULT_MAX_TOLERANCE_CAP)

    # train-edge
    te = sub.add_parser("train-edge", help="Train EDGE model from ./train/edge for edge cases.")
    te.add_argument("--eps", type=float, default=0.48)
    te.add_argument("--min-cluster-size", type=int, default=3)
    te.add_argument("--top-k", type=int, default=10)
    te.add_argument("--train-model", choices=["hog", "cnn"], default="cnn")
    te.add_argument("--train-jitters", type=int, default=2)

    # check
    c = sub.add_parser("check", help="Check a single image against db/family_db.pkl")
    c.add_argument("--image", required=True, help="Path to image")
    c.add_argument("--model", choices=["hog", "cnn"], default=DEFAULT_FACE_MODEL_CHECK)
    c.add_argument("--decision", choices=["any_known", "majority_known", "all_known"], default=DEFAULT_DECISION)
    c.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    c.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    c.add_argument("--jitters", type=int, default=DEFAULT_JITTERS_CHECK)
    c.add_argument("--two-tier", action="store_true", help="Use two-tier detection (fast + full fallback)")
    c.add_argument("--use-edge", action="store_true", help="Also use edge model as final fallback")

    # batch
    b = sub.add_parser("batch", help="Scan a folder; move to folder/family and folder/non-family")
    b.add_argument("--folder", required=True, help="Folder path")
    b.add_argument("--model", choices=["hog", "cnn"], default=DEFAULT_FACE_MODEL_CHECK)
    b.add_argument("--decision", choices=["any_known", "majority_known", "all_known"], default=DEFAULT_DECISION)
    b.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    b.add_argument("--two-tier", action="store_true", help="Use two-tier detection (fast + full fallback)")
    b.add_argument("--use-edge", action="store_true", help="Also use edge model as final fallback")

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        train_auto(
            eps=args.eps,
            min_cluster_size=args.min_cluster_size,
            top_k=args.top_k,
            min_unique_images=args.min_unique_images,
            thresh_quantile=args.thresh_quantile,
            thresh_margin=args.thresh_margin,
            max_tolerance_cap=args.max_tolerance_cap,
            face_model=args.train_model,
            num_jitters=args.train_jitters,
            max_faces_per_image=args.max_faces_per_image,
        )

    elif args.cmd == "train-edge":
        train_edge(
            eps=args.eps,
            min_cluster_size=args.min_cluster_size,
            top_k=args.top_k,
            face_model=args.train_model,
            num_jitters=args.train_jitters,
        )

    elif args.cmd == "check":
        db = load_db()
        edge_db = load_edge_db() if args.use_edge else None
        p = Path(args.image)
        
        if args.two_tier or args.use_edge:
            if args.use_edge and edge_db:
                result = classify_image_three_tier(p, db, edge_db, decision=args.decision)
            else:
                result = classify_image_two_tier(p, db, decision=args.decision)
            print(result.to_dict())
        else:
            res = classify_image(
                p,
                db,
                tolerance=args.tolerance,
                decision=args.decision,
                face_model=args.model,
                max_side=args.max_side,
                num_jitters=args.jitters,
            )
            print(res)

    elif args.cmd == "batch":
        folder = Path(args.folder)
        if not folder.exists():
            raise FileNotFoundError(folder)
        batch_organize(
            folder=folder,
            decision=args.decision,
            tolerance=args.tolerance,
            face_model=args.model,
            use_two_tier=args.two_tier,
            use_edge=args.use_edge,
        )


if __name__ == "__main__":
    main()