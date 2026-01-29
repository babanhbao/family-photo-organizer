#!/usr/bin/env python3
"""
family_photo_detector.py

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

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN


# ---------------- CONFIG ----------------
TRAIN_FAMILY_DIR = Path("train/family")
DB_DIR = Path("db")
DB_PATH = DB_DIR / "family_db.pkl"

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

# Face extraction
DEFAULT_FACE_MODEL_TRAIN = "hog"  # default for TRAIN if not specified
DEFAULT_JITTERS_TRAIN = 1
DEFAULT_MAX_FACES_PER_IMAGE = 10

# Check defaults
DEFAULT_FACE_MODEL_CHECK = "cnn"  # default check model = cnn
DEFAULT_JITTERS_CHECK = 2         # 2 is usually a good trade-off
DEFAULT_MAX_SIDE = 1600           # resize for speed during CHECK/CLASSIFY

DEFAULT_TOLERANCE = 0.45
DEFAULT_DECISION = "majority_known"
# ---------------------------------------


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


def load_image_rgb_fast(path: Path, max_side: int = DEFAULT_MAX_SIDE):
    """
    Read image + downscale so the longer side <= max_side.
    Huge speedup for face detection/encoding on high-res photos.
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


# ---------------- CLASSIFY ----------------
def classify_image(
    image: Path,
    db,
    tolerance=DEFAULT_TOLERANCE,
    decision=DEFAULT_DECISION,
    face_model=DEFAULT_FACE_MODEL_CHECK,
    max_side: int = DEFAULT_MAX_SIDE,
    num_jitters: int = DEFAULT_JITTERS_CHECK,
):
    """
    Matching logic:
    - load + resize (max_side)
    - face_locations with face_model (+ upsample)
    - face_encodings (num_jitters)
    - match to closest centroid if dist <= min(per-cluster-threshold, global_tolerance)
    """
    rgb = load_image_rgb_fast(image, max_side=max_side)

    upsample = 1 if face_model == "cnn" else 0
    locs = face_recognition.face_locations(rgb, model=face_model, number_of_times_to_upsample=upsample)

    if not locs:
        return {"family": False, "faces": 0, "recognized": 0, "matches": []}

    encs = face_recognition.face_encodings(rgb, locs, num_jitters=num_jitters)

    centroids = db["centroids"]
    thresholds = db["thresholds"]
    names = db["names"]

    recognized = 0
    matches = []
    global_cap = float(tolerance)

    for enc in encs:
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

    total = len(encs)

    if decision == "any_known":
        is_family = recognized >= 1
    elif decision == "all_known":
        is_family = recognized == total
    elif decision == "majority_known":
        is_family = recognized > (total / 2.0)
    else:
        raise ValueError("Invalid decision. Use: any_known | majority_known | all_known")

    return {"family": is_family, "faces": total, "recognized": recognized, "matches": matches}


# ---------------- BATCH ORGANIZE ----------------
def batch_organize(folder: Path, decision: str, tolerance: float, face_model: str):
    """
    Scan `folder` recursively.
    Move:
      family -> folder/family/
      non-family -> folder/non-family/
    """
    db = load_db()

    family_dir = folder / "family"
    nonfamily_dir = folder / "non-family"

    images = list(iter_images(folder))
    # ignore already-sorted outputs
    images = [p for p in images if family_dir not in p.parents and nonfamily_dir not in p.parents]

    total = len(images)
    if total == 0:
        print("[INFO] No images found.")
        return

    print(f"[INFO] Batch scanning {total} images in: {folder}")

    fam = 0
    non = 0
    t0 = time.time()

    for i, p in enumerate(images, start=1):
        try:
            res = classify_image(p, db, decision=decision, tolerance=tolerance, face_model=face_model)
            if res["family"]:
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
            print(f"\r[INFO] {i}/{total} done | YES={fam} NO={non} | elapsed={elapsed:.1f}s", end="", flush=True)

    print("\n[OK] Batch complete")
    print(f"     YES (family): {fam} -> {family_dir}")
    print(f"     NO  (non-family): {non} -> {nonfamily_dir}")


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

    # check
    c = sub.add_parser("check", help="Check a single image against db/family_db.pkl")
    c.add_argument("--image", required=True, help="Path to image")
    c.add_argument("--model", choices=["hog", "cnn"], default=DEFAULT_FACE_MODEL_CHECK)
    c.add_argument("--decision", choices=["any_known", "majority_known", "all_known"], default=DEFAULT_DECISION)
    c.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    c.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    c.add_argument("--jitters", type=int, default=DEFAULT_JITTERS_CHECK)

    # batch
    b = sub.add_parser("batch", help="Scan a folder; move to folder/family and folder/non-family")
    b.add_argument("--folder", required=True, help="Folder path")
    b.add_argument("--model", choices=["hog", "cnn"], default=DEFAULT_FACE_MODEL_CHECK)
    b.add_argument("--decision", choices=["any_known", "majority_known", "all_known"], default=DEFAULT_DECISION)
    b.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)

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

    elif args.cmd == "check":
        db = load_db()
        p = Path(args.image)
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
        )


if __name__ == "__main__":
    main()
