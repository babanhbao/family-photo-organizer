#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED_DIR="$ROOT_DIR/electron/seed"

mkdir -p "$SEED_DIR"

copy_dir_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    rm -rf "$dst"
    cp -R "$src" "$dst"
    echo "[seed] copied dir: $src -> $dst"
  fi
}

copy_file_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "[seed] copied file: $src -> $dst"
  fi
}

copy_dir_if_exists "$ROOT_DIR/db" "$SEED_DIR/db"
copy_dir_if_exists "$ROOT_DIR/trained_faces" "$SEED_DIR/trained_faces"
copy_file_if_exists "$ROOT_DIR/yolov8n-face.pt" "$SEED_DIR/yolov8n-face.pt"
copy_file_if_exists "$ROOT_DIR/yolov8n.pt" "$SEED_DIR/yolov8n.pt"

if [[ ! -d "$SEED_DIR/db" ]]; then
  mkdir -p "$SEED_DIR/db"
fi
if [[ ! -d "$SEED_DIR/trained_faces" ]]; then
  mkdir -p "$SEED_DIR/trained_faces"
fi

echo "[seed] ready at $SEED_DIR"
