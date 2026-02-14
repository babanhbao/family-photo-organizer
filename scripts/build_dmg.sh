#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$ROOT_DIR/env" ]]; then
  echo "Missing virtualenv at $ROOT_DIR/env"
  exit 1
fi

if [[ ! -x "$ROOT_DIR/env/bin/pyinstaller" ]]; then
  echo "Installing pyinstaller into virtualenv..."
  "$ROOT_DIR/env/bin/pip" install pyinstaller
fi

if [[ ! -d "$ROOT_DIR/node_modules" ]]; then
  echo "Installing npm dependencies..."
  npm install
fi

echo "Preparing seed data..."
bash "$ROOT_DIR/scripts/prepare_desktop_seed.sh"

echo "Building DMG..."
npm run build:dmg

echo "Done. Check: $ROOT_DIR/release"
