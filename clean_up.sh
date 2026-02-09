#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DRY_RUN=0
ASSUME_YES=0
KILL_APP=0
KEEP_DAYS=0

usage() {
  cat <<EOF
Usage: ./$SCRIPT_NAME [options]

Clean temporary files created by family_photo_organizer app.

Options:
  --dry-run         Show what would be removed, do not delete
  --yes             Skip confirmation prompt
  --kill-app        Stop related Python processes before cleanup
  --keep-days N     Only remove job folders/files older than N days
  -h, --help        Show this help

Examples:
  ./$SCRIPT_NAME --dry-run
  ./$SCRIPT_NAME --kill-app --yes
  ./$SCRIPT_NAME --keep-days 2 --yes
EOF
}

is_non_negative_int() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

size_kb() {
  if [ -d "$1" ]; then
    du -sk "$1" 2>/dev/null | awk '{print $1+0}'
  else
    echo 0
  fi
}

confirm() {
  if [ "$ASSUME_YES" -eq 1 ]; then
    return 0
  fi
  printf "Proceed cleanup? [y/N] "
  read -r answer
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

safe_root_name() {
  case "$(basename "$1")" in
    family_organizer_yolo|family_photo_organizer_yolo) return 0 ;;
    *) return 1 ;;
  esac
}

delete_path() {
  local target="$1"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] rm -rf $target"
    return 0
  fi
  rm -rf "$target"
}

cleanup_root_full() {
  local root="$1"
  if ! safe_root_name "$root"; then
    echo "[SKIP] Unsafe root name: $root"
    return 0
  fi
  delete_path "$root"
}

cleanup_root_by_age() {
  local root="$1"
  local days="$2"
  if ! safe_root_name "$root"; then
    echo "[SKIP] Unsafe root name: $root"
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] find $root -mindepth 1 -maxdepth 1 -type d -name 'job_*' -mtime +$days -print"
    find "$root" -mindepth 1 -maxdepth 1 -type d -name 'job_*' -mtime "+$days" -print 2>/dev/null || true
    echo "[DRY] find $root -type f -name 'results.zip' -mtime +$days -print"
    find "$root" -type f -name 'results.zip' -mtime "+$days" -print 2>/dev/null || true
    return 0
  fi

  while IFS= read -r path; do
    rm -rf "$path"
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'job_*' -mtime "+$days" -print 2>/dev/null || true)

  while IFS= read -r path; do
    rm -f "$path"
  done < <(find "$root" -type f -name 'results.zip' -mtime "+$days" -print 2>/dev/null || true)

  find "$root" -type d -empty -delete 2>/dev/null || true
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --kill-app)
      KILL_APP=1
      shift
      ;;
    --keep-days)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --keep-days"
        exit 1
      fi
      KEEP_DAYS="$2"
      if ! is_non_negative_int "$KEEP_DAYS"; then
        echo "--keep-days must be a non-negative integer"
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [ "$KILL_APP" -eq 1 ]; then
  echo "Stopping related processes..."
  pkill -f app_yolo.py 2>/dev/null || true
  pkill -f family_photo_detector_yolo.py 2>/dev/null || true
fi

tmp_root="${TMPDIR:-/tmp}"
tmp_root="${tmp_root%/}"

CANDIDATES=(
  "$tmp_root/family_organizer_yolo"
  "$tmp_root/family_photo_organizer_yolo"
  "/tmp/family_organizer_yolo"
  "/tmp/family_photo_organizer_yolo"
  "/private/tmp/family_organizer_yolo"
  "/private/tmp/family_photo_organizer_yolo"
)

TARGETS=()
for d in "${CANDIDATES[@]}"; do
  [ -d "$d" ] || continue
  is_dup=0
  for t in "${TARGETS[@]:-}"; do
    if [ "$t" = "$d" ]; then
      is_dup=1
      break
    fi
  done
  if [ "$is_dup" -eq 0 ]; then
    TARGETS+=("$d")
  fi
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "No family organizer temp folders found."
  exit 0
fi

echo "Targets:"
total_before=0
for t in "${TARGETS[@]}"; do
  kb="$(size_kb "$t")"
  total_before=$((total_before + kb))
  echo "  - $t ($(du -sh "$t" 2>/dev/null | awk '{print $1}'))"
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[DRY] No files will be deleted."
fi

if ! confirm; then
  echo "Cancelled."
  exit 1
fi

for t in "${TARGETS[@]}"; do
  echo "Cleaning: $t"
  if [ "$KEEP_DAYS" -gt 0 ]; then
    cleanup_root_by_age "$t" "$KEEP_DAYS"
  else
    cleanup_root_full "$t"
  fi
done

total_after=0
for t in "${TARGETS[@]}"; do
  total_after=$((total_after + $(size_kb "$t")))
done

reclaimed_kb=$((total_before - total_after))
if [ "$reclaimed_kb" -lt 0 ]; then
  reclaimed_kb=0
fi

echo "Done."
echo "Approx reclaimed: $((reclaimed_kb / 1024)) MB"
