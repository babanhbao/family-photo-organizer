# Build macOS DMG (Desktop App)

## 1) Prerequisites
- macOS
- Node.js + npm
- Python venv at `./env` (already used by this project)

## 2) One-command build
```bash
cd /Users/kevinvo/Documents/family_photo_organizer
bash ./scripts/build_dmg.sh
```

Output DMG:
- `release/FamilyPhotoOrganizer-<version>-<arch>.dmg`

## 3) Dev desktop run (Electron shell + local python backend)
```bash
cd /Users/kevinvo/Documents/family_photo_organizer
npm install
npm run electron:dev
```

## Notes
- The build script also snapshots current local data (if present) into `electron/seed`:
  - `db/`
  - `trained_faces/`
  - `yolov8n-face.pt`, `yolov8n.pt`
- Packaged app stores runtime data in Electron `userData/data` via `FPO_APP_DATA_DIR`.
