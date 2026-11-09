---
paths:
  - "tagger/frames.py"
  - "tests/test_frames.py"
---

# Raw footage and frame extraction

- Raw footage stays on the external drive `D:\` (`/mnt/d` in WSL). `frames.py` reads from the drive and writes only kept frames to `data/frames/`. Never write to D: except `D:\underwater_tagger_extracted\`.
- If `/mnt/d` is empty, the drive isn't mounted: ask me to run `sudo mount -a`.
- Sources are GoPro files only (names starting with `G`; `PXL_*` and date-named files are phone media):
  - `Videos\Azores_scuba`, `Videos\Bahamas diving`, `Videos\Mexico`, `Videos\Philippines diving`
  - `D:\underwater_tagger_extracted\philippines_gopro\`: 31 unique clips extracted from `Pictures\PhilippinesTaiwanPhotos.zip`, verified against zip CRCs
  - Skip `Azores_scuba\Mobula Ray.zip` (duplicates the `Mobula Ray` folder).
  - `D:\underwater_tagger_extracted\bahamas_shared\`: trip-mates' media from `Videos\Bahamas diving\Winter 2024 - Bahamas-*.zip` (3 Google Drive parts, CRC-verified on extraction). Only the `Day 1`–`Day 3` dive folders are used, via `--any-camera-source` (iPhone 15 Pro in a SportDiver housing: ~451 photos, 7 short videos). `Misc Pics and Vids` is all topside and is skipped.
  - ~7.2 hours of unique GoPro video plus ~94 GoPro JPGs.
- `--source` folders take GoPro files only (my own phone media sits in the same folders). `--any-camera-source` takes every photo and video and is only for folders of other people's dive media.
- Filter settings (calibrated, see `docs/PLAN.md`): keyframes every 2 s; drop ruined frames where sharpest-tile Laplacian variance < 15; drop near-duplicates with thumbnail correlation ≥ 0.90 against the last kept frame. Photos in one folder are compared the same way in name order, so phone bursts count as duplicates.
- Phone media needs rotation: EXIF orientation for JPEG/HEIC, display rotation for video. HEIC loads through Pillow + `pillow-heif`.
