# Underwater Media Tagger — Project Plan

Portfolio project: a local app where divers upload underwater photos or dive videos, a computer vision model tags each image with one of 7 subject categories, and color correction is tuned to the subject's location. The goal is an interview-ready demo by **Nov 23, 2026**. It runs locally; there is no public deployment.

## Context for Claude Code

- Developer background: backend/data engineer (Python, Go, TypeScript, AWS). New to PyTorch and computer vision, so explain ML choices briefly when making them.
- Main goals: learn new skills and build a resume booster. Explain the why, not only the what.
- Priorities, in order: working end to end > measurable model results > polish.
- Keep all ML and image logic in the core package. The UI and API stay thin so Streamlit can later be swapped for React without touching the core.
- Never put internet training images into the test set. The test set is only frames from my own dive videos.
- Test set labels come from me by hand, never from an AI model or agent.
- Workflow: make changes locally and stop at checkpoints so I can review files. Never commit or push until I approve. Once approved, push directly to `main` in small, well-described commits.

## Decisions (locked)

- **Labels:** single label per image. Tag the most prominent subject; an animal beats scenery or diver.
- **Model:** fine-tune a pretrained ConvNeXt-Tiny (via `timm`). Grad-CAM works well on CNNs, and color correction depends on it.
- **Stretch model:** CLIP + linear probe (logistic regression on frozen CLIP embeddings), only if time allows after the main model.
- **Baseline:** zero-shot CLIP with text prompts per category.
- **Training data:** internet images. Mostly iNaturalist (API), with species mapped to categories by taxonomy. I source a scenery dataset separately. Non-underwater photos get filtered out; an AI/agent filter is fine for training data.
- **Test data:** frames extracted from my own dive videos (~30–60 per category where the footage allows) spread across videos, labeled by hand in a Streamlit labeling page.
  - Avoid selection bias: frames come from a random sample, browsing clips I remember, and loose CLIP candidates (category anywhere in CLIP's top 3), never only CLIP's top-1 picks. Each label records its selection method. CLIP's guess is hidden while labeling. Report results on the random-only subset too.
  - The labeling page picks the next frame automatically: it counts my labels so far, targets the category furthest below its quota (random unlabeled frame from that category's top-3 pool), and mixes in pure random frames so I can't tell which pool a frame came from. A category stops being targeted when it hits quota or its pool runs out. Browsing clips is a separate manual mode.
- **Raw footage:** stays on the external drive `D:\` (`/mnt/d` in WSL). Never copy it into the repo or WSL. `frames.py` reads from the drive and writes only kept frames to `data/`. C: has limited space and the WSL disk lives on C:.
  - The originals on D: are my only copies: **never modify, move, or delete anything there.** Only write to `D:\underwater_tagger_extracted\`.
  - Sources (GoPro files only, names starting with `G`; `PXL_*` and date-named files are phone media): `Videos\Azores_scuba`, `Videos\Bahamas diving`, `Videos\Mexico`, `Videos\Philippines diving`, plus `D:\underwater_tagger_extracted\philippines_gopro\` (31 unique clips extracted from `Pictures\PhilippinesTaiwanPhotos.zip`, verified against zip CRCs). `Azores_scuba\Mobula Ray.zip` duplicates the `Mobula Ray` folder and is skipped. ~7.2 hours of unique video plus ~94 GoPro JPGs.
  - The Philippines zip was cleaned up on Nov 2: the Google Photos export had 43 duplicate entries and one broken 45 GB file (first 4 GB duplicated GH030348, rest zeros, no MP4 index). The cleaned zip (30 GB, 34 clips + phone photos) was verified byte-for-byte before the original was deleted.
- **Unusable frames** (fogged lens, on the boat, at the surface, black) are skipped during labeling, not labeled `scenery`.
- **Order:** classification first, then subject-aware color correction built on it.
- **Subject location:** Grad-CAM attention maps from the classifier (no bounding-box labeling).
- **Frontend:** Streamlit now, React after Nov 23.
- **Licensing:** not a concern; personal use only.

## Categories (7)

| Category | Includes |
| --- | --- |
| turtles | All sea turtles |
| sharks | All sharks (my footage has only 1 blurry shark, so evaluate on held-out internet images) |
| rays | Stingrays, eagle rays, mantas |
| fish | All other fish, including eels |
| invertebrates | Nudibranchs, crabs, shrimp, lobsters, sea stars, octopus, squid, cuttlefish |
| diver | People as the main subject |
| scenery | Coral and reef, wide shots, open water, sand, blurry frames: anything with no main animal or diver |

## Tech stack

| Layer | Choice |
| --- | --- |
| Environment | Python 3.12, uv, ruff, pytest, GitHub Actions CI (lint + tests on push) |
| Core package | Python (`tagger/`) |
| Model | PyTorch + `timm`, fine-tuning ConvNeXt-Tiny |
| Baseline | Zero-shot CLIP with text prompts per category |
| Explainability | Grad-CAM (e.g., `pytorch-grad-cam`) |
| Image/video processing | OpenCV (ffmpeg if needed for codecs) |
| API | FastAPI (skip if behind schedule) |
| UI | Streamlit |
| Storage | Local files + SQLite |
| Training compute | Local GPU: NVIDIA RTX 3060 (12 GB) in WSL2 |

## Repo layout

```
underwater_media_tagger/
├── tagger/                 # core package: no UI code
│   ├── frames.py           # video -> sharp, de-duplicated frames
│   ├── classify.py         # load model, predict label + confidence
│   ├── baseline.py         # zero-shot CLIP baseline
│   ├── locate.py           # Grad-CAM subject heatmap / region
│   ├── color.py            # classic + subject-aware color correction
│   └── store.py            # SQLite: images, labels, predictions
├── training/
│   ├── collect_inat.py     # download iNaturalist images, map taxonomy -> category
│   ├── train.py            # fine-tuning script (local GPU)
│   ├── evaluate.py         # accuracy, per-class P/R, confusion matrix
│   └── notebooks/
├── api/                    # FastAPI wrapper around tagger/
├── app/                    # Streamlit UI (calls api/ or tagger/), incl. labeling page
├── data/                   # gitignored
│   ├── frames/             # all kept frames extracted from my footage
│   ├── train/<category>/
│   ├── val/<category>/
│   └── test/<category>/    # my own video frames only
├── models/                 # gitignored weights
├── tests/
├── .github/workflows/      # CI
└── README.md
```

## Timeline and tasks

### Week 1 — Foundations (Oct 29 – Nov 4)
- [ ] Repo setup: uv environment (CUDA PyTorch), ruff, pytest, basic tests
- [ ] GitHub Actions CI: ruff + pytest on every push
- [ ] `frames.py`: keyframes every 2s, drop clearly ruined frames (sharpest-tile Laplacian variance < 15), drop near-duplicates (thumbnail correlation ≥ 0.90 with last kept frame)
- [ ] Extract frames from all my dive videos on `/mnt/d` into `data/frames/`
- [ ] SQLite schema for images, labels, predictions
- [ ] Streamlit: upload photos/videos, gallery view
- [ ] Zero-shot CLIP baseline wired end to end (upload -> tag -> gallery)

### Week 2 — My own model (Nov 5 – Nov 11)
- [ ] `collect_inat.py`: iNaturalist images for the animal categories (target ~40–100 per class), filter out non-underwater photos
- [ ] Add my scenery dataset; source diver images
- [ ] Streamlit labeling page (one frame, 7 category buttons plus "unusable", keyboard shortcuts); hand-label test frames into `data/test/`
- [ ] Fine-tune ConvNeXt-Tiny locally with augmentation (color shift, blur)
- [ ] `evaluate.py`: baseline vs. fine-tuned on the test set, confusion matrix
- [ ] FastAPI wrapper around `tagger/`
- [ ] **Checkpoint (~Nov 11):** ugly but complete demo, upload -> tag -> gallery, with both models compared

### Week 3 — Color correction and search (Nov 12 – Nov 18)
- [ ] `locate.py`: Grad-CAM heatmap -> subject region
- [ ] `color.py`: classic correction (white balance, red-channel compensation), weighted to the subject region; fall back to whole image if the heatmap is weak or the label is `scenery`
- [ ] Streamlit: before/after view, filter/search by tag, manual tag correction
- [ ] Timebox color correction to ~4 days; "visibly better than the original" is the bar

### Week 4 — Ship (Nov 19 – Nov 23)
- [ ] Run the full app end to end locally; fix bugs
- [ ] (If time) CLIP + linear probe as a third row in the results table
- [ ] README: architecture, data, results table, confusion matrix, failure examples, tradeoffs
- [ ] Record a 2–3 minute demo video

## Success criteria

- Working local app plus demo video
- Fine-tuned model measured against the CLIP baseline on my own frames
- Before/after color correction on real dive frames
- README that supports a deep interview walkthrough

## Things to track for the README

- Commonly confused pairs: sharks vs. fish, rays vs. flatfish on sand, rays vs. sharks when distant, camouflaged bottom fish vs. sandy `scenery`
- Accuracy gap between internet-image validation and my own footage
- How the blurry shark frame is classified
- Cases where Grad-CAM misses the subject
- Why `no_subject` and `reef_scenery` were merged into `scenery`
- Frame filter calibration: whole-image Laplacian variance scored sharp mobula rays in open blue as low as fogged-lens frames, so sharpness uses the sharpest tile instead; pHash was too noisy in low-contrast water, so duplicates use thumbnail correlation (contact sheets in `data/calibration/`)

## Out of scope (until after Nov 23)

- React frontend
- Species-level identification, multi-label tagging
- Color-correcting video (frame-to-frame flicker)
- Stretch: dive log, learned color correction from before/after pairs, OBIS-based "where to see it" recommender
