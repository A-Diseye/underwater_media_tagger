# Underwater Media Tagger

A local app where divers upload underwater photos or dive videos. A computer vision model tags each image with one of 6 subject categories, and color correction is tuned to the subject's location. Portfolio project, interview-ready demo due **Nov 23, 2026**. Runs locally only; no public deployment.

Timeline, task list, and README notes: `docs/PLAN.md` (read it when planning or picking the next task). Topic rules in `.claude/rules/` load automatically when you touch the matching files.

## About me and how to work with me

- Backend/data engineer (Python, Go, TypeScript, AWS), new to PyTorch and computer vision. Explain ML choices briefly, and explain the why, not only the what. Goals: learn new skills and build a resume booster.
- Priorities, in order: working end to end > measurable model results > polish.
- Propose a design before writing code for anything new, and stop at checkpoints so I can review files.
- Never commit or push until I approve. Once approved, push directly to `main` in small, well-described commits.

## Commands

```bash
uv sync                                      # all groups: dev, ml (CUDA torch, open-clip), app
uv run pytest                                # tests
uv run ruff check . && uv run ruff format --check .
uv run streamlit run app/streamlit_app.py    # app at http://localhost:8501 (localhost only)
uv run python -m tagger.frames --source TRIP=FOLDER [--source ...]   # extract frames
uv run python -m tagger.baseline             # tag all stored images with zero-shot CLIP
```

- CI runs `uv sync --locked --no-group ml`, so tests that need torch/CLIP must skip via `pytest.importorskip`.
- Streamlit only hot-reloads the app script. After changing anything in `tagger/`, restart the server (`/restart-app`).

## Architecture

All ML, image, and storage logic lives in `tagger/`. `app/` and `api/` stay thin, so Streamlit can be swapped for React later without touching the core.

```
tagger/      core package, no UI code: frames, baseline (CLIP), classify, locate (Grad-CAM),
             color, store (SQLite), ingest (uploads), labeling, search (agent tools)
training/    collect_inat.py (curation agent), train.py, evaluate.py, notebooks/
api/         FastAPI wrapper + MCP server around tagger/
app/         Streamlit UI: gallery, upload, labeling, search pages
data/        gitignored: tagger.db, frames/<trip>/, uploads/, curation/, train/, val/
models/      gitignored weights
docs/        PLAN.md, README images
```

Stack: Python 3.12, uv, ruff (line length 100), pytest, PyTorch + `timm`, open-clip, OpenCV/PyAV, Streamlit, SQLite, Anthropic SDK. Training runs on a local RTX 3060 (12 GB) in WSL2.

## Hard rules

- **D: drive (`/mnt/d`) holds my only copies of the raw footage. Never modify, move, or delete anything there.** The only writable location is `D:\underwater_tagger_extracted\`. Never copy raw footage into the repo or WSL (C: is low on space). Enforced by `.claude/settings.json` and a hook.
- The test set is only media from my own dives: my GoPro footage plus trip-mates' photos and videos from the same dives. Never put internet images into it, and never train on trip-mates' media (same dives and animals as my test frames, so it would leak).
- Test labels come from me by hand. No model or agent labels, pre-fills, or suggests a label for a test frame.
- Tuning anything (models, prompts, agents) uses validation metrics only, never test-set metrics.
- Secrets (e.g. `ANTHROPIC_API_KEY`) come from the environment and are never committed.

## Key decisions (locked)

- **Labels:** single label per image, the most prominent subject; an animal beats scenery or diver. Frames that are fogged, on the boat, at the surface, or black are labeled `unusable` and excluded from evaluation.
- **Model:** fine-tune pretrained ConvNeXt-Tiny (`timm`); Grad-CAM works well on CNNs and color correction depends on it. Stretch: CLIP + linear probe.
- **Baseline:** zero-shot CLIP with text prompts per category.
- **Training data:** internet images only. Animals: iNaturalist, mapped to categories by taxonomy and gathered by a curation agent. Divers: Wikimedia Commons. Scenery: Commons underwater scenes, iNaturalist corals and sponges, and agent rejects with no main animal. Divers and scenery use fixed workflows, not agents (`training/workflow.py`).
- **Agents:** a training-data curation agent and a read-only library search agent (details in `.claude/rules/agents.md`).
- **Order:** classification first, then subject-aware color correction located by Grad-CAM (no bounding boxes).
- **Frontend:** Streamlit now, React after Nov 23. **Licensing:** personal use only.

## Categories (6)

| Category | Includes |
| --- | --- |
| turtles | All sea turtles |
| rays | Stingrays, eagle rays, mantas |
| fish | All other fish, including sharks and eels |
| invertebrates | Nudibranchs, crabs, shrimp, lobsters, sea stars, octopus, squid, cuttlefish, and clams or scallops that are clearly the subject |
| diver | People as the main subject |
| scenery | Coral, sponges, lone anemones, and reef, wide shots, open water, sand, blurry frames: anything with no main animal or diver |
