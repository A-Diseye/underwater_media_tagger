# Project plan

Timeline, tasks, and notes for the README. Day-to-day instructions for Claude Code live in `CLAUDE.md` and `.claude/rules/`; this file is the plan and the history behind them.

Goal: an interview-ready local demo by **Nov 23, 2026**.

## Timeline and tasks

### Week 1 — Foundations (Oct 29 – Nov 4) ✅
- [x] Repo setup: uv environment (CUDA PyTorch), ruff, pytest, basic tests
- [x] GitHub Actions CI: ruff + pytest on every push
- [x] `frames.py`: keyframes every 2s, drop clearly ruined frames (sharpest-tile Laplacian variance < 15), drop near-duplicates (thumbnail correlation ≥ 0.90 with last kept frame)
- [x] Extract frames from all my dive videos on `/mnt/d` into `data/frames/`
- [x] SQLite schema for images, labels, predictions
- [x] Streamlit: upload photos/videos, gallery view
- [x] Zero-shot CLIP baseline wired end to end (upload -> tag -> gallery)

### Week 2 — My own model (Nov 5 – Nov 13) ✅
- [x] `collect_inat.py` curation agent: iNaturalist images for the animal categories (100 per class), Claude vision filter, decision log
- [x] Diver photos from Wikimedia Commons and scenery from Commons + iNaturalist (`collect_commons.py`, `collect_scenery.py`); my own scenery frames are test-only
- [x] Hand-check of the filter: 98% of keeps, 93% of rejects correct (95 decisions)
- [x] Streamlit labeling page; hand-label test frames (592 usable)
- [x] Fine-tune ConvNeXt-Tiny with augmentation; depth-fade color cast chosen on validation (3 seeds each)
- [x] `evaluate.py`: v1 results in `docs/results/v1` (balanced accuracy CLIP 62%, fine-tuned 60%; fish and invertebrates called scenery)
- [ ] ~~FastAPI wrapper~~ cut

### Week 3 — Classifier v2 (Nov 14 – Nov 18)
- [x] Split labeled frames by clip into dev and locked holdout halves (stratified so rare categories land in both); holdout grown to 635 frames
- [x] Error analysis on the dev half only
- [x] v2 training data: ~200 wide-framing photos (agent `--wide`), plus zoom-out augmentation
- [x] Retrain, choose v2 on validation + dev; score once on the holdout half: v2 69% vs. CLIP 56% balanced accuracy (`docs/results/holdout-v2`)
- [x] `tagger/classify.py` wired into uploads; Streamlit: v2 labels with a CLIP switch, filter/search by tag, manual tag correction
- [ ] Search agent: `tagger/search.py` tools + agent loop, Streamlit chat page (MCP server if time)

### Week 4 — Ship (Nov 19 – Nov 23)
- [ ] Run the full app end to end locally; fix bugs
- [ ] (If time) CLIP + linear probe as a third row in the results table
- [ ] README: architecture, data, results (v1 and v2), confusion matrix, failure examples, tradeoffs
- [ ] Record a 2–3 minute demo video

### Free time (secondary)
- [ ] `locate.py`: Grad-CAM heatmap -> subject region
- [ ] `color.py`: classic correction (white balance, red-channel compensation), weighted to the subject region
- [ ] Streamlit: before/after view

## Success criteria

- Working local app plus demo video
- Fine-tuned model measured against the CLIP baseline on my own frames
- Before/after color correction on real dive frames
- README that supports a deep interview walkthrough

## Things to track for the README

- Commonly confused pairs: rays vs. flatfish on sand, distant rays vs. fish, camouflaged bottom fish vs. sandy `scenery`
- Accuracy gap between internet-image validation and my own footage
- GoPro vs. phone case study: accuracy on my GoPro frames vs. trip-mates' iPhone-in-housing photos from the same Bahamas dives (camera from source folder: `bahamas_shared` = phone, everything else = GoPro; filenames aren't reliable, e.g. `Group pic 1.jpg`). Practical question: is a better GoPro or a phone housing the better buy?
- How the blurry shark frame is classified (should be `fish`)
- Why sharks were merged into `fish`: 80 CLIP shark candidates shown during labeling and none were sharks, so the class couldn't be evaluated on my footage and every shark prediction would count as an error
- Cases where Grad-CAM misses the subject
- Curation agent: keep/reject counts per class, filter accuracy on my hand-checked sample, examples of good and bad rejections
- Search agent: example questions with the tool calls it made, and questions it got wrong
- Where agents are deliberately not used (test labels, tuning on test metrics) and why
- Why `no_subject` and `reef_scenery` were merged into `scenery`
- Frame filter calibration: whole-image Laplacian variance scored sharp mobula rays in open blue as low as fogged-lens frames, so sharpness uses the sharpest tile instead; pHash was too noisy in low-contrast water, so duplicates use thumbnail correlation (contact sheets in `data/calibration/`)
- Claude Code setup: hard rules enforced in `.claude/settings.json` and a hook, not just written down; instructions scoped with `.claude/rules/`

## Out of scope (until after Nov 23)

- React frontend
- Species-level identification, multi-label tagging
- Color-correcting video (frame-to-frame flicker)
- Stretch: dive log, learned color correction from before/after pairs, OBIS-based "where to see it" recommender
- Stretch: error-analysis agent that groups misclassified test frames into failure modes and drafts the README section for me to edit
- Not planned: agent-driven color correction, autonomous training/hyperparameter agent (test-set leakage risk, little benefit with one model)

## History

- **Nov 2, Philippines zip cleanup:** the Google Photos export had 43 duplicate entries and one broken 45 GB file (first 4 GB duplicated GH030348, rest zeros, no MP4 index). The cleaned zip (30 GB, 34 clips + phone photos) was verified byte-for-byte before the original was deleted.
- **Category merge:** `no_subject` and `reef_scenery` were merged into `scenery`.
- **Nov 9, sharks merged into `fish`:** 7 categories became 6, and CLIP was re-run under a new `MODEL_ID`.
