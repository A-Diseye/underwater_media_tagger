---
paths:
  - "tagger/labeling.py"
  - "app/labeling_page.py"
  - "tests/test_labeling.py"
  - "training/evaluate.py"
---

# Test-set labeling

Test frames come from my own dive videos (~30–60 per category where the footage allows), spread across videos, and are labeled by hand on the Streamlit labeling page.

## Avoiding selection bias

- Frames come from a random sample, from browsing clips I remember, and from loose CLIP candidates (category anywhere in CLIP's top 3), never only CLIP's top-1 picks.
- Each label records its selection method: `random`, `browse`, or `clip_top3:<category>`. Report results on the random-only subset too.
- CLIP's guess is never shown while labeling.
- Auto mode picks the next frame: it counts my labels so far, targets the category furthest below quota (a random unlabeled frame from that category's top-3 pool), and mixes in pure random frames so I can't tell which pool a frame came from. A category stops being targeted when it hits quota or its pool runs out. Browsing clips is a separate manual mode.
- Uploads are never in the pool.

## Parameters

Quota 40 per category; 30% of auto-picked frames are pure random; skip frames within ±10 s of an already-labeled frame from the same clip (`unusable` labels don't block neighbors); stop targeting a category after 80 CLIP-candidate frames without reaching quota.

## Labeling rules (also shown on the page; keep both in sync)

1. An animal that is clearly a subject wins (ray + diver → `rays`; turtle over reef → `turtles`).
2. Several animals → the most prominent one (largest, closest, sharpest, or what the shot is about).
3. Background animals don't count: tiny scattered fish over a wide reef → `scenery`. Test: crop the fish out; if it still works as a scenery shot, it's `scenery`.
4. Diver as the main subject → `diver`, unless an animal is also a clear subject (rule 1).
5. A school of fish as the subject → `fish`.
6. Can't decide in ~3 seconds → Skip (not labeled).

Unusable = fogged lens, on the boat, at the surface, black frame.
