---
paths:
  - "tagger/baseline.py"
  - "tests/test_baseline.py"
---

# CLIP baseline

- The zero-shot prompts in `PROMPTS` are **locked**. Don't tune them: the baseline must stay a fair, untuned comparison point, and every image in `data/tagger.db` was scored with these prompts under `MODEL_ID`.
- Changing the model or prompts means a new `MODEL_ID` and re-tagging, never overwriting the existing predictions. Ask me first.
- The prompts changed once, on Nov 9, when sharks merged into `fish` (decided from label counts, not test metrics). The shark prompts were removed and the `fish` prompts left as they were: adding even one shark prompt to `fish` nearly halved its predictions. That run is `clip-vitb32-zeroshot-6cat`; the 7-category predictions stay in the database under the old `MODEL_ID`.
- Where CLIP fails (e.g. the lizardfish case), document it in the README instead of fixing the prompts.
