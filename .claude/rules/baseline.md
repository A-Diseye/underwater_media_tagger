---
paths:
  - "tagger/baseline.py"
  - "tests/test_baseline.py"
---

# CLIP baseline

- The zero-shot prompts in `PROMPTS` are **locked**. Don't tune them: the baseline must stay a fair, untuned comparison point, and every image in `data/tagger.db` was scored with these prompts under `MODEL_ID`.
- Changing the model or prompts means a new `MODEL_ID` and re-tagging, never overwriting the existing predictions. Ask me first.
- Where CLIP fails (e.g. the lizardfish case), document it in the README instead of fixing the prompts.
