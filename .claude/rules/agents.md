---
paths:
  - "training/collect_inat.py"
  - "tagger/search.py"
  - "api/**"
  - "app/search_page.py"
  - "tests/test_collect_inat.py"
  - "tests/test_search.py"
---

# Agents

## 1. Training-data curation agent (`training/collect_inat.py`)

- Given per-category targets (40–100 images), it searches iNaturalist by taxon, downloads candidates, checks each with Claude vision (underwater? animal is the main subject? which category?), keeps or rejects it, tracks counts per class, and chooses which taxa to query next for classes that are short.
- Every keep/reject is logged with a reason to `data/curation/`. I hand-check a random sample of the log, and the README reports the filter's accuracy on that sample.
- Writes only to training/validation data, never to the test set.

## 2. Library search agent (`tagger/search.py`)

- Answers natural-language questions about my tagged library ("turtles from the Azores where CLIP wasn't confident", "which trip had the most rays?") by calling `tagger` functions as tools (gallery query, trips, label and prediction counts).
- The tools live in the core package. A Streamlit chat page and an MCP server (under `api/`) are thin clients of them, so any agent client can query the library.
- Read-only: tools never change labels, predictions, or files.

## Shared rules

- Models: Claude Haiku 5.5 for bulk image checks, Claude Sonnet 5.5 for agent decisions (curation and search).
- API key from `ANTHROPIC_API_KEY`; never committed. Tests mock the API so CI needs no key.
- Agents never label, pre-fill, or suggest labels for test frames, and never see test-set metrics.
