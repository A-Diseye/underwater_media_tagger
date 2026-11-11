---
paths:
  - "training/collect_inat.py"
  - "training/collect_commons.py"
  - "training/collect_scenery.py"
  - "training/workflow.py"
  - "tagger/search.py"
  - "api/**"
  - "app/search_page.py"
  - "tests/test_collect_inat.py"
  - "tests/test_collect_commons.py"
  - "tests/test_search.py"
---

# Agents

## 1. Training-data curation agent (`training/collect_inat.py`)

- Given per-category targets (40–100 images), it searches iNaturalist by taxon, downloads candidates, checks each with Claude vision (underwater? animal is the main subject? which category?), keeps or rejects it, tracks counts per class, and chooses which taxa to query next for classes that are short.
- Every keep/reject is logged with a reason to `data/curation/`. I hand-check a random sample of the log, and the README reports the filter's accuracy on that sample.
- Writes only to training/validation data, never to the test set.

## Divers and scenery: workflows, not agents (`training/workflow.py`)

- Where photos come from is fixed in code, so no model decides what to do next. Sources are taken in turn until the target; Haiku does the same check as above.
- Divers (`collect_commons.py`): Wikimedia Commons diver categories. An animal that's clearly a subject means it isn't a diver photo.
- Scenery (`collect_scenery.py`): Commons underwater scenes, iNaturalist corals and sponges (capped at 30%, mostly close-ups), and the agent's rejects that Haiku saw as underwater scenes with no main animal (capped at 40%, no new check needed). Kept only if underwater, category scenery, and no main animal.
- Per-uploader cap, train/val split by photographer or uploader, and one log and budget per category (`data/curation/<category>_log.jsonl`), so all collectors can run at once.
- Wikimedia asks for a contact in the User-Agent: the repo URL, never an email.

## 2. Library search agent (`tagger/search.py`)

- Answers natural-language questions about my tagged library ("turtles from the Azores where CLIP wasn't confident", "which trip had the most rays?") by calling `tagger` functions as tools (gallery query, trips, label and prediction counts).
- The tools live in the core package. A Streamlit chat page and an MCP server (under `api/`) are thin clients of them, so any agent client can query the library.
- Read-only: tools never change labels, predictions, or files.

## Shared rules

- Models: Claude Haiku 5.5 for bulk image checks, Claude Sonnet 5.5 for agent decisions (curation and search).
- API key from `ANTHROPIC_API_KEY`; never committed. Tests mock the API so CI needs no key.
- Agents never label, pre-fill, or suggest labels for test frames, and never see test-set metrics.
