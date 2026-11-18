# Test-set results: the **dev** half (used for error analysis and choosing fixes)

Hand-labeled frames from my own dives, `unusable` excluded: **277** frames, the same frames for every model. Balanced accuracy is the average of per-category recall, so every category counts equally.

| Model | Role | Balanced accuracy | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| `clip-vitb32-zeroshot-6cat` | zero-shot CLIP baseline | **65%** | 63% | 72% |
| `v3` | fine-tuned candidate | **77%** | 80% | 83% |
| `v2_both_s0` | fine-tuned candidate | **75%** | 65% | 75% |

## Recall per category (frames it rests on)

| Category | n | `clip-vitb32-zeroshot-6cat` | `v3` | `v2_both_s0` |
| --- | --- | --- | --- | --- |
| turtles | 3 | 67% | 100% | 100% |
| rays | 18 | 67% | 94% | 72% |
| fish | 28 | 50% | 36% | 39% |
| invertebrates | 9 | 56% | 56% | 78% |
| diver | 93 | 77% | 86% | 92% |
| scenery | 126 | 75% | 90% | 69% |

## Precision per category (tags given)

Of the frames each model tagged with a category, the share that truly were it. Low precision on a rare category means its real frames are buried among false tags.

| Category | true n | `clip-vitb32-zeroshot-6cat` | `v3` | `v2_both_s0` |
| --- | --- | --- | --- | --- |
| turtles | 3 | 50% (4) | 100% (3) | 50% (6) |
| rays | 18 | 48% (25) | 94% (18) | 68% (19) |
| fish | 28 | 40% (35) | 100% (10) | 50% (22) |
| invertebrates | 9 | 62% (8) | 71% (7) | 29% (24) |
| diver | 93 | 97% (74) | 88% (91) | 85% (101) |
| scenery | 126 | 73% (131) | 77% (148) | 83% (105) |

## Slices: balanced accuracy (frames)

| Slice | n | `clip-vitb32-zeroshot-6cat` | `v3` | `v2_both_s0` |
| --- | --- | --- | --- | --- |
| all | 277 | 65% | 77% | 75% |
| random-only | 110 | 51% | 74% | 79% |
| gopro | 258 | 69% | 75% | 74% |
| phone | 19 | 74% | 97% | 82% |
| trip: azores | 105 | 48% | 57% | 68% |
| trip: bahamas | 61 | 70% | 93% | 85% |
| trip: mexico | 34 | 39% | 62% | 57% |
| trip: philippines | 77 | 79% | 77% | 82% |

Confusion matrices: [`clip-vitb32-zeroshot-6cat`](confusion_clip-vitb32-zeroshot-6cat.png), [`v3`](confusion_v3.png), [`v2_both_s0`](confusion_v2_both_s0.png).
Frames the primary model got wrong: [failures.jpg](failures.jpg).
