# Test-set results: the **holdout** half, *second look* (already scored once for v2, so this number is less clean than v2's was)

Hand-labeled frames from my own dives, `unusable` excluded: **640** frames, the same frames for every model. Balanced accuracy is the average of per-category recall, so every category counts equally.

| Model | Role | Balanced accuracy | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| `clip-vitb32-zeroshot-6cat` | zero-shot CLIP baseline | **56%** | 53% | 74% |
| `v3` | fine-tuned (chosen) | **67%** | 68% | 81% |
| `v2_both_s0` | fine-tuned (comparison, didn't affect the choice) | **69%** | 55% | 74% |

## Recall per category (frames it rests on)

| Category | n | `clip-vitb32-zeroshot-6cat` | `v3` | `v2_both_s0` |
| --- | --- | --- | --- | --- |
| turtles | 11 | 55% | 73% | 64% |
| rays | 19 | 53% | 68% | 84% |
| fish | 56 | 29% | 18% | 27% |
| invertebrates | 14 | 43% | 64% | 79% |
| diver | 235 | 75% | 89% | 94% |
| scenery | 305 | 84% | 88% | 67% |

## Precision per category (tags given)

Of the frames each model tagged with a category, the share that truly were it. Low precision on a rare category means its real frames are buried among false tags.

| Category | true n | `clip-vitb32-zeroshot-6cat` | `v3` | `v2_both_s0` |
| --- | --- | --- | --- | --- |
| turtles | 11 | 40% (15) | 67% (12) | 32% (22) |
| rays | 19 | 26% (38) | 87% (15) | 37% (43) |
| fish | 56 | 37% (43) | 77% (13) | 56% (27) |
| invertebrates | 14 | 43% (14) | 60% (15) | 21% (53) |
| diver | 235 | 95% (186) | 86% (245) | 86% (258) |
| scenery | 305 | 74% (344) | 79% (340) | 86% (237) |

## Slices: balanced accuracy (frames)

| Slice | n | `clip-vitb32-zeroshot-6cat` | `v3` | `v2_both_s0` |
| --- | --- | --- | --- | --- |
| all | 640 | 56% | 67% | 69% |
| random-only | 359 | 41% | 62% | 62% |
| gopro | 583 | 51% | 55% | 63% |
| phone | 57 | 56% | 82% | 76% |
| trip: azores | 196 | 42% | 51% | 60% |
| trip: bahamas | 299 | 55% | 74% | 73% |
| trip: mexico | 44 | 50% | 45% | 49% |
| trip: philippines | 101 | 75% | 73% | 74% |

Confusion matrices: [`clip-vitb32-zeroshot-6cat`](confusion_clip-vitb32-zeroshot-6cat.png), [`v3`](confusion_v3.png), [`v2_both_s0`](confusion_v2_both_s0.png).
Frames the primary model got wrong: [failures.jpg](failures.jpg).
