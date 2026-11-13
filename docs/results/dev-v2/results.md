# Test-set results: the **dev** half (used for error analysis and choosing fixes)

Hand-labeled frames from my own dives, `unusable` excluded: **294** frames, the same frames for every model. Balanced accuracy is the average of per-category recall, so every category counts equally.

| Model | Role | Balanced accuracy | Accuracy |
| --- | --- | --- | --- |
| `clip-vitb32-zeroshot-6cat` | zero-shot CLIP baseline | **65%** | 73% |
| `convnext_tiny_depth_s0` | fine-tuned candidate | **69%** | 77% |
| `v2_base_s0` | fine-tuned candidate | **71%** | 76% |
| `v2_wide_s0` | fine-tuned candidate | **72%** | 74% |
| `v2_zoom_s0` | fine-tuned candidate | **69%** | 72% |
| `v2_both_s0` | fine-tuned candidate | **75%** | 75% |

## Recall per category (frames it rests on)

| Category | n | `clip-vitb32-zeroshot-6cat` | `convnext_tiny_depth_s0` | `v2_base_s0` | `v2_wide_s0` | `v2_zoom_s0` | `v2_both_s0` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| turtles | 3 | 67% | 100% | 100% | 100% | 67% | 100% |
| rays | 18 | 67% | 83% | 83% | 83% | 94% | 72% |
| fish | 30 | 47% | 13% | 17% | 30% | 27% | 37% |
| invertebrates | 9 | 56% | 44% | 56% | 56% | 67% | 78% |
| diver | 94 | 77% | 91% | 90% | 90% | 90% | 91% |
| scenery | 140 | 78% | 81% | 79% | 72% | 66% | 72% |

## Slices: balanced accuracy (frames)

| Slice | n | `clip-vitb32-zeroshot-6cat` | `convnext_tiny_depth_s0` | `v2_base_s0` | `v2_wide_s0` | `v2_zoom_s0` | `v2_both_s0` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 294 | 65% | 69% | 71% | 72% | 69% | 75% |
| random-only | 110 | 51% | 66% | 66% | 62% | 46% | 79% |
| gopro | 275 | 68% | 70% | 72% | 73% | 73% | 74% |
| phone | 19 | 74% | 58% | 55% | 58% | 58% | 82% |
| trip: azores | 105 | 48% | 52% | 51% | 53% | 50% | 68% |
| trip: bahamas | 61 | 70% | 57% | 55% | 60% | 61% | 85% |
| trip: mexico | 34 | 39% | 33% | 57% | 48% | 76% | 57% |
| trip: philippines | 94 | 77% | 77% | 77% | 81% | 75% | 80% |

Confusion matrices: [`clip-vitb32-zeroshot-6cat`](confusion_clip-vitb32-zeroshot-6cat.png), [`convnext_tiny_depth_s0`](confusion_convnext_tiny_depth_s0.png), [`v2_base_s0`](confusion_v2_base_s0.png), [`v2_wide_s0`](confusion_v2_wide_s0.png), [`v2_zoom_s0`](confusion_v2_zoom_s0.png), [`v2_both_s0`](confusion_v2_both_s0.png).
Frames the primary model got wrong: [failures.jpg](failures.jpg).
