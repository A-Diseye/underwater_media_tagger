# Test-set results: the **holdout** half (locked until now, scored once)

Hand-labeled frames from my own dives, `unusable` excluded: **635** frames, the same frames for every model. Balanced accuracy is the average of per-category recall, so every category counts equally.

| Model | Role | Balanced accuracy | Accuracy |
| --- | --- | --- | --- |
| `clip-vitb32-zeroshot-6cat` | zero-shot CLIP baseline | **56%** | 73% |
| `v2_both_s0` | fine-tuned (chosen) | **69%** | 74% |
| `convnext_tiny_depth_s0` | fine-tuned (comparison, didn't affect the choice) | **52%** | 77% |

## Recall per category (frames it rests on)

| Category | n | `clip-vitb32-zeroshot-6cat` | `v2_both_s0` | `convnext_tiny_depth_s0` |
| --- | --- | --- | --- | --- |
| turtles | 11 | 55% | 64% | 36% |
| rays | 19 | 53% | 84% | 89% |
| fish | 56 | 29% | 27% | 4% |
| invertebrates | 14 | 43% | 79% | 7% |
| diver | 229 | 75% | 95% | 97% |
| scenery | 306 | 84% | 67% | 80% |

## Slices: balanced accuracy (frames)

| Slice | n | `clip-vitb32-zeroshot-6cat` | `v2_both_s0` | `convnext_tiny_depth_s0` |
| --- | --- | --- | --- | --- |
| all | 635 | 56% | 69% | 52% |
| random-only | 358 | 41% | 62% | 50% |
| gopro | 578 | 51% | 64% | 47% |
| phone | 57 | 56% | 76% | 51% |
| trip: azores | 196 | 42% | 60% | 58% |
| trip: bahamas | 259 | 53% | 73% | 51% |
| trip: mexico | 43 | 49% | 49% | 43% |
| trip: philippines | 137 | 80% | 79% | 60% |

Confusion matrices: [`clip-vitb32-zeroshot-6cat`](confusion_clip-vitb32-zeroshot-6cat.png), [`v2_both_s0`](confusion_v2_both_s0.png), [`convnext_tiny_depth_s0`](confusion_convnext_tiny_depth_s0.png).
Frames the primary model got wrong: [failures.jpg](failures.jpg).
