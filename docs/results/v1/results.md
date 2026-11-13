# Test-set results

Hand-labeled frames from my own dives, `unusable` excluded: **592** frames, the same frames for every model. Balanced accuracy is the average of per-category recall, so every category counts equally.

| Model | Role | Balanced accuracy | Accuracy |
| --- | --- | --- | --- |
| `clip-vitb32-zeroshot-6cat` | zero-shot CLIP baseline | **62%** | 72% |
| `convnext_tiny_depth_s0` | fine-tuned (chosen) | **60%** | 77% |
| `convnext_tiny_none_s0` | fine-tuned (ablation, didn't affect the choice) | **58%** | 74% |
| `convnext_tiny_boost_s0` | fine-tuned (ablation, didn't affect the choice) | **59%** | 76% |

## Recall per category (frames it rests on)

| Category | n | `clip-vitb32-zeroshot-6cat` | `convnext_tiny_depth_s0` | `convnext_tiny_none_s0` | `convnext_tiny_boost_s0` |
| --- | --- | --- | --- | --- | --- |
| turtles | 12 | 58% | 58% | 50% | 58% |
| rays | 36 | 61% | 86% | 92% | 83% |
| fish | 58 | 36% | 9% | 10% | 12% |
| invertebrates | 17 | 59% | 29% | 29% | 29% |
| diver | 184 | 76% | 95% | 94% | 93% |
| scenery | 285 | 80% | 81% | 75% | 80% |

## Slices: balanced accuracy (frames)

| Slice | n | `clip-vitb32-zeroshot-6cat` | `convnext_tiny_depth_s0` | `convnext_tiny_none_s0` | `convnext_tiny_boost_s0` |
| --- | --- | --- | --- | --- | --- |
| all | 592 | 62% | 60% | 58% | 59% |
| random-only | 237 | 40% | 59% | 58% | 59% |
| gopro | 545 | 59% | 57% | 54% | 57% |
| phone | 47 | 73% | 55% | 56% | 55% |
| trip: azores | 200 | 45% | 56% | 55% | 55% |
| trip: bahamas | 171 | 68% | 54% | 53% | 56% |
| trip: mexico | 52 | 40% | 33% | 53% | 33% |
| trip: philippines | 169 | 82% | 74% | 65% | 73% |

Confusion matrices: [`clip-vitb32-zeroshot-6cat`](confusion_clip-vitb32-zeroshot-6cat.png), [`convnext_tiny_depth_s0`](confusion_convnext_tiny_depth_s0.png), [`convnext_tiny_none_s0`](confusion_convnext_tiny_none_s0.png), [`convnext_tiny_boost_s0`](confusion_convnext_tiny_boost_s0.png).
Frames the primary model got wrong: [failures.jpg](failures.jpg).
