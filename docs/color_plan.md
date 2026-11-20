# Subject-aware color correction: plan

Approved Nov 18, 2026.

## Goal

Water absorbs red light first, then green, so dive photos come out blue-green and flat. Standard
correction tools estimate the color cast from the whole frame. In a dive photo, most of the frame
is water, so the water decides the correction and the animal can come out wrong.

This feature corrects the colors **for the subject**:

1. **Locate:** Grad-CAM finds where the subject is.
2. **Correct:** the color fix is measured on that region, so the turtle looks natural even if
   the water shifts.

No bounding boxes are drawn by hand, and nothing is trained: the classifier already knows where
it looked.

**Done means:**

- In the app, any photo or video frame has a before/after view: original, corrected, and the
  Grad-CAM heatmap. A strength slider and a download button for the corrected JPEG sit alongside.
- There's a measured result: how often the subject-aware correction beats a whole-frame
  correction in a blind side-by-side test that you judge.
- Grad-CAM's hit rate on the subject is measured, with examples of misses.
- The README has a section with before/after examples, both results, and the failure cases.

## How it fits the code

```
tagger/locate.py   Grad-CAM on the v3 ensemble -> heatmap -> soft subject mask
tagger/color.py    classic correction, weighted by a mask (numpy + OpenCV, no torch)
app/               before/after view in the gallery details; "Color test" page for the blind test
training/          nothing changes
```

- **`color.py` doesn't import torch.** It takes an image and a mask, so it's fast to test and
  runs in CI. It doesn't care where the mask came from.
- **`locate.py` uses the same checkpoints as tagging,** and skips in CI like the other
  torch-based tests.
- **No writes to D: and no bulk output.** Corrections are computed when you open an image and
  kept in memory. The download button serves them from memory. A "save" writes only to
  `data/corrected/`.

## Stage 1: find the subject (`tagger/locate.py`)

**How Grad-CAM works:**

1. Run the image through the model and keep the last convolutional feature map: 768 channels
   on a coarse grid, 10×10 for a 320 px square.
2. Take the gradient of the chosen category's score (say "turtle") with respect to that map.
   Averaging the gradient over the grid gives one weight per channel: how much that channel
   pushed the answer toward turtle.
3. Weighted-sum the channels and keep only the positive part. Hot cells are where the evidence
   for "turtle" was. Upsample the result to the image size.

It works on CNNs because their last layer still has a spatial layout. This is why we chose
ConvNeXt over a ViT.

**Design choices:**

- **Cover the whole frame with square crops.** Tagging crops the center 320 px square, which
  drops the left and right quarters of a 16:9 frame (see "Checked while planning" below). The
  model is much less reliable on the full wide frame (dev balanced accuracy 82% → 69%), because
  it learned on square photos. So Grad-CAM runs on 3 overlapping 320 px squares across the frame
  (resized to 320 px tall). Each crop's 10×10 heatmap is placed back where it came from, and the
  overlaps are averaged. That covers the whole frame without the accuracy drop.
- **Average all 3 v3 seeds,** like tagging. A single model's heatmap is noisy, and the average
  costs 9 forward and backward passes (3 seeds × 3 crops), about 0.2 s on the GPU.
- **Which category to explain:** the tag the gallery shows (your correction if you made one,
  else v3's answer).
  - If the tag is `scenery`, there's no subject, so the mask is the whole frame.
  - If the tag is `diver`, the diver is the subject.
- **When the tag is wrong** (e.g. a mossy boulder tagged as a turtle), Grad-CAM still highlights
  what fooled the model: the boulder. The correction would then balance the boulder's colors.
  That's often harmless (gray rock is a fair color reference, better than blue water), but
  algae-covered rock would push the image too cyan. Three safeguards:
  1. **Confidence gate:** subject mode only when v3's confidence is at least 0.5; below that,
     the whole frame. False tags are mostly low-confidence. On the library and the dev half
     (allowed for choosing fixes):

     | v3 tags | Frames | Median confidence | Share ≥ 0.5 |
     | --- | --- | --- | --- |
     | Azores turtles (all false) | 13 | 0.37 | — |
     | Mexico rays (all false) | 50 | 0.38 | — |
     | Dev: correct animal/diver tags | 115 | 0.74 | 86% |
     | Dev: wrong animal/diver tags | 33 | 0.52 | 58% |

     The 0.5 is checked on set A before the blind test, and never changed after it.
  2. **Your corrections win:** a frame you retagged as scenery is corrected on the whole frame.
  3. **Damage stays bounded when a false tag passes:** the 0.8 / 0.2 weighting and the gain
     limits keep the result close to a plausible whole-frame correction, not a broken image.
- **From heatmap to mask:**
  1. Normalize the heatmap to 0–1.
  2. Threshold at a fraction of the peak, 0.5 to start.
  3. Keep the connected blob that holds the peak, so a second hot spot on the reef doesn't
     merge in.
  4. Grow it a little and feather the edge, so the correction doesn't show a seam.
- **When Grad-CAM can't decide:** if the blob covers more than about 60% of the frame (the
  evidence is everywhere), fall back to the whole frame. The app then says "no clear subject".
- **Output:** `Located(heatmap, mask, box, category, confident)`. The heatmap is for the overlay,
  the mask for color, and the box for the README figures.

**Checkpoint 1: contact sheets for you to review.**

1. **Tuning sheet, set A:** 24 unlabeled library frames, about 4 per category by v3's tag, from
   different trips. Each frame shows the original, the heatmap overlay, and the mask outline.
   The threshold and the fallback area are tuned on this sheet only.
2. **Scoring sheet, set B:** 30 different unlabeled frames, scored once with the tuned settings.
   For each one you mark whether the mask lands on the subject: hit, partial, or miss. The README
   reports that hit rate, with the misses as examples (already a README item: "cases where
   Grad-CAM misses the subject").

**Why unlabeled frames:**

- **No test-set frames.** They're excluded, so nothing here touches the test set.
- **Keeping the two sets separate** is the same tune-then-score discipline as the classifier.
- **The rule on test labels doesn't apply.** Marking whether a heatmap hits the animal isn't
  labeling a test frame's category, and these frames aren't test frames anyway.

## Stage 2: correct the color (`tagger/color.py`)

Revised Nov 19 after looking at how underwater photographers edit (sources at the end of this
section). They don't make the whole frame neutral. In Lightroom they white-balance on something
that should be gray, mask the subject and give it extra warmth and contrast, and mask the water
separately to keep it blue and slightly darker (Adobe's own tutorial neutralizes a shark and
"the water was kept blue"). Video editors without a red filter rebuild the red channel from
green in DaVinci Resolve, then balance. One-tap apps (Dive+, DiveColor) correct the whole frame
automatically. Our Grad-CAM mask plays the part of Lightroom's "Select Subject".

**Two modes, and that's the blind test's question** ("does a mask-based workflow beat one-tap
correction?"):

| Mode | What it does | Photographer analogue |
| --- | --- | --- |
| `whole` | steps 1-4 on the full frame, no mask | one-tap apps |
| `subject` | steps 1-6: measured on the subject, water treated separately | Lightroom with subject and water masks |

Frames that Stage 1 sends to the whole frame (scenery, low confidence, no clear subject) get
`whole`. The `local` stretch mode from the first plan is folded into `subject`.

**Steps** (full resolution, floats in 0-1):

1. **Recover red** (Ancuti et al., 2018; the Resolve channel-mixer trick). Where red is weak
   and green strong, red is partly rebuilt from green:
   `R' = R + α · (mean G − mean R) · (1 − R) · G`, α = 1. Blue gets the same in green water.
   - **Why:** white balance alone can't fix a nearly empty channel; multiplying it just
     amplifies noise into magenta blotches.
2. **White balance on likely-gray pixels** (the eyedropper, automated). A provisional gray-world
   correction is applied, then only the least colorful pixels (lowest 60% by chroma, not
   near-black or clipped) are averaged, weighted by mask × 0.8 + 0.2 in `subject` mode. Gains
   are relative to green.
   - **Only corrections water could have caused:** water removes red, never adds it, so the
     red gain is at least 1 (at most 3); blue stays within 0.5-3. An orange nudibranch "asks"
     for less red, which the clamp refuses.
   - **Why gray pixels:** a gray-world average over a colorful subject would strip its color
     (risk 1 below). Gray-pixel color constancy is a published family of methods; this is the
     simplest version.
3. **Levels** (exposure, whites, blacks). One stretch for all three channels, so it doesn't
   re-cast the balance: black point at the 0.5th percentile of each pixel's darkest channel,
   white point at the 99.5th of its brightest, measured on the whole frame.
4. **Clarity:** light CLAHE on the lightness channel (Lab), clip limit 1.5; skipped on very
   dark frames, where it mostly amplifies noise. Photographers warn that heavy clarity or
   dehaze makes water look dirty, so it stays light.
5. **Subject** (`subject` mode): full gains inside the mask.
6. **Water** (`subject` mode): outside the mask the gains are applied at half strength (in log
   terms, g^0.5) so the water stays blue instead of going gray or purple, and it's darkened by
   5% (which also hides backscatter). Steps 5 and 6 blend through the feathered mask, so there's
   no seam.
7. **Strength:** `out = original + s · (corrected − original)`, s from 0 to 1 on the app slider,
   default 0.8.

**Estimate, then apply.** `color.estimate(img, mask, mode) -> Settings` (compensation amounts,
gains for subject and water, levels) and `color.apply(img, mask, settings, strength)`. For
stills that's just `correct()`. For video it allows the editors' approach later: estimate per
frame, smooth the settings over the clip, apply. No rewrite needed.

**Tuning:** the defaults (α, chroma cut, gain limits, 0.8 / 0.2 weighting, water strength,
percentiles, CLAHE clip) are set by looking at set A, never at the blind-test frames.

**Checkpoint 2:** a before/after sheet for set A (original | `whole` | `subject`) for you to
review before anything goes into the app. Colorful subjects (orange or red invertebrates and
fish) get a specific look.

**Speed:** about 50-100 ms per 1920x1080 frame in numpy. Grad-CAM adds about 0.2 s on the GPU,
or several seconds on CPU.

**Two risks and how they're handled:**

- **Colorful subjects lose their color.** The eyedropper works because people click on
  something gray; measuring on an orange subject would strip the orange. This would hit
  invertebrates and colorful fish hardest (turtles, rays and divers are mostly neutral), so it
  could make subject mode lose the blind test for the wrong reason, and in a public tool it
  would fail on the most-photographed subjects. Handled by the red clamp and gray-pixel
  measuring above, a unit test (an orange patch under a blue cast stays orange), a look at
  colorful set A frames, and per-category blind-test results. In reserve: measure on a ring
  around the subject (the grown mask minus its core), since sand at the same distance carries
  the same cast.
- **Video flicker.** Per-frame settings jump between frames, and subject mode more than whole
  (the mask moves, and frames crossing the 0.5 confidence gate switch modes). The blind test
  judges stills, so its result applies to stills only, and the README says so. Handled later by
  the estimate/apply split: smooth the settings over the clip and blend between modes instead
  of switching. Video export is out of scope until Nov 23.

**Sources:** [Adobe: underwater photo editing](https://www.adobe.com/learn/lightroom-cc/web/underwater-photo-editing),
[Lightroom workflow order for underwater photos](https://robertherb.blogspot.com/2025/11/back-to-basics-part-3-ultimate.html),
[DivePhotoGuide: white balance when editing](https://www.divephotoguide.com/underwater-photography-techniques/article/white-balance-editing-underwater-photos),
[Blackmagic forum: editing GoPro scuba video](https://forum.blackmagicdesign.com/viewtopic.php?p=429068),
[Divernet: underwater photo-editing apps](https://divernet.com/scuba-gear/photo-editing-apps/).

## Stage 3: the app

- **In the gallery's details dialog:**
  - A **Color** section with three tabs: Original / Corrected / Heatmap.
  - Controls for mode (whole frame vs. subject), strength, and CLAHE on/off.
  - A **Download corrected JPEG** button.
  - A caption saying what was used: "Corrected for: turtle (0.82)", or "Whole frame: low
    confidence", "Whole frame: no clear subject", or "Whole frame: scenery". The Heatmap tab
    shows where it looked, and the mode switch lets you override.
- **Uploads get the same view,** so the demo can be: upload a photo, it gets tagged, see the
  correction.
- **Caching:** heatmap and mask are cached per image (`st.cache_data`), so moving the slider
  only re-runs `color.py`, which is fast.
- **Restart after changes:** after editing `tagger/`, the app needs a restart (`/restart-app`).

## Stage 4: does subject-aware actually look better?

There's no ground truth for "correct colors" underwater: nobody knows what the turtle looked like
under white light. So you judge, blind.

**Blind A/B test** ("Color test" page, like the Filter check page):

- **Frames:** about 30 frames from a fresh set C where subject mode actually applies: tag not
  `scenery`, confidence at least 0.5, and a clear Grad-CAM subject. Elsewhere the two modes give
  the same image. Up to 6 each across turtles, rays, fish, invertebrates and diver, spread over
  trips (set C has only 2 confident turtle frames and 4 invertebrate frames).
- **The comparison:** each frame shows `whole` and `subject` side by side. Left and right are
  randomized and the methods are unnamed. You pick **left / right / no difference**.
- **False tags that pass the gate are still in the set.** Each frame has a checkbox: "there's no
  [turtle] in this frame". The README then reports separately how frames with wrong tags came
  out. These are unlabeled library frames, so this isn't test labeling.
- **Storage:** votes go to `data/color_votes.json`. The page reveals nothing until all 30 are in.
- **The rule, set before the test:** subject-aware counts as better if it wins at least 60% of
  the non-tie votes. The README reports the win rate with its uncertainty (30 votes is small, so
  20/30 is about p = 0.05 against a coin flip).
- **Stills only:** the result says nothing about video, where subject mode flickers more.
- **Per-category win rates** go in the README too. A likely finding is that subject-aware helps
  most on close-up animals and divers and least on small fish in wide shots.

**Not used as the main measure:** no-reference underwater quality scores (UCIQE, UIQM). They're
known to reward oversaturation. They may appear as a secondary column, labeled as such.

## Tests (CI runs everything except torch-based tests)

- **`color.py`, on synthetic images:**
  - A gray chart with a blue-green cast comes back near neutral.
  - Strength 0 gives back the original.
  - The gain limits hold on an all-blue frame.
  - A mask over a red patch moves the balance differently than the whole frame does.
  - An orange patch under a blue cast stays orange; the red gain is never below 1.
  - In `subject` mode the water outside the mask stays blue.
  - `apply(estimate(...))` equals `correct(...)`.
  - Output is always 0–255 uint8, the same shape as the input.
- **`locate.py`, on the tiny CPU checkpoint already used in `test_classify.py`:**
  - The heatmap and mask have the image's shape, with values in 0–1.
  - `scenery` gives the whole-frame mask.
  - Wide and tall inputs: the stitched heatmap covers the whole frame, and overlaps are averaged.
- **Blind-test bookkeeping:**
  - The order is randomized but reproducible.
  - Votes map back to the right method.
  - A half-done test doesn't reveal results.

## Schedule (about 3½ working days)

| Day | Work | Checkpoint for you |
| --- | --- | --- |
| **Wed 18 (rest of day)** | `locate.py` + tests; heatmap sheet for set A | 1a: do the heatmaps land on the animals? |
| **Thu 19 (morning)** | tune the mask settings on set A; score set B; Haiku hit check (agent A) on set B + ~300 frames | 1b: mark hit/partial/miss on 30 frames (~10 min) |
| **Thu 19 (afternoon)** | `color.py` + tests; before/after sheet for set A | 2: do the defaults look right? |
| **Fri 20** | app: Color section in details and uploads; Color test page; Sonnet second judge (agent B) after your votes | 3: run the blind test (30 votes, ~15 min) |
| **Mon 23** | README section with examples and both results; commits | final review |

You do the e2e run, README fixes and demo video in your free time. For the demo, the upload →
tag → corrected-view flow from Stage 3 makes a good closing shot.

Commits: one per stage, after your approval at each checkpoint.

## Risks

- **Grad-CAM is coarse:** one grid cell is about 107 px on a 1920 px frame. Small fish will get
  a blob around them, not an outline. That's acceptable here, because the mask only weights
  color statistics; it doesn't cut anything out.
- **Grad-CAM can light up context, not the animal** (sand around a ray, reef behind a fish). The
  set B hit rate measures this, and the misses go in the README.
- **Green Azores water and dark frames:** red is close to zero, so compensation amplifies
  noise and JPEG blocks. The gain limits and CLAHE off in the dark contain it. The README shows
  the worst case honestly.
- **Video frames are compressed JPEGs.** Strong correction shows banding, so default strength is
  0.8, not 1.0.
- **Time:** if Thursday slips, the `local` variant is dropped first. The
  blind test isn't dropped: it's the measured result.

## Checked while planning: does the center crop or the shrinking cause misclassifications?

Tagging resizes each frame so its short side is 365 px, then crops the center 320×320. On 16:9
footage that keeps 49% of the width and 88% of the height. I scored v3 on the dev half three
ways (277 frames, mirror-image averaging, no smoothing):

| v3 on dev | Macro-F1 | Balanced acc. | Fish found |
| --- | --- | --- | --- |
| **Center crop (current)** | **80%** | **82%** | **14 / 28** |
| Full frame, no crop (569×320) | 75% | 69% | 8 / 28 |
| 3 crops across the frame, averaged | 77% | 74% | 10 / 28 |

- **The crop isn't the problem.** Showing the model more of the frame made it worse:
  - Subjects are usually framed near the center, so the crop rarely cuts them off.
  - The model learned on square, center-cropped photos. A squeezed wide frame makes subjects
    smaller, in a shape it never trained on.
  - Edge crops are mostly water and reef, and they pull the answer toward scenery.
- **Resolution is the bigger limit.** Even with the crop, only half the dev fish are found. At a
  320 px input, a small fish in a 1920 px frame keeps only a few pixels, whatever the source
  resolution. The fix would be a larger input and training on wide frames. That needs
  retraining, plus newly labeled footage to score it fairly (the holdout has been used twice), so
  it's a README next step, not this week's work.
- **Color correction isn't affected by the shrinking:**
  - It runs on the full-resolution image.
  - The 320 px copy only places the subject mask, and the mask only weights color averages.
  - When v3 mislabels a frame as scenery, that frame falls back to a whole-frame correction.
- **This changed one choice in the plan:** Grad-CAM uses 3 square crops instead of the full
  wide frame (Stage 1).

## Out of scope

- Correcting whole videos (frame-to-frame flicker)
- Learned correction (training on before/after pairs)
- Comparing against my own Photoshop edits (no originals kept with the edits)
- Bounding boxes or segmentation models
- Changing the classifier (see above)

## Where an agent could help

The training-data work used two patterns:
- **A vision check:** Haiku looks at one photo and answers fixed questions. Used for divers and
  scenery, as a workflow.
- **A real agent:** Sonnet decides what to do next (which taxa to search). That's the curation
  agent.

Both were trusted only after I hand-checked a sample. The same patterns fit here, under the same
rules:
- No test-set frames.
- Haiku for bulk checks, Sonnet for decisions.
- The API key comes from `.env`.
- Tests mock the API.

**A. Grad-CAM hit check, a vision check (chosen).**
- **What it does:** Haiku sees each frame with its heatmap outline and answers: is the outlined
  region on the main animal or diver? (yes / partly / no, plus a reason).
- **Why it helps:**
  - Measuring the hit rate by hand stops at about 30 frames. With Haiku it covers about 300
    unlabeled frames, so the README number rests on 10× more frames.
  - It finds misses worth showing, like sand around a ray or reef behind a fish.
- **How it's trusted:** the same as the curation filter. Haiku also grades set B, the 30 frames
  you mark, and the README reports how often it agreed with you. If agreement is low, the
  300-frame number isn't used.
- **Cost:** about $0.10 for 300 frames (Haiku, about 1,600 tokens per frame).
- **Fits the rules:** it judges where the model looked, not what category a frame is, and only
  on unlabeled frames.

**B. A second judge for the blind test (chosen).**
- **What it does:** after your 30 votes are in, Sonnet sees the same pairs, also blind and
  randomized, and picks the better-looking correction.
- **What it shows:**
  - The README reports how often Claude agrees with your taste.
  - It's an interesting result either way. If agreement is high, an AI judge could stand in for
    a human on later experiments. If it's low, it shows why I didn't trust one here.
- **What it doesn't do:** it never replaces your votes, and the win rate comes from you.
- **Cost:** under $0.25.

**C. An auto-grading agent: the most "agentic" option (not chosen).**
- **What it does:** Sonnet gets tools to correct a photo with given settings (`color.correct`
  with mode, strength, CLAHE) and to look at the result, and iterates until it's happy. That's
  about 3–4 rounds per photo, like an editor making adjustments in Photoshop.
- **How it's measured:** a third option in the blind test (agent vs. `subject` with defaults).
- **Why it's a stretch:**
  - **It's slower and costs more:** about $0.04 per photo and 20–40 s.
  - **It's harder to make reproducible.**
  - **It reverses a choice in PLAN.md**, which lists "agent-driven color correction" as not
    planned.
  - **The strongest objection:** a model judging its own edits tends to like them, so only your
    blind votes could show it's better.
- **Decision (Nov 18):** not built. A and B are.

**Not worth an agent:** choosing the frames for sets A, B and C. A fixed stratified random draw
(per category and trip) is simpler, reproducible, and can't pick favorites. It's the same reason
divers and scenery use workflows, not agents.

## Decisions

- **Nov 18:** plan approved; agents A (Grad-CAM hit check) and B (second judge) chosen, C not.
  No Photoshop pairs (no originals kept with the edits).
- **Nov 18, building Stage 1:**
  - **Sets are dealt by stratum, not hashed** (`tagger/color_sets.py`). With a plain hash, set A
    got 0 confident turtles. Each clip is now grouped by the rarest confident tag it contains, and
    each group is dealt to A, B, C, pool, pool.
  - **Confident turtles come from about 4 clips and 7 photos,** almost all from two Bahamas
    dives, so they can't be spread evenly: set A has 12 turtle frames, B 3, C 2. The blind test
    takes *up to* 6 per category, and the README says how many it got.
  - **Crop heatmaps cross-fade** (Hann-window weights) instead of plain averaging. Plain
    averaging left straight seams at crop edges.
  - **Mask keeps secondary hot regions:** any region whose peak is at least 0.6 × the top one
    (`MIN_PEAK`), chosen on set A. Keeping only the hottest region caught one diver of two, or
    the spot by a pointing finger instead of the sea star. Threshold stays 0.5.
  - **Divers get a lower cut-off (0.25; animals keep 0.5),** after your review of set A: masks
    covered only fins and legs (#11590), missed the head (#12308), or barely touched the diver
    (#5705). Grad-CAM marks what was *distinctive* for "diver" (fins, legs, tank), not the whole
    person. 0.25 spreads diver masks over more of the body. For animals, 0.35 and 0.25 spilled
    small fish masks into the sand, so they stay at 0.5. Not fixable by any threshold: #5705,
    whose evidence sits at the frame's edge. README limitation: diver masks favor gear over
    faces. The blind test's diver win rate shows whether that matters for color.
- **Nov 18, set B scored (my marks, 27 frames, settings locked first):** 17 hit, 2 partial, 3
  miss, 3 no such subject, 2 wrong tag with the outline on the real subject (an eel tagged as
  invertebrates; a fifth mark added mid-marking for this case, with 5 earlier marks redone).
  Hit rate 17/22 = 77% (about 57–90% at this size), 86% counting partials. 5 of 27 (19%) were
  false tags that passed the gate, including two Mexico "rays" at 0.76 and 0.84.
- **Agent A's bar, set before its first run:** Haiku's marks fall into three groups (hit / off =
  partial or miss / wrong tag). Its pool count is used only if it lands in my group on at least
  80% of set B. Otherwise the README reports the agreement and not the larger number.
- **Agent A, round 1: 59% agreement, under the bar.** Haiku couldn't see faint subjects in
  murky water (5 frames), was stricter than me on hit vs. partial (4), and misplaced the outline
  twice. On point 2 Haiku was right: for color correction the outline must cover the *whole*
  subject, and a little beyond is fine, but I had marked "centered on it, covering most" as a
  hit. On point 1: a future public tool can't rely on knowing the dive either, so judging only
  from the frame is the right standard.
- **Round 2: stricter definitions for both of us, set B re-marked.** hit = covers the whole
  subject (all of them if several); partial = cuts off part, or mostly background. Round 1 is kept
  in `data/color/check_B/round1/`. The definition changed after seeing Haiku's reasons, so round
  2's agreement is a second look, not a first. Mask settings stay locked, so set B is still a
  fair test of Grad-CAM. If round 2 shows the masks are too tight, any growth is tuned on set A
  and scored on fresh pool frames, since set B will be spent.
- **Round 2 results (strict marks, judged from the frame alone):** 7 hit, 11 partial, 1 miss,
  8 no visible subject. Strict hit rate 7/19 = 37%; on the subject at all (hit or partial)
  18/19 = 95%. The masks are in the right place but too tight. A third of the frames that pass
  the gate have no subject a stranger could see. Haiku: 74% group agreement (20/27), still under
  the 80% bar, so its pool count is not used. 5 of the 7 disagreements are hit vs. partial.
- **Next: grow the masks.** Growth is tuned on set A by eye (3 levels, about 4% / 8% / 12% of
  the frame's short side), then scored once on about 25 fresh frames from the pool (no longer
  needed for Haiku), marked strictly. Set B stays as the record of the tight masks.
- **Growth: 12% of the short side, plus gap joining (locked).** Set A was compared at 4 / 8 / 12
  / 20%. 12% covers whole subjects on most frames; 20% spilled well past small subjects. One
  frame (#12967, an anemone) had two hot spots on one animal and got two outlines with the
  tentacles between them uncovered. Fix: before growing, fill gaps narrower than 20% of the short
  side between kept regions (a morphological closing, `BRIDGE`). The gaps on set A were 11%,
  14% and 18% for parts of one subject, and 21% and up for separate things, so 20% joins the
  first group only. 10% was tried first and was too narrow for #12967.
- **Known misses that no mask setting fixes** (the heatmap itself is wrong, not the outline):
  #5705 (diver: evidence on a hand and the regulator, not the body), #3990 (a fish camouflaged
  on sand: evidence on the sand texture), #8667 (a second hot spot on a finger, kept by the same
  rule that keeps a second diver), and #3216 (an eel tagged invertebrate). For the README.
- **Final score of the grown masks** (`final` set: 25 fresh pool frames, 5 per subject category,
  marked once, strictly, by me): 13 hit, 9 partial, 0 miss, 3 no subject there (2 of them Azores
  "turtles", where there are none). Strict hit rate 13/22 = 59% (set B's tight masks: 7/19 =
  37%); on the subject at all 22/22 = 100%. Different frames from set B, and both samples are
  small (95% intervals about 39-76% vs. 19-59%), so "growing helped" is likely but not proven.
  Weakest: divers (2/5 hit) and rays (2/5), whose fins, tails and limbs reach past the mask;
  invertebrates 4/4 and fish 4/5.
- **Stage 2 redesigned after looking at photographers' workflows** (Nov 19): white balance on
  likely-gray pixels, a physical clamp (red gain at least 1), water kept blue at half strength
  outside the mask, and an estimate/apply split for future video. `local` folded into
  `subject`. See Stage 2.
- **The mask measures; color finds the water** (Nov 19, first set A sheet). Painting with the
  Grad-CAM mask (full correction inside, half outside) showed the mask as a milky blob around
  small animals in open water (#4153, #8228, #12608): the grown mask holds a lot of water, and
  that water got the full warm correction. Lightroom's masks follow real edges; ours doesn't.
  Now the mask only steers the white balance measurement, and water is found by color (the
  median of the least red 30% of pixels outside the mask; pixels near it count as water, inside
  the mask or not), like Lightroom's color-range and "Select Water" masks.
- **Red amount 0.8** (Nov 20). Full red correction looked too warm on set A (orange sand, warm
  reef); the red recovery and red gain compound. `RED_AMOUNT` scales both back (the red gain is
  still never below 1). Compared 0.7 / 0.75 / 0.8 / 0.85 / 1.0; 0.75 and 0.8 were hard to tell
  apart, and 0.8 is the smaller change. Photographers also leave a little of the cool cast.
- **Finish: contrast and sharpening, on trial** (Nov 20). A gentle S-curve and an unsharp mask,
  both on lightness only, the sharpening masked to real edges (Lightroom's "Masking") so flat
  water, compression blocks and backscatter aren't sharpened. Same in both modes, so the blind
  test still compares only where the correction is measured. Light (0.1, 0.4) vs. medium (0.2,
  0.8) rendered on set A.
- **Shallow frames get less correction, on trial** (Nov 20). Near the surface the water hasn't
  taken the red yet, so correcting as if deep made red things and the sunlit surface glare
  (#12608). "Red left" (95th percentile of red / green) separates them on set A: deep frames
  0.23-0.67, shallow ones 0.83-1.63. The whole correction scales from full at 0.7 to 30% at 1.0
  and above.
- **Stage 2 defaults locked** (Nov 20, checkpoint 2): red amount 0.8, medium finish (contrast
  0.2, sharpening 0.8), shallow rule on. Some frames still look off to me; tuning further, or
  learning a correction instead of hand-built steps, would take more time than is left before
  Nov 23. Noted as a limitation for the README, and the blind test measures where it stands.

