"""Color test page: the blind A/B test of subject-aware vs. whole-frame color correction.

UI only. Pairs, votes, and results live in training/color_test.py. Which side is which stays
hidden until every pair has a vote. These are unlabeled library frames, not test frames.
"""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from training import color_test

KEYS = {"left": "1", "same": "2", "right": "3"}
NAMES = {"left": "Left is better", "same": "No difference", "right": "Right is better"}


def _data_uri(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _flip_view(f: dict) -> None:
    """Both versions stacked in one spot; the arrow keys swap them instantly in the browser
    (no server round trip), which makes small differences easier to see than side by side.
    The script is ours and the images are local files, so running JavaScript here is safe."""
    ids = {side: f"flip-{f['image_id']}-{side}" for side in ("left", "right")}
    label = f"flip-{f['image_id']}-label"
    st.html(
        f"""
<div style="position:relative">
  <img id="{ids["left"]}" src="{_data_uri(f["left_path"])}" style="width:100%;display:block">
  <img id="{ids["right"]}" src="{_data_uri(f["right_path"])}" style="width:100%;display:none">
  <div id="{label}" style="position:absolute;top:10px;left:10px;padding:4px 12px;
       border-radius:6px;background:rgba(0,0,0,.65);color:#fff;font:600 16px sans-serif">
    ← Left</div>
</div>
<script>
(() => {{
  const show = (side) => {{
    document.getElementById("{ids["left"]}").style.display = side === "left" ? "block" : "none";
    document.getElementById("{ids["right"]}").style.display = side === "right" ? "block" : "none";
    document.getElementById("{label}").textContent = side === "left" ? "← Left" : "Right →";
  }};
  // One listener at a time: each rerun replaces the previous frame's.
  if (window.__colorTestFlip) document.removeEventListener("keydown", window.__colorTestFlip);
  window.__colorTestFlip = (e) => {{
    if (e.key === "ArrowLeft") {{ show("left"); e.preventDefault(); }}
    if (e.key === "ArrowRight") {{ show("right"); e.preventDefault(); }}
  }};
  document.addEventListener("keydown", window.__colorTestFlip);
}})();
</script>
""",
        unsafe_allow_javascript=True,
    )
    st.caption("← / → flip between Left and Right · 1 / 2 / 3 to vote")


def _vote(frame: dict, choice: str, which: str) -> None:
    no_subject = st.session_state.get(f"no-subject-{frame['image_id']}", False)
    color_test.save_vote(frame, choice, no_subject, which=which)
    st.session_state.color_test_last = (which, frame["image_id"])


def _undo() -> None:
    last = st.session_state.pop("color_test_last", None)
    if last is not None:
        which, image_id = last
        color_test.undo_vote(image_id, which=which)


def _rate(t: dict) -> str:
    if t["win_rate"] is None:
        return "–"
    lo, hi = t["interval"]
    return f"{t['win_rate']:.0%} ({lo:.0%}–{hi:.0%})"


def _results(manifest: list[dict], votes: dict, which: str) -> None:
    r = color_test.results(manifest, votes)
    o = r["overall"]
    verdict = "better" if r["passed"] else "not shown to be better"
    st.success(
        f"All {r['frames']} pairs voted. Subject-aware is **{verdict}** by the rule set "
        f"before the test (wins at least {color_test.WIN_BAR:.0%} of non-tie votes)."
    )
    cols = st.columns(4)
    cols[0].metric("subject-aware wins", o["subject"])
    cols[1].metric("whole-frame wins", o["whole"])
    cols[2].metric("no difference", o["same"])
    cols[3].metric("win rate (95% range)", _rate(o), help=f"one-sided p = {o['p_value']:.3f}")
    rows = [
        {
            "category": k,
            "subject": t["subject"],
            "whole": t["whole"],
            "same": t["same"],
            "win rate": _rate(t),
        }
        for k, t in r["by_category"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    w = r["wrong_tag"]
    if w["subject"] + w["whole"] + w["same"]:
        st.caption(
            f'Flagged "no subject here" (wrong tags): subject-aware {w["subject"]}, '
            f"whole-frame {w['whole']}, no difference {w['same']}. Not in the win rate."
        )
    sonnet = color_test.load_votes(who="sonnet", which=which)
    if sonnet:
        a = color_test.agreement(manifest, votes, sonnet)
        st.metric("Sonnet agrees with you", f"{a['same_winner']:.0%}", help=f"{a['frames']} pairs")
    else:
        st.caption(f"Second judge: `uv run python -m training.color_test --sonnet --round {which}`")


def render() -> None:
    st.title("Color test")
    rounds = color_test.exported_rounds()
    if not rounds:
        st.info("No pairs yet: run `uv run python -m training.color_test --export`.")
        return
    names = {"C": "Round 1 (set C)", "pool": "Round 2 (fresh pool frames)"}
    which = st.sidebar.radio(
        "Test", rounds, index=len(rounds) - 1, format_func=names.get, help="Each is run once."
    )
    manifest = color_test.load_manifest(which=which)
    votes = color_test.load_votes(which=which)
    st.caption(
        "Each pair is the same frame corrected two ways; the sides are shuffled and unnamed. "
        "Pick the one you'd keep, or no difference. Tick the box if the tagged subject isn't "
        "in the frame. Results stay hidden until every pair has a vote."
    )
    todo = [f for f in manifest if str(f["image_id"]) not in votes]
    st.progress(len(votes) / len(manifest), text=f"{len(votes)}/{len(manifest)} voted")
    if "color_test_last" in st.session_state:
        st.button("Undo last vote", on_click=_undo)
    if not todo:
        _results(manifest, votes, which)
        return

    f = todo[0]
    st.subheader(f"Subject: {f['label']} · {f['trip']} · #{f['image_id']}")
    view = st.segmented_control(
        "View", ["Flip (← →)", "Side by side"], default="Flip (← →)", key="color-test-view"
    )
    if view == "Side by side":
        left, right = st.columns(2)
        left.image(f["left_path"], caption="Left", width="stretch")
        right.image(f["right_path"], caption="Right", width="stretch")
    else:  # also when the selected view is clicked again and cleared
        _flip_view(f)
    st.checkbox(f"There's no {f['label']} in this frame", key=f"no-subject-{f['image_id']}")
    with st.container(horizontal=True, horizontal_alignment="center"):
        for choice, key in KEYS.items():
            st.button(
                f"{key} · {NAMES[choice]}",
                shortcut=key,
                on_click=_vote,
                args=(f, choice, which),
                key=f"{choice}-{f['image_id']}",
            )
