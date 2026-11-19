"""Color check page: mark whether Grad-CAM's subject outline lands on the subject.

UI only. Frames, marks, and the hit rate live in training/color_check.py. These are unlabeled
library frames (no test frames), and the mark is about where the model looked, not a category
label.
"""

import streamlit as st

from training import color_check

KEYS = {"hit": "1", "partial": "2", "miss": "3", "not_there": "4", "wrong_tag_on_subject": "5"}


def _mark(which: str, image_id: int, mark: str) -> None:
    color_check.save_mark(which, image_id, mark)


def render() -> None:
    st.title("Color check")
    sets = color_check.exported_sets()
    if not sets:
        st.info("No frame set exported yet: run `uv run python -m training.color_check --set B`.")
        return
    which = st.sidebar.radio("Set", sets, index=len(sets) - 1, help="B is scored once.")
    frames = color_check.load_manifest(which)
    marks = color_check.load_marks(which)
    s = color_check.summary(frames, marks)
    st.caption(
        "Does the white outline land on the main subject? Left: the frame. Right: Grad-CAM's "
        "heatmap (red = most evidence for the tag) and the subject outline. "
        + " · ".join(f"**{k}**: {v}" for k, v in color_check.MARK_HELP.items())
    )
    cols = st.columns(3)
    cols[0].metric("marked", f"{s['marked']}/{s['to_mark']}")
    rate = s["hit_rate"]
    cols[1].metric("hit rate", "–" if rate is None else f"{rate:.0%}")
    cols[2].metric("no clear subject", s["fallbacks"], help="Whole-frame correction; not marked.")

    todo = [f for f in frames if f["reason"] == "subject" and str(f["image_id"]) not in marks]
    if not todo:
        st.success(f"Set {which} done.")
        st.write(s["counts"], s["by_category"])
        return
    f = todo[0]
    st.subheader(f"Tag: {f['label']} ({f['confidence']:.2f}) · {f['trip']} · #{f['image_id']}")
    left, right = st.columns(2)
    left.image(f["original"], width="stretch")
    right.image(f["overlay"], width="stretch")
    with st.container(horizontal=True, horizontal_alignment="center"):
        for mark, key in KEYS.items():
            name = {
                "not_there": f"no {f['label']} here",
                "wrong_tag_on_subject": "wrong tag, outline on the real subject",
            }.get(mark, mark)
            st.button(
                f"{key} · {name}",
                shortcut=key,
                on_click=_mark,
                args=(which, f["image_id"], mark),
                key=f"{mark}-{f['image_id']}",
            )
