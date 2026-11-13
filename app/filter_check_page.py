"""Filter check page: grade a sample of Haiku's keep/reject decisions on training photos.

UI only. The sample, verdict storage, and accuracy live in training/filter_check.py. These are
internet training photos, not test frames, so showing Haiku's decision here is fine.
"""

from pathlib import Path

import streamlit as st

from training import filter_check

CURATION_DIR = Path("data/curation")


def _verdict(decision: dict, verdict: str) -> None:
    filter_check.save_verdict(CURATION_DIR, decision, verdict)


def _skip(decision_id: str) -> None:
    st.session_state.setdefault("filter_skipped", set()).add(decision_id)


def render() -> None:
    st.title("Filter check")
    st.caption(
        "Was Haiku right to keep or reject each training photo? The README reports this "
        "accuracy. Keeps: is it a usable underwater photo of this category? Rejects: should "
        "it really have been left out?"
    )
    checked = filter_check.sample(filter_check.decisions(CURATION_DIR))
    verdicts = filter_check.load_verdicts(CURATION_DIR)
    skipped = st.session_state.get("filter_skipped", set())

    totals = filter_check.summary(checked, verdicts)
    cols = st.columns(3)
    for col, side in zip(cols, ("keeps", "rejects"), strict=False):
        t = totals[side]
        rate = f"{t['right'] / t['checked']:.0%}" if t["checked"] else "–"
        col.metric(f"{side} right", rate, help=f"{t['right']} of {t['checked']} checked")
    cols[2].metric("checked", f"{len(verdicts)}/{len(checked)}")

    todo = [d for d in checked if d["id"] not in verdicts and d["id"] not in skipped]
    if not todo:
        st.success("Sample done.")
        if totals["wrong_by_category"]:
            st.write("Wrong decisions by category:", totals["wrong_by_category"])
        return

    d = todo[0]
    decision = "KEPT" if d["kept"] else f"REJECTED ({d['reason']})"
    st.subheader(f"{d['category']}: {decision}")
    st.caption(f"Haiku: {d['haiku_reason']}")
    _, middle, _ = st.columns([1, 3, 1])
    middle.image(d["image"], width="stretch")
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.button("1 · Haiku was right", shortcut="1", on_click=_verdict, args=(d, "right"))
        st.button("2 · Haiku was wrong", shortcut="2", on_click=_verdict, args=(d, "wrong"))
        st.button("S · skip", shortcut="S", on_click=_skip, args=(d["id"],))
