"""Labeling page: hand-label frames from my footage to build the test set (the answer key).

UI only. Which frame to show next, and all label storage, live in tagger/labeling.py and
tagger/store.py. CLIP's guess is never shown here, so it can't anchor my labels.
"""

import random

import streamlit as st

from tagger import CATEGORIES, labeling, store
from tagger.ingest import UPLOAD_TRIP

RULES = """
1. **An animal that is clearly a subject wins.** Ray + diver → `rays`.
   Turtle over a reef → `turtles`.
2. **Several animals → the most prominent one:** largest, closest, sharpest,
   or what the shot is about.
3. **Background animals don't count.** Tiny scattered fish over a wide reef → `scenery`.
   *Test: crop the fish out. If it still works as a scenery shot, it's `scenery`.*
4. **Diver as the main subject → `diver`**, unless an animal is also a clear subject (rule 1).
5. **A school of fish as the subject → `fish`.**
6. **Reef animals vs. the reef itself.** A distinct animal you'd point your light at →
   `invertebrates` (a clam, scallop, crab, or sea star that's clearly the subject).
   What makes up the reef → `scenery` (coral, sponges, sea fans, whip coral, a lone anemone).
   An anemone with a clownfish in it → `fish`.
7. **Can't decide in ~3 seconds → Skip.** It isn't labeled and doesn't count.

**Unusable** = fogged lens, on the boat, at the surface, black frame.
"""


def _state():
    s = st.session_state
    s.setdefault("label_current", None)  # labeling.Pick shown in auto mode
    s.setdefault("label_skipped", set())  # image ids skipped this session
    s.setdefault("label_history", [])  # [(Pick, mode)] labeled this session, for undo
    s.setdefault("browse_index", 0)
    return s


def _save(pick: labeling.Pick, label: str, mode: str) -> None:
    s = _state()
    conn = store.connect()
    with conn:  # commit
        store.set_label(conn, pick.image_id, label, labeling.SOURCE, pick.selection)
    s.label_history.append((pick, mode))
    if mode == "auto":
        s.label_current = None
    else:
        s.browse_index += 1  # move on to the next frame in the clip


def _skip(pick: labeling.Pick) -> None:
    s = _state()
    s.label_skipped.add(pick.image_id)
    s.label_current = None


def _undo() -> None:
    s = _state()
    if not s.label_history:
        return
    pick, mode = s.label_history.pop()
    conn = store.connect()
    with conn:
        store.delete_label(conn, pick.image_id, labeling.SOURCE)
    if mode == "auto":
        s.label_current = pick  # show the same frame again
    else:
        s.browse_index = max(s.browse_index - 1, 0)


def _progress(conn) -> None:
    counts = labeling.label_counts(conn)
    status = labeling.category_status(conn)
    cols = st.columns(len(CATEGORIES) + 1)
    for col, c in zip(cols, CATEGORIES, strict=False):
        mark = {"done": " ✓", "gave up": " (pool used up)"}.get(status[c], "")
        col.metric(c, f"{counts[c]}/{labeling.QUOTA}", help=f"Status: {status[c]}")
        if mark:
            col.caption(mark.strip())
    cols[-1].metric("unusable", counts[store.UNUSABLE])


def _label_buttons(pick: labeling.Pick, mode: str, allow_skip: bool) -> None:
    with st.container(horizontal=True, horizontal_alignment="center"):
        for i, c in enumerate(CATEGORIES, start=1):
            st.button(
                f"{i} · {c}",
                key=f"{mode}-label-{c}",
                shortcut=str(i),
                on_click=_save,
                args=(pick, c, mode),
            )
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.button(
            "0 · unusable",
            key=f"{mode}-label-unusable",
            shortcut="0",
            on_click=_save,
            args=(pick, store.UNUSABLE, mode),
        )
        if allow_skip:
            st.button(
                "S · skip / not sure", key="auto-skip", shortcut="S", on_click=_skip, args=(pick,)
            )
        st.button(
            "U · undo last",
            key=f"{mode}-undo",
            shortcut="U",
            on_click=_undo,
            disabled=not st.session_state.label_history,
        )


def _show_frame(file_path: str) -> None:
    _, middle, _ = st.columns([1, 4, 1])
    middle.image(file_path, width="stretch")


def _auto_mode(conn) -> None:
    s = _state()
    if s.label_current is None:
        s.label_current = labeling.next_frame(conn, random.Random(), exclude_ids=s.label_skipped)
    pick = s.label_current
    if pick is None:
        st.success("Nothing left to label: every category hit its quota or ran out of candidates.")
        return
    _show_frame(pick.file_path)
    _label_buttons(pick, "auto", allow_skip=True)


def _browse_mode(conn) -> None:
    s = _state()
    trips = [t for t in store.trips(conn) if t != UPLOAD_TRIP]
    left, right = st.columns(2)
    trip = left.selectbox("Trip", trips, key="browse_trip")
    clips = {c["id"]: c for c in labeling.library_clips(conn, trip)}

    def describe(media_id: int) -> str:
        c = clips[media_id]
        frames = f"{c['frames']} frame{'s' if c['frames'] != 1 else ''}"
        return f"{c['path'].rsplit('/', 1)[-1]}  ({frames}, {c['labeled']} labeled)"

    # Options are the fixed clip ids and the widget has a fixed key, so the selection stays
    # put when the displayed label counts change after labeling a frame.
    clip_id = right.selectbox("Clip", list(clips), format_func=describe, key=f"browse_clip-{trip}")
    if clip_id is None:
        return
    clip = clips[clip_id]
    if s.get("browse_clip") != clip["id"]:  # new clip: start at its first frame
        s.browse_clip, s.browse_index = clip["id"], 0

    frames = labeling.clip_frames(conn, clip["id"])
    s.browse_index = min(s.browse_index, len(frames) - 1)
    frame = frames[s.browse_index]
    t = "" if frame["t_s"] is None else f" @ {int(frame['t_s'] // 60)}:{int(frame['t_s'] % 60):02d}"
    with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center"):
        st.button(
            "← Previous",
            shortcut="Left",
            disabled=s.browse_index == 0,
            on_click=lambda: s.update(browse_index=s.browse_index - 1),
        )
        st.markdown(f"Frame {s.browse_index + 1} of {len(frames)}{t}", width="content")
        st.button(
            "Next →",
            shortcut="Right",
            disabled=s.browse_index >= len(frames) - 1,
            on_click=lambda: s.update(browse_index=s.browse_index + 1),
        )
    if frame["label"]:
        st.info(f"Your label: **{frame['label']}** (press a key to change it)")
    _show_frame(frame["file_path"])
    pick = labeling.Pick(frame["id"], frame["file_path"], "browse")
    _label_buttons(pick, "browse", allow_skip=False)


def render(conn) -> None:
    st.title("Label test frames")
    with st.expander("Labeling rules"):
        st.markdown(RULES)
    _progress(conn)
    mode = st.segmented_control(
        "Mode",
        ["Auto", "Browse clips"],
        default="Auto",
        help="Auto picks frames for you (random + rare-category candidates). "
        "Browse lets you step through a clip you remember.",
    )
    if mode == "Browse clips":
        _browse_mode(conn)
    else:
        _auto_mode(conn)
