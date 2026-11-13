"""Streamlit UI: browse the tagged library and upload new photos or videos.

Thin by design: all image, ML, and storage logic lives in tagger/. This file only lays out
widgets and calls tagger functions, so it can later be replaced by a React frontend.

Run with:  uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import filter_check_page  # app/ pages (Streamlit adds this folder to the import path)
import labeling_page
import pandas as pd
import streamlit as st

from tagger import CATEGORIES, classify, ingest, store
from tagger.baseline import MODEL_ID as CLIP_MODEL_ID
from tagger.baseline import ClipBaseline

UPLOAD_TYPES = ["jpg", "jpeg", "png", "heic", "heif", "mp4", "mov"]
GRID_COLUMNS = 6

st.set_page_config(page_title="Underwater Media Tagger", page_icon="🐢", layout="wide")


@st.cache_resource
def get_clip() -> ClipBaseline:
    # Loaded once per server process, not on every rerun.
    return ClipBaseline()


@st.cache_resource
def get_classifier() -> classify.Classifier:
    return classify.Classifier(classify.ACTIVE_CHECKPOINT)


def models() -> dict[str, str]:
    """Display name -> model id, the app's model first. CLIP only if v2's weights are missing."""
    found = {}
    if classify.active_checkpoint():
        found["v2 (fine-tuned)"] = classify.ACTIVE_MODEL_ID
    found["CLIP baseline"] = CLIP_MODEL_ID
    return found


def upload_taggers() -> dict:
    """Every available model tags uploads, so they can be compared like the library."""
    taggers = {CLIP_MODEL_ID: get_clip()}
    if classify.active_checkpoint():
        taggers[classify.ACTIVE_MODEL_ID] = get_classifier()
    return taggers


def timestamp(t_s: float | None) -> str:
    return "" if t_s is None else f" @ {int(t_s // 60)}:{int(t_s % 60):02d}"


@st.dialog("Image details", width="large")
def show_details(row: dict) -> None:
    st.image(row["file_path"], width="stretch")
    st.caption(f"{Path(row['source']).name}{timestamp(row['t_s'])} · trip: {row['trip']}")
    # Every model side by side: where they disagree is the interesting part.
    answers = store.predictions_for(store.connect(), row["image_id"])
    shown = [(name, m) for name, m in models().items() if m in answers]
    for col, (name, model_id) in zip(st.columns(len(shown)), shown, strict=True):
        answer = answers[model_id]
        col.markdown(f"**{name}:** {answer['label']} ({answer['confidence']:.2f})")
        scores = pd.DataFrame(
            {"category": list(answer["scores"]), "probability": list(answer["scores"].values())}
        )
        col.bar_chart(
            scores, x="category", y="probability", horizontal=True, sort="-probability", height=220
        )
    if row["trip"] == ingest.UPLOAD_TRIP and st.button("Remove this upload", type="primary"):
        remove_upload(row["media_id"])
        st.rerun()  # closes the dialog and redraws the page without the removed upload


def remove_upload(media_id: int) -> None:
    ingest.remove_upload(store.connect(), media_id)
    results = st.session_state.get("upload_results", [])
    st.session_state.upload_results = [r for r in results if r["media_id"] != media_id]


def clear_uploads() -> None:
    ingest.clear_uploads(store.connect())
    st.session_state.upload_results = []


def image_grid(rows: list[dict]) -> None:
    for start in range(0, len(rows), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, row in zip(cols, rows[start : start + GRID_COLUMNS], strict=False):
            with col:
                # Full frame, scaled down by the browser, so fullscreen shows the real image.
                # Fine for a local app (~20 MB per 48-image page over localhost).
                st.image(row["file_path"], width="stretch")
                st.caption(f"**{row['label']}** {row['confidence']:.2f}")
                # Widget keys must be unique on the page, so each image appears only once.
                if st.button("Details", key=f"details-{row['image_id']}"):
                    show_details(row)


def gallery_page(conn) -> None:
    st.title("Gallery")
    available = models()
    if classify.ACTIVE_MODEL_ID not in available.values():
        st.info(
            f"Fine-tuned model weights not found at `{classify.ACTIVE_CHECKPOINT}`, so labels "
            "come from the CLIP baseline. Train v2 with `training/train.py` to use it here."
        )
    with st.sidebar:
        st.header("Filters")
        model_label = st.radio("Model", list(available), help="Whose guesses the grid shows.")
        model_id = available[model_label]
        labels = st.multiselect("Predicted category", CATEGORIES)
        trips = st.multiselect("Trip", store.trips(conn))
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        order = st.selectbox("Sort", list(store.GALLERY_ORDERS))
        page_size = st.select_slider("Images per page", [24, 48, 96], value=48)

    filters = dict(labels=labels, trips=trips, min_confidence=min_conf, order=order)
    _, total = store.gallery(conn, model_id, limit=0, **filters)
    pages = max(1, -(-total // page_size))  # ceiling division

    # Back to page 1 whenever the filters or page size change.
    view = repr((model_id, filters, page_size))
    if st.session_state.get("gallery_view") != view:
        st.session_state.gallery_view = view
        _set_page(1, pages)

    st.caption(f"{total:,} images match · labels are {model_label} guesses, not ground truth")
    pager(pages, "top")
    page = st.session_state.page
    rows, _ = store.gallery(
        conn, model_id, limit=page_size, offset=(page - 1) * page_size, **filters
    )
    image_grid(rows)
    pager(pages, "bottom")


PAGER_POSITIONS = ("top", "bottom")


def _set_page(page: int, pages: int) -> None:
    """Set the current page (clamped to 1..pages) and show it in both page boxes."""
    page = min(max(page, 1), pages)
    st.session_state.page = page
    for position in PAGER_POSITIONS:
        st.session_state[f"page-input-{position}"] = str(page)


def _typed_page(position: str, pages: int) -> None:
    raw = st.session_state[f"page-input-{position}"].strip()
    # Anything that isn't a number puts the current page back in the box.
    _set_page(int(raw) if raw.isdigit() else st.session_state.page, pages)


def pager(pages: int, position: str) -> None:
    """Centered Previous / [page] of N / Next controls, shared by both copies of the pager."""
    with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center"):
        st.button(
            "◀ Previous",
            key=f"prev-{position}",
            disabled=st.session_state.page <= 1,
            on_click=lambda: _set_page(st.session_state.page - 1, pages),
        )
        st.text_input(
            "Page",
            key=f"page-input-{position}",
            label_visibility="collapsed",
            width=70,
            on_change=_typed_page,
            args=(position, pages),
        )
        st.markdown(f"of {pages}", width="content")
        st.button(
            "Next ▶",
            key=f"next-{position}",
            disabled=st.session_state.page >= pages,
            on_click=lambda: _set_page(st.session_state.page + 1, pages),
        )


def upload_page(conn) -> None:
    st.title("Upload")
    st.write("Add photos or videos. Videos are split into frames every 2 seconds, then tagged.")
    files = st.file_uploader("Photos or videos", type=UPLOAD_TYPES, accept_multiple_files=True)
    if files and st.button("Tag uploads", type="primary"):
        with st.spinner("Loading models (first upload only)..."):
            taggers = upload_taggers()
        results = []
        for f in files:
            # One bad file shouldn't stop (or hide) the rest of the batch.
            result = {"name": f.name, "media_id": None, "image_ids": [], "note": "", "error": ""}
            try:
                with st.spinner(f"Processing {f.name}..."):
                    r = ingest.ingest_upload(conn, f.name, f.getvalue(), taggers)
                result.update(media_id=r.media_id, image_ids=r.image_ids)
                if not r.new:
                    result["note"] = "already uploaded earlier, not reprocessed"
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"
            results.append(result)
        # Streamlit re-runs this script on every click (e.g. "Details"), and the Tag button
        # only reads as clicked once. Keep results in session state so they stay on screen.
        st.session_state.upload_results = results

    shown_media = set()
    for i, result in enumerate(st.session_state.get("upload_results", [])):
        name, media_id = result["name"], result["media_id"]
        if result["error"]:
            st.subheader(name)
            st.error(f"Could not process this file: {result['error']}")
            continue
        if media_id in shown_media:  # same content twice in one batch
            st.subheader(name)
            st.info("Same file as one shown above.")
            continue
        shown_media.add(media_id)

        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f"{name}: {len(result['image_ids'])} image(s)", width="content")
            st.button("Remove", key=f"remove-{i}", on_click=remove_upload, args=(media_id,))
        if result["note"]:
            st.caption(result["note"])
        if not result["image_ids"]:  # e.g. every frame of a video was dropped as blurry
            continue
        try:
            rows, _ = store.gallery(
                conn,
                next(iter(models().values())),  # the app's model
                image_ids=result["image_ids"],
                order="as filmed",
                limit=len(result["image_ids"]),
            )
            image_grid(rows)
        except Exception as e:
            st.error(f"Could not display results: {type(e).__name__}: {e}")

    st.divider()
    n_uploads = conn.execute(
        "SELECT COUNT(*) FROM media WHERE trip = ?", (ingest.UPLOAD_TRIP,)
    ).fetchone()[0]
    with st.popover(f"Clear all uploads ({n_uploads})", disabled=n_uploads == 0):
        st.write(
            f"Delete all {n_uploads} uploaded file(s), their frames, and their tags? "
            "Your dive library is not affected."
        )
        st.button("Yes, delete all uploads", type="primary", on_click=clear_uploads)


conn = store.connect()
page = st.sidebar.radio("Page", ["Gallery", "Upload", "Label", "Filter check"])
if page == "Gallery":
    gallery_page(conn)
elif page == "Upload":
    upload_page(conn)
elif page == "Label":
    labeling_page.render(conn)
else:
    filter_check_page.render()
