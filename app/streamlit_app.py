"""Streamlit UI: browse the tagged library and upload new photos or videos.

Thin by design: all image, ML, and storage logic lives in tagger/. This file only lays out
widgets and calls tagger functions, so it can later be replaced by a React frontend.

Run with:  uv run streamlit run app/streamlit_app.py
"""

import io
from pathlib import Path

import color_check_page  # app/ pages (Streamlit adds this folder to the import path)
import filter_check_page
import labeling_page
import numpy as np
import pandas as pd
import search_page
import streamlit as st
from PIL import Image

from tagger import CATEGORIES, classify, color, ingest, locate, similar, store
from tagger.baseline import MODEL_ID as CLIP_MODEL_ID
from tagger.baseline import ClipBaseline

UPLOAD_TYPES = ["jpg", "jpeg", "png", "heic", "heif", "mp4", "mov"]
GRID_COLUMNS = 6
# What a tag can be corrected to: the categories, or unusable (fogged, on the boat, at the
# surface, too dark or blurred to tell). The models themselves only predict the categories.
CORRECTABLE = (*CATEGORIES, store.UNUSABLE)

st.set_page_config(page_title="Underwater Media Tagger", page_icon="🐢", layout="wide")


@st.cache_resource
def get_clip() -> ClipBaseline:
    # Loaded once per server process, not on every rerun.
    return ClipBaseline()


@st.cache_resource
def get_classifier() -> classify.Ensemble:
    return classify.Ensemble(classify.ACTIVE_CHECKPOINTS)


@st.cache_resource
def get_locator() -> locate.Locator:
    return locate.Locator(get_classifier().members)


def models() -> dict[str, str]:
    """Display name -> model id, the app's model first. CLIP only if v3's weights are missing."""
    found = {}
    if classify.active_checkpoints():
        found["v3 (fine-tuned)"] = classify.ACTIVE_MODEL_ID
    found["CLIP baseline"] = CLIP_MODEL_ID
    return found


def upload_taggers() -> dict:
    """Every available model tags uploads, so they can be compared like the library."""
    taggers = {CLIP_MODEL_ID: get_clip()}
    if classify.active_checkpoints():
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
    correction_controls(row["image_id"])
    color_section(row)
    if row["trip"] == ingest.UPLOAD_TRIP and st.button("Remove this upload", type="primary"):
        remove_upload(row["media_id"])
        st.rerun()  # closes the dialog and redraws the page without the removed upload


def correction_controls(image_id: int) -> None:
    """Fix a wrong tag. The correction only changes what the gallery shows and searches: my
    test labels and the models stay as they are."""
    conn = store.connect()
    current = store.correction_for(conn, image_id)
    st.divider()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        choice = st.selectbox(
            "Correct tag",
            CORRECTABLE,
            index=CORRECTABLE.index(current) if current else None,
            placeholder="Pick the right category",
            key=f"correct-{image_id}",
        )
        if st.button("Save correction", type="primary", disabled=choice in (None, current)):
            with conn:
                store.set_label(conn, image_id, choice, store.CORRECTION)
            st.rerun()  # closes the dialog; the grid shows the corrected tag
        if current and st.button("Remove correction"):
            with conn:
                store.delete_label(conn, image_id, store.CORRECTION)
            st.rerun()
    if current:
        st.caption(f"Corrected by you to **{current}**.")


FINISH_LABELS = [level.capitalize() for level in color.FINISHES]  # None, Light, Medium
WHY_WHOLE = {
    locate.SCENERY: "Whole frame: scenery has no single subject",
    locate.UNUSABLE: "Whole frame: marked unusable",
    locate.LOW_CONFIDENCE: "Whole frame: tag confidence under 0.5, so it may be wrong",
    locate.NO_CLEAR_SUBJECT: "Whole frame: no clear subject in the heatmap",
}


@st.cache_data(max_entries=16, show_spinner="Finding the subject...")
def located(image_id: int, file_path: str, label: str, confidence: float) -> locate.Located:
    """Grad-CAM for one image, cached so the strength slider doesn't re-run it."""
    with Image.open(file_path) as img:
        return get_locator().locate(img.convert("RGB"), label, confidence)


@st.cache_data(max_entries=32, show_spinner="Correcting colors...")
def corrected(
    file_path: str,
    _found: locate.Located,
    image_id: int,
    mode: str,
    strength: float,
    finish_level: str,
):
    """(corrected RGB, settings). `image_id` keys the cache in place of the unhashable mask."""
    rgb = np.asarray(Image.open(file_path).convert("RGB"))
    mask = _found.mask if _found.has_subject else None
    settings = color.estimate(rgb, mask, mode)
    return color.apply(rgb, mask, settings, strength, finish_level), settings


def color_section(row: dict) -> None:
    """Subject-aware color correction (docs/color_plan.md): the Grad-CAM subject steers the
    white balance, the water is kept blue. Computed only when switched on, kept in memory."""
    st.divider()
    image_id = row["image_id"]
    if not st.toggle("Color correction", key=f"color-on-{image_id}"):
        return
    if not classify.active_checkpoints():
        st.warning("The v3 model's weights are missing, so the subject can't be found.")
        return
    label, confidence = locate.shown_tag(store.connect(), image_id, classify.ACTIVE_MODEL_ID)
    found = located(image_id, row["file_path"], label, confidence)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        mode = st.segmented_control(
            "Measure the color on",
            ["Subject", "Whole frame"],
            default="Subject" if found.has_subject else "Whole frame",
            disabled=not found.has_subject,
            key=f"color-mode-{image_id}",
        )
        strength = st.slider(
            "Strength", 0.0, 1.0, color.STRENGTH, 0.05, key=f"color-strength-{image_id}"
        )
        # Three presets tuned on set A rather than free sliders: enough to adjust a photo,
        # not enough to over-sharpen it into the crunchy look.
        finish_level = st.segmented_control(
            "Contrast and sharpness",
            FINISH_LABELS,
            default="Medium",
            key=f"color-finish-{image_id}",
        )
    mode = color.SUBJECT if mode == "Subject" and found.has_subject else color.WHOLE
    finish_level = (finish_level or "Medium").lower()  # clicking the selected one clears it
    out, settings = corrected(row["file_path"], found, image_id, mode, strength, finish_level)

    if found.has_subject and mode == color.SUBJECT:
        what = f"Corrected for: **{label}** ({confidence:.2f})"
    elif found.has_subject:
        what = "Whole frame: your choice"
    else:
        what = WHY_WHOLE[found.reason]
    if settings.amount < 1:
        what += f" · Shallow: light correction ({settings.amount:.0%})"
    st.caption(what)

    corrected_tab, heatmap_tab, original_tab = st.tabs(["Corrected", "Heatmap", "Original"])
    with Image.open(row["file_path"]) as img:
        rgb = np.asarray(img.convert("RGB"))
    corrected_tab.image(out, width="stretch")
    if found.heatmap is not None:
        heatmap_tab.image(locate.overlay(rgb, found), width="stretch")
        heatmap_tab.caption("Red: where the model saw the tag. White: the subject it found.")
    else:
        heatmap_tab.caption("No heatmap for scenery or unusable frames.")
    original_tab.image(rgb, width="stretch")

    buffer = io.BytesIO()
    Image.fromarray(out).save(buffer, format="JPEG", quality=92)
    st.download_button(
        "Download corrected JPEG",
        buffer.getvalue(),
        file_name=f"{Path(row['file_path']).stem}_corrected.jpg",
        mime="image/jpeg",
        key=f"color-download-{image_id}",
        on_click="ignore",  # a rerun would close the details dialog
    )


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
                if row["corrected"]:
                    st.caption(f"**{row['label']}** ✎ corrected (model: {row['model_label']})")
                elif row["labeled_unusable"]:
                    st.caption(f"**unusable** ✔ your label (model: {row['model_label']})")
                else:
                    st.caption(f"**{row['label']}** {row['confidence']:.2f}")
                # Widget keys must be unique on the page, so each image appears only once.
                if st.button("Details", key=f"details-{row['image_id']}"):
                    show_details(row)


def gallery_page(conn) -> None:
    st.title("Gallery")
    available = models()
    if classify.ACTIVE_MODEL_ID not in available.values():
        st.info(
            "Fine-tuned v3 weights not found in `models/`, so labels come from the CLIP "
            "baseline. Train v3 with `training/sweep.py --steps train` to use it here."
        )
    with st.sidebar:
        st.header("Filters")
        model_label = st.radio("Model", list(available), help="Whose guesses the grid shows.")
        model_id = available[model_label]
        labels = st.multiselect(
            "Category",
            CORRECTABLE,
            help="The model's guess, or your correction where you made one. Only corrections "
            "can be unusable.",
        )
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
                    if r.image_ids:  # so search can find similar frames to (and from) uploads
                        similar.add(conn, get_clip(), r.image_ids)
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
PAGES = ["Gallery", "Search", "Upload", "Label", "Filter check", "Color check"]
page = st.sidebar.radio("Page", PAGES)
if page == "Gallery":
    gallery_page(conn)
elif page == "Search":
    search_page.render()
elif page == "Upload":
    upload_page(conn)
elif page == "Label":
    labeling_page.render(conn)
elif page == "Filter check":
    filter_check_page.render()
else:
    color_check_page.render()
