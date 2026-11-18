"""Search page: ask questions about the library in plain English.

UI only. The tools, the agent loop, and cost tracking live in tagger/search.py; the library is
opened read-only for it. The API key loads from .env here, never from the shell.
"""

import os
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

from tagger import search, similar, store
from tagger.classify import ACTIVE_MODEL_ID

GRID_COLUMNS = 6
EXAMPLES = (
    "Which trip had the most rays?",
    "Show me turtles the model wasn't sure about",
    "Where do the fine-tuned model and CLIP disagree on fish?",
)


@st.cache_resource
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@st.cache_resource
def _index(mtime: float) -> similar.Index | None:
    # Keyed by the file's modification time, so new uploads' embeddings are picked up.
    return similar.Index.load()


def _show_images(image_ids: list[int]) -> None:
    if not image_ids:
        return
    rows, _ = store.gallery(
        store.connect(), ACTIVE_MODEL_ID, image_ids=image_ids, limit=len(image_ids)
    )
    by_id = {r["image_id"]: r for r in rows}
    ordered = [by_id[i] for i in image_ids if i in by_id]  # in the order the tools returned
    for start in range(0, len(ordered), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, row in zip(cols, ordered[start : start + GRID_COLUMNS], strict=False):
            at = (
                ""
                if row["t_s"] is None
                else f" @ {int(row['t_s'] // 60)}:{int(row['t_s'] % 60):02d}"
            )
            col.image(row["file_path"], width="stretch")
            col.caption(f"**{row['label']}** · {Path(row['source']).name}{at} · #{row['image_id']}")


def _show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["text"] or "_(no answer)_")
        _show_images(turn["image_ids"])
        if turn["tool_calls"]:
            with st.expander(f"How I found this ({len(turn['tool_calls'])} tool calls)"):
                for call in turn["tool_calls"]:
                    st.code(f"{call['name']}({call['input']})", language=None)
        st.caption(f"Cost: ${turn['cost_usd']:.4f}")


def render() -> None:
    st.title("Search")
    st.caption(
        "Ask about your dives in plain English. Answers use the fine-tuned model's tags (or your "
        "corrections), so they can be wrong. Image numbers (#) can be used in follow-ups."
    )
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("Add ANTHROPIC_API_KEY to the .env file at the repo root to use search.")
        return

    s = st.session_state
    s.setdefault("search_history", [])  # the API conversation, append-only
    s.setdefault("search_turns", [])  # what's shown: question, answer, images, tool calls, cost
    if not s.search_turns:
        st.markdown("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES))

    for turn in s.search_turns:
        _show_turn(turn)

    if s.search_turns:  # after the last answer, so the total is in view without scrolling up
        total = sum(t["cost_usd"] for t in s.search_turns)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"This conversation: ${total:.4f}", width="content")
            if st.button("New conversation"):
                s.search_history, s.search_turns = [], []
                st.rerun()

    question = st.chat_input("Ask about your library")
    if not question:
        return
    embeddings = similar.EMBEDDINGS
    index = _index(embeddings.stat().st_mtime if embeddings.exists() else 0.0)
    lib = search.LibraryTools(store.connect_readonly(), index)
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Searching your library..."):
        try:
            answer, history = search.ask(_client(), lib, s.search_history, question)
        except anthropic.APIError as e:
            st.error(f"The search failed: {type(e).__name__}: {e}")
            return
    s.search_history = history
    s.search_turns.append(
        {
            "question": question,
            "text": answer.text,
            "image_ids": answer.image_ids,
            "tool_calls": answer.tool_calls,
            "cost_usd": answer.cost_usd,
        }
    )
    st.rerun()  # redraw the whole conversation, including this answer
