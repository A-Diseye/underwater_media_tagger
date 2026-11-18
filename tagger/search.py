"""Library search agent: answers questions about my tagged dive library.

Claude Sonnet calls read-only tools over the library and writes the answer from what they
return. The SDK's tool runner owns the loop (ask -> run the tools Claude picks -> send results
back -> repeat); this module only defines the tools and reads each turn as it happens.

Design choices:
- Fixed, tested queries instead of letting Claude write SQL: less flexible, but no wrong or slow
  queries, and the model never gets raw database access.
- Read-only at the database level: the tools get a connection opened with mode=ro, so even a bug
  can't change a label, correction, or prediction.
- The tags are the gallery's: the fine-tuned model's guess, unless I corrected it or labeled the
  frame unusable. The agent never sees my test labels or any accuracy numbers.
- The images shown under an answer are the ones the tools actually returned, never ids taken
  from Claude's text, so it can't point at an image that doesn't exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from anthropic import beta_tool

from tagger import CATEGORIES, store
from tagger.baseline import MODEL_ID as CLIP_MODEL_ID
from tagger.classify import ACTIVE_MODEL_ID
from tagger.pricing import SONNET, cost_usd
from tagger.similar import Index

MODEL = SONNET
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TURNS = 8
MAX_RESULTS = 24  # images a tool returns at most, and the most shown under an answer

SYSTEM = f"""You answer questions about the user's own library of underwater photos and \
frames from their dive videos, using the tools. Each image has one tag: \
{", ".join(CATEGORIES)}, or unusable where the user flagged it. Tags are the fine-tuned model's \
guesses unless the user corrected them, so they can be wrong; mention it when that matters (for \
example, low confidence). Base every fact on tool results, and never invent image ids, counts, \
or trips.

The images your tools return are shown to the user automatically under your answer, so describe \
them ("the first few are mantas over sand") rather than listing ids. If a result is close but not \
quite what they asked for, say so and suggest a follow-up, such as other frames from the same \
clip or similar-looking frames. Keep answers short."""


def _at(t_s: float | None) -> str | None:
    return None if t_s is None else f"{int(t_s // 60)}:{int(t_s % 60):02d}"


class LibraryTools:
    """The tools' logic, over a read-only connection. Records every image a tool returns."""

    def __init__(
        self,
        conn,
        index: Index | None = None,
        model_id: str = ACTIVE_MODEL_ID,
        clip_id: str = CLIP_MODEL_ID,
    ):
        self.conn, self.index, self.model_id, self.clip_id = conn, index, model_id, clip_id
        self.shown: list[int] = []

    def _images(self, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            self.shown.append(r["image_id"])
            item = {
                "image_id": r["image_id"],
                "tag": r["label"],
                "confidence": round(r["confidence"], 2),
                "trip": r["trip"],
                "clip": Path(r["source"]).name,
                "at": _at(r["t_s"]),
            }
            if r["corrected"]:
                item["tag_source"] = f"corrected by user (model said {r['model_label']})"
            elif r["labeled_unusable"]:
                item["tag_source"] = "flagged unusable by user"
            out.append(item)
        return out

    def _rows(self, image_ids: list[int], order: str = "as filmed") -> list[dict]:
        if not image_ids:
            return []
        rows, _ = store.gallery(
            self.conn, self.model_id, image_ids=image_ids, order=order, limit=len(image_ids)
        )
        return rows

    def find_images(
        self,
        categories: list[str] | None = None,
        trips: list[str] | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        order: str = "most confident",
        limit: int = 12,
    ) -> dict:
        rows, total = store.gallery(
            self.conn,
            self.model_id,
            labels=categories,
            trips=trips,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            order=order,
            limit=min(max(limit, 1), MAX_RESULTS),
        )
        return {"total_matching": total, "images": self._images(rows)}

    def count_by(
        self,
        group: str,
        categories: list[str] | None = None,
        trips: list[str] | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> list[dict]:
        counts = store.count_by(
            self.conn, self.model_id, group, categories, trips, min_confidence, max_confidence
        )
        if group == "clip":
            counts = [{**c, "group": Path(c["group"]).name} for c in counts[:50]]
        return counts

    def list_trips(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT m.trip, COUNT(DISTINCT m.id) AS clips, COUNT(i.id) AS images
            FROM media m JOIN images i ON i.media_id = m.id
            WHERE m.trip IS NOT NULL GROUP BY m.trip ORDER BY m.trip
            """
        )
        return [dict(r) for r in rows]

    def model_disagreements(
        self, category: str | None = None, trip: str | None = None, limit: int = 12
    ) -> dict:
        where, filters = ["a.label != b.label"], []
        if category:
            where.append("(a.label = ? OR b.label = ?)")
            filters += [category, category]
        if trip:
            where.append("m.trip = ?")
            filters.append(trip)
        base = f"""
            FROM predictions a
            JOIN predictions b ON b.image_id = a.image_id AND b.model = ?
            JOIN images i ON i.id = a.image_id JOIN media m ON m.id = i.media_id
            WHERE a.model = ? AND {" AND ".join(where)}
        """
        params = [self.clip_id, self.model_id, *filters]  # in the order the ? appear
        total = self.conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
        picked = self.conn.execute(
            f"""
            SELECT a.image_id, a.label AS fine_tuned, b.label AS clip {base}
            ORDER BY a.confidence DESC LIMIT ?
            """,
            [*params, min(max(limit, 1), MAX_RESULTS)],
        ).fetchall()
        guesses = {r["image_id"]: (r["fine_tuned"], r["clip"]) for r in picked}
        images = self._images(self._rows(list(guesses), order="most confident"))
        for image in images:
            image["fine_tuned_said"], image["clip_said"] = guesses[image["image_id"]]
        return {"total_disagreements": total, "images": images}

    def image_details(self, image_id: int) -> dict:
        rows = self._rows([image_id])
        if not rows:
            raise ValueError(f"no image with id {image_id}")
        [image] = self._images(rows)
        for model, answer in store.predictions_for(self.conn, image_id).items():
            name = (
                "fine_tuned"
                if model == self.model_id
                else "clip"
                if model == self.clip_id
                else None
            )
            if name:
                top = sorted(answer["scores"].items(), key=lambda kv: -kv[1])[:3]
                image[f"{name}_top3"] = {c: round(p, 2) for c, p in top}
        return image

    def clip_frames(self, image_id: int, window_s: float = 20.0) -> dict:
        frame = self.conn.execute(
            "SELECT media_id, t_s FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if frame is None:
            raise ValueError(f"no image with id {image_id}")
        if frame["t_s"] is None:  # a photo: its "clip" is just itself
            return {"images": self._images(self._rows([image_id]))}
        ids = [
            r["id"]
            for r in self.conn.execute(
                "SELECT id FROM images WHERE media_id = ? AND ABS(t_s - ?) <= ? ORDER BY t_s",
                (frame["media_id"], frame["t_s"], window_s),
            )
        ][:MAX_RESULTS]
        return {"images": self._images(self._rows(ids))}

    def similar_images(self, image_id: int, limit: int = 12, other_clips_only: bool = True) -> dict:
        if self.index is None:
            raise ValueError("similar-image search isn't set up (no embeddings file)")
        if image_id not in self.index:
            raise ValueError(f"no embedding for image {image_id}")
        exclude: set[int] = set()
        if other_clips_only:
            exclude = {
                r["id"]
                for r in self.conn.execute(
                    "SELECT id FROM images WHERE media_id = "
                    "(SELECT media_id FROM images WHERE id = ?)",
                    (image_id,),
                )
            }
        # Ask for extra, since images removed from the library may linger in the embeddings file.
        limit = min(max(limit, 1), MAX_RESULTS)
        nearest = self.index.nearest(image_id, limit * 2, exclude)
        rows = {r["image_id"]: r for r in self._rows([i for i, _ in nearest])}
        ranked = [rows[i] for i, _ in nearest if i in rows][:limit]
        images = self._images(ranked)
        score = dict(nearest)
        for image in images:
            image["similarity"] = round(score[image["image_id"]], 2)
        return {"images": images}


Category = Literal["turtles", "rays", "fish", "invertebrates", "diver", "scenery", "unusable"]
Trip = Literal["azores", "bahamas", "mexico", "philippines", "uploads"]
Order = Literal["most confident", "least confident", "as filmed"]


def make_tools(lib: LibraryTools) -> list:
    """The tools Claude can call, as SDK tool functions. The SDK builds each tool's schema from
    the type hints and docstring, and calls the function when Claude asks for it."""

    @beta_tool
    def find_images(
        categories: list[Category] | None = None,
        trips: list[Trip] | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        order: Order = "most confident",
        limit: int = 12,
    ) -> str:
        """Find images by tag, trip, and the model's confidence. Returns how many match in total
        and up to `limit` of them.

        Args:
            categories: Tags to include; omit for all.
            trips: Trips to include; omit for all.
            min_confidence: Lowest model confidence to include, 0 to 1.
            max_confidence: Highest model confidence to include, 0 to 1. Use a low value to find
                images the model was unsure about.
            order: "most confident", "least confident", or "as filmed" (clip by clip, in time
                order).
            limit: How many images to return, at most 24.
        """
        return json.dumps(
            lib.find_images(categories, trips, min_confidence, max_confidence, order, limit)
        )

    @beta_tool
    def count_by(
        group: Literal["category", "trip", "clip"],
        categories: list[Category] | None = None,
        trips: list[Trip] | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> str:
        """Count images grouped by tag, trip, or clip, largest first, e.g. "which trip had the
        most rays" is count_by(group="trip", categories=["rays"]).

        Args:
            group: What to group by.
            categories: Only count images with these tags.
            trips: Only count images from these trips.
            min_confidence: Lowest model confidence to count, 0 to 1.
            max_confidence: Highest model confidence to count, 0 to 1.
        """
        return json.dumps(lib.count_by(group, categories, trips, min_confidence, max_confidence))

    @beta_tool
    def list_trips() -> str:
        """List the trips in the library, with how many clips and images each has."""
        return json.dumps(lib.list_trips())

    @beta_tool
    def model_disagreements(
        category: Category | None = None, trip: Trip | None = None, limit: int = 12
    ) -> str:
        """Find images where the fine-tuned model and the zero-shot CLIP baseline disagree.

        Args:
            category: Only disagreements where either model said this tag.
            trip: Only images from this trip.
            limit: How many images to return, at most 24.
        """
        return json.dumps(lib.model_disagreements(category, trip, limit))

    @beta_tool
    def image_details(image_id: int) -> str:
        """Everything about one image: its tag and where it came from (trip, clip, time), and
        both models' top three guesses with confidence.

        Args:
            image_id: An image id from an earlier tool result.
        """
        return json.dumps(lib.image_details(image_id))

    @beta_tool
    def clip_frames(image_id: int, window_s: float = 20.0) -> str:
        """Frames from the same video clip around an image, in filmed order. Use it to find a
        better frame of the same moment (sharper, closer, better angle).

        Args:
            image_id: An image id from an earlier tool result.
            window_s: How many seconds before and after to include.
        """
        return json.dumps(lib.clip_frames(image_id, window_s))

    @beta_tool
    def similar_images(image_id: int, limit: int = 12, other_clips_only: bool = True) -> str:
        """Images anywhere in the library that look similar to this one.

        Args:
            image_id: An image id from an earlier tool result.
            limit: How many images to return, at most 24.
            other_clips_only: Leave out frames from the same clip (use clip_frames for those),
                to find other encounters that looked alike.
        """
        return json.dumps(lib.similar_images(image_id, limit, other_clips_only))

    return [
        find_images,
        count_by,
        list_trips,
        model_disagreements,
        image_details,
        clip_frames,
        similar_images,
    ]


@dataclass
class Answer:
    text: str
    image_ids: list[int]  # images the tools returned this turn, in order, no repeats
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "input"} per call
    cost_usd: float = 0.0
    stop_reason: str | None = None


def ask(client, lib: LibraryTools, history: list, question: str) -> tuple[Answer, list]:
    """Answer one question. Returns the answer and the conversation to pass to the next question.

    The history only ever grows, with every reply exactly as Claude sent it (reasoning included),
    which the API requires for follow-up questions.
    """
    lib.shown = []
    messages = [*history, {"role": "user", "content": question}]
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        tools=make_tools(lib),
        messages=messages,
        max_iterations=MAX_TURNS,
        thread=False,  # send the whole conversation each turn: simpler, and easy to test
        cache_control={"type": "ephemeral"},  # the instructions and tool list are reused
        output_config={"effort": "medium"},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    tool_calls, cost, last = [], 0.0, None
    for message in runner:
        last = message
        cost += cost_usd(MODEL, message.usage)
        messages.append(message.to_param())
        tool_calls += [
            {"name": b.name, "input": b.input} for b in message.content if b.type == "tool_use"
        ]
        results = runner.generate_tool_call_response()  # cached: the runner reuses it
        if results is not None:
            messages.append(results)

    text = " ".join(b.text for b in last.content if b.type == "text").strip() if last else ""
    if last is not None and last.stop_reason == "refusal":
        text = "I can't help with that one."
    elif last is not None and last.stop_reason == "tool_use":
        text = (text + "\n\n" if text else "") + "(Stopped after the tool-call limit.)"
    seen, image_ids = set(), []
    for image_id in lib.shown:
        if image_id not in seen:
            seen.add(image_id)
            image_ids.append(image_id)
    answer = Answer(
        text=text,
        image_ids=image_ids[:MAX_RESULTS],
        tool_calls=tool_calls,
        cost_usd=cost,
        stop_reason=last.stop_reason if last else None,
    )
    return answer, messages
