"""Precedents: how the most similar already-resolved Metaculus questions turned out.

Looking up similar resolved questions was done by 34% of Fall 2025 prize
winners and 0% of non-winners (Fisher p = 0.04), and the tournament's
questions often come in series, so a question's own history is often the
best reference class there is. Metaculus ranks posts by embedding similarity
(`similar_to_post_id`, posts/services/feed.py in Metaculus's repo); when a new
post has no embedding yet, a title search stands in.

Resolved questions only: an outcome is a historical fact, never another
forecaster's live opinion.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

POSTS_URL = "https://www.metaculus.com/api/posts/"
TIMEOUT_SECONDS = 20
FETCH_LIMIT = 12
SHOWN = 8
SEARCH_CHARS = 120
_UNSCORED = {"annulled", "ambiguous"}

Http = Callable[..., requests.Response]


def _resolved(post: dict[str, Any], exclude_post_id: int) -> tuple[str, str, str | None] | None:
    """(title, resolution, resolved date) for a usable single-question post."""
    question = post.get("question")
    if not isinstance(question, dict) or post.get("id") == exclude_post_id:
        return None
    resolution = question.get("resolution")
    if resolution is None or str(resolution).lower() in _UNSCORED:
        return None
    title = str(question.get("title") or post.get("title") or "").strip()
    if not title:
        return None
    when = str(question.get("actual_resolve_time") or "")[:10] or None
    if question.get("type") in ("numeric", "discrete") and question.get("unit"):
        resolution = f"{resolution} {question['unit']}"
    return title, str(resolution), when


def summarise_precedents(posts: list[dict[str, Any]], exclude_post_id: int) -> str:
    rows = [r for r in (_resolved(p, exclude_post_id) for p in posts) if r][:SHOWN]
    if not rows:
        return ""
    lines = ["The most similar questions Metaculus has already resolved (closest first):"]
    for title, resolution, when in rows:
        shown = resolution.upper() if resolution.lower() in ("yes", "no") else resolution
        lines.append(f'- "{title}" → resolved {shown}' + (f" ({when})" if when else ""))
    binary = [r for r in rows if r[1].lower() in ("yes", "no")]
    if binary:
        yes = sum(1 for r in binary if r[1].lower() == "yes")
        lines.append(f"Yes/no base rate among these: {yes} of {len(binary)} resolved Yes.")
    lines.append("Similar is not identical: check each one's deadline, threshold and wording before using it as a reference class.")
    return "\n".join(lines)


def _fetch(params: dict[str, Any], token: str, get: Http) -> list[dict[str, Any]]:
    response = get(POSTS_URL, params=params, headers={"Authorization": f"Token {token}"}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    results = (response.json() or {}).get("results")
    return results if isinstance(results, list) else []


def precedent_research(post_id: int, title: str, token: str, *, get: Http = requests.get) -> str:
    base = {"statuses": "resolved", "limit": FETCH_LIMIT}
    similar = _fetch({**base, "similar_to_post_id": post_id}, token, get)
    usable = sum(1 for p in similar if _resolved(p, post_id))
    posts = similar
    if similar and not usable:
        # Results but none usable: say what came back, so a changed API shape shows in the log.
        questions = [p.get("question") for p in similar if isinstance(p, dict) and isinstance(p.get("question"), dict)]
        statuses = sorted({str(q.get("status")) for q in questions})
        resolutions = sorted({repr(q.get("resolution")) for q in questions})[:5]
        logger.warning(
            f"precedents: none of {len(similar)} results usable; "
            f"{len(questions)} single questions, statuses {statuses}, resolutions {resolutions}"
        )
    if not usable:
        posts = _fetch({**base, "search": title[:SEARCH_CHARS]}, token, get)
    logger.info(
        f"precedents for post {post_id}: {len(similar)} similar ({usable} usable)"
        + ("" if usable else f", {len(posts)} by title search")
    )
    # Nothing usable adds nothing to the prompt; the log line above says why.
    return summarise_precedents(posts, post_id)
