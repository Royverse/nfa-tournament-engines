"""Pure helpers for assembling the research report a forecast is built on.

The report is published inside the bot's Metaculus comment, which is private
while the question is open and becomes public after it closes. So a failed
source is recorded by exception type only — never by message, which can carry
URLs or request details.
"""

from __future__ import annotations

from typing import Iterable

NEWS_QUERY_MAX_CHARS = 1500


def news_query(title: str, resolution_criteria: str | None) -> str:
    """The natural-language query sent to the news search."""
    text = title.strip()
    if resolution_criteria:
        text += "\n" + resolution_criteria.strip()
    return text[:NEWS_QUERY_MAX_CHARS]


def compose_research(sections: Iterable[tuple[str, object]]) -> str:
    """Join (title, result) pairs into one markdown report.

    A result is a string (possibly empty — the source had nothing to say) or
    an exception (the source failed). Every attempted source is accounted for,
    so a reader of the comment can tell "no news" from "news search broke".
    """
    parts: list[str] = []
    for title, result in sections:
        if isinstance(result, BaseException):
            parts.append(f"## {title}\nUnavailable this run ({type(result).__name__}).")
        elif isinstance(result, str) and result.strip():
            parts.append(f"## {title}\n{result.strip()}")
    return "\n\n".join(parts) if parts else "No research source returned anything this run."
