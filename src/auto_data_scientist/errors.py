"""Exception types. Every user-facing failure is an ``AdsError`` with an actionable message."""

from __future__ import annotations

from dataclasses import dataclass


class AdsError(Exception):
    """Base class for expected, user-correctable failures."""


class DataError(AdsError):
    """The input data cannot be loaded or is unusable for the requested analysis."""


class UnsafeQueryError(AdsError):
    """A SQL statement was rejected by the read-only guard."""


class QueryTimeoutError(AdsError):
    """A query exceeded the configured time budget and was interrupted."""


class TranslationError(AdsError):
    """A natural-language question could not be translated without guessing."""


@dataclass(frozen=True)
class Explanation:
    """A failure in words for a person: what happened, why, and what to do next."""

    title: str
    reason: str
    hint: str


_KINDS: tuple[tuple[type[Exception], str, str], ...] = (
    (QueryTimeoutError, "The query was stopped",
     "Filter or aggregate earlier so it has less to do, then run it again."),
    (UnsafeQueryError, "The query was refused",
     "Only one read-only SELECT on the loaded tables can run. Nothing was executed."),
    (TranslationError, "The question was not translated",
     "Ask for a total, average, count or top N of a named column, or write the SQL "
     "yourself. The translator refuses instead of guessing."),
    (DataError, "This data cannot be used for that",
     "Check the file or pick different columns, then try again."),
    (AdsError, "The request could not be completed", "Adjust the request and try again."),
    (ValueError, "A setting is out of range", "Change the value and try again."),
)


def explain(exc: Exception) -> Explanation:
    """Describe an expected failure for display. Unexpected exception types are re-raised,
    so a programming error is never dressed up as a user mistake."""
    for kind, title, hint in _KINDS:
        if isinstance(exc, kind):
            reason = " ".join(str(exc).split()) or kind.__name__
            reason = reason[0].upper() + reason[1:]
            return Explanation(title, reason if reason.endswith((".", "!", "?")) else reason + ".",
                               hint)
    raise exc
