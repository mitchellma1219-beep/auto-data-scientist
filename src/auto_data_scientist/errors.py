"""Exception types. Every user-facing failure is an ``AdsError`` with an actionable message."""


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
