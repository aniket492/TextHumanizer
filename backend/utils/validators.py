"""Input validation for text endpoints."""
from __future__ import annotations

from ml.preprocessing.normalize_text import normalize_whitespace, word_count


class TextValidationError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def validate_text(text: str | None, *, min_words: int, max_words: int, max_chars: int) -> str:
    """Normalise and validate user text. Returns the normalised text or raises TextValidationError."""
    if text is None or not text.strip():
        raise TextValidationError("empty_input", "Please enter some text.")
    if len(text) > max_chars * 2:
        # Reject absurdly large payloads before doing any work on them.
        raise TextValidationError(
            "input_too_long", f"Text is too long. The limit is {max_chars:,} characters.", 413
        )
    clean = normalize_whitespace(text)
    if not clean:
        raise TextValidationError("empty_input", "Please enter some text.")
    if len(clean) > max_chars:
        raise TextValidationError(
            "input_too_long", f"Text is too long. The limit is {max_chars:,} characters.", 413
        )
    letters = sum(c.isalpha() for c in clean)
    if letters < 0.4 * len(clean.replace(" ", "")):
        raise TextValidationError("invalid_text", "The input does not look like natural-language text.")
    n = word_count(clean)
    if n < min_words:
        raise TextValidationError("input_too_short", f"Please enter at least {min_words} words.")
    if n > max_words:
        raise TextValidationError(
            "input_too_long", f"Text is too long ({n:,} words). The limit is {max_words:,} words.", 413
        )
    return clean
