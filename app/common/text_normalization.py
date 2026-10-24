import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace. Deterministic, no AI involved —
    this is the cheap first pass before embedding-based similarity."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())
