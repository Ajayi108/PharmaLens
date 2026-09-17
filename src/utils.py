import re


def sentences(text: str) -> list[str]:
    """Keep source substrings intact; conservative English/German boundaries."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])|\n\s*\n", text) if part.strip()]


def clean_text(text: str) -> str:
    # Retain the original extracted characters, normalizing whitespace only.
    return re.sub(r"\s+", " ", text).strip()
