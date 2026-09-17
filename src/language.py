"""Deterministic local language identification; never translate source text."""
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0
LANGUAGE_NAMES = {"en": "English", "de": "German", "und": "Unknown / uncertain"}


def detect_language(text: str) -> str:
    if sum(character.isalpha() for character in text) < 20:
        return "und"
    # Sample throughout the document, rather than only an English abstract.
    if len(text) > 18000:
        step = (len(text) - 3000) // 5
        text = "\n".join(text[start:start + 3000] for start in range(0, len(text) - 2999, step))
    try:
        candidates = detect_langs(text)
        return candidates[0].lang if candidates and candidates[0].prob >= .75 else "und"
    except LangDetectException:
        return "und"


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, f"Other ({code})")
