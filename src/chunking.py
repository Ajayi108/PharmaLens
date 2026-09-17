import re
from .models import DocumentChunk


def chunk_blocks(blocks: list[dict], target_words: int = 400, overlap_words: int = 75) -> list[DocumentChunk]:
    """Choose sentence/paragraph boundaries, then slice the untouched extracted text."""
    if target_words <= overlap_words or overlap_words < 0:
        raise ValueError("Target must exceed nonnegative overlap.")
    chunks = []
    for block in blocks:
        source = block["original_text"]
        words = list(re.finditer(r"\S+", source))
        start = 0
        while start < len(words):
            end = min(start + target_words, len(words))
            if end < len(words):
                for boundary in range(end, start + max(1, int(target_words * .75)), -1):
                    gap = source[words[boundary - 1].end():words[boundary].start()]
                    if re.search(r"[.!?][\"'”)]?$", words[boundary - 1].group()) or "\n\n" in gap:
                        end = boundary
                        break
            text = source[words[start].start():words[end - 1].end()]
            chunks.append(DocumentChunk(block["document"], block["page"], block["section"], text,
                str(len(chunks)), normalized_section=block.get("normalized_section", block["section"]),
                language=block.get("language", "und"), document_id=block.get("document_id", "")))
            if end == len(words):
                break
            start = max(start + 1, end - overlap_words)
    return chunks
