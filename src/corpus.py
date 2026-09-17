"""One multilingual index per review session; never partition by language."""
import numpy as np

from . import embeddings
from .embeddings import EMBEDDING_MODEL
from .language import detect_language
from .pdf_parser import normalize_section
from .vector_store import VectorStore


def prepare_article(article, document_id, model):
    """Migrate old session data without discarding reviewer decisions or notes."""
    if article.get("embedding_model") == EMBEDDING_MODEL and "embeddings" in article:
        return
    language = article.get("language") or detect_language("\n".join(c.text for c in article["chunks"]))
    for number, chunk in enumerate(article["chunks"]):
        chunk.document_id = document_id
        chunk.chunk_id = f"{document_id}:{number}"
        chunk.language = language
        chunk.normalized_section = normalize_section(chunk.section)
    article["language"] = language
    article["embeddings"] = embeddings.encode_texts(model, [chunk.text for chunk in article["chunks"]])
    article["embedding_model"] = EMBEDDING_MODEL
    article.pop("store", None)


def build_corpus(articles):
    chunks = [chunk for article in articles.values() for chunk in article["chunks"]]
    vectors = np.concatenate([article["embeddings"] for article in articles.values()], axis=0)
    return VectorStore(chunks, vectors)
