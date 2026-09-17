import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import streamlit as st
from .model_cache import cached_model_path

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource
def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cached_model_path(model_name), device="cpu",
                               trust_remote_code=False, local_files_only=True)


def encode_texts(model, texts: list[str]) -> np.ndarray:
    """Pool token windows so long chunks are not truncated at the model's token limit."""
    windows, owners = [], []
    budget = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
    for owner, text in enumerate(texts):
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, max(1, len(tokens)), budget):
            windows.append(model.tokenizer.decode(tokens[start:start + budget], skip_special_tokens=True))
            owners.append(owner)
    encoded = model.encode(windows, batch_size=32, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    pooled = np.zeros((len(texts), encoded.shape[1]), dtype="float32")
    np.add.at(pooled, owners, encoded)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(pooled)
