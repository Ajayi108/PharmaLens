import faiss
import numpy as np


class VectorStore:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        vectors = np.array(embeddings, dtype="float32", copy=True, order="C")
        if len(chunks) != len(vectors) or not len(chunks):
            raise ValueError("An index needs one embedding per nonempty chunk collection.")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, queries, k):
        queries = np.array(queries, dtype="float32", copy=True, order="C")
        faiss.normalize_L2(queries)
        scores, indices = self.index.search(queries, min(k, len(self.chunks)))
        return scores, indices
