"""Explicit real-model integration check; run from the project root.

python -m tests.smoke_models (requires locally installed non-generative models).
"""
import numpy as np

from src.embeddings import load_embedding_model, encode_texts
from src.models import DocumentChunk, ScreeningCriterion
from src.qa import load_qa_model, extract_answer
from src.retrieval import retrieve
from src.screening import default_criteria, screen_article
from src.vector_store import VectorStore


def main():
    texts = [
        "The patient developed severe liver injury two weeks after starting the medication.",
        "Zwei Wochen nach Beginn der Behandlung entwickelte der Patient eine schwere Leberschädigung.",
        "No serious adverse events were observed.",
        "Es wurden keine schwerwiegenden unerwünschten Ereignisse beobachtet.",
        "The laboratory temperature was measured every morning using a thermometer.",
    ]
    model = load_embedding_model()
    vectors = encode_texts(model, texts)
    assert vectors.shape == (5, 384) and np.isfinite(vectors).all()
    chunks = [DocumentChunk(f"synthetic-{i}.pdf", 1, "Results", text, str(i),
                            language="de" if i in (1, 3) else "en", document_id=str(i))
              for i, text in enumerate(texts)]
    store = VectorStore(chunks, vectors)
    criteria = default_criteria()
    queries = {c.id: encode_texts(model, c.queries) for c in criteria}
    for identity in range(4):
        result = screen_article(str(identity), store, criteria, queries, .3, 3,
                                ["drug", "event", "human"], document_id=str(identity))
        states = {r.criterion.id: r.result for r in result.results}
        if identity < 2:
            assert states["event"] == "Evidence found", states
            assert states["temporal"] == "Evidence found", states
        else:
            assert states["serious"] == "Negative evidence", states
    print("Real-model English/German positive, temporal and negated serious-event screening passed.", flush=True)
    for question, expected in [("What adverse event did the patient experience?", "1"),
                               ("Welche Nebenwirkung entwickelte der Patient?", "0")]:
        hits = retrieve(store, encode_texts(model, [question]), ScreeningCriterion("custom", question, [question], question), .3, 2)
        assert expected in [hit.document_id for hit in hits], [(hit.document_id, hit.similarity_score) for hit in hits]
        print("Cross-language retrieval passed:", question, [(h.document_id, round(h.similarity_score, 3)) for h in hits], flush=True)
    qa = load_qa_model()
    for context, question in [("The patient developed severe colitis after treatment.", "What adverse event did the patient develop?"),
                              ("Nach dem dritten Behandlungszyklus entwickelte der Patient eine schwere Kolitis.", "Welche Nebenwirkung entwickelte der Patient?")]:
        answer = extract_answer(qa, question, context)
        assert answer and answer["answer"] in context, (question, answer)
        print("Multilingual extractive QA passed:", answer["answer"], flush=True)


if __name__ == "__main__":
    main()
