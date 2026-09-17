from .models import RetrievedEvidence
from .negation import assess_negation


def retrieve(store, query_vectors, criterion, threshold=0.45, top_k=3, document_id=None):
    scores, indices = store.search(query_vectors, len(store.chunks))
    best = {}
    for row_scores, row_indices in zip(scores, indices):
        for score, index in zip(row_scores, row_indices):
            if index < 0 or score < threshold:
                continue
            chunk = store.chunks[index]
            if document_id is not None and chunk.document_id != document_id:
                continue
            # Reference-only matches should not drive article screening.
            if chunk.normalized_section == "References" or chunk.section.casefold() in {"references", "bibliography", "literatur", "literaturverzeichnis"}:
                continue
            best[int(index)] = max(float(score), best.get(int(index), -1))
    evidence = []
    for index, score in sorted(best.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        chunk = store.chunks[index]
        negatives, mixed = assess_negation(chunk.text, criterion.concepts)
        evidence.append(RetrievedEvidence(criterion.name, chunk.text, chunk.document_name,
                                         chunk.page_number, chunk.section, min(1.0, score),
                                         bool(negatives), chunk.chunk_id, negatives, mixed,
                                         normalized_section=chunk.normalized_section,
                                         language=chunk.language, document_id=chunk.document_id))
    return evidence
