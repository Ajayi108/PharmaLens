from .models import ScreeningResult, ArticleScreeningResult
from .retrieval import retrieve
from .screening_criteria import default_criteria

def evidence_status(evidence):
    if not evidence:
        return "Not found"
    positive = any(not item.negated for item in evidence)
    negative = any(item.negated for item in evidence)
    if (positive and negative) or any(item.mixed for item in evidence):
        return "Mixed evidence — review"
    return "Negative evidence" if negative else "Evidence found"


def preliminary_status(results, required):
    states = {result.criterion.id: result.result for result in results}
    matched = [key for key in required if states.get(key) == "Evidence found"]
    missing = [key for key in required if key not in matched]
    if required and not missing:
        return "Potentially Relevant", "Retrieval criteria satisfied: " + ", ".join(matched) + ". Reviewer confirmation is required."
    if "event" in required and states.get("event") == "Negative evidence":
        return "Potentially Not Relevant", "Retrieved adverse-event evidence appears negated. Check every passage before exclusion."
    return "Insufficient Evidence", "Required criteria without unambiguous retrieved evidence: " + (", ".join(missing) or "none selected") + ". Missing retrieval is not proof of absence."


def screen_article(name, store, criteria, vectors, threshold, top_k, required, document_id=None):
    results = []
    for criterion in criteria:
        evidence = retrieve(store, vectors[criterion.id], criterion, threshold, top_k, document_id=document_id)
        results.append(ScreeningResult(criterion, evidence_status(evidence), evidence))
    status, explanation = preliminary_status(results, required)
    return ArticleScreeningResult(name, results, status, explanation)
