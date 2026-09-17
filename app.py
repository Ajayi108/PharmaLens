"""PharmaLens: local, evidence-first literature screening. No generative models."""
import hashlib
import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.chunking import chunk_blocks
from src.corpus import build_corpus, prepare_article
from src.embeddings import EMBEDDING_MODEL, encode_texts, load_embedding_model
from src.entities import detect_entities
from src.language import language_name
from src.models import ScreeningCriterion
from src.pdf_parser import extract_pdf
from src.qa import QA_MODEL, extract_answer, load_qa_model
from src.retrieval import retrieve
from src.screening import default_criteria, screen_article

st.set_page_config(page_title="PharmaLens | Literature screening", page_icon="🔬", layout="wide")
st.session_state.setdefault("articles", {})
st.session_state.setdefault("upload_epoch", 0)
st.session_state.setdefault("query_vectors", {})
if st.session_state.get("query_model") != EMBEDDING_MODEL:
    st.session_state["query_vectors"] = {}
    st.session_state["qa_answers"] = {}
    st.session_state["query_model"] = EMBEDDING_MODEL


def show_evidence(evidence, question, qa_enabled):
    st.caption(f"{evidence.document_name} · {language_name(evidence.language)} · Page {evidence.page_number} · {evidence.section} · Cosine similarity {evidence.similarity_score:.3f}")
    st.caption(f"Normalized section: {evidence.normalized_section} · Article ID: {evidence.document_id[:8]} · Negation detected: {'Yes' if evidence.negated else 'No'}")
    with st.container(border=True):
        st.text(evidence.text)
    if evidence.negated:
        st.warning("Mixed polarity — manual review" if evidence.mixed else "Possible negative evidence")
        for sentence in evidence.negative_sentences:
            st.text(sentence)
    entities = detect_entities(evidence.text)
    if entities:
        st.caption("Dictionary / regex matches: " + " · ".join(dict.fromkeys(f"{e['type']}: {e['entity']}" for e in entities)))
    if qa_enabled:
        if evidence.negated:
            st.caption("Extractive answer omitted for negated or mixed evidence; inspect the full passage.")
        else:
            try:
                key = (QA_MODEL, question, evidence.text)
                cache = st.session_state.setdefault("qa_answers", {})
                if key not in cache:
                    cache[key] = extract_answer(load_qa_model(), question, evidence.text)
                answer = cache[key]
                if answer:
                    st.write("**Extracted answer span**")
                    st.text(answer["answer"])
                    st.caption(f"QA span score: {answer['score']:.3f} — not a probability of clinical truth.")
                else:
                    st.caption("No answer span above the QA score threshold.")
            except Exception as exc:
                st.warning(f"Extractive QA unavailable; retrieval remains available. {exc}")


with st.sidebar:
    st.title("🔬 PharmaLens")
    st.caption("Local pharmacovigilance literature screening. Trace each finding to its source.")
    st.divider()
    st.caption("EMBEDDING MODEL")
    st.code(EMBEDDING_MODEL, language=None)
    threshold = st.slider("Similarity threshold", 0.30, 0.90, 0.45, 0.01)
    top_k = st.slider("Retrieved passages per criterion", 1, 10, 3)
    qa_enabled = st.toggle("Enable extractive QA", value=False)
    st.caption("Optional multilingual XLM-R extracts existing English or German text spans only.")
    st.caption(f"QA model: {QA_MODEL}")
    if st.button("Clear current analysis", width="stretch"):
        epoch = st.session_state["upload_epoch"] + 1
        st.session_state.clear()
        st.session_state["upload_epoch"] = epoch
        st.rerun()
    st.caption("Models load from local files without network checks. Install once with: python -m scripts.download_models --with-qa")

st.title("Literature screening, grounded in evidence")
st.write("Find English and German passages. Inspect their original wording. Record your decision.")
st.caption("Documents are processed locally and are not sent to a generative AI service.")

criteria = default_criteria()
query_config = st.session_state.setdefault("query_config", {})
if st.session_state.get("criteria_version") != "bilingual-v1":
    for criterion in criteria:
        existing = query_config.get(criterion.id, "").splitlines()
        query_config[criterion.id] = "\n".join(dict.fromkeys(existing + criterion.queries)).strip()
        st.session_state.pop("queries_" + criterion.id, None)
    st.session_state["criteria_version"] = "bilingual-v1"
with st.expander("Screening criteria & preliminary status rules"):
    st.caption("Edit English and German semantic queries, one per line. Rules use retrieved, non-negated evidence; they do not establish clinical facts.")
    selected = st.multiselect("Required criteria for Potentially Relevant", [c.id for c in criteria],
                              default=["drug", "event", "human"],
                              format_func=lambda value: next(c.name for c in criteria if c.id == value))
    chosen = st.selectbox("Edit criterion", [c.id for c in criteria],
                          format_func=lambda value: next(c.name for c in criteria if c.id == value))
    for criterion in criteria:
        key = "queries_" + criterion.id
        query_config.setdefault(criterion.id, "\n".join(criterion.queries))
        if criterion.id == chosen:
            query_config[criterion.id] = st.text_area("Semantic queries (one per line)",
                value=query_config[criterion.id], key=key)
        criterion.queries = [q.strip() for q in query_config[criterion.id].splitlines() if q.strip()]

uploads = st.file_uploader("Upload literature PDFs", type=["pdf"], accept_multiple_files=True,
                           key=f"uploads_{st.session_state['upload_epoch']}")
if st.button("Analyze PDFs", type="primary", disabled=not uploads):
    progress = st.progress(0, text="Preparing local embedding model…")
    try:
        model = load_embedding_model()
        for number, upload in enumerate(uploads):
            data = upload.getvalue()
            identity = hashlib.sha256(data + upload.name.encode()).hexdigest()
            if identity not in st.session_state.articles or st.session_state.articles[identity].get("error"):
                try:
                    progress.progress(number / len(uploads), text=f"Extracting text: {upload.name}")
                    blocks, pages, empty = extract_pdf(data, upload.name)
                    for block in blocks:
                        block["document_id"] = identity
                    chunks = chunk_blocks(blocks)
                    if not chunks:
                        raise ValueError("No extractable text. Scanned PDFs require OCR before upload.")
                    article = dict(name=upload.name, pages=pages,
                        chunks=chunks, language=blocks[0]["language"], empty=empty,
                        decision="Maybe / Manual Review", notes="")
                    progress.progress(number / len(uploads), text=f"Embedding {len(chunks)} passages: {upload.name}")
                    prepare_article(article, identity, model)
                    st.session_state.articles[identity] = article
                except Exception as exc:
                    st.session_state.articles[identity] = dict(name=upload.name, error=str(exc))
            progress.progress((number + 1) / len(uploads), text=f"Processed {upload.name}")
    except Exception as exc:
        st.error(f"Local embedding model could not load. {exc}")
    finally:
        progress.empty()

articles = st.session_state.articles
if not articles:
    st.info("Upload one or more research PDFs, then select Analyze PDFs to begin.")
    st.stop()

ready = {key: value for key, value in articles.items() if "error" not in value}
if not ready:
    st.dataframe(pd.DataFrame([{"Document": a["name"], "Status": a["error"]} for a in articles.values()]), hide_index=True)
    st.stop()

try:
    model = load_embedding_model()
    for identity, article in ready.items():
        prepare_article(article, identity, model)
    signature = (EMBEDDING_MODEL, tuple((identity, len(a["chunks"])) for identity, a in ready.items()))
    if st.session_state.get("corpus_signature") != signature:
        st.session_state["corpus"] = build_corpus(ready)
        st.session_state["corpus_signature"] = signature
    store = st.session_state["corpus"]
    query_vectors = {}
    for criterion in criteria:
        if not criterion.queries:
            st.error(f"Add at least one query for {criterion.name}.")
            st.stop()
        key = (EMBEDDING_MODEL, *criterion.queries)
        if key not in st.session_state.query_vectors:
            st.session_state.query_vectors[key] = encode_texts(model, criterion.queries)
        query_vectors[criterion.id] = st.session_state.query_vectors[key]
    analyses = {key: screen_article(a["name"], store, criteria, query_vectors, threshold, top_k, selected, document_id=key)
                for key, a in ready.items()}
except Exception as exc:
    st.error(f"Screening failed: {exc}")
    st.stop()

st.dataframe(pd.DataFrame([{"Document": a["name"], "Language": language_name(a.get("language", "und")), "Pages": a.get("pages", 0),
    "Chunks": len(a.get("chunks", [])), "Status": a.get("error", "Ready"),
    "Pages without text": ", ".join(map(str, a.get("empty", [])))} for a in articles.values()]), hide_index=True, width="stretch")
st.caption(f"Shared multilingual index: {len(store.chunks)} passages across {len(ready)} articles. No translation is performed.")

comparison = [{"Document": ready[key]["name"], "Language": language_name(ready[key]["language"]), "Article ID": key[:8], "Preliminary status": result.status,
               "Evidence criteria": sum(r.result == "Evidence found" for r in result.results),
               "Reviewer decision": ready[key]["decision"]} for key, result in analyses.items()]
with st.expander("Compare articles", expanded=len(ready) > 1):
    st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")

article_id = st.selectbox("Select article", list(ready), format_func=lambda key: f"{ready[key]['name']} · {key[:8]}")
article, analysis = ready[article_id], analyses[article_id]
st.caption(f"Detected language: {language_name(article['language'])}")
if article["language"] not in {"en", "de"}:
    st.warning("Language is uncertain or outside English/German. Retrieval remains available, but screening rules are designed for English and German.")
st.subheader(analysis.status)
st.write(analysis.explanation)
st.caption("Preliminary retrieval-based screening only. Similarity is not clinical confidence. All findings require reviewer confirmation.")
if article["empty"]:
    st.warning("Some pages have no extractable text. Evidence may be missing; OCR is not included.")

summary, evidence_tab, questions, review = st.tabs(["Screening summary", "Source evidence", "Custom question", "Reviewer decision"])
rows = []
for result in analysis.results:
    best = result.evidence[0] if result.evidence else None
    rows.append({"Criterion": result.criterion.name, "Result": result.result,
                 "Score": round(best.similarity_score, 3) if best else None,
                 "Evidence": best.text if best else "", "Document": analysis.document_name,
                 "Language": language_name(article["language"]),
                 "Page": best.page_number if best else None, "Section": best.section if best else "",
                 "Normalized section": best.normalized_section if best else ""})
with summary:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("For mixed evidence, open Source evidence to inspect all retrieved passages and polarity.")
with evidence_tab:
    criterion_name = st.selectbox("Screening criterion", [r.criterion.name for r in analysis.results])
    result = next(r for r in analysis.results if r.criterion.name == criterion_name)
    st.write(result.result)
    for i, evidence in enumerate(result.evidence):
        with st.expander(f"Passage {i + 1} · Page {evidence.page_number} · {evidence.similarity_score:.3f}", expanded=i == 0):
            qa_question = result.criterion.question_de if evidence.language == "de" else result.criterion.question
            show_evidence(evidence, qa_question, qa_enabled)
with questions:
    scope = st.radio("Search scope", ["All English and German documents", "Selected article"], horizontal=True)
    with st.form("custom_question"):
        question = st.text_input("Ask a screening question in English or German", placeholder="Was the patient hospitalized? / Wurde der Patient ins Krankenhaus eingeliefert?")
        submitted = st.form_submit_button("Find supporting passages")
    if submitted and question.strip():
        st.session_state["custom_" + article_id] = question.strip()
    question = st.session_state.get("custom_" + article_id)
    if question:
        st.write(question)
        key = (EMBEDDING_MODEL, question)
        if key not in st.session_state.query_vectors:
            st.session_state.query_vectors[key] = encode_texts(model, [question])
        found = retrieve(store, st.session_state.query_vectors[key],
                         ScreeningCriterion("custom", question, [question], question), threshold, top_k,
                         document_id=article_id if scope == "Selected article" else None)
        if not found:
            st.info("No evidence above the selected similarity threshold.")
        for evidence in found:
            show_evidence(evidence, question, qa_enabled)
with review:
    with st.form("review_" + article_id):
        options = ["Include", "Exclude", "Maybe / Manual Review"]
        decision = st.radio("Final reviewer decision", options, index=options.index(article["decision"]), horizontal=True)
        notes = st.text_area("Reviewer notes", value=article["notes"])
        if st.form_submit_button("Save reviewer decision", type="primary"):
            article.update(decision=decision, notes=notes)
            st.rerun()
    st.caption("Saved in this browser session. Export to keep a durable copy.")

st.divider()
exports = [dict(article_id=key, **asdict(result), reviewer_decision=ready[key]["decision"],
                language=ready[key]["language"], reviewer_notes=ready[key]["notes"], pages_without_text=ready[key]["empty"])
           for key, result in analyses.items()]
payload = dict(embedding_model=EMBEDDING_MODEL, qa_model=QA_MODEL, similarity_threshold=threshold, top_k=top_k,
               required_criteria=selected, articles=exports)
left, right = st.columns(2)
left.download_button("Export all evidence & decisions (JSON)", json.dumps(payload, indent=2, ensure_ascii=False),
                     "pharmalens-screening.json", "application/json")
# Neutralize formula prefixes when opening CSV in a spreadsheet.
frame = pd.DataFrame(rows)
for column in frame.select_dtypes(include="object"):
    frame[column] = frame[column].map(lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
right.download_button("Export article summary (CSV)", frame.to_csv(index=False).encode("utf-8-sig"),
                      "pharmalens-summary.csv", "text/csv")
