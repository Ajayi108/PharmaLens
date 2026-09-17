# PharmaLens

Local Streamlit application for English and German PDF pharmacovigilance literature screening. Uses PyMuPDF, multilingual SentenceTransformers, one shared FAISS index and bilingual rule-based NLP. Optional multilingual XLM-RoBERTa question answering extracts source spans. **No generative model, translation service, OpenAI API, or cloud inference is used.**

## Run

Use Python 3.11 or 3.12 for the broadest availability of PyTorch/FAISS wheels.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.download_models --with-qa
streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate`. Open http://localhost:8501. The checked-in configuration binds to localhost and disables Streamlit usage telemetry.

Install model weights once with `python -m scripts.download_models --with-qa` (omit `--with-qa` if you only need retrieval). This explicit setup step requires internet access and downloads multilingual MiniLM and optional XLM-R from Hugging Face. The application itself loads installed snapshots with `local_files_only=True` and never checks the network for model updates. Missing weights produce an installation message instead of repeated network retries. Both models run locally on CPU; no PDF text is included in installation requests.

When updating an existing installation, rerun `pip install -r requirements.txt` for `langdetect` and `sentencepiece`. Existing session articles are re-embedded with the multilingual model while keeping saved reviewer decisions and notes. Query caches and the shared index are versioned by embedding model, so old English vectors are not mixed with new multilingual vectors.

If a running instance reports an import error for a function present in the source (for example `normalize_section`), restart the Streamlit process with Ctrl+C followed by `streamlit run app.py`. A browser refresh alone may retain previously imported modules after a multi-file update. Export any accessible review decisions before restarting: process restarts clear in-memory sessions.

## Review workflow

1. Upload English and/or German PDFs (50 MB maximum each), click **Analyze PDFs**. All documents use the same multilingual model and the same FAISS index, regardless of language. Identical filename/content pairs are deduplicated; same filenames with different contents remain separate through document IDs.
2. Inspect the automatically detected language, page/chunk counts and extraction warnings. Local `langdetect` samples text throughout long documents; short/uncertain text is labeled Unknown. Scanned pages require external OCR. Password-protected files report an error without stopping other files.
3. Select an article. Adjust cosine threshold (0.30–0.90), Top-K (1–10), and optionally QA. These changes reuse article embeddings.
4. Inspect original-language evidence with filename, language, page, original heading and normalized section. German headings such as `Ergebnisse` map to `Results` without rewriting the source heading. Negative/mixed evidence is flagged. Ask custom questions in English or German: by default they search all indexed documents, with an optional selected-article filter. Article screening always filters the shared index by document ID before selecting Top-K.
5. Record Include, Exclude or Maybe / Manual Review with notes. Click **Save reviewer decision**. Compare articles and export JSON (all evidence, settings, decisions) or CSV (selected article summary).
6. Clear current analysis to remove session-held documents, indices, answers and decisions. Cached model weights remain. Session state is not durable storage; export before closing/restarting.

## Interpretation and limitations

- Similarity scores measure semantic proximity, **not calibrated clinical confidence**. “Evidence found” means a passage crossed the retrieval threshold; it does not prove that a criterion is clinically satisfied. Preliminary relevance never replaces review.
- The default rule requires retrieved drug, adverse-event and human evidence. Change required criteria and semantic queries in the interface. Negated or mixed criteria cannot satisfy the positive rule. Negative adverse-event evidence can assign Potentially Not Relevant; missing evidence alone assigns Insufficient Evidence.
- Negation is a conservative English/German clause-level heuristic, not validated clinical NLP. It recognizes phrases such as `no adverse events`, `keine Nebenwirkungen`, `nicht beobachtet`, and `ohne Komplikationen`, with common pseudo-negation exceptions. It can miss scope, double negatives, hypothetical statements, and background versus patient-specific findings. Mixed passages require manual review. Seriousness and severity are not equivalent; reviewers must assess the source.
- Small bilingual entity dictionaries and regexes recognize some drugs/events, German ages, sex, decimal-comma dosages and dates. They preserve source offsets, are incomplete and do not decide relevance. Extend `src/entities.py` or replace it with a biomedical NER provider.
- QA uses the extractive `question-answering` pipeline, with SQuAD2 no-answer handling enabled. A span-score cutoff and exact offset/substring validation are enforced. Model abstention and answers are not clinically validated; returned spans require review. German evidence uses German predefined QA questions; custom questions remain in the user's language. QA is suppressed on negated/mixed passages.
- Extraction preserves source wording and whitespace in displayed chunks. No PDF translation or paraphrasing occurs. Multi-column reading order, tables, scanned pages and unusual headings may need manual inspection. References/`Literatur`/`Literaturverzeichnis` are excluded when detected.
- Chunks target 400 words with 75 words overlap and prefer sentence/paragraph endings; page/section boundaries produce shorter chunks. To avoid multilingual MiniLM's input truncation, token windows are embedded and pooled into a normalized chunk vector. Pooling may dilute small findings; tune queries and inspect context.
- Primary language detection is a heuristic: mixed-language papers or short text can be mislabeled. Language labels do not restrict retrieval. Unknown/other languages remain searchable, with a warning that clinical rules target English/German. The user interface remains English; evidence stays in its source language.
- Uploaded data and embeddings remain in session memory, not shared Streamlit data caches. Only models use `st.cache_resource`. Running this on a remote server sends uploads to that server; the local-only guarantee assumes the documented localhost deployment.
- Documents accumulate until Clear; analyze manageable batches on CPU. No OCR, regulatory validation, persistent database or automatic case submission is included.

## Architecture

`app.py` contains the UI and session lifecycle. `src/pdf_parser.py` extracts page/section blocks; `language.py` detects primary language locally; `chunking.py` slices original text; `embeddings.py` loads and encodes; `corpus.py` migrates articles and builds the single session-wide `vector_store.py` FAISS inner-product index. `retrieval.py` merges bilingual query hits by maximum cosine similarity and removes duplicate chunk hits, with an optional document-ID filter. `screening_criteria.py` defines 17 bilingual criteria (including hospitalization and death) and English/German QA questions. `screening.py` assigns evidence and preliminary statuses. `negation.py`, `entities.py` and `qa.py` enrich evidence. `models.py` carries language, document ID and original/normalized section metadata into exports.

API references: [SentenceTransformers](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html), [Streamlit resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).

Model references: [multilingual MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [deepset multilingual extractive XLM-R model card](https://huggingface.co/deepset/xlm-roberta-base-squad2). The QA model is trained on SQuAD2 and its model card reports English and German evaluations; it is not a pharmacovigilance-specific model. XLM-R model authors: deepset (CC BY 4.0).

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic passages and deterministic vectors; they do not download models. Real-model integration requires the model weights and is separate from these deterministic regression checks.

Run `python -m tests.smoke_models` for the real multilingual MiniLM + shared FAISS + XLM-R integration check (requires the models installed by the setup command above). It checks English/German adverse events and temporal relationships, negated serious events in both languages, retrieval across languages in both directions, and exact source-span QA in English and German. This is a small functional check, not a clinical performance evaluation or threshold calibration.
