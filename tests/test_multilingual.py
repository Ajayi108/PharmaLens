import unittest
from unittest.mock import Mock, patch

import numpy as np
import pymupdf

from src.chunking import chunk_blocks
from src.corpus import prepare_article, build_corpus
from src.embeddings import EMBEDDING_MODEL
from src.entities import detect_entities
from src.language import detect_language
from src.models import DocumentChunk, ScreeningCriterion
from src.negation import assess_negation
from src.pdf_parser import extract_pdf, normalize_section
from src.qa import extract_answer
from src.retrieval import retrieve
from src.screening import evidence_status, screen_article
from src.screening_criteria import default_criteria, DEFINITIONS
from src.vector_store import VectorStore

EN = "The patient developed severe liver injury two weeks after starting the medication."
DE = "Zwei Wochen nach Beginn der Behandlung entwickelte der Patient eine schwere Leberschädigung."


class MultilingualTests(unittest.TestCase):
    def test_language_detection(self):
        for text, expected in [(EN, "en"), (DE, "de"), ("", "und"), ("200 mg", "und")]:
            with self.subTest(text=text):
                self.assertEqual(detect_language(text), expected)

    def test_bilingual_configuration(self):
        self.assertEqual(len(default_criteria()), 17)
        for key, name, en, de, question, question_de, concepts in DEFINITIONS:
            self.assertTrue(en and de and question and question_de and concepts, key)

    def test_german_pdf_preserves_heading_text_and_metadata(self):
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "2. Ergebnisse:\n" + DE)
            pdf.new_page().insert_text((72, 72), "Literaturverzeichnis\nReferenz eins.")
            blocks, pages, empty = extract_pdf(pdf.tobytes(), "bericht.pdf")
        self.assertEqual(blocks[0]["section"], "2. Ergebnisse:")
        self.assertEqual(blocks[0]["normalized_section"], "Results")
        self.assertEqual(blocks[0]["language"], "de")
        self.assertEqual(blocks[1]["normalized_section"], "References")
        chunks = chunk_blocks(blocks)
        self.assertEqual(chunks[0].text, DE)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].language, "de")
        self.assertEqual(normalize_section("Unerwünschte Ereignisse"), "Adverse Events")

    def test_chunk_whitespace_is_unchanged(self):
        source = "Nach der Behandlung\nentwickelte  der Patient eine schwere Kolitis.\n\nDie Behandlung endete."
        chunks = chunk_blocks([dict(document="a", page=1, section="Fallbericht", original_text=source)])
        self.assertEqual(chunks[0].text, source)

    def test_bilingual_polarity_and_article_filtering(self):
        serious = next(c for c in default_criteria() if c.id == "serious")
        texts = ["No serious adverse events were observed.",
                 "Es wurden keine schwerwiegenden unerwünschten Ereignisse beobachtet."]
        chunks = [DocumentChunk("same.pdf", 1, "Results", text, str(i), language=language, document_id=str(i))
                  for i, (text, language) in enumerate(zip(texts, ["en", "de"]))]
        store = VectorStore(chunks, [[1, 0], [1, 0]])
        for identity in ["0", "1"]:
            result = screen_article("same.pdf", store, [serious], {"serious": [[1, 0]]}, .3, 3, ["serious"], document_id=identity)
            self.assertEqual(result.results[0].result, "Negative evidence")
            self.assertEqual(len(result.results[0].evidence), 1)
            self.assertEqual(result.results[0].evidence[0].document_id, identity)
            self.assertNotEqual(result.status, "Potentially Relevant")
        for text in [EN, DE]:
            for key in ["event", "temporal"]:
                criterion = next(c for c in default_criteria() if c.id == key)
                passage_store = VectorStore([DocumentChunk("a", 1, "Results", text, "0")], [[1, 0]])
                self.assertEqual(evidence_status(retrieve(passage_store, [[1, 0]], criterion, .3, 3)), "Evidence found")

    def test_german_negation_variants_and_mixed_polarity(self):
        for text in ["keine unerwünschten Ereignisse", "keine Nebenwirkungen", "keine unerwünschten Wirkungen",
                     "entwickelte keine Beschwerden", "es gab keine Hinweise auf Kolitis", "nicht im Zusammenhang mit dem Arzneimittel",
                     "ohne Komplikationen", "wurde nicht beobachtet", "trat nicht auf"]:
            self.assertTrue(assess_negation(text)[0], text)
        self.assertTrue(assess_negation("Keine Kolitis, aber ein Hautausschlag trat auf.", "kolitis|hautausschlag")[1])
        self.assertFalse(assess_negation("Die Reaktion kann nicht ausgeschlossen werden.")[0])
        self.assertFalse(assess_negation("Nicht nur Kolitis wurde beobachtet.")[0])

    def test_shared_index_dedup_and_german_reference_exclusion(self):
        chunks = [DocumentChunk("same.pdf", 1, section, text, str(i), normalized_section=normalized,
                                language=lang, document_id=str(i))
                  for i, (section, normalized, text, lang) in enumerate([
                      ("Ergebnisse", "Results", DE, "de"), ("Results", "Results", EN, "en"),
                      ("3. Literatur", "References", DE, "de")])]
        store = VectorStore(chunks, [[1, 0], [.9, .1], [1, 0]])
        criterion = ScreeningCriterion("custom", "Question", [], "")
        found = retrieve(store, [[1, 0], [1, 0]], criterion, .3, 10)
        self.assertEqual(len(found), 2)
        self.assertEqual({e.language for e in found}, {"de", "en"})
        self.assertEqual(len(retrieve(store, [[1, 0]], criterion, .3, 1, document_id="1")), 1)

    def test_german_qa_requires_exact_span_and_can_abstain(self):
        context = "Der Patient entwickelte eine schwere Kolitis."
        span = "eine schwere Kolitis"
        start = context.index(span)
        model = Mock(return_value=dict(answer=span, start=start, end=start + len(span), score=.9))
        self.assertEqual(extract_answer(model, "Welche Nebenwirkung?", context)["answer"], span)
        self.assertTrue(model.call_args.kwargs["handle_impossible_answer"])
        model.return_value["answer"] = "severe colitis"
        self.assertIsNone(extract_answer(model, "Welche Nebenwirkung?", context))
        model.return_value = dict(answer="", start=0, end=0, score=.99)
        self.assertIsNone(extract_answer(model, "Welche Nebenwirkung?", context))

    def test_german_entities_keep_source_offsets(self):
        text = "Eine 64-jährige Frau erhielt 2,5 mg/kg Methotrexat am 16.09.2026 und entwickelte Leberschädigung."
        entities = detect_entities(text)
        self.assertTrue({"Age", "Sex", "Dosage", "Drug", "Date", "Adverse event"} <= {e["type"] for e in entities})
        for entity in entities:
            self.assertEqual(entity["entity"], text[entity["start"]:entity["end"]])

    def test_old_session_migrates_once_without_losing_decision(self):
        article = dict(chunks=[DocumentChunk("a", 1, "Ergebnisse", DE, "0")],
                       store="old English index", decision="Include", notes="Keep me")
        with patch("src.embeddings.encode_texts", return_value=np.array([[1., 0.]], dtype="float32")) as encode:
            prepare_article(article, "id-de", object())
            prepare_article(article, "id-de", object())
            encode.assert_called_once()
        self.assertEqual(article["embedding_model"], EMBEDDING_MODEL)
        self.assertEqual(article["language"], "de")
        self.assertEqual(article["decision"], "Include")
        self.assertEqual(article["notes"], "Keep me")
        self.assertNotIn("store", article)
        self.assertEqual(build_corpus({"id-de": article}).index.ntotal, 1)


if __name__ == "__main__":
    unittest.main()
