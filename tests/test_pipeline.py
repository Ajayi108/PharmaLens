import unittest
from unittest.mock import Mock

import pymupdf as fitz
import numpy as np

from src.chunking import chunk_blocks
from src.models import DocumentChunk, ScreeningCriterion
from src.negation import assess_negation
from src.pdf_parser import extract_pdf
from src.qa import extract_answer
from src.retrieval import retrieve
from src.screening import default_criteria, evidence_status, preliminary_status, screen_article
from src.vector_store import VectorStore
from src.embeddings import encode_texts


class PipelineTests(unittest.TestCase):
    def test_page_and_section_provenance(self):
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Case Presentation\nA 64-year-old male received pembrolizumab.")
            page = pdf.new_page()
            page.insert_text((72, 72), "Results\nThe patient developed colitis.")
            pdf.new_page()
            data = pdf.tobytes()
        blocks, count, empty = extract_pdf(data, "case.pdf")
        self.assertEqual(count, 3)
        self.assertEqual(empty, [3])
        chunks = chunk_blocks(blocks)
        self.assertEqual([(c.page_number, c.section) for c in chunks], [(1, "Case Presentation"), (2, "Results")])
        self.assertIn("pembrolizumab", chunks[0].text)

    def test_chunking_covers_long_text_and_tail(self):
        text = " ".join(f"Word{i}." for i in range(1000))
        chunks = chunk_blocks([dict(document="a", page=1, section="Results", original_text=text)])
        covered = {word for chunk in chunks for word in chunk.text.split()}
        self.assertEqual(covered, set(text.split()))
        self.assertTrue(all(len(c.text.split()) <= 400 for c in chunks))
        self.assertTrue(set(chunks[0].text.split()) & set(chunks[1].text.split()))

    def test_negation_scope_and_mixed(self):
        negative, mixed = assess_negation("No adverse events were observed.", "adverse|rash")
        self.assertTrue(negative)
        self.assertFalse(mixed)
        negative, mixed = assess_negation("No adverse events occurred, but rash developed.", "adverse|rash")
        self.assertTrue(mixed)
        self.assertFalse(assess_negation("The patient had no fever. The drug was restarted.", "restart")[0])

    def test_retrieval_deduplicates_and_excludes_references(self):
        chunks = [DocumentChunk("a.pdf", i+1, "References" if i == 0 else "Results", text, str(i))
                  for i, text in enumerate(["Adverse events bibliography.", "No adverse events were observed.", "Patient recovered."])]
        store = VectorStore(chunks, np.array([[1, 0], [1, 0], [0, 1]]))
        criterion = ScreeningCriterion("event", "Event", [], "", "adverse")
        hits = retrieve(store, np.array([[1, 0], [1, 0]]), criterion, .5, 3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].page_number, 2)
        self.assertEqual(evidence_status(hits), "Negative evidence")
        self.assertEqual(retrieve(store, np.array([[-1, 0]]), criterion, .5, 3), [])

    def test_negative_event_cannot_be_relevant(self):
        criterion = next(c for c in default_criteria() if c.id == "event")
        store = VectorStore([DocumentChunk("a", 1, "Results", "No adverse events were observed.", "0")], [[1, 0]])
        analysis = screen_article("a", store, [criterion], {"event": np.array([[1, 0]])}, .4, 3, ["event"])
        self.assertEqual(analysis.status, "Potentially Not Relevant")
        self.assertEqual(preliminary_status([], ["drug"])[0], "Insufficient Evidence")

    def test_qa_only_accepts_exact_source_span(self):
        context = "Severe colitis developed."
        model = Mock(return_value=dict(answer="colitis", start=7, end=14, score=.9))
        self.assertEqual(extract_answer(model, "What?", context)["answer"], "colitis")
        model.return_value["answer"] = "invented answer"
        self.assertIsNone(extract_answer(model, "What?", context))

    def test_embedding_windows_cover_tail(self):
        class Tokenizer:
            def num_special_tokens_to_add(self, pair=False): return 2
            def encode(self, text, add_special_tokens=False): return list(range(len(text.split())))
            def decode(self, tokens, skip_special_tokens=True): return " ".join(map(str, tokens))
        model = Mock()
        model.max_seq_length = 6
        model.tokenizer = Tokenizer()
        model.encode.return_value = np.array([[1., 0.], [0., 1.], [0., 1.]])
        result = encode_texts(model, ["a b c d e f g h i"])
        self.assertEqual(model.encode.call_args.args[0], ["0 1 2 3", "4 5 6 7", "8"])
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1., places=6)


if __name__ == "__main__":
    unittest.main()
