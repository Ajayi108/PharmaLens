import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pymupdf as fitz
from io import BytesIO
from streamlit.testing.v1 import AppTest

from src.models import DocumentChunk
from src.embeddings import EMBEDDING_MODEL
from src.vector_store import VectorStore


class AppTests(unittest.TestCase):
    def test_custom_question_searches_shared_corpus_and_scope_can_narrow(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        articles = {}
        texts = {"en": "The patient developed liver injury after medication.",
                 "de": "Der Patient entwickelte nach der Behandlung eine schwere Leberschädigung."}
        for language, text in texts.items():
            chunk = DocumentChunk("same.pdf", 1, "Results", text, language + ":0",
                                  language=language, normalized_section="Results", document_id=language)
            articles[language] = dict(name="same.pdf", pages=1, chunks=[chunk], language=language,
                                     embeddings=np.array([[1., 0.]], dtype="float32"), embedding_model=EMBEDDING_MODEL,
                                     empty=[], decision="Maybe / Manual Review", notes="")
        app.session_state["articles"] = articles
        with patch("src.embeddings.load_embedding_model", return_value=object()), \
             patch("src.embeddings.encode_texts", side_effect=lambda model, texts: np.tile([1., 0.], (len(texts), 1))) as encode:
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["corpus"].index.ntotal, 2)
            self.assertEqual(set(app.dataframe[0].value["Language"]), {"English", "German"})
            app.text_input[0].set_value("What adverse event did the patient experience?")
            next(item for item in app.button if item.label == "Find supporting passages").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn(texts["de"], [item.value for item in app.text])
            calls = encode.call_count
            next(item for item in app.radio if item.label == "Search scope").set_value("Selected article").run()
            self.assertEqual(len(app.exception), 0)
            self.assertNotIn(texts["de"], [item.value for item in app.text])
            self.assertEqual(encode.call_count, calls)

    def test_upload_processes_valid_pdf_and_reports_corrupt_file(self):
        with fitz.open() as document:
            document.new_page().insert_text((72, 72), "Results\nA patient received aspirin and developed rash.")
            valid = BytesIO(document.tobytes())
        valid.name = "valid.pdf"
        corrupt = BytesIO(b"not a pdf")
        corrupt.name = "broken.pdf"
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        with patch("streamlit.file_uploader", return_value=[valid, corrupt]), \
             patch("src.embeddings.load_embedding_model", return_value=object()), \
             patch("src.embeddings.encode_texts", side_effect=lambda model, texts: np.tile([1., 0.], (len(texts), 1))):
            app.run(timeout=30)
            next(item for item in app.button if item.label == "Analyze PDFs").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            articles = list(app.session_state["articles"].values())
            self.assertEqual(len(articles), 2)
            self.assertEqual(sum("error" in article for article in articles), 1)
            self.assertEqual(next(a for a in articles if "error" not in a)["pages"], 1)

    def test_empty_app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Upload one or more", app.info[0].value)

    def test_article_review_and_clear(self):
        chunk = DocumentChunk("case.pdf", 1, "Case Presentation",
                              "A 64-year-old male received pembrolizumab and developed colitis.", "0")
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.session_state["articles"] = {"abc12345": dict(name="case.pdf", pages=1,
            chunks=[chunk], store=VectorStore([chunk], [[1., 0.]]), empty=[],
            decision="Maybe / Manual Review", notes="")}
        with patch("src.embeddings.load_embedding_model", return_value=object()), \
             patch("src.embeddings.encode_texts", side_effect=lambda model, texts: np.tile([1., 0.], (len(texts), 1))):
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.subheader[0].value, "Potentially Relevant")
            next(item for item in app.radio if item.label == "Final reviewer decision").set_value("Include")
            next(item for item in app.text_area if item.label == "Reviewer notes").set_value("Reviewed source.")
            next(item for item in app.button if item.label == "Save reviewer decision").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["articles"]["abc12345"]["decision"], "Include")
            next(item for item in app.button if item.label == "Clear current analysis").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["articles"], {})


if __name__ == "__main__":
    unittest.main()
