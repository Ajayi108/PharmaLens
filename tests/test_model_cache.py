import unittest
from unittest.mock import patch
from huggingface_hub.errors import LocalEntryNotFoundError
from src.model_cache import cached_model_path


class ModelCacheTests(unittest.TestCase):
    def test_resolution_is_local_only(self):
        with patch("src.model_cache.snapshot_download", return_value="local/snapshot") as download:
            self.assertEqual(cached_model_path("model-id"), "local/snapshot")
            download.assert_called_once_with(repo_id="model-id", local_files_only=True)

    def test_missing_weights_have_actionable_error(self):
        with patch("src.model_cache.snapshot_download", side_effect=LocalEntryNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "scripts.download_models"):
                cached_model_path("missing-model")
