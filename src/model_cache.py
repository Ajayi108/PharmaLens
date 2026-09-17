"""Resolve installed model snapshots without making network requests."""
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def cached_model_path(model_name: str) -> str:
    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"Model {model_name} is not installed locally. "
            "Run `python -m scripts.download_models --with-qa` once with internet access, then retry."
        ) from exc
