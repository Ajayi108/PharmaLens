"""Explicit one-time model installation; never receives document text."""
import argparse
from huggingface_hub import snapshot_download
from src.embeddings import EMBEDDING_MODEL
from src.qa import QA_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-qa", action="store_true", help="Also install the optional extractive QA model")
    args = parser.parse_args()
    for model in [EMBEDDING_MODEL] + ([QA_MODEL] if args.with_qa else []):
        print(f"Installing {model}", flush=True)
        snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    print("Models installed. Start or retry PharmaLens.")


if __name__ == "__main__":
    main()
