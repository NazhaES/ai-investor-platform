import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the local embedding model once, downloading it to MODELS_CACHE_DIR on first use.

    Returns:
        HuggingFace embedding model.
    """
    embedding_model = config.EMBEDDING_MODEL

    model_path = Path(os.environ["MODELS_CACHE_DIR"]) / embedding_model.split("/")[-1]
    if model_path.exists():
        print(f"Embedding model '{embedding_model}' found in cache.")
    else:
        print(f"Embedding model '{embedding_model}' not in cache. Downloading...")
        tmp = model_path.with_name(model_path.name + ".downloading")
        snapshot_download(repo_id=embedding_model, local_dir=tmp, allow_patterns=["*.json", "*.txt", "*.safetensors"])
        tmp.rename(model_path)
        print(f"Download complete: {model_path}")

    embeddings = HuggingFaceEmbeddings(
        model_name=str(model_path),
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Embedding model '{embedding_model}' loaded and ready.")

    return embeddings


if __name__ == "__main__":
    get_embeddings()
