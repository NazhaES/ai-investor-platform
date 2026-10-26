from dotenv import load_dotenv
import os

from pinecone import Pinecone, ServerlessSpec

import config

load_dotenv()


def create_index(
    api_key: str,
    index_name: str,
    embedding_dimensions: int,
    cloud: str = "aws",
    region: str = "us-east-1"
) -> None:
    """
    Create Pinecone index.

    Args:
        api_key: Pinecone API key.
        index_name: Index name.
        embedding_dimensions: Embedding dimensions (384 for bge-small-en-v1.5).
        cloud: Serverless cloud provider.
        region: Serverless region.
    """
    client = Pinecone(api_key=api_key)

    if client.has_index(index_name):
        print(f"Index '{index_name}' already exists.")
        return

    client.create_index(
        name=index_name,
        dimension=embedding_dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

    print(f"Index '{index_name}' created successfully.")


def create_index_from_settings() -> None:
    """
    Create the Pinecone index described in config.py.
    """
    create_index(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=config.PINECONE_INDEX_NAME,
        embedding_dimensions=config.EMBEDDING_DIMENSIONS,
        cloud=config.PINECONE_CLOUD,
        region=config.PINECONE_REGION
    )


if __name__ == "__main__":
    create_index_from_settings()
