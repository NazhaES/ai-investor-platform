import os
import sys
from dotenv import load_dotenv

load_dotenv()

from ingestion.embeddings import get_embeddings
from vectorstore.pinecone_store import PineconeVectorStore, Retriever
import config


def search_vectorstore(query: str, top: int = 5):
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = config.PINECONE_INDEX_NAME

    if not api_key or not index_name:
        raise RuntimeError(
            "Missing Pinecone configuration. Set PINECONE_API_KEY in .env and PINECONE_INDEX_NAME in config.py."
        )

    embeddings = get_embeddings()

    store = PineconeVectorStore(
        api_key=api_key,
        index_name=index_name,
        namespace=config.PINECONE_NAMESPACE
    )
    retriever = Retriever(store.client, embeddings, store.namespace)
    results = retriever.invoke(query, top_k=top)

    print(f"Query: {query!r}")
    print(f"Top: {top}")
    print(f"Results: {len(results)}\n")

    for idx, result in enumerate(results, start=1):
        snippet = result.page_content.strip().replace("\n", " ") or "<no content>"
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"  content snippet: {snippet}")
        print("  " + "-" * 60)

    if not results:
        print("No results returned. Verify your index contents or try a different query.")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retrieval_debug \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_vectorstore(query)


if __name__ == "__main__":
    main()
