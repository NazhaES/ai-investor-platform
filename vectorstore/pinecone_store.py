import re
from types import SimpleNamespace

from pinecone import Pinecone
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class PineconeVectorStore:
    """Pinecone vector store."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str
    ) -> None:
        self.client = Pinecone(api_key=api_key).Index(index_name)
        self.namespace = namespace

    def upload_chunks(
        self,
        chunks,
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """
        Upload chunks to Pinecone, replacing previous chunks from the same file.
        """
        prefix = f"{source_file}#"

        old_ids = [
            item.id
            for page in self.client.list(prefix=prefix, namespace=self.namespace)
            for item in page.vectors
        ]
        for i in range(0, len(old_ids), 1000):
            self.client.delete(ids=old_ids[i:i + 1000], namespace=self.namespace)

        vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])

        documents = [
            {
                "id": f"{prefix}{i}",
                "values": vector,
                "metadata": {
                    "company": company,
                    "year": year,
                    "source_file": source_file,
                    "content": chunk.page_content
                }
            }
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        for i in range(0, len(documents), 100):
            self.client.upsert(vectors=documents[i:i + 100], namespace=self.namespace)

        print(f"Uploaded {len(documents)} chunks.")


class Retriever:
    """Hybrid retriever: Pinecone vector search + BM25 keyword search.
    Mirrors the Retriever used in the RAG extractor.
    """
    def __init__(self, client, embeddings, namespace: str):
        self.client = client
        self.embeddings = embeddings
        self.namespace = namespace
        self._bm25 = {}

    def _load_chunks(self, company: str | None, year: int | None) -> list[str]:
        """Fetch chunk texts from Pinecone, filtered by company and year."""
        chunks = []
        for page in self.client.list(namespace=self.namespace):
            ids = [item.id for item in page.vectors]
            fetched = self.client.fetch(ids=ids, namespace=self.namespace)
            for vector in fetched.vectors.values():
                metadata = vector.metadata or {}
                if company and year and (metadata.get("company") != company or metadata.get("year") != str(year)):
                    continue
                chunks.append(metadata.get("content", ""))
        return chunks

    def _keyword_search(self, query: str, company: str | None, year: int | None, top_k: int) -> list[str]:
        """BM25 keyword search, index built once per company/year."""
        key = (company, year) if company and year else None
        if key not in self._bm25:
            chunks = self._load_chunks(company, year)
            self._bm25[key] = (chunks, BM25Okapi([tokenize(c) for c in chunks]) if chunks else None)

        chunks, bm25 = self._bm25[key]
        if not bm25:
            return []

        scores = bm25.get_scores(tokenize(query))
        best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [chunks[i] for i in best if scores[i] > 0]

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """Retrieve relevant chunks with vector + keyword search, merged by rank fusion.
        Returns a list of SimpleNamespace objects with `page_content`.
        """
        filter_expr = None
        if company and year:
            filter_expr = {
                "company": {"$eq": company},
                "year": {"$eq": str(year)}
            }
        results = self.client.query(
            vector=self.embeddings.embed_query(query),
            top_k=top_k,
            filter=filter_expr,
            include_metadata=True,
            namespace=self.namespace
        )
        vector_hits = [(match.metadata or {}).get("content", "") for match in results.matches]
        keyword_hits = self._keyword_search(query, company, year, top_k)

        fused = {}
        for hits in (vector_hits, keyword_hits):
            for rank, content in enumerate(hits):
                fused[content] = fused.get(content, 0) + 1 / (60 + rank)

        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [SimpleNamespace(page_content=content) for content in ranked]
