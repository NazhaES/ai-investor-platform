from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hard cap per chunk: keeps Pinecone metadata under its 40 KB limit
# and roughly matches the embedding model's effective window.
MAX_CHUNK_CHARS = 6000

from dotenv import load_dotenv
load_dotenv()



def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.

    Args:
        markdown_file: Markdown file path.

    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown.

    Args:
        markdown_file: Markdown file path.
        embeddings: LangChain embedding model.

    Returns:
        List of semantic chunks.
    """
    markdown_content = read_markdown(markdown_file)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks = splitter.create_documents([markdown_content])

    return enforce_max_chunk_size(chunks)


def enforce_max_chunk_size(
    chunks: list[Document],
    max_chars: int = MAX_CHUNK_CHARS
) -> list[Document]:
    """
    Split any oversized chunk into smaller pieces.

    The semantic chunker can merge long uniform sections (e.g. financial
    statement notes) into huge chunks that exceed Pinecone's 40 KB
    metadata limit and the embedding model's input window.
    """
    oversized = [c for c in chunks if len(c.page_content) > max_chars]
    if not oversized:
        return chunks

    print(
        f"Splitting {len(oversized)} oversized chunk(s) "
        f"(>{max_chars} chars) into smaller pieces..."
    )

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=200,
    )

    result = []
    for chunk in chunks:
        if len(chunk.page_content) <= max_chars:
            result.append(chunk)
        else:
            result.extend(
                fallback_splitter.create_documents([chunk.page_content])
            )

    return result

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from ingestion.embeddings import get_embeddings

    embeddings = get_embeddings()

    markdown_file = "data/markdown/2024_Apple.md"
    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()
