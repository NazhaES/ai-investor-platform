import os
from pathlib import Path

from dotenv import load_dotenv

from ingestion.embeddings import get_embeddings
from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from utils import parse_company_year
from vectorstore.pinecone_store import PineconeVectorStore
import config

load_dotenv()


def ingest_document(
    pdf_path: str,
    embeddings,
    vector_store
) -> None:
    """
    Ingest a single PDF document.
    """
    pdf_file = Path(pdf_path)

    company, year = parse_company_year(pdf_file)
    print(f"Ingesting {pdf_file.name} as company={company!r}, year={year!r}")

    converter = PDFToMarkdownConverter()

    markdown_file = converter.convert_pdf(
        pdf_path=pdf_path,
        output_dir=config.MARKDOWN_DIR
    )

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks for {pdf_file.name}")

    vector_store.upload_chunks(
        chunks=chunks,
        embeddings=embeddings,
        company=company,
        year=year,
        source_file=pdf_file.name
    )


def ingest_directory(input_dir: str) -> None:
    """
    Ingest all PDFs from a directory.
    """
    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE
    )

    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_file in pdf_files:
        ingest_document(
            pdf_path=str(pdf_file),
            embeddings=embeddings,
            vector_store=vector_store
        )


if __name__ == "__main__":
    ingest_directory(config.RAW_PDFS_DIR)
