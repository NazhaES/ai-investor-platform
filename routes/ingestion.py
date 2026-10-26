import shutil
from fastapi import APIRouter, File, UploadFile
from pathlib import Path
import os
from ingestion.embeddings import get_embeddings
from vectorstore.pinecone_store import PineconeVectorStore, Retriever
from ingestion.ingest_documents import ingest_document
from rag.kpi_extractor_rag import extract_financial_metrics
from database.metrics import save_metrics
from utils import parse_company_year
import config

router = APIRouter()


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...)
):
    upload_dir = Path(config.RAW_PDFS_DIR)
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE
    )

    ingest_document(
        pdf_path=str(file_path),
        embeddings=embeddings,
        vector_store=vector_store
    )

    company, year = parse_company_year(file_path)
    year = int(year) if year.isdigit() else None

    metrics = extract_financial_metrics(
        retriever=Retriever(vector_store.client, embeddings, vector_store.namespace),
        company=company,
        year=year
    )

    save_metrics(
        company=company,
        year=year,
        metrics=metrics
    )

    return {
        "message": "Document uploaded successfully",
        "file_name": file.filename
    }
