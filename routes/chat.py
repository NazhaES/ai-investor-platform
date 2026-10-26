import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ingestion.embeddings import get_embeddings
from vectorstore.pinecone_store import PineconeVectorStore, Retriever
from llm.llm import get_openai_client
import config

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None
    detailed: bool = False

@router.post("/chat")
def chat(request: ChatRequest):
    try:
        embeddings = get_embeddings()

        vector_store = PineconeVectorStore(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=config.PINECONE_INDEX_NAME,
            namespace=config.PINECONE_NAMESPACE
        )
        retriever = Retriever(vector_store.client, embeddings, vector_store.namespace)

        context = ""
        if request.company and request.year:
            docs = retriever.invoke(
                query=request.question,
                company=request.company,
                year=request.year
            )
        else:
            docs = retriever.invoke(
                query=request.question
            )
        context = "\n\n".join(doc.page_content for doc in docs)

        if request.detailed:
            style = "Provide a thorough, well-structured answer."
        else:
            style = (
                "Be concise: answer in a few sentences (under ~150 words). "
                "Use a short bullet list only if necessary."
            )

        rules = (
            "Rules for numbers (critical): "
            "Use ONLY figures that appear verbatim in the context. "
            "Never invent, round, convert units, or calculate figures yourself. "
            "If a figure is not present in the context, say it is not available."
        )

        prompt = f"You are an expert financial analyst. {style} {rules} Use the following context from corporate reports to answer the user's question. If the context does not contain relevant information, politely indicate that you do not have enough data.\n\nContext:\n{context}\n\nUser Question: {request.question}\n\nAnswer:"

        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
