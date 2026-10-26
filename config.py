APP_HOST = "0.0.0.0"
APP_PORT = 8000

RAW_PDFS_DIR = "data/raw_pdfs"
MARKDOWN_DIR = "data/markdown"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384

PINECONE_INDEX_NAME = "ai-investor-platform"
PINECONE_NAMESPACE = "financial-reports"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-120b"

KPI_TOP_K = 20
