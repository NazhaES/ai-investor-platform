from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from database.metrics import get_metrics
from database.create_table import create_tables
from ingestion.embeddings import get_embeddings
from vectorstore.create_index import create_index_from_settings
import config
from routes.ingestion import router as ingestion_router
from routes.chat import router as chat_router

load_dotenv()

app = FastAPI(
    title="Fundamenta API"
)


@app.on_event("startup")
def startup_event():
    """
    Initialize database, embedding model and vector index on app startup.
    """
    create_tables()

    get_embeddings()

    try:
        create_index_from_settings()
    except Exception as e:
        print(f"Warning: Could not create vector index: {e}")

    print("Startup complete. App is ready.")

app.include_router(
    ingestion_router,
    prefix="/api",
    tags=["Ingestion"]
)

app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"]
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)


@app.get("/")
def dashboard(request: Request):
    """
    Render dashboard UI.
    """
    metrics = get_metrics()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "total_companies": len(metrics),
            "total_reports": len(metrics)
        }
    )


@app.get("/api/metrics")
def metrics():
    """
    Return KPI metrics.
    """
    return get_metrics()


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.APP_HOST,
        port=config.APP_PORT
    )
