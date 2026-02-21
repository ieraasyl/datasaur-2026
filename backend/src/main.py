"""FastAPI application: POST /diagnose + serves Astro static build."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import settings
from src.models import DiagnoseRequest, DiagnoseResponse
from src.rag import pipeline
from src.rag.embedder import get_embedder
from src.rag.vectorstore import get_vectorstore
from src.rag.bm25 import get_bm25
from src.rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Use class-based pipeline for better error handling and health checks
pipeline_instance = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RAG indexes...")
    t0 = time.time()
    # Pre-load components
    get_embedder()
    get_vectorstore()
    get_bm25()
    # Try to initialize pipeline
    ok = pipeline_instance.load_indexes()
    if not ok:
        logger.warning(
            "Indexes not fully loaded! Run: python scripts/index_corpus.py first.\n"
            "Falling back to function-based pipeline — may have limited functionality."
        )
    elapsed = time.time() - t0
    logger.info(f"Startup complete in {elapsed:.1f}s")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Datasaur 2026 — Medical Diagnosis Assistant",
    description="AI-powered clinical decision support based on Kazakhstan clinical protocols",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint with pipeline status."""
    return {
        "status": "ok",
        "pipeline_ready": pipeline_instance.is_ready(),
    }


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """Diagnose endpoint - uses class-based pipeline if ready, falls back to function-based."""
    if not request.symptoms or not request.symptoms.strip():
        raise HTTPException(status_code=422, detail="symptoms field must not be empty.")
    
    # Use class-based pipeline if ready
    if pipeline_instance.is_ready():
        try:
            return await pipeline_instance.diagnose(request.symptoms)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("Unhandled error in /diagnose")
            raise HTTPException(status_code=500, detail="Internal server error.")
    else:
        # Fallback to function-based pipeline
        logger.warning("Using function-based pipeline fallback")
    return await pipeline.diagnose(request.symptoms)


# Serve Astro static build
static_dir = settings.static_dir
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        # Serve index.html for all non-API routes (SPA fallback)
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Frontend not found"}
else:
    logger.warning(f"Static frontend not found at {static_dir}. Run 'bun run build' in frontend/.")

    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Backend running. Frontend not built yet."}
