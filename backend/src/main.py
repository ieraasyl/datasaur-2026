import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.models import DiagnoseRequest, DiagnoseResponse
from src.rag import pipeline
from src.rag.embedder import get_embedder
from src.rag.vectorstore import get_vectorstore
from src.rag.bm25 import get_bm25
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading indexes and models...")
    t0 = time.time()
    get_embedder()
    get_vectorstore()
    get_bm25()
    print(f"[Startup] Ready in {time.time() - t0:.1f}s")
    yield


app = FastAPI(title="Datasaur 2026 — Medical Diagnosis Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    return await pipeline.diagnose(request.symptoms)


# Serve Astro static build
static_dir = settings.static_dir
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
