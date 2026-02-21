"""Orchestrates: embed → hybrid retrieve → rerank → prompt → LLM → parse."""
import json
import logging

from src.config import settings, TOP_K, TOP_N_DIAG
from src.models import Diagnosis, DiagnoseResponse
from src.rag.embedder import get_embedder
from src.rag.vectorstore import get_vectorstore
from src.rag.bm25 import get_bm25
from src.rag.retriever import HybridRetriever, hybrid_search, aggregate_by_protocol
from src.rag.prompt import build_prompt
from src.rag.llm import LLMClient

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""

    def __init__(self):
        self.embedder = None
        self.vs = None
        self.bm25 = None
        self.retriever: HybridRetriever | None = None
        self.llm = LLMClient()
        self._ready = False
        self._reranker = None
        if settings.use_reranker:
            try:
                from src.rag.reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
                logger.info("Cross-encoder reranker initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}. Continuing without reranker.")

    def load_indexes(self) -> bool:
        """Load pre-built FAISS + BM25 indexes from disk."""
        try:
            self.vs = get_vectorstore()
            self.bm25 = get_bm25()
            self.embedder = get_embedder()
            if self.vs.index is not None and self.bm25.bm25 is not None:
                self.retriever = HybridRetriever(self.vs, self.bm25)
                self._ready = True
                logger.info("RAG pipeline ready (FAISS + BM25 loaded).")
                return True
        except Exception as e:
            logger.warning(f"Index load failed: {e}")
        logger.warning("Indexes not fully loaded — pipeline not ready.")
        return False

    def is_ready(self) -> bool:
        return self._ready

    async def diagnose(self, symptoms: str, top_n: int = TOP_N_DIAG) -> DiagnoseResponse:
        """Main diagnosis method."""
        if not self._ready:
            raise RuntimeError("Pipeline not initialized — indexes not loaded.")

        q_vec = self.embedder.encode_query(symptoms)
        chunks = self.retriever.search(symptoms, q_vec, k=TOP_K)
        logger.info(f"Retrieved {len(chunks)} chunks for query (before re-ranking).")

        if self._reranker is not None:
            try:
                chunks = self._reranker.rerank(symptoms, chunks, top_k=TOP_K)
                logger.info(f"Chunks re-ranked with cross-encoder (top {len(chunks)} chunks).")
            except Exception as exc:
                logger.warning(f"Reranker failed, falling back to hybrid ranking only: {exc}")

        chunks = aggregate_by_protocol(chunks, top_protocols=5)
        logger.info(f"After protocol aggregation: {len(chunks)} chunks.")

        prompt = build_prompt(symptoms, chunks, top_n=top_n)
        raw_diagnoses = await self.llm.diagnose(prompt, chunks, top_n=top_n)

        diagnoses = []
        for i, d in enumerate(raw_diagnoses[:top_n]):
            try:
                diagnoses.append(Diagnosis(
                    rank=d.get("rank", i + 1),
                    diagnosis=str(d.get("diagnosis", "Неизвестный диагноз")),
                    icd10_code=str(d.get("icd10_code", "Z99")),
                    explanation=str(d.get("explanation", "")),
                ))
            except Exception as exc:
                logger.warning(f"Skipping malformed diagnosis entry: {exc}")

        return DiagnoseResponse(diagnoses=diagnoses)


async def diagnose(symptoms: str | None) -> DiagnoseResponse:
    """Legacy function interface - uses singleton pipeline."""
    symptoms = symptoms or ""
    embedder = get_embedder()
    query_embedding = embedder.encode_query(symptoms)

    chunks = hybrid_search(symptoms, query_embedding)

    if settings.use_reranker:
        try:
            from src.rag.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            chunks = reranker.rerank(symptoms, chunks, top_k=TOP_K)
            logger.debug("Legacy path: chunks re-ranked with cross-encoder.")
        except Exception as _exc:
            logger.warning(f"[Pipeline] Legacy reranker failed, ignoring: {_exc}")

    chunks = aggregate_by_protocol(chunks, top_protocols=5)

    from src.rag.prompt import build_prompt_messages
    messages = build_prompt_messages(symptoms, chunks)

    from src.rag import llm
    raw = await llm.complete(messages, chunks)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            diagnoses = [Diagnosis(**d) for d in data]
        else:
            diagnoses = [Diagnosis(**d) for d in data.get("diagnoses", data.get("results", []))]
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to parse LLM response: {e}\nRaw: {raw}")
        diagnoses = []
        for rank, chunk in enumerate(chunks[:3], start=1):
            codes = chunk.get("icd_codes", [])
            if codes:
                diagnoses.append(Diagnosis(
                    rank=rank,
                    diagnosis=chunk.get("source_file", "Unknown").replace(".pdf", ""),
                    icd10_code=codes[0],
                    explanation="Определено на основе поиска по протоколам.",
                ))

    return DiagnoseResponse(diagnoses=diagnoses)
