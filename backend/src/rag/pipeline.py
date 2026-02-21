import json
from src.rag.embedder import get_embedder
from src.rag.retriever import hybrid_search
from src.rag.prompt import build_prompt
from src.rag import llm
from src.models import DiagnoseResponse, Diagnosis


async def diagnose(symptoms: str | None) -> DiagnoseResponse:
    symptoms = symptoms or ""
    embedder = get_embedder()
    query_embedding = embedder.encode(symptoms)

    chunks = hybrid_search(symptoms, query_embedding)

    messages = build_prompt(symptoms, chunks)
    raw = await llm.complete(messages, chunks)

    try:
        data = json.loads(raw)
        diagnoses = [Diagnosis(**d) for d in data.get("diagnoses", [])]
    except Exception as e:
        print(f"[Pipeline] Failed to parse LLM response: {e}\nRaw: {raw}")
        # Fallback: return top chunk's ICD codes
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
