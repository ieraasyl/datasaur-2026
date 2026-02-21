import json
import logging
from openai import AsyncOpenAI
from src.config import settings

logger = logging.getLogger(__name__)


def _get_client() -> AsyncOpenAI | None:
    if not settings.gpt_oss_api_key:
        return None
    return AsyncOpenAI(
        base_url=settings.gpt_oss_url,
        api_key=settings.gpt_oss_api_key,
    )


def _mock_diagnoses(chunks: list[dict], top_n: int = 3) -> list[dict]:
    """Return top ICD codes from retrieved chunks when no API key is set."""
    seen: set[str] = set()
    diagnoses: list[dict] = []
    rank = 1
    for chunk in chunks:
        for code in chunk.get("icd_codes", []):
            if code not in seen:
                seen.add(code)
                diagnoses.append({
                    "rank": rank,
                    "diagnosis": chunk.get("source_file", "Unknown").replace(".pdf", ""),
                    "icd10_code": code,
                    "explanation": f"[Mock] На основе протокола {chunk.get('protocol_id', '')}.",
                })
                rank += 1
            if rank > top_n:
                break
        if rank > top_n:
            break
    return diagnoses


class LLMClient:
    """Wrapper around the gpt-oss OpenAI-compatible API."""

    def __init__(self):
        self._client = _get_client()

    async def diagnose(self, prompt: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
        """Send prompt to LLM and return parsed list of diagnosis dicts."""
        if self._client is None:
            logger.warning("[LLM] No API key — running in mock mode")
            return _mock_diagnoses(chunks, top_n)

        from src.rag.prompt import SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self._client.chat.completions.create(
            model=settings.gpt_oss_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            return data.get("diagnoses", data.get("results", []))
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}\nRaw: {raw}")
            return _mock_diagnoses(chunks, top_n)


async def complete(messages: list[dict], chunks: list[dict]) -> str:
    """Legacy function interface for backward compatibility."""
    client = _get_client()

    if client is None:
        logger.warning("[LLM] No API key — running in mock mode")
        return json.dumps({"diagnoses": _mock_diagnoses(chunks)}, ensure_ascii=False)

    response = await client.chat.completions.create(
        model=settings.gpt_oss_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content
