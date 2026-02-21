"""GPT-OSS client using OpenAI SDK pointed at the organizer LiteLLM proxy."""
import json
import logging
import re

from openai import AsyncOpenAI

from src.config import settings, TOP_N_DIAG

logger = logging.getLogger(__name__)


def _parse_json_diagnoses(text: str) -> list[dict] | None:
    """Extract and parse a JSON array/object from LLM response text."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "diagnoses" in data:
            return data["diagnoses"]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return None


def mock_response(chunks: list[dict], top_n: int = TOP_N_DIAG) -> list[dict]:
    """Fallback: build diagnoses directly from retrieved chunk metadata (no LLM)."""
    icd_scores: dict[str, float] = {}
    icd_meta: dict[str, dict] = {}
    icd_chunk_count: dict[str, int] = {}

    for c in chunks:
        base_score = c.get("rrf_score", c.get("dense_score", c.get("sparse_score", 0.5)))
        score = float(base_score) ** 1.2
        seen_here: set[str] = set()
        for icd in c.get("icd_codes", []):
            if not icd:
                continue
            icd_scores[icd] = icd_scores.get(icd, 0.0) + score
            if icd not in icd_meta:
                icd_meta[icd] = c
            if icd not in seen_here:
                seen_here.add(icd)
                icd_chunk_count[icd] = icd_chunk_count.get(icd, 0) + 1

    def _specificity_bonus(code: str) -> float:
        return 1.1 if "." in code else 1.0

    def _final_score(item: tuple[str, float]) -> float:
        code, base = item
        freq = icd_chunk_count.get(code, 1)
        return base * _specificity_bonus(code) * (1.0 + 0.05 * min(freq, 10))

    ranked = sorted(icd_scores.items(), key=_final_score, reverse=True)[:top_n]
    results = []
    for rank, (icd, _) in enumerate(ranked, 1):
        meta = icd_meta[icd]
        chunk_text = meta.get("chunk", meta.get("text", ""))
        results.append({
            "rank": rank,
            "diagnosis": meta.get("title", meta.get("source_file", "Диагноз")),
            "icd10_code": icd,
            "explanation": (
                f"Код {icd} из протокола {meta['source_file']}. "
                f"{chunk_text[:150].strip()}..."
            ),
        })
    return results


class LLMClient:
    """LLM client with fallback to mock mode."""

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI | None:
        """Lazy initialize OpenAI client."""
        if self._client is None:
            if not settings.gpt_oss_api_key:
                return None
            self._client = AsyncOpenAI(
                base_url=settings.gpt_oss_url,
                api_key=settings.gpt_oss_api_key,
            )
        return self._client

    async def diagnose(
        self,
        user_prompt: str,
        chunks: list[dict],
        top_n: int = TOP_N_DIAG,
    ) -> list[dict]:
        """Call GPT-OSS and return parsed diagnoses. Falls back to mock on error."""
        from src.rag.prompt import SYSTEM_PROMPT

        client = self._get_client()
        if settings.mock_llm or client is None:
            logger.warning("MOCK_LLM=true or no API key — using retrieval-based fallback.")
            return mock_response(chunks, top_n)

        try:
            response = await client.chat.completions.create(
                model=settings.gpt_oss_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or ""
            logger.debug(f"LLM raw response (first 300 chars): {raw[:300]}")
            parsed = _parse_json_diagnoses(raw)
            if parsed:
                for i, d in enumerate(parsed[:top_n]):
                    d["rank"] = i + 1
                return parsed[:top_n]
            else:
                logger.warning("LLM returned unparseable JSON — falling back to mock.")
                return mock_response(chunks, top_n)
        except Exception as exc:
            logger.error(f"LLM call failed: {exc} — falling back to mock.")
            return mock_response(chunks, top_n)


async def complete(messages: list[dict], chunks: list[dict]) -> str:
    """Legacy function interface - returns JSON string."""
    client = LLMClient()
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    diagnoses = await client.diagnose(user_msg, chunks)
    return json.dumps({"diagnoses": diagnoses}, ensure_ascii=False)
