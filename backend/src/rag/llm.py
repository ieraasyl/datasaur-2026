import json
from openai import AsyncOpenAI
from src.config import settings


def _get_client() -> AsyncOpenAI | None:
    if not settings.gpt_oss_api_key:
        return None
    return AsyncOpenAI(
        base_url=settings.gpt_oss_url,
        api_key=settings.gpt_oss_api_key,
    )


async def complete(messages: list[dict], chunks: list[dict]) -> str:
    client = _get_client()

    if client is None:
        # Mock mode: return top ICD codes from retrieved chunks
        print("[LLM] No API key — running in mock mode")
        seen = set()
        diagnoses = []
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
                if rank > 3:
                    break
            if rank > 3:
                break

        return json.dumps({"diagnoses": diagnoses}, ensure_ascii=False)

    response = await client.chat.completions.create(
        model=settings.gpt_oss_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content
