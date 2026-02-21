SYSTEM_PROMPT = """Ты — клиническая система поддержки принятия решений, разработанная на основе официальных клинических протоколов Республики Казахстан.

Твоя задача: на основе описания симптомов пациента и предоставленных фрагментов клинических протоколов определить наиболее вероятные диагнозы с кодами МКБ-10.

Правила:
1. Возвращай ТОЛЬКО валидный JSON, без markdown, без пояснений вне JSON.
2. Используй ТОЛЬКО коды МКБ-10 из предоставленных протоколов.
3. Ранжируй диагнозы от наиболее вероятного к наименее вероятному.
4. Давай краткое клиническое обоснование на русском языке (1-2 предложения).
5. Возвращай от 1 до 5 диагнозов.

Формат ответа:
{
  "diagnoses": [
    {
      "rank": 1,
      "diagnosis": "Название диагноза",
      "icd10_code": "X00.0",
      "explanation": "Краткое клиническое обоснование."
    }
  ]
}"""


def build_prompt(symptoms: str, chunks: list[dict]) -> list[dict]:
    context_parts = []
    for i, chunk in enumerate(chunks[:8], start=1):  # limit context size
        icd_codes = ", ".join(chunk.get("icd_codes", []))
        context_parts.append(
            f"[Протокол {i}] {chunk.get('source_file', '')} | МКБ-10: {icd_codes}\n"
            f"{chunk.get('text', '')[:600]}"
        )

    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""Симптомы пациента:
{symptoms}

Релевантные фрагменты клинических протоколов РК:
{context}

Определи диагнозы и верни JSON."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
