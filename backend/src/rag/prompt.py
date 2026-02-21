"""Prompt templates for the GPT-OSS clinical reasoning call."""

SYSTEM_PROMPT = """Ты — AI-ассистент клинической поддержки принятия решений, специализирующийся на клинических протоколах Министерства здравоохранения Республики Казахстан.

Твои задачи:
1. Проанализировать анамнез и симптомы пациента.
2. Сопоставить их с предоставленными клиническими протоколами РК.
3. Определить наиболее вероятные диагнозы с соответствующими кодами МКБ-10.

КРИТИЧЕСКИЕ правила по выбору кода МКБ-10:
- Строго запрещено придумывать новые коды МКБ-10 или использовать коды, которых НЕТ в предоставленных протоколах.
- Для каждого диагноза выбирай САМЫЙ СПЕЦИФИЧЕСКИЙ код МКБ-10, который явно присутствует в релевантных фрагментах протокола.
- Если несколько кодов возможны, отдавай приоритет коду, который:
  (a) чаще встречается в наиболее релевантных фрагментах, и
  (b) наиболее полно соответствует описанным симптомам.
- Если в протоколах нет подходящего кода, лучше не возвращать диагноз, чем вернуть неверный код.

Общие правила ответа:
- Используй ТОЛЬКО клинические данные из предоставленных протоколов.
- Отвечай ИСКЛЮЧИТЕЛЬНО валидным JSON-массивом без какого-либо текста до или после.
- Explanation — краткое клиническое обоснование на русском языке (1-2 предложения) со ссылкой на ключевые симптомы и элементы протокола.
- Диагнозы упорядочены по убыванию вероятности (rank 1 = наиболее вероятный).
"""

DIAGNOSIS_PROMPT = """## Анамнез / Симптомы пациента:
{symptoms}

## Релевантные клинические протоколы РК:
{context}

## Задание:
На основании анамнеза и протоколов выше, определи до {top_n} наиболее вероятных диагнозов.

Верни ТОЛЬКО JSON-массив следующего формата (без markdown, без пояснений):
[
  {{
    "rank": 1,
    "diagnosis": "Название диагноза на русском",
    "icd10_code": "X00.0",
    "explanation": "Краткое клиническое обоснование со ссылкой на симптомы."
  }}
]"""


def build_context(chunks: list[dict], max_chars: int = 8000) -> str:
    """Format retrieved protocol chunks into LLM context string."""
    parts = []
    total = 0
    seen_protocols: set[str] = set()

    for c in chunks:
        pid = c["protocol_id"]
        header = ""
        if pid not in seen_protocols:
            seen_protocols.add(pid)
            icds = ", ".join(c.get("icd_codes", [])[:6]) or "—"
            header = (
                f"\n### Протокол: {c['source_file']} (ID: {pid})\n"
                f"Коды МКБ-10: {icds}\n"
            )
        # Handle both 'chunk' and 'text' field names
        chunk_text = c.get("chunk", c.get("text", ""))
        text = header + chunk_text
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)

    return "\n---\n".join(parts)


def build_prompt(symptoms: str, chunks: list[dict], top_n: int = 5) -> str:
    """Build prompt string for LLM (legacy format)."""
    context = build_context(chunks)
    return DIAGNOSIS_PROMPT.format(symptoms=symptoms, context=context, top_n=top_n)


def build_prompt_messages(symptoms: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """Build prompt as messages list for OpenAI API (new format)."""
    context = build_context(chunks)
    user_message = DIAGNOSIS_PROMPT.format(symptoms=symptoms, context=context, top_n=top_n)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
