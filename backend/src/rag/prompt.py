"""Prompt templates for the GPT-OSS clinical reasoning call."""

SYSTEM_PROMPT = """Ты — AI-ассистент клинической диагностики по протоколам Минздрава РК.

Правила:
1. Используй ТОЛЬКО коды МКБ-10, явно указанные в предоставленных фрагментах протоколов. ЗАПРЕЩЕНО придумывать коды.
2. Выбирай САМЫЙ СПЕЦИФИЧЕСКИЙ код (с десятичной частью, если есть).
3. Ранг 1 = наиболее вероятный диагноз.
4. Верни ТОЛЬКО валидный JSON-массив, без markdown и пояснений до/после.

Формат ответа — JSON-массив:
[{"rank":1,"diagnosis":"Название","icd10_code":"X00.0","explanation":"1 предложение"}]"""

DIAGNOSIS_PROMPT = """## Симптомы:
{symptoms}

## Возможные коды МКБ-10 из найденных протоколов:
{icd_list}

## Релевантные клинические протоколы РК:
{context}

Определи до {top_n} наиболее вероятных диагнозов. Используй ТОЛЬКО коды из протоколов выше.
Верни JSON-массив:"""


def build_context(chunks: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved protocol chunks into LLM context string."""
    parts = []
    total = 0
    seen_protocols: set[str] = set()

    for c in chunks:
        pid = c["protocol_id"]
        header = ""
        if pid not in seen_protocols:
            seen_protocols.add(pid)
            icds = ", ".join(c.get("icd_codes", [])[:8]) or "—"
            header = f"\n### {c['source_file']} | МКБ-10: {icds}\n"
        chunk_text = c.get("chunk", c.get("text", ""))
        text = header + chunk_text
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)

    return "\n---\n".join(parts)


def _collect_icd_list(chunks: list[dict], max_codes: int = 20) -> str:
    """Collect a compact, unique list of ICD-10 codes from retrieved chunks."""
    codes: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        for code in c.get("icd_codes", []):
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
            if len(codes) >= max_codes:
                break
        if len(codes) >= max_codes:
            break
    return ", ".join(codes) if codes else "нет явных кандидатов"


def build_prompt(symptoms: str, chunks: list[dict], top_n: int = 5) -> str:
    """Build prompt string for LLM (legacy format)."""
    context = build_context(chunks)
    icd_list = _collect_icd_list(chunks)
    return DIAGNOSIS_PROMPT.format(
        symptoms=symptoms,
        context=context,
        top_n=top_n,
        icd_list=icd_list,
    )


def build_prompt_messages(symptoms: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
    """Build prompt as messages list for OpenAI API."""
    context = build_context(chunks)
    icd_list = _collect_icd_list(chunks)
    user_message = DIAGNOSIS_PROMPT.format(
        symptoms=symptoms,
        context=context,
        top_n=top_n,
        icd_list=icd_list,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
