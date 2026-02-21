"""Prompt templates for the GPT-OSS clinical reasoning call."""
from collections import defaultdict

SYSTEM_PROMPT = """Ты — AI-ассистент клинической диагностики по протоколам Минздрава Республики Казахстан.

Задача: по описанию симптомов пациента определить наиболее вероятный диагноз и его код МКБ-10, опираясь ТОЛЬКО на предоставленные фрагменты клинических протоколов.

Алгоритм:
1. Внимательно прочитай описание симптомов пациента.
2. Определи, какой из предоставленных протоколов наиболее соответствует описанным симптомам.
3. Из этого протокола выбери САМЫЙ СПЕЦИФИЧЕСКИЙ код МКБ-10 (с десятичной частью, если есть), который лучше всего описывает состояние пациента.
4. ЗАПРЕЩЕНО придумывать коды — используй ТОЛЬКО коды, явно указанные в протоколах.

Формат ответа — строго JSON:
{"diagnoses":[{"rank":1,"diagnosis":"Название диагноза","icd10_code":"X00.0","explanation":"Краткое обоснование"}]}"""

DIAGNOSIS_PROMPT = """## Симптомы пациента:
{symptoms}

## Найденные клинические протоколы РК:
{context}

## Доступные коды МКБ-10 (из протоколов выше):
{icd_list}

Проанализируй симптомы и определи до {top_n} наиболее вероятных диагнозов.
Для каждого диагноза укажи конкретный код МКБ-10 из списка выше, который ЛУЧШЕ ВСЕГО соответствует описанным симптомам.
Верни JSON:"""


def build_context(chunks: list[dict], max_chars: int = 14000) -> str:
    """Format retrieved chunks grouped by protocol for clearer LLM reasoning."""
    protocols: dict[str, dict] = defaultdict(
        lambda: {"source_file": "", "icd_codes": [], "chunks": []}
    )

    for c in chunks:
        pid = c["protocol_id"]
        if not protocols[pid]["source_file"]:
            protocols[pid]["source_file"] = c.get("source_file", "")
            protocols[pid]["icd_codes"] = c.get("icd_codes", [])
        protocols[pid]["chunks"].append(c.get("chunk", c.get("text", "")))

    parts: list[str] = []
    total = 0
    for pid, data in protocols.items():
        src = data["source_file"]
        icds = ", ".join(data["icd_codes"][:10]) or "—"
        header = f"\n### Протокол: {src}\nКоды МКБ-10: {icds}\n"

        protocol_text = header
        for chunk_text in data["chunks"]:
            candidate = protocol_text + "\n" + chunk_text
            if total + len(candidate) > max_chars:
                if protocol_text != header:
                    break
                chunk_text = chunk_text[: max_chars - total - len(header) - 10]
                protocol_text += "\n" + chunk_text
                break
            protocol_text = candidate

        if total + len(protocol_text) > max_chars:
            if parts:
                break
        parts.append(protocol_text)
        total += len(protocol_text)

    return "\n---\n".join(parts)


def _collect_icd_list(chunks: list[dict], max_codes: int = 30) -> str:
    """Collect ICD-10 codes grouped by protocol for clarity."""
    protocol_codes: dict[str, list[str]] = defaultdict(list)
    seen_global: set[str] = set()
    total = 0

    for c in chunks:
        src = c.get("source_file", "Unknown")
        for code in c.get("icd_codes", []):
            if code and code not in seen_global:
                seen_global.add(code)
                protocol_codes[src].append(code)
                total += 1
            if total >= max_codes:
                break
        if total >= max_codes:
            break

    if not protocol_codes:
        return "нет явных кандидатов"

    parts = []
    for src, codes in protocol_codes.items():
        name = src.replace(".pdf", "")
        parts.append(f"{name}: {', '.join(codes)}")
    return "\n".join(parts)


def build_prompt(symptoms: str, chunks: list[dict], top_n: int = 5) -> str:
    """Build prompt string for LLM."""
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
