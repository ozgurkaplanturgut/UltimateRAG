from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from app.core.config import get_settings

settings = get_settings()


def _normalize_queries(qs: List[str], fallback: str) -> List[str]:
    """Normalize rewrite queries: trim, dedupe, ensure exactly 3 with fallback."""
    out: List[str] = []
    for q in qs or []:
        q = str(q).strip()
        if q and q not in out:
            out.append(q)
        if len(out) == 3:
            break
    while len(out) < 3:
        out.append(fallback)
    return out


def _parse_json_obj(text: str) -> Optional[dict]:
    """Parse STRICT JSON object from text, return None on failure."""
    t = (text or "").strip()
    if not t:
        return None

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    return None


async def build_doc_profile_llm(client: AsyncOpenAI, doc_sample: str) -> Dict[str, Any]:
    """
    Build a lightweight document profile via LLM.
    Returns STRICT JSON object, defaulting to empty fields on failure.
    """
    prompt = f"""
Return ONLY valid JSON (no markdown, no extra text).

Schema:
{{
  "doc_type": string|null,
  "language": string|null,
  "summary": string|null,
  "keywords": [string, ...],
  "entities": [string, ...]
}}

TEXT SAMPLE:
{doc_sample}
""".strip()

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "You output JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    txt = (resp.choices[0].message.content or "").strip()
    obj = _parse_json_obj(txt)
    if not obj:
        return {"doc_type": None, "language": None, "summary": None, "keywords": [], "entities": []}

    return {
        "doc_type": obj.get("doc_type"),
        "language": obj.get("language"),
        "summary": obj.get("summary"),
        "keywords": obj.get("keywords") or [],
        "entities": obj.get("entities") or [],
    }


def build_rewrite_prompt(
    question: str,
    doc_profile: Dict[str, Any],
    prf_snippets: List[str],
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build prompt for rewriting queries."""
    history_block = ""
    if history:
        take = history[-settings.rag_rewrite_history_max_messages:]
        lines: List[str] = []
        for m in take:
            role = (m.get("role") or "user").upper()
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        if lines:
            history_block = "\n\nCHAT HISTORY (recent):\n" + "\n".join(lines)

    snippets_block = "\n".join(
        (prf_snippets or [])[: settings.rag_prf_snippets])

    return f"""
            You are rewriting a user question for better retrieval in a RAG system.

            Return ONLY valid JSON (no markdown, no extra text) with this exact schema:
            {{"queries": ["...", "..."]}}

            Rules:
            - "queries" MUST contain exactly 2 strings.
            - Each query must be standalone and good for retrieval.
            - No numbering, no bullets, no explanations.
            - Preserve the user's intent.
            - Use document-specific terminology when helpful.
            - If question references previous turns ("why", "that", etc.), use CHAT HISTORY to resolve references.

            DOCUMENT PROFILE (JSON):
            {json.dumps(doc_profile, ensure_ascii=False)}

            PRF SNIPPETS:
            {snippets_block}

            USER QUESTION:
            {question}
            {history_block}
            """.strip()


async def rewrite_queries(
    client: AsyncOpenAI,
    question: str,
    doc_profile: Dict[str, Any],
    prf_snippets: List[str],
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Generate exactly 3 rewrite queries, with normalization fallback."""
    prompt = build_rewrite_prompt(question, doc_profile, prf_snippets, history)

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system",
                "content": "Output JSON only. No markdown. Follow schema exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    txt = (resp.choices[0].message.content or "").strip()
    obj = _parse_json_obj(txt)

    if obj and isinstance(obj.get("queries"), list):
        return _normalize_queries(obj["queries"], fallback=question)

    return _normalize_queries([], fallback=question)
