from __future__ import annotations

from typing import AsyncIterator, List

from openai import AsyncOpenAI

from app.core.config import get_settings

settings = get_settings()


def build_answer_prompt(question: str, contexts: List[str], conversation: str) -> str:
    """
    Single universal prompt that works for BOTH:
    - FOLLOWUP (answerable from conversation)
    - RETRIEVE (answerable from doc context)
    It enforces groundedness and avoids the "Evidence must be from CONTEXT" deadlock.
    """
    ctx = "\n\n---\n\n".join([c for c in (contexts or []) if c]).strip()
    conv = (conversation or "").strip()

    return f"""
You are a grounded assistant. You MUST answer using ONLY the sources provided below:
- CONVERSATION HISTORY
- DOCUMENT CONTEXT

You must follow this exact decision procedure:

STEP 1 — Decide the best source:
A) If the user question is about the assistant's prior message(s) (e.g., "why did you think that", "explain", "clarify", "what do you mean", "expand on that", "justify"), then prefer CONVERSATION HISTORY.
B) Otherwise, prefer DOCUMENT CONTEXT.
C) If the preferred source is insufficient, fall back to the other source.
D) If BOTH are insufficient, output the insufficient message.

STEP 2 — Compose the answer:
- Use ONLY the chosen source(s) (conversation and/or document).
- Do NOT use any outside knowledge.
- If you answer using conversation, you may refer to what was previously said (e.g., "In the previous answer, I said ...").

STEP 3 — Evidence (mandatory if you provide a non-insufficient answer):
- Provide 1–3 DISTINCT verbatim quotes copied EXACTLY from the SAME source(s) you used.
- If you used conversation: quotes must come from CONVERSATION HISTORY.
- If you used document context: quotes must come from DOCUMENT CONTEXT.
- If you used both: you may mix, but every quote must clearly exist verbatim in one of the sources.

Hard rules:
- Never invent quotes.
- Never provide "Evidence" that is paraphrased.
- If you cannot find any verbatim quote in either source that supports your answer, then you MUST output the insufficient message instead.

INSUFFICIENT MESSAGE (must match exactly):
"I'm sorry, but the provided documents do not contain enough information to answer this question."

Output format (exact):
Answer: <your answer OR the insufficient message>
Evidence:
1) "<quote 1>"
2) "<quote 2>"
3) "<quote 3>"

If insufficient, output:
Answer: I'm sorry, but the provided documents do not contain enough information to answer this question.
Evidence: [none]

SOURCES START

CONVERSATION HISTORY:
{conv if conv else "[No prior conversation]"}

DOCUMENT CONTEXT:
{ctx if ctx else "[No document context available]"}

USER QUESTION:
{question}

SOURCES END
""".strip()


async def stream_answer(
    client: AsyncOpenAI,
    question: str,
    contexts: List[str],
    conversation: str = "",
) -> AsyncIterator[str]:
    """Stream LLM answer token deltas."""
    prompt = build_answer_prompt(question, contexts, conversation)

    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "You are a strictly grounded assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
