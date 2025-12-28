from __future__ import annotations

from typing import AsyncIterator, List

from openai import AsyncOpenAI

from app.core.config import get_settings

settings = get_settings()


def build_answer_prompt(question: str, contexts: List[str], conversation: str) -> str:
    """Build a balanced, grounded prompt for better RAG performance."""
    ctx = "\n\n---\n\n".join([c for c in contexts if c]).strip()
    conv = (conversation or "").strip()

    return f"""
You are an expert assistant that provides helpful and accurate answers based ONLY on the provided context and conversation history.

### INSTRUCTIONS:
1. **Groundedness:** Your answer must be derived directly from the CONTEXT or CONVERSATION. Do not use any internal knowledge or make up facts.
2. **Handling Missing Information:** 
   - If the information is not present in the sources, state: "I'm sorry, but the provided documents do not contain enough information to answer this question."
   - Do not attempt to guess or use outside information.
3. **Synthesis:** Combine information from different parts of the context to provide a comprehensive answer.
4. **Tone:** Be professional, direct, and concise.

### OUTPUT FORMAT:
- **Answer:** <Your structured answer here>
- **Evidence:** Provide 1-3 distinct, verbatim quotes from the CONTEXT that directly support your answer. Use the format: "..."

### DATA:
CONVERSATION HISTORY:
{conv if conv else "[No prior conversation]"}

DOKÜMAN CONTEXT:
{ctx if ctx else "[No document context available]"}

USER QUESTION:
{question}
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
