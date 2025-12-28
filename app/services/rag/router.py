from __future__ import annotations

import json
import logging
from typing import Literal, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RagRouteDecision(BaseModel):
    route: Literal["FOLLOWUP", "RETRIEVE"] = Field(...)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = Field(default="")


ROUTER_SYSTEM = """You are a routing classifier for a Retrieval-Augmented Generation (RAG) system.

Goal:
Choose whether the user's message should be handled as:
- FOLLOWUP: answer using ONLY the ongoing conversation / the assistant's previous message(s), without retrieving from the document
- RETRIEVE: run document retrieval because the user likely needs new information grounded in the document

Key idea:
- Prefer FOLLOWUP for meta-questions about the conversation or the assistant's previous answer.
- Use RETRIEVE only when the user asks for new document-grounded facts, specific details, or evidence.

Definitions:

FOLLOWUP (no retrieval):
Choose FOLLOWUP when the user is asking to:
- explain, clarify, justify, restate, or expand the assistant's previous answer
- ask "why did you say/think that", "what do you mean", "how did you get that", "can you elaborate"
- request reformatting of the previous answer (shorter/longer/bullets/translate)
- ask for the reasoning behind the assistant's answer WITHOUT asking for quotes/citations/passages
- ask about the conversation state (session, previous messages, what you said earlier)

Typical FOLLOWUP examples:
- "Why did you think that?"
- "Why did you say that?"
- "What do you mean by that?"
- "Can you explain more?"
- "Summarize your answer in 2 sentences."
- "Translate that to Turkish."

RETRIEVE (needs document search):
Choose RETRIEVE when the user:
- asks a new question about the document/story/content
- asks for specific facts, names, events, definitions from the document
- asks for quotes, passages, citations, evidence, or “where does it say”
- asks for chapter/section/page/line location or verbatim text
- asks anything that must be grounded in the document beyond what was already stated in the conversation

Special rule (hard override to RETRIEVE):
If the user asks for any of the following, ALWAYS choose RETRIEVE:
- quotes / passage / excerpt / exact wording
- citations / evidence / proof / "where does it say"
- chapter/section/page/line references

Disambiguation rule:
If the message is ambiguous but looks like a meta-question about the assistant's previous answer (e.g., "why", "how do you know", "explain that"),
choose FOLLOWUP. If the user explicitly requests evidence/quotes/location in the text, choose RETRIEVE.

Output format:
Return ONLY valid JSON (no markdown, no extra text) with keys:
- "route": "FOLLOWUP" or "RETRIEVE"
- "confidence": a float between 0 and 1
- "reason": a short string explaining the decision
"""



async def route_rag_query(
    client: AsyncOpenAI,
    user_question: str,
    conversation: str,
    model: Optional[str] = None,
) -> RagRouteDecision:
    """LLM router. Conservative: defaults to RETRIEVE on any parsing/LLM issues."""
    model_name = model or getattr(
        settings, "rag_router_model", None) or settings.openai_model

    prompt = {
        "conversation": (conversation or "").strip(),
        "question": user_question.strip(),
        "note": "Choose FOLLOWUP only if conversation alone is enough. Otherwise RETRIEVE.",
    }

    try:
        resp = await client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": json.dumps(
                    prompt, ensure_ascii=False)},
            ],
        )

        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        decision = RagRouteDecision.model_validate(data)

        if decision.route == "FOLLOWUP" and decision.confidence < 0.70:
            return RagRouteDecision(
                route="RETRIEVE",
                confidence=0.69,
                reason="Router confidence too low for FOLLOWUP; fallback to RETRIEVE.",
            )

        return decision

    except Exception as e:
        logger.exception("Router failed; fallback to RETRIEVE. err=%s", e)
        return RagRouteDecision(route="RETRIEVE", confidence=0.0, reason=f"router_error_fallback: {type(e).__name__}")
