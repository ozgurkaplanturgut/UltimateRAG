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

Task:
Decide whether the user's message needs document retrieval (RETRIEVE) or can be answered using only the immediate conversation history (FOLLOWUP).

You must be conservative:
- Choose FOLLOWUP only when you are highly confident that the conversation alone is sufficient.
- If unsure, choose RETRIEVE.

Definitions:
- FOLLOWUP:
  The user asks about the assistant's previous answer or the conversation itself (explain, clarify, justify, restate),
  WITHOUT asking for new information from the document.

- RETRIEVE:
  The user asks for new facts/details from the document, a new question about the document, changes topic,
  requests specific evidence/quotes/where-it-is-written, or asks anything that likely requires looking at the document again.

Special rule for evidence/quotes:
- If the user asks for citations, quotes, passages, "where does it say", proof, or source from the document, choose RETRIEVE.

Output format:
Return ONLY valid JSON (no markdown, no extra text) with keys:
- "route": "FOLLOWUP" or "RETRIEVE"
- "confidence": a float between 0 and 1
- "reason": a short string explaining the decision

Remember: If not highly confident that FOLLOWUP is sufficient, choose RETRIEVE.
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
