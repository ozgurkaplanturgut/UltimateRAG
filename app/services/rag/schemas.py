from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RagRequest(BaseModel):
    """Kafka request message schema."""
    model_config = ConfigDict(extra="forbid")

    request_id: str
    action: Literal["query", "upload", "delete"]

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    doc_id: str

    # query
    prompt: Optional[str] = None

    # upload
    source_url: Optional[str] = None
    filename: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_by_action(self) -> "RagRequest":
        if self.action == "upload":
            if not self.source_url:
                raise ValueError("source_url is required for upload")
        elif self.action == "query":
            if not self.session_id:
                raise ValueError("session_id is required for query")
            if not self.prompt:
                raise ValueError("prompt is required for query")
        return self


class RagResponseEvent(BaseModel):
    """Kafka response event schema."""
    request_id: str
    event_type: Literal["meta", "chunk", "end", "error"]
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedWindow(BaseModel):
    """A retrieved document window."""
    point_id: str
    score: float
    text: str
    chunk_text: str
    payload: Dict[str, Any]


class DocProfile(BaseModel):
    """Document profile metadata extracted by LLM."""
    doc_type: Optional[str] = None
    language: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = []
    entities: List[str] = []
