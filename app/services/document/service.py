from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import httpx
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.services.rag.chunking import paragraph_aware_chunks
from app.services.rag.retrieve import embed_dense
from app.services.rag.rewrite import build_doc_profile_llm
from app.services.rag.sparse import sparse_encode

settings = get_settings()
logger = logging.getLogger(__name__)


async def download_text(source_url: str) -> str:
    """Download document text from URL (same behavior: UA header, follow redirects, 120s timeout)."""
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as h:
        r = await h.get(source_url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text


async def build_chunks(text: str) -> List[Tuple[str, int]]:
    """Chunk document text (paragraph-aware)."""
    return paragraph_aware_chunks(text, settings.rag_chunk_tokens, settings.rag_overlap_tokens)


async def upsert_doc_meta(
    qdrant: QdrantClient,
    *,
    user_id: Optional[str],
    doc_id: str,
    meta_id: int,
    client: AsyncOpenAI,
    text: str,
) -> dict:
    """Create doc_profile + summary embedding and upsert doc_meta point."""
    sample = text[: settings.rag_doc_profile_sample_chars]
    doc_profile = await build_doc_profile_llm(client, sample)
    summary = (doc_profile.get("summary") or "")[:4000]

    meta_vec = (await embed_dense(client, [summary or doc_id]))[0]

    qdrant.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=meta_id,
                vector={settings.qdrant_dense_vector_name: meta_vec},
                payload={
                    "type": "doc_meta",
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "doc_profile": doc_profile,
                    "text": summary,
                },
            )
        ],
        wait=True,
    )
    return doc_profile


async def upsert_chunks(
    qdrant: QdrantClient,
    *,
    user_id: Optional[str],
    doc_id: str,
    chunks: List[Tuple[str, int]],
    client: AsyncOpenAI,
) -> int:
    """Embed + sparse encode chunks and upsert to Qdrant in batches (preserves original logic)."""
    texts = [c[0] for c in chunks]
    dense_vecs: List[List[float]] = []

    bs = int(settings.rag_embed_batch)
    for i in range(0, len(texts), bs):
        dense_vecs.extend(await embed_dense(client, texts[i: i + bs]))

    points: List[models.PointStruct] = []
    for idx, ((chunk_text, tok_offset), dvec) in enumerate(zip(chunks, dense_vecs)):
        svec = sparse_encode(chunk_text)
        pid = abs(hash((user_id, doc_id, idx))) % (2**63)

        points.append(
            models.PointStruct(
                id=pid,
                vector={
                    settings.qdrant_dense_vector_name: dvec,
                    settings.qdrant_sparse_vector_name: svec,
                },
                payload={
                    "type": "chunk",
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "offset": tok_offset,
                    "text": chunk_text,
                },
            )
        )

    upsert_batch = int(settings.rag_qdrant_upsert_batch)
    for i in range(0, len(points), upsert_batch):
        qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i: i + upsert_batch],
            wait=True,
        )

    return len(points)
