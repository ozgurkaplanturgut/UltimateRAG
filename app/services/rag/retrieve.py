from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from app.core.config import get_settings
from app.services.rag.sparse import sparse_encode

settings = get_settings()


async def run_blocking(fn, *args, **kwargs):
    """Run sync / CPU-heavy work in a thread so we don't block the event loop."""
    return await asyncio.to_thread(partial(fn, *args, **kwargs))


def rrf_fuse(
    results_lists: List[List[Tuple[str, float, Dict[str, Any]]]],
    k: int = 60,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Reciprocal Rank Fusion (rank-based).
    Each item: (point_id, score, payload)
    """
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for lst in results_lists:
        for rank, (pid, _score, payload) in enumerate(lst, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            payloads[pid] = payload

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(pid, s, payloads.get(pid, {})) for pid, s in fused]


async def embed_dense(client: AsyncOpenAI, texts: List[str]) -> List[List[float]]:
    """Dense embeddings using OpenAI embeddings endpoint."""
    resp = await client.embeddings.create(model=settings.openai_embed_model, input=texts)
    return [d.embedding for d in resp.data]


async def qdrant_dense_search(
    qdrant: QdrantClient,
    query_vec: List[float],
    flt: Filter,
    limit: int,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Dense search on named vector (sync qdrant call executed in thread)."""
    res_obj = await run_blocking(
        qdrant.query_points,
        collection_name=settings.qdrant_collection,
        query=query_vec,
        using=settings.qdrant_dense_vector_name,
        query_filter=flt,
        with_payload=True,
        limit=limit,
    )
    points = getattr(res_obj, "points", res_obj)

    out: List[Tuple[str, float, Dict[str, Any]]] = []
    for p in points:
        out.append((str(p.id), float(p.score or 0.0), p.payload or {}))
    return out


async def qdrant_sparse_search(
    qdrant: QdrantClient,
    sparse_vec,
    flt: Filter,
    limit: int,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Sparse search on named sparse vector (sync qdrant call executed in thread)."""
    res_obj = await run_blocking(
        qdrant.query_points,
        collection_name=settings.qdrant_collection,
        query=sparse_vec,
        using=settings.qdrant_sparse_vector_name,
        query_filter=flt,
        with_payload=True,
        limit=limit,
    )
    points = getattr(res_obj, "points", res_obj)

    out: List[Tuple[str, float, Dict[str, Any]]] = []
    for p in points:
        out.append((str(p.id), float(p.score or 0.0), p.payload or {}))
    return out


async def prf_snippets(
    client: AsyncOpenAI,
    qdrant: QdrantClient,
    question: str,
    flt: Filter,
    limit: int = 6,
) -> List[str]:
    """Pseudo-Relevance Feedback: retrieve top snippets to help rewrite."""
    qvec = (await embed_dense(client, [question]))[0]
    svec = sparse_encode(question)

    dense_task = qdrant_dense_search(qdrant, qvec, flt, limit)
    sparse_task = qdrant_sparse_search(qdrant, svec, flt, limit)
    dense, sparse = await asyncio.gather(dense_task, sparse_task)

    fused = rrf_fuse([dense, sparse])[:limit]

    snippets: List[str] = []
    for _pid, _s, payload in fused:
        t = (payload.get("text") or "").replace("\n", " ")[:240]
        if t:
            snippets.append(t)
    return snippets


async def hybrid_retrieve_with_rewrites(
    client: AsyncOpenAI,
    qdrant: QdrantClient,
    rewrites: List[str],
    flt: Filter,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    For each rewrite:
      - dense embed
      - sparse encode
      - dense+sparse search concurrently
      - RRF fuse per rewrite
    Then:
      - RRF fuse across rewrites
    """

    async def _one_query(q: str) -> List[Tuple[str, float, Dict[str, Any]]]:
        qvec = (await embed_dense(client, [q]))[0]
        svec = sparse_encode(q)

        dense_task = qdrant_dense_search(qdrant, qvec, flt, settings.rag_top_k)
        sparse_task = qdrant_sparse_search(
            qdrant, svec, flt, settings.rag_top_k)
        dense, sparse = await asyncio.gather(dense_task, sparse_task)

        return rrf_fuse([dense, sparse])[: settings.rag_top_k]

    per_lists = await asyncio.gather(*[_one_query(q) for q in rewrites])
    fused_all = rrf_fuse(per_lists)[: settings.rag_top_k * 2]
    return fused_all
