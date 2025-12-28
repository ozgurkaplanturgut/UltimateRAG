from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient, models
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_qdrant() -> QdrantClient:
    """Create a Qdrant client configured for gRPC."""
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_grpc_port,
        prefer_grpc=True,
        timeout=60,
    )


def ensure_collection(client: QdrantClient) -> None:
    """
    Ensure collection exists with:
    - named dense vector (size=1536, cosine)
    - named sparse vector
    """
    name = settings.qdrant_collection
    if client.collection_exists(name):
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            settings.qdrant_dense_vector_name: models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            settings.qdrant_sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000),
    )
    logger.info("Created Qdrant collection=%s", name)


def doc_meta_point_id(user_id: str, doc_id: str) -> int:
    """Deterministic point id for doc_meta record."""
    return abs(hash(f"meta:{user_id}:{doc_id}")) % (2**63)


def filter_doc(user_id: Optional[str], doc_id: str) -> Filter:
    """Build filter for doc_id (+ optional user_id)."""
    must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if user_id is not None:
        must.append(FieldCondition(key="user_id",
                    match=MatchValue(value=user_id)))
    return Filter(must=must)


def delete_doc_points(client: QdrantClient, user_id: Optional[str], doc_id: str) -> None:
    """Delete all points for a given doc (and optional user scope)."""
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=filter_doc(user_id, doc_id)),
        wait=True,
    )
