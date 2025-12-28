from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from qdrant_client.models import SparseVector
from transformers import AutoTokenizer

from app.core.config import get_settings

settings = get_settings()
_tokenizer: Optional[AutoTokenizer] = None


def get_tokenizer() -> AutoTokenizer:
    """Lazy-load tokenizer for sparse encoding."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.rag_sparse_tokenizer_model)
    return _tokenizer


def sparse_encode(text: str) -> SparseVector:
    """
    Create a sparse vector from token ids using:
    - tf -> log1p(tf)
    - L2 normalization

    Behavior preserved from original implementation.
    """
    tok = get_tokenizer()
    ids = tok.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=settings.rag_sparse_max_tokens,
    )

    if not ids:
        return SparseVector(indices=[], values=[])

    tf = Counter(ids)
    indices = list(tf.keys())
    values = [math.log1p(tf[i]) for i in indices]

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    values = [v / norm for v in values]

    return SparseVector(indices=indices, values=values)
