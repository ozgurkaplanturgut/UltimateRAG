from __future__ import annotations

from typing import Any, Dict, List, Tuple

import tiktoken
from sentence_transformers import CrossEncoder

from app.core.config import get_settings

settings = get_settings()
enc = tiktoken.get_encoding("cl100k_base")


def _windows(text: str, window_tokens: int, stride_tokens: int) -> List[str]:
    """Split text into overlapping token windows."""
    toks = enc.encode(text)
    out: List[str] = []
    i = 0
    while i < len(toks):
        j = min(i + window_tokens, len(toks))
        out.append(enc.decode(toks[i:j]))
        if j == len(toks):
            break
        i = max(0, j - stride_tokens)
    return out


def rerank_windows(
    cross_encoder: CrossEncoder,
    candidates: List[Tuple[str, float, Dict[str, Any]]],
    rewrites: List[str],
) -> List[Tuple[str, float, str, Dict[str, Any]]]:
    """
    Window-level rerank with CrossEncoder.

    Input candidates: (point_id, rrf_score, payload)
    Output: (point_id, best_score, best_window_text, payload)
    """
    out: List[Tuple[str, float, str, Dict[str, Any]]] = []

    for pid, _rrf_score, payload in candidates:
        chunk_text = payload.get("text") or ""
        wins = _windows(chunk_text, settings.rag_rerank_window_tokens,
                        settings.rag_rerank_stride_tokens)
        if not wins:
            continue

        best = -1e9
        best_w = wins[0]

        # rewrite-aware
        for w in wins:
            pairs = [(q, w) for q in rewrites]
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
            s = float(max(scores)) if len(scores) else -1e9
            if s > best:
                best = s
                best_w = w

        out.append((pid, best, best_w, payload))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[: settings.rag_rerank_top_k]
