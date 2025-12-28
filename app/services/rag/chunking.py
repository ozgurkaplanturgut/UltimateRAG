from __future__ import annotations

from typing import List, Tuple

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def tokenize_len(text: str) -> int:
    """Return token length for cl100k_base."""
    return len(enc.encode(text))


def hard_token_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> List[Tuple[str, int]]:
    """
    Hard split by tokens with overlap.
    Returns (chunk_text, start_token_offset).
    """
    toks = enc.encode(text)
    out: List[Tuple[str, int]] = []
    i = 0
    while i < len(toks):
        j = min(i + chunk_tokens, len(toks))
        chunk = enc.decode(toks[i:j])
        out.append((chunk, i))
        if j == len(toks):
            break
        i = max(0, j - overlap_tokens)
    return out


def paragraph_aware_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> List[Tuple[str, int]]:
    """
    Paragraph-aware chunking:
    - combine paragraphs until chunk_tokens
    - if paragraph exceeds chunk_tokens -> hard split
    - add overlap via tail tokens
    Returns (chunk_text, start_token_offset).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Tuple[str, int]] = []

    buf: List[str] = []
    buf_tok = 0
    global_tok_offset = 0

    for p in paragraphs:
        p_tok = tokenize_len(p)

        if p_tok > chunk_tokens:
            if buf:
                joined = "\n\n".join(buf)
                # Preserve original behavior: compute 'start' but not used in hard split output.
                _start = global_tok_offset - tokenize_len(joined)
                _ = _start
                chunks.extend(hard_token_chunks(
                    joined, chunk_tokens, overlap_tokens))
                buf, buf_tok = [], 0

            chunks.extend(hard_token_chunks(p, chunk_tokens, overlap_tokens))
            global_tok_offset += p_tok
            continue

        if buf_tok + p_tok + 2 <= chunk_tokens:
            buf.append(p)
            buf_tok += p_tok + 2
        else:
            joined = "\n\n".join(buf)
            start = global_tok_offset - tokenize_len(joined)
            chunks.append((joined, start))

            if overlap_tokens > 0:
                tail = enc.decode(enc.encode(joined)[-overlap_tokens:])
                buf = [tail, p]
                buf_tok = tokenize_len(tail) + p_tok + 2
            else:
                buf = [p]
                buf_tok = p_tok + 2

        global_tok_offset += p_tok

    if buf:
        joined = "\n\n".join(buf)
        start = max(0, global_tok_offset - tokenize_len(joined))
        chunks.append((joined, start))

    return chunks
