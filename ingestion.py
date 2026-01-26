"""Ingestion pipeline for semantic chunking."""

from __future__ import annotations

from typing import List, Optional

from semantic_chunker import Chunk, DEFAULT_TOKENIZER_MODEL, chunk_text, load_tokenizer


def ingest_text(
    text: str,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    min_tokens: int = 40,
    max_tokens: int = 150,
    tokenizer: Optional[object] = None,
) -> List[Chunk]:
    """Return semantic chunks with metadata."""
    tokenizer_obj = tokenizer or load_tokenizer(tokenizer_model)
    return chunk_text(
        text,
        tokenizer_obj,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
