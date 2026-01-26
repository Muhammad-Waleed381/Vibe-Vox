"""Semantic chunking utilities for VibeVox-Core.

Uses NLTK sentence segmentation and a model-accurate tokenizer to
assemble chunks within a token budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import re

import nltk
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Chunk:
    chunk_id: int
    start_char: int
    end_char: int
    token_count: int
    sent_ids: List[int]
    text: str


def ensure_punkt() -> None:
    """Ensure NLTK punkt data is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_tokenizer(model_id: str = DEFAULT_TOKENIZER_MODEL) -> PreTrainedTokenizerBase:
    """Load tokenizer for accurate token counts."""
    return AutoTokenizer.from_pretrained(model_id)


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Return token count for the given text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_paragraphs(text: str) -> List[tuple[int, int, str]]:
    """Split text into paragraphs and preserve character offsets."""
    paragraphs: List[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n", text):
        end = match.start()
        if start < end:
            paragraphs.append((start, end, text[start:end]))
        start = match.end()
    if start < len(text):
        paragraphs.append((start, len(text), text[start:]))
    return paragraphs


def split_sentences(text: str) -> List[tuple[int, int, str]]:
    """Split text into sentences with offsets using NLTK."""
    ensure_punkt()
    sentences = nltk.sent_tokenize(text)
    results: List[tuple[int, int, str]] = []
    search_start = 0
    for sent in sentences:
        idx = text.find(sent, search_start)
        if idx == -1:
            continue
        start = idx
        end = idx + len(sent)
        results.append((start, end, sent))
        search_start = end
    return results


def split_long_sentence(sentence: str) -> List[str]:
    """Split very long sentences on soft punctuation."""
    parts = re.split(r"([,;:—-])", sentence)
    if len(parts) == 1:
        return [sentence]
    merged: List[str] = []
    buf = ""
    for part in parts:
        if part == "":
            continue
        buf += part
        if part in {",", ";", ":", "—", "-"}:
            merged.append(buf.strip())
            buf = ""
    if buf.strip():
        merged.append(buf.strip())
    return [p for p in merged if p]


def _iter_sentences_with_offsets(text: str) -> Iterable[tuple[int, int, str, int]]:
    sent_id = 0
    for para_start, _, para_text in split_paragraphs(text):
        for start, end, sent in split_sentences(para_text):
            yield para_start + start, para_start + end, sent, sent_id
            sent_id += 1


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    min_tokens: int = 40,
    max_tokens: int = 150,
) -> List[Chunk]:
    """Chunk text into semantic units using a token budget."""
    chunks: List[Chunk] = []
    buffer_text = ""
    buffer_start = 0
    buffer_sent_ids: List[int] = []

    def flush_buffer(end_char: int) -> None:
        nonlocal buffer_text, buffer_start, buffer_sent_ids
        if not buffer_text.strip():
            buffer_text = ""
            buffer_sent_ids = []
            return
        token_count = count_tokens(buffer_text, tokenizer)
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                start_char=buffer_start,
                end_char=end_char,
                token_count=token_count,
                sent_ids=list(buffer_sent_ids),
                text=buffer_text.strip(),
            )
        )
        buffer_text = ""
        buffer_sent_ids = []

    for start, end, sent, sent_id in _iter_sentences_with_offsets(text):
        if not buffer_text:
            buffer_start = start
        candidate = (buffer_text + " " + sent).strip() if buffer_text else sent
        candidate_tokens = count_tokens(candidate, tokenizer)

        if candidate_tokens <= max_tokens:
            buffer_text = candidate
            buffer_sent_ids.append(sent_id)
            continue

        if buffer_text:
            flush_buffer(end_char=start)

        if count_tokens(sent, tokenizer) <= max_tokens:
            buffer_text = sent
            buffer_sent_ids = [sent_id]
            buffer_start = start
            continue

        parts = split_long_sentence(sent)
        for part in parts:
            part_tokens = count_tokens(part, tokenizer)
            if part_tokens > max_tokens:
                for i in range(0, len(part), 200):
                    sub = part[i : i + 200]
                    if count_tokens(sub, tokenizer) > max_tokens:
                        continue
                    buffer_text = sub
                    buffer_sent_ids = [sent_id]
                    flush_buffer(end_char=start + len(sub))
                continue
            if not buffer_text:
                buffer_start = start
            candidate = (buffer_text + " " + part).strip() if buffer_text else part
            if count_tokens(candidate, tokenizer) <= max_tokens:
                buffer_text = candidate
                buffer_sent_ids.append(sent_id)
            else:
                flush_buffer(end_char=start)
                buffer_text = part
                buffer_sent_ids = [sent_id]

    flush_buffer(end_char=len(text))

    if chunks and chunks[-1].token_count < min_tokens and len(chunks) > 1:
        last = chunks.pop()
        prev = chunks.pop()
        merged_text = prev.text + " " + last.text
        merged = Chunk(
            chunk_id=prev.chunk_id,
            start_char=prev.start_char,
            end_char=last.end_char,
            token_count=count_tokens(merged_text, tokenizer),
            sent_ids=prev.sent_ids + last.sent_ids,
            text=merged_text,
        )
        chunks.append(merged)
        for i in range(merged.chunk_id + 1, len(chunks)):
            chunks[i].chunk_id = i

    return chunks


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    sample = (
        "The hallway was quiet, too quiet. He paused, listening for any sound. "
        "A faint whisper drifted past, and his heart raced.\n\n"
        "Then, the door creaked open — slowly, deliberately — as if the house itself "
        "were breathing. He stepped forward, unsure if he wanted the answer."
    )
    chunks = chunk_text(sample, tokenizer, max_tokens=150)
    for chunk in chunks:
        print(chunk.chunk_id, chunk.token_count, repr(chunk.text))
