"""Chapter -> training-sample chunking for LoRA fine-tuning.

Strategy
--------
- Each chapter is read as plain UTF-8 text.
- Paragraphs are demarcated by blank-line breaks (`\\n\\n+`).
- Chunks are built greedily: pack paragraphs in order until adding the next one
  would exceed `max_seq_len`. The chunk is emitted, a new one starts.
- A paragraph that on its own exceeds `max_seq_len` is split on sentence
  boundaries (regex sentence-end approximation) into pieces that fit.
- The final chunk of a chapter has an EOS token appended (signals "story end"
  to the model). Mid-chapter chunks do NOT get EOS — they're continuations.
- Short final chunks are kept (training on partial-length samples is fine —
  the trainer pads to seq_len).

Returns chunk records with the actual token IDs already produced; no double
tokenization downstream. Pure CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Paragraph splitter: one or more blank lines (with optional whitespace).
_RE_PARA = re.compile(r"\n\s*\n+")
# Crude sentence boundary: period/question/exclam followed by whitespace + capital.
# Handles most narrative prose; misses some edge cases (Mr. Smith etc.) but
# fallback splitting is rare so the imprecision doesn't matter much.
_RE_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")


@dataclass(frozen=True)
class Chunk:
    input_ids: list[int]
    n_tokens: int
    is_chapter_end: bool
    chunk_idx: int  # 0-based within the chapter


def _split_long_paragraph(
    text: str, tokenizer, max_seq_len: int
) -> list[list[int]]:
    """Fallback for paragraphs that exceed max_seq_len on their own. Splits on
    sentence boundaries; if a single sentence is still too long, hard-truncates.
    """
    out: list[list[int]] = []
    sentences = _RE_SENTENCE.split(text)
    cur_text_parts: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if not sent_ids:
            continue
        if len(sent_ids) > max_seq_len:
            # Single sentence too long — flush whatever we have, then hard-split
            if cur_text_parts:
                out.append(
                    tokenizer.encode(" ".join(cur_text_parts), add_special_tokens=False)
                )
                cur_text_parts, cur_tokens = [], 0
            # Hard chunk the over-long sentence
            for i in range(0, len(sent_ids), max_seq_len):
                out.append(sent_ids[i : i + max_seq_len])
            continue
        if cur_tokens + len(sent_ids) > max_seq_len:
            out.append(
                tokenizer.encode(" ".join(cur_text_parts), add_special_tokens=False)
            )
            cur_text_parts, cur_tokens = [], 0
        cur_text_parts.append(sent)
        cur_tokens += len(sent_ids)
    if cur_text_parts:
        out.append(
            tokenizer.encode(" ".join(cur_text_parts), add_special_tokens=False)
        )
    return out


def chunk_chapter(
    text: str,
    tokenizer,
    max_seq_len: int = 2048,
    eos_token_id: int | None = None,
) -> list[Chunk]:
    """Chunk a chapter's text into training-ready token-id lists."""
    paragraphs = [p.strip() for p in _RE_PARA.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[list[int]] = []
    cur: list[int] = []
    para_sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    for para in paragraphs:
        para_ids = tokenizer.encode(para, add_special_tokens=False)
        if len(para_ids) > max_seq_len:
            # Flush current chunk first
            if cur:
                chunks.append(cur)
                cur = []
            # Split the long paragraph and emit each piece
            for piece in _split_long_paragraph(para, tokenizer, max_seq_len):
                chunks.append(piece)
            continue

        addition = (para_sep_ids if cur else []) + para_ids
        if len(cur) + len(addition) > max_seq_len:
            chunks.append(cur)
            cur = list(para_ids)
        else:
            cur.extend(addition)

    if cur:
        chunks.append(cur)

    # Append EOS to the final chunk only — signals end-of-document
    if eos_token_id is not None and chunks:
        if len(chunks[-1]) < max_seq_len:
            chunks[-1] = chunks[-1] + [eos_token_id]
        else:
            chunks[-1] = chunks[-1][:-1] + [eos_token_id]

    return [
        Chunk(
            input_ids=ids,
            n_tokens=len(ids),
            is_chapter_end=(i == len(chunks) - 1),
            chunk_idx=i,
        )
        for i, ids in enumerate(chunks)
    ]
