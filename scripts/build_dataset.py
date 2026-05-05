"""Build the per-register HDF5 training dataset.

Pipeline
--------
1. Read `source/gutenberg_imported.jsonl` (filtered to one cluster).
2. Read `source/briefs.jsonl` (per-chunk briefs from synthesize_briefs.py).
3. For each chapter: chunk with the Mistral tokenizer at the configured seq_len.
4. For each chunk: build the instruction-format sequence
       `[INST] Continue this scene in {register} style: {brief} [/INST] {chunk}</s>`,
   pad to seq_len, record the prompt prefix length.
5. Apply the per-author chunk cap (`--max-chunks-per-author`) to prevent any
   single author from dominating.
6. Stratified train/val/test split per author.
7. Pack into `source/{register}.h5` with:
     - `input_ids`        int32 [N, L]
     - `attention_mask`   int8  [N, L]
     - `prompt_lengths`   int16 [N]
     - `n_tokens`         int16 [N]    real (pre-pad) length of full sequence
     - `author_id`        int8  [N]
     - `chapter_id`       int32 [N]
     - `chunk_idx`        int16 [N]
     - `is_chapter_end`   int8  [N]
     - `split`            int8  [N]
     - attrs: `register`, `tokenizer`, `seq_len`, `pad_token_id`, `eos_token_id`

Usage
-----
  python scripts/build_dataset.py --register victorian-formal
  python scripts/build_dataset.py --register gothic-dark --max-chunks-per-author 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from genfic.data.chunking import chunk_chapter  # noqa: E402
from genfic.registers import REGISTERS  # noqa: E402


def gather_chapters(repo_root: Path, cluster: str) -> list[dict]:
    p = repo_root / "source" / "gutenberg_imported.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    rows = [r for r in rows if r.get("cluster") == cluster]
    for r in rows:
        r["_path"] = repo_root / r["path"]
    return rows


def load_briefs(repo_root: Path, cluster: str) -> dict[tuple[str, int], str]:
    p = repo_root / "source" / "briefs.jsonl"
    if not p.exists():
        return {}
    out: dict[tuple[str, int], str] = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("cluster") != cluster:
            continue
        out[(r["chapter_path"], r["chunk_idx"])] = r["brief"]
    return out


def stratified_split(
    chunks_by_author: dict[str, list[int]],
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> dict[int, int]:
    assignment: dict[int, int] = {}
    for author, idxs in chunks_by_author.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(round(n * test_frac))) if n >= 10 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 10 else 0
        n_train = n - n_test - n_val
        for i, ci in enumerate(shuffled):
            if i < n_train:
                assignment[ci] = 0
            elif i < n_train + n_val:
                assignment[ci] = 1
            else:
                assignment[ci] = 2
    return assignment


def cap_per_author(
    all_chunks: list[dict], cap: int, rng: random.Random,
) -> list[dict]:
    """Keep at most `cap` chunks per author, sampled uniformly."""
    by_author: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(all_chunks):
        by_author[c["author"]].append(i)
    keep_idx: set[int] = set()
    for author, idxs in by_author.items():
        if len(idxs) <= cap:
            keep_idx.update(idxs)
        else:
            keep_idx.update(rng.sample(idxs, cap))
    return [all_chunks[i] for i in sorted(keep_idx)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", required=True, choices=sorted(REGISTERS))
    p.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-chunks-per-author", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Override output path (default: source/{register}.h5)")
    args = p.parse_args()

    register = REGISTERS[args.register]
    out_path = REPO_ROOT / (args.out or f"source/{args.register}.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = gather_chapters(REPO_ROOT, args.register)
    if not rows:
        print(f"No chapters found for cluster {args.register}. "
              f"Run scripts/ingest_gutenberg.py first.", file=sys.stderr)
        return 2
    briefs = load_briefs(REPO_ROOT, args.register)
    if not briefs:
        print(f"No briefs found for cluster {args.register}. "
              f"Run scripts/synthesize_briefs.py first.", file=sys.stderr)
        return 2

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tok.eos_token_id  # Mistral has no pad — convention is reuse EOS
    eos_id = tok.eos_token_id

    print(f"\nChunking + formatting {len(rows)} chapters @ seq_len={args.seq_len}...")
    L = args.seq_len
    # Reserve room for the [INST] ... [/INST] prompt + closing EOS so a chunk
    # plus its prompt fits in L without truncation. Must match the value used
    # by synthesize_briefs.py so chunk_idx alignment is preserved.
    PROMPT_TOKEN_BUDGET = 64
    chunk_max = L - PROMPT_TOKEN_BUDGET
    all_chunks: list[dict] = []
    skipped_no_brief = 0
    t0 = time.time()
    for ri, rec in enumerate(rows, 1):
        text = rec["_path"].read_text(encoding="utf-8")
        chunks = chunk_chapter(text, tok, max_seq_len=chunk_max, eos_token_id=None)
        for ch in chunks:
            brief = briefs.get((rec["path"], ch.chunk_idx))
            if not brief:
                skipped_no_brief += 1
                continue

            prompt_text = (
                f"[INST] Continue this scene in {register.display} style: "
                f"{brief} [/INST] "
            )
            prompt_ids = tok.encode(prompt_text, add_special_tokens=True)
            chunk_ids = list(ch.input_ids)
            # Always end with EOS so the model learns to stop after the response
            full_ids = prompt_ids + chunk_ids
            if not full_ids or full_ids[-1] != eos_id:
                full_ids.append(eos_id)
            if len(full_ids) > L:
                full_ids = full_ids[: L - 1] + [eos_id]
            all_chunks.append({
                "input_ids": full_ids,
                "prompt_len": min(len(prompt_ids), L),
                "n_tokens": len(full_ids),
                "author": rec["author"],
                "chapter_id": rec.get("pg_id", 0),
                "chunk_idx": ch.chunk_idx,
                "is_chapter_end": ch.is_chapter_end,
            })
        if ri % 250 == 0:
            print(f"  {ri}/{len(rows)} chapters -> {len(all_chunks)} chunks "
                  f"({time.time()-t0:.0f}s)")
    print(f"  Done: {len(all_chunks)} chunks; skipped {skipped_no_brief} for missing brief")

    rng = random.Random(args.seed)
    if args.max_chunks_per_author > 0:
        before = len(all_chunks)
        all_chunks = cap_per_author(all_chunks, args.max_chunks_per_author, rng)
        print(f"  Per-author cap @ {args.max_chunks_per_author}: {before} -> {len(all_chunks)} chunks")

    chunks_by_author: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(all_chunks):
        chunks_by_author[c["author"]].append(i)
    split_assignment = stratified_split(
        chunks_by_author, args.val_frac, args.test_frac, rng,
    )

    authors_sorted = sorted(chunks_by_author.keys(),
                            key=lambda a: -len(chunks_by_author[a]))
    author_id = {a: i for i, a in enumerate(authors_sorted)}

    N = len(all_chunks)
    print(f"\nWriting HDF5: {out_path} (N={N}, L={L}, est size ~{N*L*4/1024/1024:.0f} MB)")
    with h5py.File(out_path, "w") as f:
        f.attrs["register"] = args.register
        f.attrs["tokenizer"] = args.tokenizer
        f.attrs["seq_len"] = L
        f.attrs["pad_token_id"] = pad_id
        f.attrs["eos_token_id"] = eos_id
        f.attrs["seed"] = args.seed
        f.attrs["val_frac"] = args.val_frac
        f.attrs["test_frac"] = args.test_frac
        f.attrs["max_chunks_per_author"] = args.max_chunks_per_author
        f.create_dataset(
            "authors",
            data=np.array(authors_sorted, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        ds_input = f.create_dataset("input_ids", shape=(N, L), dtype=np.int32,
                                    compression="gzip", compression_opts=4)
        ds_mask = f.create_dataset("attention_mask", shape=(N, L), dtype=np.int8,
                                   compression="gzip", compression_opts=4)
        ds_plen = f.create_dataset("prompt_lengths", shape=(N,), dtype=np.int16)
        ds_ntok = f.create_dataset("n_tokens", shape=(N,), dtype=np.int16)
        ds_aid = f.create_dataset("author_id", shape=(N,), dtype=np.int8)
        ds_cid = f.create_dataset("chapter_id", shape=(N,), dtype=np.int32)
        ds_cidx = f.create_dataset("chunk_idx", shape=(N,), dtype=np.int16)
        ds_end = f.create_dataset("is_chapter_end", shape=(N,), dtype=np.int8)
        ds_split = f.create_dataset("split", shape=(N,), dtype=np.int8)

        BATCH = 1000
        for b_start in range(0, N, BATCH):
            b_end = min(b_start + BATCH, N)
            batch = all_chunks[b_start:b_end]
            input_buf = np.full((len(batch), L), pad_id, dtype=np.int32)
            mask_buf = np.zeros((len(batch), L), dtype=np.int8)
            for j, c in enumerate(batch):
                ids = c["input_ids"][:L]
                input_buf[j, : len(ids)] = ids
                mask_buf[j, : len(ids)] = 1
            ds_input[b_start:b_end] = input_buf
            ds_mask[b_start:b_end] = mask_buf
            ds_plen[b_start:b_end] = np.array([c["prompt_len"] for c in batch], dtype=np.int16)
            ds_ntok[b_start:b_end] = np.array([min(c["n_tokens"], L) for c in batch], dtype=np.int16)
            ds_aid[b_start:b_end] = np.array([author_id[c["author"]] for c in batch], dtype=np.int8)
            ds_cid[b_start:b_end] = np.array([c["chapter_id"] for c in batch], dtype=np.int32)
            ds_cidx[b_start:b_end] = np.array([c["chunk_idx"] for c in batch], dtype=np.int16)
            ds_end[b_start:b_end] = np.array([1 if c["is_chapter_end"] else 0 for c in batch], dtype=np.int8)
            ds_split[b_start:b_end] = np.array(
                [split_assignment[i] for i in range(b_start, b_end)], dtype=np.int8,
            )
            if b_start % 5000 == 0:
                print(f"  written {b_end}/{N}")

    print("\n=== per-author / per-split chunk tally ===")
    print(f"{'Author':28s} {'train':>6s} {'val':>5s} {'test':>5s} {'total':>6s} {'tokens':>11s}")
    print("-" * 70)
    grand = Counter()
    grand_tokens = 0
    for a in authors_sorted:
        idxs = chunks_by_author[a]
        sp = Counter(split_assignment[i] for i in idxs)
        n_tok = sum(all_chunks[i]["n_tokens"] for i in idxs)
        grand_tokens += n_tok
        for k in (0, 1, 2):
            grand[k] += sp[k]
        print(f"{a:28s} {sp[0]:6d} {sp[1]:5d} {sp[2]:5d} {len(idxs):6d} {n_tok:11,d}")
    print("-" * 70)
    print(f"{'TOTAL':28s} {grand[0]:6d} {grand[1]:5d} {grand[2]:5d} {N:6d} {grand_tokens:11,d}")
    print(f"\nDataset: {out_path.relative_to(REPO_ROOT)}  "
          f"({out_path.stat().st_size/1024/1024:.0f} MB on disk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
