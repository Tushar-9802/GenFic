"""Synthesize one brief per chunk for the register-training corpus.

Reads ingested Gutenberg chapters, chunks them with the same tokenizer +
seq_len that build_dataset.py will use, and uses base Mistral-7B-Instruct
to produce a single-sentence brief describing each chunk's scene. The brief
will be paired with the chunk as the target response at training time.

GPU-heavy: ~2 s/chunk on RTX 5070 Ti at bf16. Plan for ~1–2 hr per cluster.
Resumable: skips chunks already in the output jsonl.

Usage
-----
  python scripts/synthesize_briefs.py --cluster victorian-formal
  python scripts/synthesize_briefs.py --cluster gothic-dark --max-new-tokens 60
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PREFIX_RE = re.compile(
    r"^(?:Brief|Sentence|Description|Summary|Scene|Answer)\s*[:\-—]\s*",
    re.IGNORECASE,
)

# Title patterns that strongly indicate non-prose content (poetry, indexes,
# essay collections, plays). Skip these at chapter-gather time.
NON_PROSE_TITLE_RE = re.compile(
    r"\b(index of|poems?|poetry|verse|ballads?|rhymes?|sonnets?|"
    r"musical comed|play in (?:one|two|three|four|five) acts?|"
    r"chamber music|black riders|symptoms of being)\b",
    re.IGNORECASE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from genfic.data.chunking import chunk_chapter  # noqa: E402
from genfic.registers import REGISTERS  # noqa: E402

BRIEF_PROMPT = (
    "In one short sentence, describe what happens in the scene below. "
    "Use plain modern English; do not imitate the prose style. "
    "No preamble, no quotation, no preface — just the sentence."
)


def gather_chapters(
    repo_root: Path,
    cluster: str | None,
    max_chapters_per_author: int,
    seed: int,
) -> list[dict]:
    p = repo_root / "source" / "gutenberg_imported.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    if cluster:
        rows = [r for r in rows if r.get("cluster") == cluster]
    # Drop clearly non-prose entries by title
    rows = [r for r in rows if not NON_PROSE_TITLE_RE.search(r.get("work_title", ""))]
    # Cap chapters per author with a deterministic shuffle
    rng = random.Random(seed)
    by_author: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_author[r["author"]].append(r)
    capped: list[dict] = []
    for author, items in by_author.items():
        if max_chapters_per_author > 0 and len(items) > max_chapters_per_author:
            rng.shuffle(items)
            items = items[:max_chapters_per_author]
        capped.extend(items)
    # Stable order for resumability: sort by (author, work_slug, chapter_num)
    capped.sort(key=lambda r: (r["author"], r["work_slug"], r["chapter_num"]))
    for r in capped:
        r["_path"] = repo_root / r["path"]
    return capped


def load_existing(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    seen = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        seen.add((r["chapter_path"], r["chunk_idx"]))
    return seen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", choices=sorted(REGISTERS),
                   help="Restrict to one cluster (default: all)")
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Chunker seq_len (must match build_dataset.py)")
    p.add_argument("--brief-context-chars", type=int, default=1500,
                   help="How many chars of the chunk to feed the synth model")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-chapters-per-author", type=int, default=50,
                   help="Cap chapters per author to bound GPU work; 0 = no cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="source/briefs.jsonl")
    args = p.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_existing(out_path)

    rows = gather_chapters(
        REPO_ROOT, args.cluster, args.max_chapters_per_author, args.seed,
    )
    if not rows:
        print(f"No chapters found{f' in cluster {args.cluster}' if args.cluster else ''}.",
              file=sys.stderr)
        return 2
    by_author: dict[str, int] = defaultdict(int)
    for r in rows:
        by_author[r["author"]] += 1
    print(f"Brief-synth chapter pool: {len(rows)} chapters across {len(by_author)} authors")
    for a in sorted(by_author):
        print(f"  {a:28s} {by_author[a]:3d} chapters")
    print(f"Loading tokenizer + base model: {args.model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    print(f"\nSynthesizing briefs for {len(rows)} chapters "
          f"(skipping {len(seen)} already done)...")
    n_done = 0
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for ri, rec in enumerate(rows, 1):
            text = rec["_path"].read_text(encoding="utf-8")
            # Chunker reserves PROMPT_TOKEN_BUDGET=64 in build_dataset.py;
            # match it here so chunk_idx alignment is preserved between scripts.
            chunks = chunk_chapter(text, tok, max_seq_len=args.seq_len - 64, eos_token_id=None)
            for ch in chunks:
                key = (rec["path"], ch.chunk_idx)
                if key in seen:
                    continue
                # Decode the full chunk and trim to brief_context_chars characters
                preview_text = tok.decode(ch.input_ids, skip_special_tokens=True)[
                    : args.brief_context_chars
                ]
                prompt = (
                    f"[INST] {BRIEF_PROMPT}\n\nScene:\n{preview_text}\n[/INST]"
                )
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tok.eos_token_id,
                    )
                full = tok.decode(out[0], skip_special_tokens=True)
                brief = full.split("[/INST]", 1)[-1].strip() if "[/INST]" in full else full
                # Single line; strip any leading "Brief:" / "Summary:" / etc. label;
                # strip surrounding quotes the model may add.
                brief = brief.split("\n", 1)[0].strip()
                brief = PREFIX_RE.sub("", brief, count=1).strip()
                brief = brief.strip("\"' ")
                if not brief:
                    continue
                fout.write(json.dumps({
                    "chapter_path": rec["path"],
                    "chunk_idx": ch.chunk_idx,
                    "brief": brief,
                    "author": rec["author"],
                    "cluster": rec["cluster"],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
            if ri % 25 == 0:
                rate = n_done / max(1.0, time.time() - t_start)
                print(f"  {ri}/{len(rows)} chapters · {n_done} new briefs · {rate:.1f} briefs/s")

    print(f"\n=== done: {n_done} new briefs -> {out_path.relative_to(REPO_ROOT)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
