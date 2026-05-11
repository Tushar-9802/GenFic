"""Sample evaluation briefs from the held-out test splits.

The adapters were trained on briefs paired with train-split chunks. Briefs
paired with test-split chunks were never shown to the model, so they're
distributionally identical to training briefs but legitimately unseen — the
textbook source for an unbiased generation eval.

Identifying test-split briefs
-----------------------------
build_dataset.py stores `chapter_id = pg_id` in the `.h5`, but pg_id is a
*work*-level identifier shared across all chapters of one book. So we cannot
cross-walk an `.h5` test row back to a unique (chapter_path, chunk_idx) brief
via the `.h5` alone.

Instead, we deterministically reproduce build_dataset.py's split assignment
in-memory. We can do this without re-tokenizing because:

  - briefs.jsonl has briefs for every chunk build_dataset.py processed
    (synthesize_briefs.py runs first; build_dataset.py drops chunks with no
    brief, but in this corpus every kept chunk has one — confirmed: 831/831
    chapters have contiguous chunk_idx sequences in briefs.jsonl).
  - chunk_chapter() emits chunks in chunk_idx 0..K order.
  - So replaying (manifest order × sorted-chunk_idx) reproduces build_dataset's
    `all_chunks` list exactly.

We then apply the same per-author cap (200, seed=42, same `random.Random` call
sequence) and the same stratified train/val/test split (90/10/10 with min=1
when n>=10, same rng). This gives us, for every brief, the same split label
the trainer actually assigned.

Outputs
-------
  source/eval/briefs_register.jsonl     N=2 briefs per register × 5 = 10
                                        Used for the register-transfer sweep.

  source/eval/briefs_instruction.jsonl  8 briefs, each wrapped with one
                                        auto-checkable constraint. Used for
                                        the instruction-following probe.

Both files are deterministic given --seed.

Usage
-----
  python scripts/build_eval_briefs.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from genfic.registers import REGISTERS  # noqa: E402

# Eight auto-checkable instruction-following constraints. The constraint_id is
# the dispatch key for the per-constraint checker in eval_adapters.py.
INSTRUCTION_CONSTRAINTS = [
    ("Use first-person past-tense narration.", "first_person"),
    ("Include a character named Eleanor.", "named_eleanor"),
    ("Include exactly one weather observation.", "weather"),
    ("Use no spoken dialogue at all.", "no_dialogue"),
    ("End the scene with a question mark.", "ends_question"),
    ("Limit the scene to under 200 words.", "under_200_words"),
    ("Set the scene at dawn.", "set_at_dawn"),
    ("Mention an object the protagonist is holding or carrying.", "object_held"),
]

# Catalog/leakage filters — briefs synthesized from Gutenberg metadata pages,
# title-listing prefaces, or other non-prose surface that slipped through
# chapter detection. These don't describe scenes and produce nonsense
# continuations regardless of the adapter.
_RE_CATALOG_LEAKAGE = re.compile(
    r"(?:"
    r"\bclassic novels\b|"
    r"\bThese classic\b|"
    r"\bedited by\b|"
    r"\b(?:1[89]\d{2}) edition\b|"
    r'(?:"[A-Z][A-Za-z ]+",?\s+(?:by|including)\s+){2,}'  # 2+ title-by-author chains
    r")",
    re.IGNORECASE,
)

# Train / val / test build_dataset constants. Must match the defaults in
# scripts/build_dataset.py for the replay to be exact.
DEFAULT_CAP = 200
DEFAULT_VAL_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.10
DEFAULT_BUILD_SEED = 42


def _looks_like_catalog_leakage(brief: str) -> bool:
    return _RE_CATALOG_LEAKAGE.search(brief) is not None


def _reproduce_split_assignment(
    repo_root: Path,
    cluster: str,
    cap: int = DEFAULT_CAP,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_BUILD_SEED,
) -> dict[tuple[str, int], int]:
    """Replay build_dataset.py's per-cluster split assignment exactly.

    Returns: {(chapter_path, chunk_idx): split} where split = 0/1/2 for
    train/val/test.
    """
    # 1. Walk manifest in original order, filtered to this cluster.
    chapters = []
    with open(repo_root / "source" / "gutenberg_imported.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("cluster") == cluster:
                chapters.append(r)

    # 2. Build per-chapter brief index sorted by chunk_idx.
    brief_idxs_by_chap: dict[str, list[int]] = defaultdict(list)
    with open(repo_root / "source" / "briefs.jsonl", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b.get("cluster") == cluster:
                brief_idxs_by_chap[b["chapter_path"]].append(int(b["chunk_idx"]))
    for k in brief_idxs_by_chap:
        brief_idxs_by_chap[k].sort()

    # 3. Reproduce build_dataset.py's `all_chunks` order: manifest order ×
    #    chunk_idx order within chapter, skipping chapters with no briefs.
    all_chunks: list[dict] = []
    for chap in chapters:
        path = chap["path"]
        if path not in brief_idxs_by_chap:
            continue
        for chunk_idx in brief_idxs_by_chap[path]:
            all_chunks.append({
                "path": path,
                "chunk_idx": chunk_idx,
                "author": chap["author"],
            })

    # 4. Per-author cap (matches `cap_per_author` in build_dataset.py:101).
    rng = random.Random(seed)
    by_author: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(all_chunks):
        by_author[c["author"]].append(i)
    keep_idx: set[int] = set()
    for author, idxs in by_author.items():
        if len(idxs) <= cap:
            keep_idx.update(idxs)
        else:
            keep_idx.update(rng.sample(idxs, cap))
    all_chunks = [all_chunks[i] for i in sorted(keep_idx)]

    # 5. Stratified split per author (matches `stratified_split` in
    #    build_dataset.py:77, same rng instance).
    chunks_by_author_post: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(all_chunks):
        chunks_by_author_post[c["author"]].append(i)

    split_per_post_idx: dict[int, int] = {}
    for author, idxs in chunks_by_author_post.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(round(n * test_frac))) if n >= 10 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 10 else 0
        n_train = n - n_test - n_val
        for i, ci in enumerate(shuffled):
            if i < n_train:
                split_per_post_idx[ci] = 0
            elif i < n_train + n_val:
                split_per_post_idx[ci] = 1
            else:
                split_per_post_idx[ci] = 2

    # 6. Map back to (path, chunk_idx) -> split.
    out: dict[tuple[str, int], int] = {}
    for post_idx, c in enumerate(all_chunks):
        out[(c["path"], c["chunk_idx"])] = split_per_post_idx[post_idx]
    return out


def _collect_test_briefs(repo_root: Path, cluster: str) -> list[dict]:
    """Return briefs.jsonl entries for the cluster's test-split chunks."""
    split_map = _reproduce_split_assignment(repo_root, cluster)
    out: list[dict] = []
    with open(repo_root / "source" / "briefs.jsonl", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b.get("cluster") != cluster:
                continue
            key = (b["chapter_path"], int(b["chunk_idx"]))
            if split_map.get(key) == 2:  # test
                out.append({
                    "register": cluster,
                    "chapter_path": b["chapter_path"],
                    "chunk_idx": int(b["chunk_idx"]),
                    "brief": b["brief"].strip(),
                    "author": b.get("author", ""),
                })
    return out


def _verify_against_h5(repo_root: Path, cluster: str, split_map: dict[tuple[str, int], int]) -> dict:
    """Spot-check: count by split should approximately match the .h5's split
    distribution for this cluster. Approximate because .h5 stores pg_id (work-
    level) not chapter-level, but the totals must match exactly."""
    import h5py
    h5_path = repo_root / "source" / f"{cluster}.h5"
    with h5py.File(h5_path, "r") as f:
        h5_split = f["split"][:]
    h5_counts = {0: int((h5_split == 0).sum()),
                 1: int((h5_split == 1).sum()),
                 2: int((h5_split == 2).sum())}
    replay_counts = {0: 0, 1: 0, 2: 0}
    for s in split_map.values():
        replay_counts[s] += 1
    return {"h5": h5_counts, "replay": replay_counts}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-register", type=int, default=2,
                   help="Register-transfer briefs to sample per register")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="source/eval")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Reproduce split assignments + verify they match the trained .h5 ---
    print("Replaying build_dataset.py split assignment per register...")
    test_pools_raw: dict[str, list[dict]] = {}
    for cname in sorted(REGISTERS):
        split_map = _reproduce_split_assignment(REPO_ROOT, cname)
        verification = _verify_against_h5(REPO_ROOT, cname, split_map)
        h5_ok = verification["h5"] == verification["replay"]
        print(f"  {cname:<22}  replay vs h5 split counts: "
              f"{verification['replay']} vs {verification['h5']}  "
              f"{'OK exact match' if h5_ok else 'MISMATCH'}")
        if not h5_ok:
            print(f"    ABORT: replay diverges from h5 — would mean train-set leakage. "
                  f"Investigate cap_per_author / stratified_split parity.",
                  file=sys.stderr)
            return 2
        # Collect raw test briefs for this cluster.
        pool = _collect_test_briefs(REPO_ROOT, cname)
        test_pools_raw[cname] = pool

    # --- Filter (length + catalog leakage) and report attrition ---
    print("\nFiltering briefs (length 10-80 words, no catalog leakage)...")
    test_pools: dict[str, list[dict]] = {}
    for cname, raw in test_pools_raw.items():
        filtered = []
        leakage_dropped = 0
        length_dropped = 0
        for b in raw:
            wc = len(b["brief"].split())
            if not (10 <= wc <= 80):
                length_dropped += 1
                continue
            if _looks_like_catalog_leakage(b["brief"]):
                leakage_dropped += 1
                continue
            filtered.append(b)
        test_pools[cname] = filtered
        print(f"  {cname:<22}  raw={len(raw)}  kept={len(filtered)}  "
              f"(dropped {length_dropped} length, {leakage_dropped} leakage)")

    # --- Sample register-transfer briefs ---
    rng = random.Random(args.seed)
    register_briefs: list[dict] = []
    for cname in sorted(REGISTERS):
        pool = test_pools[cname]
        if len(pool) < args.per_register:
            print(f"WARN: {cname} has only {len(pool)} candidates, using all", file=sys.stderr)
            picked = pool
        else:
            picked = rng.sample(pool, args.per_register)
        for b in picked:
            register_briefs.append({
                "brief_id": f"{cname}-{b['chunk_idx']:03d}",
                "source_register": cname,
                "brief": b["brief"],
                "source_chapter": b["chapter_path"],
                "source_chunk_idx": b["chunk_idx"],
            })

    out_reg = out_dir / "briefs_register.jsonl"
    with open(out_reg, "w", encoding="utf-8") as f:
        for b in register_briefs:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(register_briefs)} register-transfer briefs -> "
          f"{out_reg.relative_to(REPO_ROOT)}")

    # --- Sample instruction-probe briefs (one per constraint, drawn across registers) ---
    all_pool = []
    for pool in test_pools.values():
        all_pool.extend(pool)
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(all_pool)
    instr_briefs: list[dict] = []
    for i, (constraint_text, constraint_id) in enumerate(INSTRUCTION_CONSTRAINTS):
        if i >= len(all_pool):
            break
        b = all_pool[i]
        wrapped = f"{b['brief']} {constraint_text}"
        instr_briefs.append({
            "brief_id": f"instr-{i:02d}-{constraint_id}",
            "source_register": b["register"],
            "constraint_id": constraint_id,
            "constraint_text": constraint_text,
            "brief": wrapped,
            "source_chapter": b["chapter_path"],
            "source_chunk_idx": b["chunk_idx"],
        })

    out_instr = out_dir / "briefs_instruction.jsonl"
    with open(out_instr, "w", encoding="utf-8") as f:
        for b in instr_briefs:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"Wrote {len(instr_briefs)} instruction-probe briefs -> "
          f"{out_instr.relative_to(REPO_ROOT)}")

    # --- Preview ---
    print("\n=== Register-transfer briefs ===")
    for b in register_briefs:
        print(f"\n[{b['brief_id']}] (source={b['source_register']})")
        print(f"  {b['brief']}")
    print("\n=== Instruction-following briefs ===")
    for b in instr_briefs:
        print(f"\n[{b['brief_id']}] (constraint={b['constraint_id']}, source={b['source_register']})")
        print(f"  {b['brief']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
