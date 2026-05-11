"""Build the per-cluster Burrows' Delta MFW-150 index from the PD corpus.

Burrows' Delta (Burrows 2002) is the canonical stylometric attribution metric.
Procedure:

  1. Choose the N most frequent words across the entire corpus (MFW-N; N=150
     here, the standard choice).
  2. Compute relative frequencies of those N words per text.
  3. Z-normalize each frequency dimension across the corpus (per-word mean and
     std), giving each text a 150-dim z-score vector.
  4. The "delta" between two texts is the mean absolute z-score difference.
     Lower = more similar style.

We persist three things to `source/burrows_centroids.json`:

  - `mfw_words`           the 150 chosen words, in fixed order
  - `corpus_mu` / `_sigma` per-word mean and std across the corpus (for
                          z-normalization at eval time)
  - `cluster_centroids`   per-cluster mean of z-score vectors over its chapters

At eval time, an adapter generation is tokenized, MFW counts are computed,
z-normalized using the saved mu/sigma, and the delta to each cluster centroid
is reported. The smallest delta = stylistically nearest cluster.

Usage
-----
  python scripts/build_burrows_index.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from genfic.registers import REGISTERS  # noqa: E402

_RE_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_RE_INTRA_NEWLINE = re.compile(r"\n+")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Collapses newlines first so the wrapped-line
    PG storage convention doesn't introduce phantom token boundaries."""
    text = _RE_INTRA_NEWLINE.sub(" ", text)
    return [w.lower() for w in _RE_WORD.findall(text)]


def _word_counts_for_chapter(path: Path) -> Counter:
    text = path.read_text(encoding="utf-8")
    return Counter(_tokenize(text))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="source/gutenberg_imported.jsonl")
    p.add_argument("--out", default="source/burrows_centroids.json")
    p.add_argument("--mfw-n", type=int, default=150,
                   help="Number of most-frequent words to keep (Burrows uses 150)")
    args = p.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    out_path = REPO_ROOT / args.out

    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    print(f"Loaded {len(rows)} chapters from {manifest_path.name}")

    # Pass 1: per-chapter word counts + global vocabulary frequency.
    print("Pass 1: counting words across all chapters...")
    t0 = time.time()
    chapter_counts: list[tuple[str, str, Counter]] = []  # (cluster, path, counter)
    global_counts: Counter = Counter()
    for i, r in enumerate(rows, 1):
        c = _word_counts_for_chapter(REPO_ROOT / r["path"])
        chapter_counts.append((r["cluster"], r["path"], c))
        global_counts.update(c)
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}  ({time.time()-t0:.0f}s)")

    # MFW-N selection.
    mfw_words = [w for w, _ in global_counts.most_common(args.mfw_n)]
    print(f"\nTop-{args.mfw_n} most frequent words selected.")
    print(f"  most common 10: {mfw_words[:10]}")
    print(f"  least common (rank {args.mfw_n}): {mfw_words[-1]} "
          f"(freq={global_counts[mfw_words[-1]]:,})")

    # Pass 2: per-chapter relative frequencies for the MFW set.
    print("\nPass 2: computing per-chapter MFW relative frequencies...")
    word_idx = {w: i for i, w in enumerate(mfw_words)}
    n_words = len(mfw_words)

    # rel_freq[chapter_i, word_j] = count(word_j in chapter_i) / total_words(chapter_i)
    chapter_vecs: list[list[float]] = []
    chapter_clusters: list[str] = []
    for cluster, _path, c in chapter_counts:
        total = sum(c.values())
        if total == 0:
            continue
        vec = [0.0] * n_words
        for w, count in c.items():
            j = word_idx.get(w)
            if j is not None:
                vec[j] = count / total
        chapter_vecs.append(vec)
        chapter_clusters.append(cluster)
    print(f"  {len(chapter_vecs)} chapter vectors built")

    # Per-word mu and sigma across the corpus (for z-normalization).
    mu = [0.0] * n_words
    for v in chapter_vecs:
        for j in range(n_words):
            mu[j] += v[j]
    mu = [m / len(chapter_vecs) for m in mu]

    sigma_sq = [0.0] * n_words
    for v in chapter_vecs:
        for j in range(n_words):
            sigma_sq[j] += (v[j] - mu[j]) ** 2
    sigma = [(s / len(chapter_vecs)) ** 0.5 for s in sigma_sq]
    # Avoid divide-by-zero on words that happen to be perfectly uniform.
    sigma = [max(s, 1e-9) for s in sigma]

    # Z-normalized per-chapter vectors.
    chapter_z: list[list[float]] = []
    for v in chapter_vecs:
        chapter_z.append([(v[j] - mu[j]) / sigma[j] for j in range(n_words)])

    # Per-cluster centroid = mean of its chapters' z-vectors.
    cluster_sums: dict[str, list[float]] = {}
    cluster_n: dict[str, int] = {}
    for cluster_name, z in zip(chapter_clusters, chapter_z):
        if cluster_name not in cluster_sums:
            cluster_sums[cluster_name] = [0.0] * n_words
            cluster_n[cluster_name] = 0
        for j in range(n_words):
            cluster_sums[cluster_name][j] += z[j]
        cluster_n[cluster_name] += 1
    centroids = {
        c: [s / cluster_n[c] for s in sums] for c, sums in cluster_sums.items()
    }

    # Sanity: pairwise delta between cluster centroids. A small number here
    # would mean the clusters are stylistically indistinguishable on the MFW
    # axis (a problem). Burrows delta = mean |z_a - z_b|.
    print("\nPairwise Burrows' Delta between cluster centroids:")
    cnames = sorted(centroids)
    print(f"{'':<22}" + "  ".join(f"{c[:12]:>12}" for c in cnames))
    for ci in cnames:
        row_vals = []
        for cj in cnames:
            if ci == cj:
                row_vals.append("    --")
                continue
            d = sum(abs(centroids[ci][k] - centroids[cj][k]) for k in range(n_words)) / n_words
            row_vals.append(f"{d:>12.4f}")
        print(f"{ci:<22}" + "  ".join(row_vals))

    # Per-cluster intra-distance: how cohesive each cluster is.
    print("\nPer-cluster intra-cluster Delta (lower = more cohesive):")
    for cname in cnames:
        idxs = [i for i, c in enumerate(chapter_clusters) if c == cname]
        # Mean delta from each chapter to the centroid
        cent = centroids[cname]
        dists = [
            sum(abs(chapter_z[i][k] - cent[k]) for k in range(n_words)) / n_words
            for i in idxs
        ]
        mean_d = sum(dists) / max(1, len(dists))
        print(f"  {cname:<22} n={len(idxs):4d}  mean_delta={mean_d:.4f}")

    out_data = {
        "mfw_n": args.mfw_n,
        "mfw_words": mfw_words,
        "corpus_mu": mu,
        "corpus_sigma": sigma,
        "cluster_centroids": centroids,
        "cluster_chapter_counts": cluster_n,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\n-> {out_path.relative_to(REPO_ROOT)} "
          f"({out_path.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
