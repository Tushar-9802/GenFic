"""Train a 5-class register classifier on the PD corpus.

Used as the strongest "did the adapter learn the register?" signal in the eval
suite: take an adapter's generation, run it through this classifier, ask
"which of the five register clusters does this look like?" If adapter outputs
are classified as their target register at high rate, register transfer is real.

Methodology
-----------
- Texts: every chapter from `source/gutenberg_imported.jsonl`, sliced into
  ~500-word windows (long enough for stable style signal, short enough to
  generate hundreds of samples per cluster).
- Features: TF-IDF over word 1- and 2-grams, sublinear_tf=True. No character
  n-grams (overkill for 5-class register; word-level gives clean signal).
- Model: Logistic Regression with L2, C=1.0, multinomial.
- 80/20 chapter-stratified holdout for honest test accuracy (split on
  chapter level, not chunk level, so adjacent chunks of the same chapter
  don't leak across train/test).

Outputs
-------
  source/register_classifier.pkl    sklearn Pipeline (TfidfVectorizer + LR)
  prints classification report + confusion matrix on the holdout split

Usage
-----
  python scripts/train_classifier.py
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from genfic.registers import REGISTERS  # noqa: E402

_RE_INTRA_NEWLINE = re.compile(r"\n+")
_RE_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _normalize(text: str) -> str:
    """Collapse intra-paragraph line wraps so PG-storage `\\n\\n` doesn't
    leak into features as whitespace artifacts."""
    return _RE_INTRA_NEWLINE.sub(" ", text)


def _slice_windows(text: str, words_per_window: int) -> list[str]:
    """Split text into ~words_per_window-word windows. Drops any tail window
    shorter than half the target size."""
    words = _RE_WORD.findall(text)
    out = []
    for start in range(0, len(words), words_per_window):
        chunk = words[start : start + words_per_window]
        if len(chunk) < words_per_window // 2:
            break
        out.append(" ".join(chunk))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="source/gutenberg_imported.jsonl")
    p.add_argument("--out", default="source/register_classifier.pkl")
    p.add_argument("--words-per-window", type=int, default=500,
                   help="Window size for slicing chapters into training samples")
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--c", type=float, default=1.0,
                   help="LogReg inverse-regularization strength")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    out_path = REPO_ROOT / args.out

    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    print(f"Loaded {len(rows)} chapters from {manifest_path.name}")

    # Build sliced windows; track chapter id (used as the grouping key for
    # GroupShuffleSplit so a chapter's windows don't straddle train/test).
    print(f"Slicing chapters into {args.words_per_window}-word windows...")
    texts: list[str] = []
    labels: list[str] = []
    groups: list[int] = []
    t0 = time.time()
    for chap_id, r in enumerate(rows):
        text = _normalize((REPO_ROOT / r["path"]).read_text(encoding="utf-8"))
        for window in _slice_windows(text, args.words_per_window):
            texts.append(window)
            labels.append(r["cluster"])
            groups.append(chap_id)
        if (chap_id + 1) % 1000 == 0:
            print(f"  {chap_id+1}/{len(rows)}  windows so far: {len(texts):,}  ({time.time()-t0:.0f}s)")
    print(f"Total: {len(texts):,} windows across {len(set(groups))} chapters")
    print(f"Class distribution: {dict(Counter(labels))}")

    # Chapter-grouped 80/20 split. GroupShuffleSplit ensures no chapter's
    # windows appear in both train and test (otherwise the classifier could
    # memorize chapter-specific tokens and inflate test accuracy).
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    (train_idx, test_idx), = splitter.split(texts, labels, groups)
    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    print(f"\nSplit: {len(X_train):,} train / {len(X_test):,} test")
    print(f"Train chapters: {len(set(groups[i] for i in train_idx))}")
    print(f"Test chapters:  {len(set(groups[i] for i in test_idx))}")

    print(f"\nFitting TF-IDF (max_features={args.max_features:,}, ngram=(1,2)) + LogReg (C={args.c})...")
    pipe = Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=args.max_features,
            min_df=5,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"\b[a-zA-Z][a-zA-Z'-]+\b",
        )),
        ("clf", LogisticRegression(
            C=args.c,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=args.seed,
        )),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"  fit in {time.time()-t0:.0f}s")

    y_pred = pipe.predict(X_test)
    print("\n=== holdout classification report ===")
    print(classification_report(y_test, y_pred, digits=3))

    cnames = sorted(REGISTERS)
    cm = confusion_matrix(y_test, y_pred, labels=cnames)
    print("=== confusion matrix (rows=true, cols=pred) ===")
    print(f"{'':<22}" + "  ".join(f"{c[:12]:>12}" for c in cnames))
    for i, c in enumerate(cnames):
        row_total = max(1, cm[i].sum())
        print(f"{c:<22}" + "  ".join(
            f"{cm[i, j]:>5d}({cm[i, j]/row_total*100:>4.0f}%)" for j in range(len(cnames))
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"\n-> {out_path.relative_to(REPO_ROOT)} "
          f"({out_path.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
