"""Compute per-author and per-register linguistic style fingerprints.

Used as the baseline against which the fine-tuned adapter's generations are
compared (see IEEE Paper §IV-D — register-discrimination via passive voice
ratio, nominalization rate, lexical density, type-token ratio, sentence length,
adverb density, and dialogue ratio).

Pure CPU; no spaCy/NLTK required. Approximations:
  - Sentence count: regex split on `[.!?]\\s+[A-Z]`
  - Paragraph count: `\\n\\n+` splits
  - Type-token ratio: |unique words| / |total words|, per chapter, then averaged
  - Mean word length, sentence length, paragraph length
  - Adverb density: count of `-ly` suffixed words / total word count
  - Punctuation density: commas + semicolons per sentence
  - Dialogue ratio: paragraphs containing at least one quoted span / total
  - Sentence-end distribution: % of sentences ending with `?` / `!` / `.`
  - Top-K distinguishing bigrams per author

Inputs
------
  source/gutenberg_imported.jsonl   (produced by scripts/ingest_gutenberg.py)

Outputs
-------
  source/style_profile.json   per-author dict of statistics
  prints a human-readable summary table

Usage
-----
  python scripts/style_profile.py
  python scripts/style_profile.py --register victorian-formal
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from genfic.registers import REGISTERS  # noqa: E402

_RE_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")
_RE_PARA = re.compile(r"\n\s*\n+")
_RE_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_RE_DIALOGUE = re.compile(r'["“][^"”]+["”]')
_RE_LY = re.compile(r"\b\w+ly\b", re.IGNORECASE)


def gather_records(repo_root: Path, register: str | None = None) -> list[dict]:
    """Read source/gutenberg_imported.jsonl, optionally filter to one cluster."""
    out = []
    p = repo_root / "source" / "gutenberg_imported.jsonl"
    if not p.exists():
        return out
    allowed_authors = (
        set(REGISTERS[register].authors) if register and register in REGISTERS else None
    )
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if allowed_authors is not None and r.get("author") not in allowed_authors:
            continue
        r["_path"] = repo_root / r["path"]
        out.append(r)
    return out


def chapter_stats(text: str) -> dict:
    paragraphs = [p.strip() for p in _RE_PARA.split(text) if p.strip()]
    if not paragraphs:
        return {}
    para_count = len(paragraphs)

    sentences: list[str] = []
    para_sentence_counts: list[int] = []
    for p in paragraphs:
        ss = _RE_SENTENCE.split(p)
        sentences.extend(ss)
        para_sentence_counts.append(len(ss))
    sent_count = max(1, len(sentences))

    words = _RE_WORD.findall(text)
    word_count = max(1, len(words))
    unique_words = {w.lower() for w in words}
    total_word_chars = sum(len(w) for w in words)

    end_chars = Counter()
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s[-1] in ".!?":
            end_chars[s[-1]] += 1

    commas = text.count(",")
    semis = text.count(";")
    ly_words = len(_RE_LY.findall(text))
    dialogue_paras = sum(1 for p in paragraphs if _RE_DIALOGUE.search(p))

    lower_words = [w.lower() for w in words]
    bigrams = Counter(zip(lower_words, lower_words[1:]))

    return {
        "n_chars": len(text),
        "n_words": word_count,
        "n_unique_words": len(unique_words),
        "n_sentences": sent_count,
        "n_paragraphs": para_count,
        "total_word_chars": total_word_chars,
        "ly_words": ly_words,
        "commas": commas,
        "semis": semis,
        "dialogue_paras": dialogue_paras,
        "end_period": end_chars.get(".", 0),
        "end_excl": end_chars.get("!", 0),
        "end_quest": end_chars.get("?", 0),
        "para_sentence_counts": para_sentence_counts,
        "bigrams": bigrams,
    }


def aggregate_author(chapter_stats_list: list[dict]) -> dict:
    if not chapter_stats_list:
        return {}
    keys_sum = (
        "n_chars n_words n_unique_words n_sentences n_paragraphs "
        "total_word_chars ly_words commas semis dialogue_paras "
        "end_period end_excl end_quest"
    ).split()
    s = Counter()
    para_sent_all: list[int] = []
    bigrams_all: Counter = Counter()
    chapter_ttrs = []
    for cs in chapter_stats_list:
        for k in keys_sum:
            s[k] += cs.get(k, 0)
        para_sent_all.extend(cs.get("para_sentence_counts", []))
        bigrams_all.update(cs.get("bigrams", Counter()))
        if cs.get("n_words"):
            chapter_ttrs.append(cs["n_unique_words"] / cs["n_words"])

    n_words = max(1, s["n_words"])
    n_sent = max(1, s["n_sentences"])
    n_para = max(1, s["n_paragraphs"])
    n_end = max(1, s["end_period"] + s["end_excl"] + s["end_quest"])

    return {
        "chapters": len(chapter_stats_list),
        "total_words": s["n_words"],
        "total_sentences": s["n_sentences"],
        "total_paragraphs": s["n_paragraphs"],
        "mean_word_length_chars": round(s["total_word_chars"] / n_words, 2),
        "mean_sentence_length_words": round(n_words / n_sent, 2),
        "mean_paragraph_length_sentences": round(n_sent / n_para, 2),
        "median_paragraph_length_sentences": (
            sorted(para_sent_all)[len(para_sent_all) // 2] if para_sent_all else 0
        ),
        "type_token_ratio_chapter_avg": round(
            sum(chapter_ttrs) / max(1, len(chapter_ttrs)), 4
        ),
        "adverb_ly_density_pct": round(s["ly_words"] / n_words * 100, 3),
        "commas_per_sentence": round(s["commas"] / n_sent, 3),
        "semicolons_per_1k_words": round(s["semis"] / n_words * 1000, 3),
        "dialogue_paragraph_pct": round(s["dialogue_paras"] / n_para * 100, 2),
        "sentence_end_pct": {
            "period": round(s["end_period"] / n_end * 100, 1),
            "excl": round(s["end_excl"] / n_end * 100, 1),
            "quest": round(s["end_quest"] / n_end * 100, 1),
        },
        "_bigrams": bigrams_all,
    }


def distinguishing_bigrams(profiles: dict[str, dict], top_k: int = 30) -> dict[str, list[tuple]]:
    global_counts: Counter = Counter()
    global_total = 0
    per_author_total: dict[str, int] = {}
    for a, prof in profiles.items():
        bg = prof["_bigrams"]
        per_author_total[a] = sum(bg.values())
        global_counts.update(bg)
        global_total += per_author_total[a]

    out: dict[str, list[tuple]] = {}
    for a, prof in profiles.items():
        bg = prof["_bigrams"]
        my_total = max(1, per_author_total[a])
        scores: list[tuple[float, tuple, float, float]] = []
        for big, c in bg.items():
            if c < 30:
                continue
            my_rate = c / my_total
            other_count = global_counts[big] - c
            other_total = max(1, global_total - my_total)
            other_rate = other_count / other_total
            if other_rate < 1e-7:
                continue
            ratio = my_rate / other_rate
            if ratio < 2.0:
                continue
            scores.append((ratio, big, my_rate, other_rate))
        scores.sort(reverse=True)
        out[a] = [
            (f"{w1} {w2}", round(my_rate * 1e4, 2), round(other_rate * 1e4, 2), round(ratio, 1))
            for ratio, (w1, w2), my_rate, other_rate in scores[:top_k]
        ]
    return out


def compute_profiles(records: list[dict]) -> dict[str, dict]:
    by_author: dict[str, list[dict]] = defaultdict(list)
    for i, rec in enumerate(records, 1):
        text = rec["_path"].read_text(encoding="utf-8")
        cs = chapter_stats(text)
        if cs:
            by_author[rec["author"]].append(cs)
        if i % 500 == 0:
            print(f"  {i}/{len(records)}")
    return {a: aggregate_author(lst) for a, lst in by_author.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", choices=sorted(REGISTERS),
                   help="Restrict profile to one register cluster's authors")
    p.add_argument("--out", default="source/style_profile.json")
    args = p.parse_args()

    records = gather_records(REPO_ROOT, register=args.register)
    if not records:
        print("No records found in source/gutenberg_imported.jsonl", file=sys.stderr)
        return 2
    print(f"Loading {len(records)} chapters across "
          f"{len(set(r['author'] for r in records))} authors"
          f"{f' (register={args.register})' if args.register else ''}...")

    profiles = compute_profiles(records)
    print("Computing distinguishing bigrams...")
    distbg = distinguishing_bigrams(profiles, top_k=30)

    out_data = {}
    for a, prof in profiles.items():
        clean = {k: v for k, v in prof.items() if not k.startswith("_")}
        clean["distinguishing_bigrams"] = distbg.get(a, [])
        out_data[a] = clean

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\n-> {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
