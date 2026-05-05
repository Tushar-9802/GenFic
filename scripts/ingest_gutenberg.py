"""Fetch Project Gutenberg public-domain works by register cluster.

Pipeline
--------
1. Download (and cache) the official Project Gutenberg catalog CSV.
2. Resolve author list from `src/genfic/registers.py` for the requested cluster(s).
3. For each author: filter the catalog to language=en, type=Text rows whose
   Authors field contains "Last, First". Cap per author with --max-works.
4. Download the plain-text body from gutenberg.org via the canonical URL
   `https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt`. Tolerate 404s
   (some IDs only ship as HTML/EPUB).
5. Strip PG header/footer markers (*** START / *** END).
6. Split into chapters via heading regex; save each chapter to
   `source/raw/gutenberg/{author_slug}/{work_slug}/chapter-NNN.txt`.
7. Emit `source/gutenberg_imported.jsonl` with one row per chapter.

Usage
-----
  # All five clusters, default 10 works/author
  python scripts/ingest_gutenberg.py

  # One cluster, smoke
  python scripts/ingest_gutenberg.py --cluster victorian-formal --max-works 3

  # Force re-download of catalog
  python scripts/ingest_gutenberg.py --refresh-catalog
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from genfic.registers import REGISTERS  # noqa: E402

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
TEXT_URL_FALLBACK = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

PG_HEADER_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", re.IGNORECASE)
PG_FOOTER_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", re.IGNORECASE)
CHAPTER_HEADING_RE = re.compile(
    r"^[ \t]*(?:CHAPTER|Chapter)\s+(?:[IVXLCDM]+|\d+|[A-Za-z]+)\.?\s*$",
    re.MULTILINE,
)
SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(s: str) -> str:
    return SLUG_RE.sub("-", s.lower()).strip("-")


def _strip_diacritics(s: str) -> str:
    """Lowercase + strip combining diacritical marks for tolerant matching."""
    import unicodedata
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def candidate_lastfirst(name: str) -> list[str]:
    """Yield possible 'Last, First' forms for `name`, accommodating compound
    surnames ('Joseph Sheridan Le Fanu' -> ['Sheridan Le Fanu, Joseph',
    'Le Fanu, Joseph Sheridan', 'Fanu, Joseph Sheridan Le'])."""
    parts = name.strip().split()
    if len(parts) < 2:
        return [name]
    out = []
    for n_first in range(1, len(parts)):
        given = " ".join(parts[:n_first])
        last = " ".join(parts[n_first:])
        out.append(f"{last}, {given}")
    return out


def cache_catalog(repo_root: Path, refresh: bool = False) -> Path:
    cache = repo_root / "source" / "pg_catalog.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return cache
    print(f"Downloading PG catalog ({CATALOG_URL}) ...")
    r = requests.get(CATALOG_URL, timeout=120, stream=True)
    r.raise_for_status()
    with open(cache, "wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            f.write(chunk)
    print(f"  cached: {cache.relative_to(repo_root)}  ({cache.stat().st_size/1024/1024:.1f} MB)")
    return cache


def load_catalog(cache: Path) -> list[dict]:
    rows: list[dict] = []
    with open(cache, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Type") != "Text":
                continue
            if r.get("Language") != "en":
                continue
            rows.append(r)
    return rows


def find_author_works(rows: list[dict], author: str, max_works: int) -> list[dict]:
    """Match rows whose Authors field contains any of the author's plausible
    'Last, First' forms. Tolerant to compound surnames and diacritics."""
    needles = [_strip_diacritics(c) for c in candidate_lastfirst(author)]
    out = []
    for r in rows:
        field = _strip_diacritics(r.get("Authors") or "")
        if not any(n in field for n in needles):
            continue
        out.append(r)
        if len(out) >= max_works:
            break
    return out


def fetch_text(pg_id: str, sleep: float = 0.5) -> str | None:
    time.sleep(sleep)
    for url in (TEXT_URL_TEMPLATE.format(id=pg_id), TEXT_URL_FALLBACK.format(id=pg_id)):
        try:
            r = requests.get(url, timeout=60)
        except Exception:
            continue
        if r.status_code == 200 and r.text and "Project Gutenberg" in r.text[:8000]:
            return r.text
    return None


def strip_gutenberg_boilerplate(text: str) -> str:
    m_start = PG_HEADER_RE.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = PG_FOOTER_RE.search(text)
    if m_end:
        text = text[:m_end.start()]
    return text.strip()


def split_chapters(text: str) -> list[str]:
    matches = list(CHAPTER_HEADING_RE.finditer(text))
    if len(matches) < 2:
        body = text.strip()
        return [body] if body and len(body.split()) > 500 else []
    chapters: list[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body.split()) > 100:
            chapters.append(body)
    return chapters


def ingest_one_author(
    author: str,
    cluster_name: str,
    max_works: int,
    catalog_rows: list[dict],
    out_root: Path,
    manifest: list[dict],
    seen_slugs: set[str],
) -> int:
    print(f"  [{author}] looking up in catalog...")
    works = find_author_works(catalog_rows, author, max_works=max_works * 3)
    if not works:
        print(f"    ! no works found in catalog for {author}")
        return 0
    n_kept = 0
    for w in works:
        if n_kept >= max_works:
            break
        title = w.get("Title", "untitled").split("\n")[0].strip()
        pg_id = w.get("Text#", "")
        if not pg_id:
            continue
        work_slug = slug(title)[:60] or f"pg-{pg_id}"
        author_slug = slug(author)
        key = f"{author_slug}/{work_slug}"
        if key in seen_slugs:
            continue
        out_dir = out_root / author_slug / work_slug
        if out_dir.exists() and any(out_dir.glob("chapter-*.txt")):
            print(f"    - already on disk: {key}")
            seen_slugs.add(key)
            continue
        text = fetch_text(pg_id)
        if not text:
            print(f"    ? skipped {pg_id} ({title[:50]}): no plain-text URL or empty")
            continue
        text = strip_gutenberg_boilerplate(text)
        chapters = split_chapters(text)
        if not chapters:
            print(f"    ? skipped {pg_id} ({title[:50]}): no chapter splits found")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for ci, body in enumerate(chapters, 1):
            chap_path = out_dir / f"chapter-{ci:03d}.txt"
            chap_path.write_text(body, encoding="utf-8")
            manifest.append({
                "author": author,
                "cluster": cluster_name,
                "work_title": title,
                "work_slug": work_slug,
                "pg_id": pg_id,
                "chapter_num": ci,
                "path": str(chap_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "words": len(body.split()),
                "chars": len(body),
            })
        total_w = sum(len(c.split()) for c in chapters)
        print(f"    + pg{pg_id} {title[:55]}: {len(chapters)} chapters, {total_w:,} words")
        seen_slugs.add(key)
        n_kept += 1
    return n_kept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", choices=sorted(REGISTERS),
                   help="Restrict to one cluster (default: all)")
    p.add_argument("--max-works", type=int, default=10,
                   help="Cap works per author (default 10)")
    p.add_argument("--refresh-catalog", action="store_true",
                   help="Re-download pg_catalog.csv")
    p.add_argument("--out", default="source/raw/gutenberg")
    p.add_argument("--manifest", default="source/gutenberg_imported.jsonl")
    args = p.parse_args()

    out_root = REPO_ROOT / args.out
    manifest_path = REPO_ROOT / args.manifest
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    cache = cache_catalog(REPO_ROOT, refresh=args.refresh_catalog)
    print("Loading catalog...")
    rows = load_catalog(cache)
    print(f"  {len(rows):,} English text entries in catalog")

    manifest: list[dict] = []
    seen_slugs: set[str] = set()
    if manifest_path.exists():
        for line in open(manifest_path, encoding="utf-8"):
            r = json.loads(line)
            manifest.append(r)
            seen_slugs.add(f"{slug(r['author'])}/{r['work_slug']}")
        print(f"Resuming: {len(manifest)} prior chapters in manifest")

    clusters = [args.cluster] if args.cluster else list(REGISTERS)
    for cname in clusters:
        reg = REGISTERS[cname]
        print(f"\n=== {cname} ({reg.display}) ===")
        for author in reg.authors:
            ingest_one_author(
                author, cname, args.max_works, rows, out_root, manifest, seen_slugs,
            )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== summary ===")
    by_cluster: dict[str, int] = {}
    by_author: dict[str, int] = {}
    by_cluster_words: dict[str, int] = {}
    for r in manifest:
        by_cluster[r["cluster"]] = by_cluster.get(r["cluster"], 0) + 1
        by_author[r["author"]] = by_author.get(r["author"], 0) + 1
        by_cluster_words[r["cluster"]] = by_cluster_words.get(r["cluster"], 0) + r.get("words", 0)
    for c in sorted(by_cluster):
        print(f"  {c:22s} {by_cluster[c]:5d} chapters  {by_cluster_words[c]:>10,d} words")
    print(f"  {'TOTAL':22s} {len(manifest):5d} chapters across {len(by_author)} authors, "
          f"{sum(by_cluster_words.values()):,} words")
    print(f"\nManifest: {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
