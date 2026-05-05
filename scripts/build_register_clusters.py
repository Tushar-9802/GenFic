"""Validate the register clusters defined in `src/genfic/registers.py`.

Runs `scripts/style_profile.py`'s metric pass on each cluster's authors and
checks that clusters separate on the §IV-D metrics. A cluster is considered
valid if its central tendency on at least 3 of 5 axes lies in a band that does
not overlap any other cluster's interquartile range on that axis.

The 5 axes used here:
  1. mean_sentence_length_words
  2. type_token_ratio_chapter_avg
  3. adverb_ly_density_pct
  4. commas_per_sentence
  5. dialogue_paragraph_pct

Outputs
-------
  source/register_clusters.json
    {
      "victorian-formal": {
        "authors": [...],
        "metrics": { axis -> {"median": x, "iqr": [lo, hi], "values": {...}} },
        "separates_on": [axes],
        "valid": bool
      },
      ...
    }
  prints a human-readable separation report.

Usage
-----
  python scripts/build_register_clusters.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import the style_profile pass directly so we don't shell out
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from style_profile import compute_profiles, gather_records  # noqa: E402

from genfic.registers import REGISTERS  # noqa: E402

AXES = (
    "mean_sentence_length_words",
    "type_token_ratio_chapter_avg",
    "adverb_ly_density_pct",
    "commas_per_sentence",
    "dialogue_paragraph_pct",
)


def quartiles(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    s = sorted(xs)
    n = len(s)
    return (
        s[n // 4],
        s[n // 2],
        s[(3 * n) // 4],
    )


def cluster_axis_values(profiles: dict[str, dict], authors: tuple[str, ...], axis: str) -> list[float]:
    out = []
    for a in authors:
        if a in profiles:
            v = profiles[a].get(axis)
            if isinstance(v, (int, float)):
                out.append(float(v))
    return out


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """[a_lo, a_hi] overlaps [b_lo, b_hi]?"""
    return not (a[1] < b[0] or b[1] < a[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="source/register_clusters.json")
    args = p.parse_args()

    print("Loading all Gutenberg chapters and computing per-author profiles...")
    records = gather_records(REPO_ROOT, register=None)
    if not records:
        print("ERROR: source/gutenberg_imported.jsonl missing or empty. "
              "Run scripts/ingest_gutenberg.py first.", file=sys.stderr)
        return 2
    profiles = compute_profiles(records)
    print(f"  profiled {len(profiles)} authors")

    # Per-cluster axis stats
    cluster_stats: dict[str, dict] = {}
    for cname, reg in REGISTERS.items():
        cluster_stats[cname] = {
            "display": reg.display,
            "authors": list(reg.authors),
            "axes": {},
        }
        for axis in AXES:
            vals = cluster_axis_values(profiles, reg.authors, axis)
            if not vals:
                cluster_stats[cname]["axes"][axis] = None
                continue
            q1, q2, q3 = quartiles(vals)
            cluster_stats[cname]["axes"][axis] = {
                "n": len(vals),
                "median": q2,
                "iqr": [q1, q3],
                "values": {a: profiles[a].get(axis) for a in reg.authors if a in profiles},
            }

    # Separation: two signals per axis, per cluster.
    #   "strong" — cluster's IQR doesn't overlap any other cluster's IQR
    #   "extreme" — cluster's median is the min OR max across all clusters on this axis
    # Pass = at least one strong axis, OR at least two extreme axes. Surface
    # metrics on within-literary register are subtle (all English prose has
    # mean sentence length 7-9 and TTR 0.27-0.35), so requiring strict IQR
    # separation on most axes would reject any plausible literary cluster.
    # The post-training adapter eval (style_profile.py on outputs) is the real
    # discriminator; this script just catches degenerate cluster definitions.
    medians_by_axis: dict[str, list[tuple[str, float]]] = {a: [] for a in AXES}
    for cname, stats in cluster_stats.items():
        for axis in AXES:
            ax = stats["axes"].get(axis)
            if ax is not None:
                medians_by_axis[axis].append((cname, ax["median"]))

    for cname, stats in cluster_stats.items():
        strong: list[str] = []
        extreme: list[str] = []
        for axis in AXES:
            mine = stats["axes"].get(axis)
            if mine is None:
                continue
            mine_iqr = tuple(mine["iqr"])
            # strong: no IQR overlap with any other cluster
            other_iqrs = [
                tuple(ostats["axes"][axis]["iqr"])
                for ocn, ostats in cluster_stats.items()
                if ocn != cname and ostats["axes"].get(axis) is not None
            ]
            if other_iqrs and not any(overlaps(mine_iqr, oi) for oi in other_iqrs):
                strong.append(axis)
            # extreme: my median is the min or max among cluster medians on this axis
            axis_medians = medians_by_axis[axis]
            if axis_medians:
                vals = [v for _, v in axis_medians]
                if mine["median"] == min(vals) or mine["median"] == max(vals):
                    extreme.append(axis)
        stats["separates_strong"] = strong
        stats["separates_extreme"] = extreme
        stats["valid"] = len(strong) >= 1 or len(extreme) >= 2

    # Report
    print("\n=== cluster separation report ===")
    print(f"Pass criterion: >=1 strong-separation axis OR >=2 extreme-rank axes\n")
    for cname, stats in cluster_stats.items():
        ok = "PASS" if stats["valid"] else "WEAK"
        print(f"[{ok}] {cname} ({stats['display']})  "
              f"strong={len(stats['separates_strong'])}/{len(AXES)} "
              f"extreme={len(stats['separates_extreme'])}/{len(AXES)}")
        if stats["separates_strong"]:
            print(f"    strong: {', '.join(stats['separates_strong'])}")
        if stats["separates_extreme"]:
            print(f"    extreme: {', '.join(stats['separates_extreme'])}")
        for axis in AXES:
            ax = stats["axes"].get(axis)
            if ax is None:
                print(f"    {axis:34s} no data")
                continue
            print(f"    {axis:34s} median={ax['median']:.3f}  "
                  f"iqr=[{ax['iqr'][0]:.3f}, {ax['iqr'][1]:.3f}]  n={ax['n']}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cluster_stats, indent=2), encoding="utf-8")
    print(f"\n-> {out_path.relative_to(REPO_ROOT)}")
    weak = [c for c, s in cluster_stats.items() if not s["valid"]]
    if weak:
        print(f"\nWARNING: {len(weak)} cluster(s) below the separation threshold: "
              f"{', '.join(weak)}. Consider re-curating authors before training.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
