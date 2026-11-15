"""
Best Earth site per similarity domain, for the Top page.

For every lunar target and every domain (topography, thermal, mineralogy / ISRU, volatiles) the
similarity engine runs with that domain's weights only and every other domain at 0, exactly like
setting three of the Analog Suite sliders to 0. The best cells are kept for two pools:

  Analog Suite   earth_candidates.csv after the false-analog filter (the known analog regions)
  Discovery      discovery/gee_cells.csv after the same filter, minus known lunar analog sites
                 (the pool gee_rank.py uses)

Each pool is scored on its own, as on its own page, so a score is comparable inside a pool but not
between the two; the feature values are comparable. `overall_score` is the default-weight score.

Ties: a domain with one or two features is matched almost perfectly by thousands of cells (volatiles:
every cell whose moisture equals the target's). Cells whose domain score is equal to 0.1 point are
therefore ordered by `overall_score`, so the pick is the most lunar-like site among the equals.

Usage: python discovery/top_by_category.py [cells.csv]   -> discovery/top_by_category.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from gee_rank import SPACING_KM, spaced  # noqa: E402
from physics_engine import (DEFAULT_WEIGHTS, FEATURE_GROUPS, FEATURES, apply_false_analog_filter,  # noqa: E402
                            compute_similarity_scores, load_candidates, load_targets)
from place_names import add_places  # noqa: E402
from scan import known_mask  # noqa: E402

TOP_PER_DOMAIN = 5
OUT = HERE / "top_by_category.csv"
KEEP = ["lat", "lon", "elevation_m", "macro_region", "category_score", "overall_score"] + FEATURES


def domain_weights(group: str) -> dict:
    """The group's default feature weights, every other feature 0."""
    return {f: (DEFAULT_WEIGHTS[f] if f in FEATURE_GROUPS[group] else 0.0) for f in FEATURES}


def pools(cells_path: Path) -> dict:
    analog, _ = apply_false_analog_filter(load_candidates(HERE.parent / "earth_candidates.csv"))
    cells = pd.read_csv(cells_path, low_memory=False).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    filtered, _ = apply_false_analog_filter(cells)
    sites = [s for s in json.loads((HERE / "known_analogs.json").read_text(encoding="utf-8"))["sites"]
             if s.get("for", "both") in ("moon", "both")]
    known = known_mask(filtered["lat"].to_numpy(), filtered["lon"].to_numpy(), sites)
    return {"Analog Suite": analog, "Discovery": filtered.loc[~known].copy()}


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gee_cells.csv"
    targets = load_targets(HERE.parent / "lunar_targets.json")
    out = []
    for source, pool in pools(src).items():
        print(f"{source}: {len(pool):,} cells")
        for name, tgt in targets.items():
            if tgt.get("body", "Moon") != "Moon":
                continue
            overall = compute_similarity_scores(pool, name, targets=targets)["similarity_score"]
            for group in FEATURE_GROUPS:
                scored = compute_similarity_scores(pool, name, domain_weights(group), targets=targets)
                scored = scored.dropna(subset=["similarity_score"]).rename(columns={"similarity_score": "category_score"})
                scored["overall_score"] = overall.reindex(scored.index)
                scored = (scored.assign(_tier=scored["category_score"].round(1))
                          .sort_values(["_tier", "overall_score"], ascending=False))
                top = spaced(scored, TOP_PER_DOMAIN, SPACING_KM)[KEEP].copy()
                top.insert(0, "rank", range(1, len(top) + 1))
                top.insert(0, "category", group)
                top.insert(0, "target", name)
                top.insert(0, "source", source)
                out.append(top)
                b = top.iloc[0]
                print(f"  {name:<18} {group:<16} best {b['category_score']:.1f} at {b['lat']:.3f}, {b['lon']:.3f}")
    res = add_places(pd.concat(out, ignore_index=True))
    res.to_csv(OUT, index=False)
    print(f"-> {OUT} ({len(res)} rows)")


if __name__ == "__main__":
    main()
