"""
Rank the Earth Engine cells (discovery/gee_cells.csv) with the Analog Suite's own engine:

1. apply_false_analog_filter   - same rules as the Analog Suite (salt playa, mobile dunes, vegetated / wet,
                                 non-barren land cover, missing data).
2. known lunar analog sites removed - discovery/known_analogs.json entries for 'moon' / 'both'.
3. compute_similarity_scores    - same engine and default weights as the Analog Suite, lunar targets only.
4. spaced top list per target   - best cells at least SPACING_KM apart, so one basin cannot fill the list.

Usage: python discovery/gee_rank.py [cells.csv]   -> discovery/gee_ranked.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from physics_engine import apply_false_analog_filter, compute_similarity_scores, load_targets  # noqa: E402
from place_names import add_places  # noqa: E402
from scan import haversine, known_mask  # noqa: E402

TOP_PER_TARGET = 100
SPACING_KM = 50.0


def spaced(df: pd.DataFrame, n: int, km: float) -> pd.DataFrame:
    keep, la, lo = [], [], []
    for i, lat, lon in zip(df.index, df["lat"].to_numpy(), df["lon"].to_numpy()):
        if la and haversine(lat, lon, np.array(la), np.array(lo)).min() < km:
            continue
        keep.append(i)
        la.append(lat)
        lo.append(lon)
        if len(keep) >= n:
            break
    return df.loc[keep]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gee_cells.csv"
    cells = pd.read_csv(src).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"cells: {len(cells):,}")

    filtered, summary = apply_false_analog_filter(cells)
    print("false-analog filter:", {k: f"{v:,}" for k, v in summary.items()})

    sites = [s for s in json.loads((HERE / "known_analogs.json").read_text(encoding="utf-8"))["sites"]
             if s.get("for", "both") in ("moon", "both")]          # lunar analog sites only
    known = known_mask(filtered["lat"].to_numpy(), filtered["lon"].to_numpy(), sites)
    pool = filtered.loc[~known].copy()
    print(f"known analog sites removed: {int(known.sum()):,} -> pool {len(pool):,}")

    targets = load_targets(HERE.parent / "lunar_targets.json")
    out = []
    for name, tgt in targets.items():
        if tgt.get("body", "Moon") != "Moon":                         # lunar targets only
            continue
        scored = compute_similarity_scores(pool, name, targets=targets).dropna(subset=["similarity_score"])
        top = spaced(scored, TOP_PER_TARGET, SPACING_KM).copy()
        top.insert(0, "target", name)
        top["rank"] = range(1, len(top) + 1)
        out.append(top)
        print(f"\n== {name}: best {top['similarity_score'].iloc[0]:.1f}")
        print(top.head(5)[["lat", "lon", "macro_region", "similarity_score", "subscore_topography",
                           "subscore_thermal", "subscore_mineralogy_isru", "subscore_volatiles"]].round(2).to_string(index=False))
    res = add_places(pd.concat(out, ignore_index=True))
    res.to_csv(HERE / "gee_ranked.csv", index=False)
    print(f"\n-> {HERE / 'gee_ranked.csv'} ({len(res)} rows)")


if __name__ == "__main__":
    main()
