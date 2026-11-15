# NESY

Planetary analog discovery and engineering transfer suite (NASA Space Apps Challenge 2026).
NESY ranks places on Earth by how closely they resemble three lunar target zones — Shackleton Ridge,
Malapert Mountain and Marius Hills — and shows where the analogy holds and where it breaks down.

## Run

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. It needs internet access for map tiles and for the Earth
elevation tiles used by the Terrain Twin panel.

## Pages

**Analog Suite** (`app.py`) scores the 9,808 cells of `earth_candidates.csv` (five known analog
regions, 5 × 5 km) against the selected target:

1. False-analog filter: rejects salt playas, mobile dunes, vegetated or wet ground, non-barren land
   cover and physically impossible sensor values.
2. Similarity score (0–100): weighted Euclidean + cosine hybrid over eight features, with sub-scores
   for topography, thermal regime, mineralogy / ISRU and volatiles. Weights are adjustable in the sidebar.
3. Engineering transfer: gravity, vacuum-thermal and radiation-shielding corrections for the selected site.
4. Terrain Twin: the target DEM and the Earth site on the same 5 km / 30 m grid.

**Discovery** (`views/discovery.py`) looks for analog terrain outside the known regions, with four rankings:

| Ranking | File | What it compares |
|---|---|---|
| Analog Suite engine (worldwide) | `discovery/gee_ranked.csv` | The same eight features and engine, on a worldwide Earth Engine scan |
| 3D shape × texture × slope | `discovery/shape_texture_candidates.csv` | Surface shape, crater / gully texture and slope |
| 3D shape only | `discovery/shape_candidates.csv` | Surface shape at 30 m (rotation and mirror allowed) |
| Slope statistics | `discovery/discovery_candidates.csv` | Slope and roughness statistics |

Known lunar analog sites listed in `discovery/known_analogs.json` are removed from every ranking.

**Top** (`views/top.py`) takes one similarity domain at a time (topography, thermal, mineralogy / ISRU or
volatiles at full weight, the other three at 0) and shows the best Analog Suite site next to the best
Discovery site for the selected target. It reads `discovery/top_by_category.csv`, written by
`python discovery/top_by_category.py`.

## Data

| File | Content |
|---|---|
| `lunar_targets.json` | Target parameters (ideal / min / max) calibrated from NASA LRO QuickMap statistics |
| `quickmap-region-data-*.csv` | The QuickMap region statistics the targets were calibrated from |
| `earth_candidates.csv` | Analog Suite cells, exported from Earth Engine (NASADEM, Landsat 9, MODIS LST, SMAP, ESA WorldCover) |
| `terrain_dems/*.json` | Target DEMs, 169 × 169 at 30 m (LOLA, SLDEM2015, Mars 2020 CTX DTM) |
| `ranked_candidates.csv` | Output of `python physics_engine.py --target "Shackleton Ridge"` |

Scores are relative to the candidate set they were computed on: compare them within one ranking,
not between the Analog Suite and Discovery pages.

## Re-running the Discovery scans

The worldwide scan needs a Google Cloud project registered for Earth Engine and `earthengine-api`
(see the note in `requirements.txt`):

```bash
python discovery/gee_scan.py --project <cloud-project-id>
```

```bash
python discovery/gee_rank.py
```

```bash
python discovery/top_by_category.py
```

`gee_scan.py` writes `discovery/gee_cells.csv` (about 1.75 million cells, ~390 MB, not tracked in git);
it resumes where it stopped. `gee_rank.py` filters, scores and names the best 100 cells per target,
at least 50 km apart; `top_by_category.py` does the same per similarity domain for the Top page. Run
both again after changing `lunar_targets.json`. Place names need `discovery/geonames/cities1000.zip` from
https://download.geonames.org/export/dump/ and use the `EE_PROJECT` environment variable for the
region / country lookup.

The terrain-only scans need no account:

```bash
python discovery/scan.py
```

```bash
python discovery/scan_shape.py
```

```bash
python discovery/rerank_texture.py
```

## Limits

- Candidates are a pre-screen from orbital data; check a site on satellite imagery before relying on it.
- The worldwide scan uses 0.045° cells (5 km north–south, narrower east–west away from the equator)
  and does not mask vegetated pixels before averaging, so its values are close to, but not identical
  with, `earth_candidates.csv`. The differences are listed at the top of `discovery/gee_scan.py`.
- Earth's day–night temperature range (about 10–35 K) is far below the lunar one (180–290 K), so the
  thermal sub-score is low for every Earth site.
