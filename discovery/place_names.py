"""
Name every candidate: the region and country it lies in, plus its nearest town.

  place     "Gökçeören, Manisa, Turkey"                     town within 3 km
            "near Karapınar, Konya, Turkey"                 town within 50 km (same country)
            "New Valley, Egypt (~304 km from Mūţ)"          remote: region / country of the point itself
  place_km  distance to the nearest town [km]

Region and country come from the administrative boundaries the point falls in (FAO GAUL level 1 on
Google Earth Engine), so a point near a border is not named after a town across it. Towns come from
GeoNames cities1000 (CC BY 4.0, discovery/geonames/). Without Earth Engine the town's region / country
is used as a fallback.

Usage: python discovery/place_names.py [--project <earth-engine-project>]
       -> adds place / place_km to every candidate CSV in discovery/
The Earth Engine project can also be set with the EE_PROJECT environment variable.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

HERE = Path(__file__).resolve().parent
GEO = HERE / "geonames"
ADMIN_CACHE = GEO / "admin_cache.csv"
CANDIDATE_FILES = ["gee_ranked.csv", "shape_texture_candidates.csv", "shape_candidates.csv", "discovery_candidates.csv"]
EARTH_R_KM = 6371.0
DEFAULT_PROJECT = os.environ.get("EE_PROJECT", "nesy-123")   # set EE_PROJECT to use your own Cloud project


@lru_cache(maxsize=1)
def _towns():
    with zipfile.ZipFile(GEO / "cities1000.zip") as z:
        raw = z.read("cities1000.txt").decode("utf-8")
    cities = pd.read_csv(io.StringIO(raw), sep="\t", header=None, quoting=3, low_memory=False,
                         usecols=[1, 4, 5, 8, 10], names=["name", "lat", "lon", "cc", "admin1"])
    admin = pd.read_csv(GEO / "admin1CodesASCII.txt", sep="\t", header=None, quoting=3,
                        names=["code", "name", "ascii", "id"], dtype=str)
    admin1 = dict(zip(admin["code"], admin["name"]))
    countries = {}
    for line in (GEO / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            parts = line.split("\t")
            countries[parts[0]] = parts[4]
    tree = BallTree(np.radians(cities[["lat", "lon"]].to_numpy()), metric="haversine")
    return cities, admin1, countries, tree


def _key(lat, lon) -> str:
    return f"{float(lat):.5f},{float(lon):.5f}"


def admin_lookup(points: pd.DataFrame, project: str | None) -> dict:
    """{'lat,lon': (region, country)} from FAO GAUL level 1, cached in discovery/geonames/admin_cache.csv."""
    cache = {}
    if ADMIN_CACHE.exists():
        c = pd.read_csv(ADMIN_CACHE, dtype=str, keep_default_na=False)
        cache = {k: (r, n) for k, r, n in zip(c["key"], c["region"], c["country"])}
    keys = [_key(a, b) for a, b in zip(points["lat"], points["lon"])]
    todo = sorted({k for k in keys if k not in cache})
    if todo and project:
        try:
            import ee
            ee.Initialize(project=project)
            # GAUL 2015 is marked deprecated on Earth Engine but still served. Its successor
            # (FAO/GAUL/2025/level1, fields GAUL1_NAME / GAUL0_NAME) spells regions differently
            # ("Región De Antofagasta" vs "Antofagasta"); switching means clearing admin_cache.csv
            # and renaming every candidate, so the labels stay on one edition here.
            gaul = ee.FeatureCollection("FAO/GAUL/2015/level1")
            for i in range(0, len(todo), 500):
                batch = todo[i:i + 500]
                pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(float(k.split(",")[1]), float(k.split(",")[0])),
                                                       {"k": k}) for k in batch])
                joined = ee.Join.saveFirst("a").apply(pts, gaul, ee.Filter.intersects(leftField=".geo", rightField=".geo"))
                res = joined.map(lambda f: ee.Feature(None, {"k": f.get("k"),
                                                             "r": ee.Feature(f.get("a")).get("ADM1_NAME"),
                                                             "c": ee.Feature(f.get("a")).get("ADM0_NAME")})).getInfo()
                for feat in res["features"]:
                    p = feat["properties"]
                    cache[p["k"]] = (p.get("r") or "", p.get("c") or "")
                for k in batch:                       # points in the sea / outside GAUL
                    cache.setdefault(k, ("", ""))
            pd.DataFrame([(k, r, n) for k, (r, n) in cache.items()], columns=["key", "region", "country"]) \
                .to_csv(ADMIN_CACHE, index=False)
        except Exception as exc:                      # no Earth Engine: fall back to the town's region
            print(f"admin lookup skipped ({str(exc)[:120]})")
    return {k: cache.get(k, ("", "")) for k in keys}


def _norm_country(name: str) -> str:
    """'Iran  (Islamic Republic of)' -> 'Iran'; collapses spaces."""
    return " ".join(re.sub(r"\(.*?\)", "", name or "").split())


def _same_country(gaul: str, geonames: str) -> bool:
    a, b = _norm_country(gaul).lower(), _norm_country(geonames).lower()
    return bool(a) and bool(b) and (a.startswith(b[:4]) or b.startswith(a[:4]))


def add_places(df: pd.DataFrame, project: str | None = DEFAULT_PROJECT) -> pd.DataFrame:
    cities, admin1, countries, tree = _towns()
    dist, idx = tree.query(np.radians(df[["lat", "lon"]].to_numpy(dtype=float)), k=1)
    adm = admin_lookup(df, project)
    labels = []
    for (lat, lon), d, i in zip(df[["lat", "lon"]].to_numpy(), dist[:, 0] * EARTH_R_KM, idx[:, 0]):
        c = cities.iloc[i]
        town = str(c["name"])
        t_region = admin1.get(f"{c['cc']}.{c['admin1']}", "")
        t_country = countries.get(c["cc"], str(c["cc"]))
        g_region, g_country = adm[_key(lat, lon)]
        if g_region.lower().startswith("administrative unit not available"):   # GAUL label for disputed areas
            g_region = ""
        same_country = not g_country or _same_country(g_country, t_country)
        region = g_region or t_region
        country = t_country if same_country else _norm_country(g_country)
        area = ", ".join(x for x in [region, country] if x)
        if d <= 3 and same_country:
            labels.append(f"{town}, {area}" if region != town else area)
        elif d <= 50 and same_country:
            labels.append(f"near {town}, {area}")
        else:
            labels.append(f"{area} (~{d:.0f} km from {town})")
    out = df.copy()
    out["place"] = labels
    out["place_km"] = np.round(dist[:, 0] * EARTH_R_KM, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="Earth Engine project for region / country lookup")
    args = ap.parse_args()
    for name in CANDIDATE_FILES:
        f = HERE / name
        if not f.exists():
            continue
        df = add_places(pd.read_csv(f).drop(columns=["place", "place_km"], errors="ignore"), args.project)
        df.to_csv(f, index=False)
        print(f"{name}: {len(df)} rows named; e.g. {df['place'].iloc[0]}")


if __name__ == "__main__":
    main()
