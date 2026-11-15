"""
Earth Engine discovery scan: rebuild the earth_candidates.csv features worldwide, from the same
sources as the team guide (ARTEMIS-SHIELD Teknik Uygulama ve Veri Rehberi), so the Analog Suite's
false-analog filter and similarity engine can run on new ground.

Differences from the team's earth_candidates.js export (kept so a worldwide scan stays affordable):
  * cells are 0.045 deg squares: 5 km north-south, but 5 km x cos(latitude) east-west (4.4 km median,
    2.5 km at 60 deg), not equal-area 5 x 5 km UTM cells; each 1 deg tile holds 22 x 22 cells (0.99 deg)
  * every pixel is averaged; the team script first masks vegetated (NDVI > 0.15), water and urban pixels
  * one year (2024) instead of 2023-2024; thermal inertia is 274.46 / delta_t_k, not T_night / delta_t_k
  * no "blue > 0" / "SWIR2 > 0" guard on the band ratios; apply_false_analog_filter rejects the
    resulting impossible values (invalid_value rule)
Measured on the 8,302 cells both data sets share: correlation 0.82-1.00 per feature, medians within
5 % except delta_t_k (-9 %) and thermal_inertia_proxy (+10 %), same filter verdict on 95.9 % of cells.

Per cell (0.045 deg grid, EPSG:4326):
  elevation_m                  NASADEM 30 m, cell mean
  slope_deg                    NASADEM slope at ~55 m, cell mean (calibrated to the existing CSV)
  tri_m                        standard deviation of NASADEM elevation in the cell (calibrated to the existing CSV)
  delta_t_k                    MODIS MOD11A1 LST day - night, yearly mean
  thermal_inertia_proxy        274.46 / delta_t_k   (fits the existing CSV: r = 0.993)
  mafic_index                  Landsat 9 L2 Red / Blue (B4 / B2), cloud-masked yearly median
  clay_hydrated_index          Landsat 9 L2 SWIR1 / SWIR2 (B6 / B7)
  albedo                       Landsat 9 visible mean (B2 + B3 + B4) / 3 (calibrated to the existing CSV)
  ndvi                         Landsat 9 (B5 - B4) / (B5 + B4)
  subsurface_ice_moisture      SMAP L4 sm_surface, yearly mean
  landcover_class              ESA WorldCover v200 majority class (60 = bare / sparse)
  bare_frac                    share of the cell that is WorldCover class 60

Usage
  python discovery/gee_scan.py --project <cloud-project-id> --pilot         # Anatolia + Qaidam boxes
  python discovery/gee_scan.py --project <cloud-project-id>                 # all bare-land tiles 56S-60N
Output: discovery/gee_cells.csv (appends per tile, resumable)
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "gee_cells.csv"
CELL_DEG = 0.045                  # ~5 km in latitude
TILE_DEG = 1.0
YEAR = ("2024-01-01", "2025-01-01")
PILOT_BOXES = {                   # [lat_min, lon_min, lat_max, lon_max], from the team guide
    "Anatolia": [37.5, 33.0, 40.0, 44.5],
    "Tibet Qaidam": [36.5, 91.0, 38.5, 95.0],
}
BANDS = ["elevation_m", "slope_deg", "tri_m", "delta_t_k", "mafic_index", "clay_hydrated_index",
         "albedo", "ndvi", "subsurface_ice_moisture", "landcover_class", "bare_frac"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def feature_image() -> ee.Image:
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").toFloat()
    # Calibrated against earth_candidates.csv (pilot tile 37N 92E): the existing slope_deg matches slope
    # computed at ~55 m, and tri_m matches the standard deviation of elevation inside the cell.
    dem55 = dem.reduceResolution(ee.Reducer.mean(), maxPixels=256).reproject(crs="EPSG:4326", scale=55)
    slope = ee.Terrain.slope(dem55)
    tri_std = dem.reduceResolution(ee.Reducer.stdDev(), maxPixels=65535).rename("tri_m")

    def lmask(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(0b11111).eq(0)          # fill, dilated cloud, cirrus, cloud, shadow
        sr = img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(0.0000275).add(-0.2)
        return sr.updateMask(clear)

    ls = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(*YEAR).map(lmask).median())
    b2, b3, b4, b5, b6, b7 = [ls.select(b) for b in ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]]
    mafic = b4.divide(b2)
    clay = b6.divide(b7)
    ndvi = b5.subtract(b4).divide(b5.add(b4))
    albedo = b2.add(b3).add(b4).divide(3)          # visible mean: matches the existing CSV (ratio 0.99)

    lst = ee.ImageCollection("MODIS/061/MOD11A1").filterDate(*YEAR)
    dt = (lst.select("LST_Day_1km").mean().subtract(lst.select("LST_Night_1km").mean())).multiply(0.02)
    # SMAP L4 has 8 images a day; the 1st of each month (96 images) keeps memory low and still covers all seasons.
    sm = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(*YEAR)
          .filter(ee.Filter.calendarRange(1, 1, "day_of_month")).select("sm_surface").mean())

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    wc300 = wc.reduceResolution(ee.Reducer.mode(), maxPixels=4096).reproject(crs="EPSG:4326", scale=300)
    bare = wc300.eq(60).toFloat()

    fine = ee.Image.cat([dem.rename("elevation_m").reduceResolution(ee.Reducer.mean(), maxPixels=65535),
                         slope.rename("slope_deg").reduceResolution(ee.Reducer.mean(), maxPixels=65535),
                         tri_std])
    coarse = ee.Image.cat([dt.rename("delta_t_k"), mafic.rename("mafic_index"), clay.rename("clay_hydrated_index"),
                           albedo.rename("albedo"), ndvi.rename("ndvi"), sm.rename("subsurface_ice_moisture")])
    lc = wc300.reduceResolution(ee.Reducer.mode(), maxPixels=4096).rename("landcover_class").toFloat()
    bf = bare.reduceResolution(ee.Reducer.mean(), maxPixels=4096).rename("bare_frac")
    return ee.Image.cat([fine, coarse, lc, bf]).select(BANDS)


def fetch_tile(img: ee.Image, lat0: float, lon0: float, size: float = TILE_DEG) -> pd.DataFrame:
    n = int(round(size / CELL_DEG))
    arr = ee.data.computePixels({
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": n, "height": n},
            "affineTransform": {"scaleX": CELL_DEG, "shearX": 0, "translateX": lon0,
                                "shearY": 0, "scaleY": -CELL_DEG, "translateY": lat0 + size},
            "crsCode": "EPSG:4326",
        },
    })
    rows, cols = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    names = arr.dtype.names
    df = pd.DataFrame({b: np.asarray(arr[b], dtype=float).ravel() for b in names})
    df.insert(0, "lon", (lon0 + (cols.ravel() + 0.5) * CELL_DEG).round(5))
    df.insert(0, "lat", (lat0 + size - (rows.ravel() + 0.5) * CELL_DEG).round(5))
    if "delta_t_k" in df:
        df["thermal_inertia_proxy"] = 274.46 / df["delta_t_k"].where(df["delta_t_k"] > 0.5)
    return df.dropna(subset=[names[0]])


GROUPS = [["elevation_m", "slope_deg", "tri_m"], ["mafic_index", "clay_hydrated_index", "albedo", "ndvi"],
          ["delta_t_k"], ["subsurface_ice_moisture"], ["landcover_class", "bare_frac"]]


def fetch_tile_all(img: ee.Image, lat0: float, lon0: float, size: float = TILE_DEG) -> pd.DataFrame:
    """One request per band group (all bands in one request exceed Earth Engine's memory limit)."""
    with ThreadPoolExecutor(len(GROUPS)) as ex:
        parts = list(ex.map(lambda b: fetch_tile(img.select(b), lat0, lon0, size), GROUPS))
    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=["lat", "lon"], how="left")
    return df[["lat", "lon"] + BANDS + ["thermal_inertia_proxy"]]


def bare_tiles() -> list[tuple[float, float]]:
    """
    1-degree tiles (56S-60N) worth scanning: a coarse screen with MODIS MCD12Q1 2023 (500 m) at 0.25 deg,
    keeping tiles where barren (16) or open shrubland (7) covers > 2 %. The real per-cell filter later
    uses ESA WorldCover exactly like the Analog Suite; this only skips tiles with no dry open land at all.
    """
    lc = ee.ImageCollection("MODIS/061/MCD12Q1").filterDate("2023-01-01", "2024-01-01").first().select("LC_Type1")
    dry = lc.eq(16).Or(lc.eq(7)).toFloat().rename("dry")
    step = 0.25
    arr = ee.data.computePixels({
        "expression": dry,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": int(360 / step), "height": int(116 / step)},
                 "affineTransform": {"scaleX": step, "shearX": 0, "translateX": -180,
                                     "shearY": 0, "scaleY": -step, "translateY": 60},
                 "crsCode": "EPSG:4326"},
    })
    a = np.asarray(arr["dry"], dtype=float)
    k = int(1 / step)
    a = np.nan_to_num(a).reshape(116, k, 360, k).mean(axis=(1, 3))
    return [(60 - i - 1.0, -180 + j * 1.0) for i in range(116) for j in range(360) if a[i, j] > 0.02]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--pilot", action="store_true", help="only the Anatolia and Qaidam boxes")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    img = feature_image()

    if args.pilot:
        tiles = []
        for a, b, c, d in PILOT_BOXES.values():
            tiles += [(la, lo) for la in np.arange(math.floor(a), math.ceil(c), TILE_DEG)
                      for lo in np.arange(math.floor(b), math.ceil(d), TILE_DEG)]
    else:
        log("finding tiles with bare land ...")
        tiles = bare_tiles()
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT, usecols=["tile"])
        done = set(prev["tile"].unique())
    todo = [t for t in tiles if f"{t[0]:.0f},{t[1]:.0f}" not in done]
    log(f"{len(tiles)} tiles, {len(todo)} to fetch")

    def one(t):
        for attempt in range(3):
            try:
                df = fetch_tile_all(img, *t)
                df["tile"] = f"{t[0]:.0f},{t[1]:.0f}"
                return t, df
            except Exception as exc:  # quota / timeout: back off and retry
                err = exc
                time.sleep(10 * (attempt + 1))
        log(f"  tile {t} failed: {str(err)[:200]}")
        return t, None

    n = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(one, t) for t in todo]):
            t, df = fut.result()
            n += 1
            if df is not None and len(df):
                df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            if n % 10 == 0 or n == len(todo):
                log(f"  {n}/{len(todo)} tiles")
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
