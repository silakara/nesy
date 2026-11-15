"""
Terrain twin: the target (Moon / Mars) and an Earth candidate measured on the same 5 km box.

* Target DEMs come from ``terrain_dems/*.json`` (169 x 169 grid, 30 m spacing, int16 little-endian
  half-metres, base64). They were produced from NASA PDS LOLA ldem_75s_30m (Shackleton, Malapert),
  SLDEM2015 (Marius Hills) and the USGS Mars 2020 CTX DTM (Jezero).
* Earth DEMs are fetched live from AWS Terrain Tiles (terrarium PNG, mostly SRTM) and resampled
  onto the same grid, so both sides go through the same slope / roughness / hillshade code.

Grid convention: row 0 = north (for polar targets: 0 deg longitude), column 0 = west.
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

CELL_M = 30.0
GRID_N = 169
M_PER_PX_Z0 = 156543.03392
M_PER_DEG = 111319.49
ELEVATION_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

DEM_DIR = Path(__file__).resolve().parent / "terrain_dems"
TARGET_DEM_FILES = {
    "Shackleton Ridge": "shackleton.json",
    "Malapert Mountain": "malapert.json",
    "Marius Hills": "marius.json",
    "Jezero Crater Rim": "jezero.json",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_target_dem(target_name: str) -> Optional[Dict[str, Any]]:
    """Return {'z': (n, n) float array in metres, 'meta': dict} or None if no DEM exists for the target."""
    fname = TARGET_DEM_FILES.get(target_name)
    if not fname or not (DEM_DIR / fname).exists():
        return None
    with open(DEM_DIR / fname, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    n = int(data["n"])
    raw = np.frombuffer(base64.b64decode(data["dem"]), dtype="<i2").astype(float) / 2.0
    meta = {k: v for k, v in data.items() if k not in ("dem", "summary")}
    return {"z": raw.reshape(n, n), "meta": meta}


def _zoom_for(lat: float, cell_m: float = CELL_M) -> int:
    z = math.ceil(math.log2(M_PER_PX_Z0 * math.cos(math.radians(lat)) / cell_m))
    return int(min(14, max(0, z)))


def _lonlat_to_pixel(lon: np.ndarray, lat: np.ndarray, z: int):
    scale = 256 * 2 ** z
    la = np.radians(lat)
    x = (lon + 180.0) / 360.0 * scale
    y = (1 - np.log(np.tan(la) + 1 / np.cos(la)) / math.pi) / 2 * scale
    return x, y


def _fetch_tile(z: int, x: int, y: int, timeout: float) -> np.ndarray:
    from PIL import Image  # pillow ships with streamlit

    url = ELEVATION_TILES.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": "nesy-terrain-twin"})
    for attempt in range(3):          # a single slow / dropped tile used to leave the 3D view empty
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    rgb = np.asarray(img, dtype=float)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def fetch_earth_dem(lat: float, lon: float, timeout: float = 12.0) -> Dict[str, Any]:
    """Sample a GRID_N x GRID_N grid (CELL_M apart) centred on lat/lon from AWS Terrain Tiles."""
    zoom = _zoom_for(lat)
    half = (GRID_N - 1) / 2
    off = (np.arange(GRID_N) - half) * CELL_M
    east, north = np.meshgrid(off, -off)  # row 0 = north
    lats = lat + north / M_PER_DEG
    lons = lon + east / (M_PER_DEG * math.cos(math.radians(lat)))
    px, py = _lonlat_to_pixel(lons, lats, zoom)
    u, v = px - 0.5, py - 0.5
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = u - i0, v - j0

    ntiles = 2 ** zoom
    tx_min, tx_max = int(i0.min()) // 256, int(i0.max() + 1) // 256
    ty_min, ty_max = int(j0.min()) // 256, int(j0.max() + 1) // 256
    keys = [(tx, ty) for tx in range(tx_min, tx_max + 1) for ty in range(ty_min, ty_max + 1)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        tiles = dict(zip(keys, ex.map(lambda k: _fetch_tile(zoom, k[0] % ntiles, k[1], timeout), keys)))

    w = (tx_max - tx_min + 1) * 256
    h = (ty_max - ty_min + 1) * 256
    mosaic = np.empty((h, w))
    for (tx, ty), arr in tiles.items():
        mosaic[(ty - ty_min) * 256:(ty - ty_min + 1) * 256, (tx - tx_min) * 256:(tx - tx_min + 1) * 256] = arr
    ii, jj = i0 - tx_min * 256, j0 - ty_min * 256
    z = (mosaic[jj, ii] * (1 - fx) * (1 - fy) + mosaic[jj, ii + 1] * fx * (1 - fy)
         + mosaic[jj + 1, ii] * (1 - fx) * fy + mosaic[jj + 1, ii + 1] * fx * fy)
    return {"z": z, "zoom": zoom, "source": "AWS Terrain Tiles (terrarium; mostly SRTM 30 m)"}


# --------------------------------------------------------------------------- #
# Measures (same definitions as NESY's terrain.js)
# --------------------------------------------------------------------------- #
def slope_deg(z: np.ndarray, cell: float = CELL_M) -> np.ndarray:
    """Horn 3x3 slope [deg] for interior cells."""
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, k = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = (c + 2 * f + k - (a + 2 * d + g)) / (8 * cell)
    dzdy = (g + 2 * h + k - (a + 2 * b + c)) / (8 * cell)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def roughness_m(z: np.ndarray) -> np.ndarray:
    """Mean absolute height step to the 8 neighbours [m] for interior cells."""
    zc = z[1:-1, 1:-1]
    n0, n1 = z.shape
    acc = np.zeros_like(zc)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                acc += np.abs(z[1 + dr:n0 - 1 + dr, 1 + dc:n1 - 1 + dc] - zc)
    return acc / 8


def summarize(z: np.ndarray, cell: float = CELL_M) -> Dict[str, Any]:
    s = slope_deg(z, cell)
    r = roughness_m(z)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    return {
        "slope_median": float(np.median(s)) if s.size else float("nan"),
        "slope_p90": float(np.percentile(s, 90)) if s.size else float("nan"),
        "roughness_median": float(np.median(r)) if r.size else float("nan"),
        "relief_m": float(np.nanmax(z) - np.nanmin(z)),
        "slopes": s,
    }


def hillshade(z: np.ndarray, az_deg: float, el_deg: float, cell: float = CELL_M,
              cast_shadows: bool = True) -> np.ndarray:
    """Lambert shading, uint8 (n, n). Azimuth clockwise from grid-up. cast_shadows=False skips the
    shadow ray-march, so slopes are shaded but nothing is blacked out."""
    n = z.shape[0]
    az, el = math.radians(az_deg), math.radians(el_deg)
    lx, ly, lz = math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)
    zp = np.pad(z, 1, mode="edge")
    dzdx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2 * cell)
    dzdn = (zp[:-2, 1:-1] - zp[2:, 1:-1]) / (2 * cell)
    lum = (-dzdx * lx - dzdn * ly + lz) / np.sqrt(dzdx ** 2 + dzdn ** 2 + 1)
    lum = np.where(np.isfinite(lum) & (lum > 0), lum, 0.0)

    if cast_shadows and el_deg < 89.9:
        dc_raw, dr_raw = math.sin(az), -math.cos(az)
        longest = max(abs(dc_raw), abs(dr_raw)) or 1.0
        dc, dr = dc_raw / longest, dr_raw / longest
        rise = math.tan(el) * math.hypot(dc, dr) * cell
        zmax = np.nanmax(z)
        rr, cc = np.mgrid[0:n, 0:n].astype(float)
        h = z.copy()
        active = lum > 0
        shadow = np.zeros_like(active)
        for _ in range(2 * n):
            rr += dr
            cc += dc
            h = h + rise
            inside = (rr >= 0) & (cc >= 0) & (rr <= n - 1) & (cc <= n - 1) & (h <= zmax)
            active &= inside
            if not active.any():
                break
            ri = np.clip(np.rint(rr), 0, n - 1).astype(int)
            ci = np.clip(np.rint(cc), 0, n - 1).astype(int)
            hit = active & (z[ri, ci] > h + 1e-6)
            shadow |= hit
            active &= ~hit
        lum = np.where(shadow, 0.0, lum)
    return (lum * 255).astype(np.uint8)
