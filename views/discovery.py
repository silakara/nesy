"""Discovery page: Earth terrain that matches the lunar / Martian targets, outside known analog sites."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from terrain_twin import TARGET_DEM_FILES  # noqa: E402
from twin_ui import _stretch, cached_earth_dem, cached_target_dem, render_terrain_twin  # noqa: E402

DISC = BASE_DIR / "discovery"
RANKINGS = {
    "Analog Suite engine (worldwide)": {
        "file": "gee_ranked.csv", "score": "similarity_score", "higher_better": True,
        "about": "Worldwide ~5 km cells (0.045°) built on Google Earth Engine from the same sources as the "
                 "Analog Suite data, passed through the same false-analog filter (no ocean, forest, cropland, towns, "
                 "salt flats, dunes, vegetated or wet ground) and scored with the same similarity engine (0–100). "
                 "Known lunar analog sites are removed; best cells at least 50 km apart.",
    },
    "3D shape × texture × slope": {
        "file": "shape_texture_candidates.csv", "score": "final_score", "higher_better": True,
        "about": "Shape, texture and slope, each scored 0–1 and multiplied (1 = identical to the target). Shape: same ridges and hollows after rotation and tilt removal. Texture: crater-like pits, no water gullies. Slope: similar steepness and relief.",
    },
    "3D shape only": {
        "file": "shape_candidates.csv", "score": "shape_corr", "higher_better": True,
        "about": "How closely the Earth surface follows the target surface at 30 m (rotation and mirror allowed). Overall form only: texture and steepness are not checked.",
    },
    "Slope statistics": {
        "file": "discovery_candidates.csv", "score": "match_distance", "higher_better": False,
        "about": "Distance between slope and roughness statistics at 30 m (lower = closer). Does not look at how the terrain is laid out, so the 3D shape can differ.",
    },
}

st.title("🧭 Discovery: Analog Terrain Not Yet Pinned")
st.caption(
    "Earth terrain that matches each target, found by scanning open satellite data. Known analog sites are excluded."
)


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def known_sites() -> list:
    """Lunar analog sites only (the ones removed before scanning); Mars-only sites stay searchable."""
    sites = json.loads((DISC / "known_analogs.json").read_text(encoding="utf-8"))["sites"]
    return [s for s in sites if s.get("for", "both") in ("moon", "both")]


LUNAR_TARGETS = [t for t in TARGET_DEM_FILES if "Jezero" not in t]   # Discovery is about the Moon only

sb = st.sidebar
target = sb.selectbox("Lunar Target Zone", LUNAR_TARGETS)
body = "Moon"
available = {k: v for k, v in RANKINGS.items() if (DISC / v["file"]).exists()}
if not available:
    st.warning("No scan results found in discovery/. Run `python discovery/scan.py` and `python discovery/scan_shape.py`.")
    st.stop()
mode = sb.radio("Ranking", list(available))
top_n = sb.slider("Candidates shown", 5, 40, 15, 5)
show_known = sb.toggle("Show excluded known analog sites", value=True)

spec = available[mode]
f = DISC / spec["file"]
df = load(str(f), f.stat().st_mtime)
d = df[(df["target"] == target) & df["lat"].between(-56, 60)].copy()
d = d.sort_values(spec["score"], ascending=not spec["higher_better"]).head(top_n).reset_index(drop=True)
d.insert(0, "#", range(1, len(d) + 1))
if d.empty:
    st.info(f"No candidates for {target} in this ranking.")
    st.stop()

st.info(f"**{mode}:** {spec['about']}")

# ---- Map ------------------------------------------------------------------ #
m = folium.Map(location=[20, 10], zoom_start=2, tiles=None, control_scale=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite",
).add_to(m)
if show_known:
    for s in known_sites():
        tip = f"Excluded: {s['name']} ({s['source']})"
        style = dict(color="#94a3b8", weight=1, fill=True, fill_opacity=0.18)
        if "box" in s:
            a, b, c, e = s["box"]
            folium.Rectangle([[a, c], [b, e]], tooltip=tip, **style).add_to(m)
        else:
            folium.Circle([s["lat"], s["lon"]], radius=s["radius_km"] * 1000, tooltip=tip, **style).add_to(m)
for _, r in d.iterrows():
    folium.CircleMarker(
        [r["lat"], r["lon"]], radius=8, color="#0f172a", weight=1.2, fill=True, fill_opacity=0.95,
        fill_color="#10b981" if r["#"] <= 3 else "#f59e0b",
        tooltip=f"#{r['#']} · {r.get('place', '')} · {r['lat']:.3f}°, {r['lon']:.3f}° · {spec['score']} {r[spec['score']]:.3f}",
    ).add_to(m)
# Every section below is a collapsible block that starts open.
with st.expander("🗺️ Candidate map", expanded=True):
    st_folium(m, height=430, use_container_width=True, key=f"disc_map_{target}_{mode}", returned_objects=[])
    st.caption("Green: top 3. Amber: other candidates. Grey: known lunar analog sites removed before the scan.")

# ---- Table ---------------------------------------------------------------- #
cols = ["#"] + (["place"] if "place" in d.columns else []) + ["lat", "lon", "elevation_m"] + [c for c in [
    "macro_region", "similarity_score", "subscore_topography", "subscore_thermal", "subscore_mineralogy_isru",
    "subscore_volatiles", "slope_deg", "tri_m", "delta_t_k", "mafic_index", "clay_hydrated_index", "albedo",
    "ndvi", "subsurface_ice_moisture",
    "final_score", "detrended_corr", "texture_similarity", "slope_similarity", "shape_corr", "rmse_m",
    "relief_m", "target_relief_m",
    "slope_median", "target_slope_median", "slope_p90", "target_slope_p90", "roughness_median",
    "target_roughness_median", "match_distance", "cover_frac"] if c in d.columns]
with st.expander("📋 Candidate table", expanded=True):
    st.dataframe(d[cols], hide_index=True, **_stretch())


def _label(i) -> str:
    place = d.loc[i, "place"] if "place" in d.columns else ""
    return f"#{d.loc[i, '#']} · {place} ({d.loc[i, 'lat']:.3f}°, {d.loc[i, 'lon']:.3f}°)"


# ---- Top places in 3D ----------------------------------------------------- #
tz = cached_target_dem(target)["z"]
surfaces = [(f"{target} (Moon)", tz, "Greys")]
failed = []
for i in list(d.index[:3]):
    rr = d.loc[i]
    try:
        ez = cached_earth_dem(round(float(rr["lat"]), 5), round(float(rr["lon"]), 5))["z"]
    except Exception:
        failed.append(_label(i))
        continue
    if "mirrored" in rr and bool(rr["mirrored"]):
        ez = ez[:, ::-1]
    if "rotation_deg" in rr and float(rr["rotation_deg"]):
        from scipy.ndimage import rotate as nd_rotate
        ez = nd_rotate(ez, float(rr["rotation_deg"]), reshape=False, order=1, mode="nearest")
    surfaces.append((_label(i), ez, "Earth"))
relief = max(float(z.max() - z.min()) for _, z, _ in surfaces)
with st.expander("🧊 Top 3 in 3D", expanded=True):
    st.caption(
        f"The {target} target and the three best Earth sites as 5 × 5 km surfaces on the same vertical scale "
        "(heights from each box's lowest point). Drag to rotate, scroll to zoom."
    )
    for row_start in range(0, len(surfaces), 2):
        cols3d = st.columns(2)
        for col, (title, z, scale) in zip(cols3d, surfaces[row_start:row_start + 2]):
            ax = np.arange(z.shape[0]) * 30.0
            fig = go.Figure(go.Surface(z=z - z.min(), x=ax, y=ax[::-1], colorscale=scale, showscale=False))
            fig.update_layout(
                title=dict(text=title, font=dict(size=13)), template="plotly_dark", height=360,
                margin=dict(l=0, r=0, t=36, b=0), paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(zaxis=dict(range=[0, relief], title="m"), xaxis_title="m", yaxis_title="m",
                           aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35)),
            )
            col.plotly_chart(fig, key=f"top3d_{target}_{mode}_{title}", **_stretch())
    if failed:
        st.warning("Elevation tiles could not be fetched for: " + "; ".join(failed) + ". Reload the page to retry.")

# ---- Candidate inspector -------------------------------------------------- #
with st.expander("🔎 Inspect a candidate", expanded=True):
    pick = st.selectbox(
        "Candidate",
        list(d.index),
        format_func=_label,
        key=f"disc_pick_{target}_{mode}",
    )
    r = d.loc[pick]
    lat, lon = float(r["lat"]), float(r["lon"])
    l1, l2 = st.columns(2)
    l1.link_button(
        "🌍 Open in NASA Worldview",
        f"https://worldview.earthdata.nasa.gov/?v={lon - 0.4:.4f},{lat - 0.3:.4f},{lon + 0.4:.4f},{lat + 0.3:.4f}"
        "&l=MODIS_Terra_CorrectedReflectance_TrueColor,Reference_Labels_15m,Reference_Features_15m,Coastlines_15m",
        **_stretch(),
    )
    l2.link_button("🗺️ Open in Google Maps (satellite)", f"https://www.google.com/maps/@{lat},{lon},13z/data=!3m1!1e3",
                   **_stretch())
render_terrain_twin(
    pd.Series({"lat": lat, "lon": lon, "macro_region": f"#{r['#']} {r.get('place', 'Discovery candidate')}"}),
    target, body, key="disc",
    rotate_deg=float(r["rotation_deg"]) if "rotation_deg" in r else 0.0,
    mirrored=bool(r["mirrored"]) if "mirrored" in r else False,
)

with st.expander("ℹ️ Method, data and limits", expanded=True):
    st.markdown(
        "**Analog Suite engine (worldwide)**\n"
        "1. `discovery/gee_scan.py` builds the same 11 features as `earth_candidates.csv` on Google Earth Engine for "
        "every 0.045° cell in 1° tiles with dry open land (56°S–60°N): NASADEM slope / roughness, Landsat 9 "
        "Red/Blue (mafic), SWIR1/SWIR2 (clay), visible albedo and NDVI, MODIS day–night ΔT, SMAP moisture, "
        "ESA WorldCover land cover. Cells are 5 km north–south and 5 km × cos(latitude) east–west "
        "(4.4 km median), and vegetated pixels are averaged in rather than masked, so values are close to the "
        "Analog Suite data but not identical. On the 8,302 cells both share: correlation 0.82–1.00 per feature, "
        "ΔT 9% lower, and the filter gives the same verdict on 95.9% of cells (98% in Qaidam, 82–84% in "
        "Anatolia and the Altiplano, where NDVI sits right at the 0.15 cut).\n"
        "2. `discovery/gee_rank.py` runs the Analog Suite's `apply_false_analog_filter` and "
        "`compute_similarity_scores` unchanged, after removing known lunar analog sites "
        "(`for: moon / both` in `discovery/known_analogs.json`; Mars-only sites such as Qaidam or Atacama stay "
        "searchable because they have not been proposed for the Moon).\n\n"
        "**3D shape scans: how it works**\n"
        "1. Keep every 5 × 5 km box between 56°S and 60°N that is mostly barren (NASA MODIS land cover), "
        "minus known analog sites. About 4.3 million boxes.\n"
        "2. Compare each box with the target shape at ~250 m and keep the best 150 per target.\n"
        "3. Re-measure those at 30 m with 24 rotations × mirror → **3D shape only**.\n"
        "4. Re-score by shape × texture × slope → **3D shape × texture × slope**.\n\n"
        "**Reading the scores**\n"
        "- final_score above ~0.5 is a close match; around 0.2 means one part is off "
        "(see `texture_similarity` and `slope_similarity`).\n"
        "- Shackleton and Malapert stay near 0.2: steep Earth slopes have water gullies, lunar ones do not.\n"
        "\n"
        "**Data:** AWS Terrain Tiles (SRTM 30 m) for Earth; NASA PDS LOLA, SLDEM2015 and USGS CTX DTM for the "
        "targets; NASA MODIS IGBP 2024 for land cover; excluded sites in `discovery/known_analogs.json`.\n\n"
        "**Limits:** the 3D shape and slope rankings use terrain only (no mineralogy, temperature, ice or "
        "radiation); the Analog Suite engine ranking uses all eight features but not the 3D shape. Every candidate "
        "is a pre-screen: check it on satellite imagery, since dunes and SRTM noise can mimic craters.\n\n"
        "**Re-run:** `python discovery/gee_scan.py --project <id>` then `python discovery/gee_rank.py`; "
        "`python discovery/scan.py`, `python discovery/scan_shape.py`, `python discovery/rerank_texture.py`."
    )
