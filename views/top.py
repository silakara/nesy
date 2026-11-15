"""Top page: for one lunar target and one similarity domain, the best Analog Suite site next to the best Discovery site."""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from physics_engine import FEATURE_GROUPS, load_targets  # noqa: E402
from terrain_twin import hillshade  # noqa: E402
from twin_ui import _stretch, _tint, cached_earth_dem, cached_target_dem  # noqa: E402

TOP_FILE = BASE_DIR / "discovery" / "top_by_category.csv"
JSON_PATH = BASE_DIR / "lunar_targets.json"

GROUP_LABELS = {
    "topography": "Topography & Trafficability",
    "thermal": "Thermal Regime",
    "mineralogy_isru": "Mineralogy & ISRU",
    "volatiles": "Volatiles & Ice Proxy",
}
FEATURE_LABELS = {
    "slope_deg": "Slope (deg)",
    "tri_m": "Roughness TRI (m)",
    "delta_t_k": "Diurnal ΔT (K)",
    "mafic_index": "Mafic / Basalt Index",
    "clay_hydrated_index": "Hydrated / Clay Index",
    "subsurface_ice_moisture": "Subsurface Moisture",
    "albedo": "Surface Albedo",
    "thermal_inertia_proxy": "Thermal Inertia Proxy",
}
SOURCES = {
    "Analog Suite": {"icon": "🛰️", "color": "#38bdf8",
                     "about": "Best cell in the known analog regions (`earth_candidates.csv`)."},
    "Discovery": {"icon": "🧭", "color": "#10b981",
                  "about": "Best cell of the worldwide Earth Engine scan, known lunar analog sites removed."},
}


@st.cache_data(show_spinner=False)
def load_top(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_targets(path: str, mtime: float) -> dict:
    return load_targets(path)


st.title("🏆 Top: Best Earth Match per Domain")
st.caption(
    "One similarity domain at a time: the lunar target, the most similar place in the Analog Suite regions and "
    "the most similar place found by Discovery, side by side."
)

if not TOP_FILE.exists():
    st.warning("No results found. Run `python discovery/top_by_category.py` (needs `discovery/gee_cells.csv`).")
    st.stop()

top = load_top(str(TOP_FILE), TOP_FILE.stat().st_mtime)
targets = cached_targets(str(JSON_PATH), JSON_PATH.stat().st_mtime)
names = [t for t in targets if t in set(top["target"])]
if not names:
    st.warning("`discovery/top_by_category.csv` has no rows for the current targets. Re-run `python discovery/top_by_category.py`.")
    st.stop()

target = st.sidebar.selectbox("Lunar Target Zone", names)
group = st.radio("Similarity domain", list(FEATURE_GROUPS), format_func=GROUP_LABELS.get, horizontal=True,
                 key="top_group")
label = GROUP_LABELS[group]
others = ", ".join(GROUP_LABELS[g] for g in FEATURE_GROUPS if g != group)
st.info(f"**{target} · {label}** at full weight; {others} at 0.")

params = targets[target]["parameters"]
feats = FEATURE_GROUPS[group]
sel = top[(top["target"] == target) & (top["category"] == group)]


def target_table() -> pd.DataFrame:
    return pd.DataFrame([{"Feature": FEATURE_LABELS[f], "Ideal": params[f]["ideal"],
                          "Range": f"{params[f]['min']:g} – {params[f]['max']:g}"} for f in feats])


def site_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{"Feature": FEATURE_LABELS[f], "This site": round(float(row[f]), 3),
                          "Status": "✅ Valid" if params[f]["min"] <= row[f] <= params[f]["max"] else "❌ Not valid"}
                         for f in feats])


SUN_AZ, SUN_EL, AMBIENT = 135, 45, 0.30     # high sun, no cast shadows, lifted by ambient light: bright plates


def plate(z, rgb):
    """Bright shaded relief: slopes are toned, nothing is blacked out. Same lighting for every plate."""
    shade = hillshade(z, SUN_AZ, SUN_EL, cast_shadows=False).astype(float)
    return _tint(255.0 * AMBIENT + (1.0 - AMBIENT) * shade, rgb)


tdata = targets[target]
tdem = cached_target_dem(target)
polar = abs(float(tdata.get("center_lat", 0))) > 80

# Every section below is a collapsible block that starts open.
best = {}
with st.expander("🌙 Moon · 🛰️ Analog Suite · 🧭 Discovery: side by side", expanded=True):
    moon_col, *site_cols = st.columns(3)
    with moon_col:
        st.subheader("🌙 Moon")
        st.caption("The target zone itself, for reference.")
        if tdem is not None:
            st.image(plate(tdem["z"], (1.0, 1.0, 1.0)), **_stretch())
        else:
            st.info("No terrain model is bundled for this target.")
        st.markdown(f"**{target}**  \n{tdata.get('region', '')}  \n"
                    f"{tdata.get('center_lat', 0):.3f}°, {tdata.get('center_lon', 0):.3f}°")
        st.dataframe(target_table(), hide_index=True, **_stretch())
        st.caption(tdata.get("scientific_context", ""))

    for col, (source, spec) in zip(site_cols, SOURCES.items()):
        with col:
            st.subheader(f"{spec['icon']} {source}")
            st.caption(spec["about"])
            s = sel[sel["source"] == source].sort_values("rank")
            if s.empty:
                st.info("No site for this target.")
                continue
            r = s.iloc[0]
            best[source] = r
            try:
                ez = cached_earth_dem(round(float(r["lat"]), 5), round(float(r["lon"]), 5))["z"]
                st.image(plate(ez, (1.0, 0.86, 0.66)), **_stretch())
            except Exception:  # network / tile errors must not break the page
                st.info("Elevation tiles could not be fetched. Reload the page to retry.")
            st.markdown(f"**{r['place']}**  \n{r['lat']:.3f}°, {r['lon']:.3f}° · {r['elevation_m']:,.0f} m")
            st.dataframe(site_table(r), hide_index=True, **_stretch())
            m1, m2 = st.columns(2)
            m1.metric("This domain", f"{r['category_score']:.1f}",
                      help=f"{label} similarity (0–100): the score with only this domain weighted.")
            m2.metric("All domains", f"{r['overall_score']:.1f}",
                      help="Score of the same site with the default weights on all four domains.")
            st.link_button("🗺️ Open in Google Maps (satellite)",
                           f"https://www.google.com/maps/@{r['lat']},{r['lon']},13z/data=!3m1!1e3", **_stretch())
    st.caption(
        "Plates: shaded relief of each 5 × 5 km box on the same 30 m grid, evenly lit (no cast shadows). "
        + ("Moon plate top = 0° longitude; " if polar else "Moon plate top = north; ") + "Earth plates top = north. "
        "Status: Valid when the site's value lies inside the target range."
    )

if best:
    with st.expander("🗺️ Where the two sites are", expanded=True):
        m = folium.Map(location=[20, 10], zoom_start=2, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Satellite",
        ).add_to(m)
        for source, r in best.items():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=9, color="#0f172a", weight=1.2, fill=True, fill_opacity=0.95,
                fill_color=SOURCES[source]["color"],
                tooltip=f"{source} · {r['place']} · {label} {r['category_score']:.1f}",
            ).add_to(m)
        st_folium(m, height=380, use_container_width=True, key=f"top_map_{target}_{group}", returned_objects=[])
        st.caption("Blue: Analog Suite site. Green: Discovery site.")

with st.expander("📋 All four domains", expanded=True):
    st.caption(f"Best site per domain for {target}.")
    overview = []
    for g in FEATURE_GROUPS:
        row = {"Domain": GROUP_LABELS[g]}
        for source in SOURCES:
            s = top[(top["target"] == target) & (top["category"] == g) & (top["source"] == source) & (top["rank"] == 1)]
            row[source] = s["place"].iloc[0] if len(s) else "–"
            row[f"{source} score"] = round(float(s["category_score"].iloc[0]), 1) if len(s) else None
        overview.append(row)
    st.dataframe(pd.DataFrame(overview), hide_index=True, **_stretch())

with st.expander("🥈 Next best sites", expanded=True):
    st.caption(f"{target} · {label}. Sites are at least 50 km apart.")
    for source in SOURCES:
        s = sel[sel["source"] == source].sort_values("rank")
        st.markdown(f"**{source}**")
        st.dataframe(
            s[["rank", "place", "lat", "lon", "category_score", "overall_score"] + feats]
            .rename(columns={"rank": "#", "category_score": f"{label} similarity", "overall_score": "Overall similarity",
                             **FEATURE_LABELS})
            .round(3),
            hide_index=True, **_stretch())

with st.expander("ℹ️ Method and limits", expanded=True):
    st.markdown(
        "- `discovery/top_by_category.py` runs the Analog Suite similarity engine once per target and domain, with "
        "that domain's weights only and the other three at 0: the same as moving three of the Analog Suite "
        "sliders to 0.\n"
        "- **Analog Suite** column: the cells of `earth_candidates.csv` that pass the false-analog filter. "
        "**Discovery** column: the worldwide Earth Engine scan after the same filter, minus known lunar analog "
        "sites (the pool behind the Discovery page's first ranking).\n"
        "- Each column is scored on its own cells, as on its own page, so compare a score within a column. "
        "Between the two columns, compare the feature values against the target.\n"
        "- A domain with one or two features is matched almost exactly by many cells (for volatiles, every cell "
        "whose moisture equals the target's). Sites whose domain score is equal to 0.1 point are ordered by "
        "overall similarity.\n"
        "- Earth's day–night temperature range is far below the lunar one, so thermal scores are low everywhere.\n\n"
        "**Re-run:** `python discovery/top_by_category.py` after changing `lunar_targets.json` or re-running the scan."
    )
