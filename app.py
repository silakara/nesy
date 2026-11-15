

"""
NESY — Planetary Analog Discovery & Engineering Transfer Suite
NASA Space Apps Challenge 2026
=======================================================================

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from physics_engine import (
    DEFAULT_WEIGHTS,
    FEATURE_GROUPS,
    FEATURES,
    MACRO_REGIONS,
    apply_false_analog_filter,
    calculate_transfer_functions,
    compute_similarity_scores,
    generate_breakdown_matrix,
    load_candidates,
    load_targets,
)
from twin_ui import render_terrain_twin

# --------------------------------------------------------------------------- #
# Page Config & Custom Aerospace Styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="NESY | Planetary Analog Engineering Suite",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: rgba(28, 34, 48, 0.55);
        border: 1px solid rgba(56, 189, 248, 0.30);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .mission-banner {
        background: linear-gradient(90deg, rgba(15,23,42,0.94) 0%, rgba(30,58,138,0.72) 100%);
        border-left: 4px solid #38bdf8;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #f1f5f9;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "earth_candidates.csv"
JSON_PATH = BASE_DIR / "lunar_targets.json"

NAMED_REGIONS = [r[0] for r in MACRO_REGIONS]

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

GROUP_LABELS = {
    "topography": "Topography & Trafficability",
    "thermal": "Thermal Regime",
    "mineralogy_isru": "Mineralogy & ISRU",
    "volatiles": "Volatiles & Ice Proxy",
}

GROUP_USE_CASES = {
    "topography": "Rover mobility, autonomous hazard avoidance, lander leg stability & site grading trials",
    "thermal": "Multi-layer insulation (MLI), radiator sizing & extreme diurnal thermal cycling field tests",
    "mineralogy_isru": "Basaltic feedstock excavation, microwave/solar sintering & carbothermal O₂ extraction",
    "volatiles": "Ground-penetrating radar (GPR), neutron spectrometer calibration & cryogenic drilling trials",
}

TARGET_USE_CASES = {
    "Shackleton Ridge": "Lunar South Pole: near-continuous solar illumination + adjacent PSR water-ice ISRU",
    "Malapert Mountain": "Lunar South Pole: direct-to-Earth communications relay & elevated power hub",
    "Marius Hills": "Oceanus Procellarum: intact lava-tube skylight shelter (40 m basalt roof) & mare ISRU",
    "Jezero Crater Rim": "Mars Benchmark: hydrated phyllosilicate/carbonate sampling & Martian ISRU prototyping",
}

BREAK_LABELS = {
    "gravity": "Gravity Scaling (1g → Target g)",
    "vacuum_thermal": "Vacuum & Radiative Thermal Regime",
    "radiation": "Space Radiation (SPE & GCR Shielding)",
    "regolith_and_dust": "Regolith Mechanics & Electrostatic Dust",
    "volatiles_and_chemistry": "Volatile Phase & Cryogenic Chemistry",
    "illumination_and_hazards": "Solar Grazing Angle, Micrometeoroids & Quakes",
}

VERDICT_ICON = {
    "Valid": "✅ Valid",
    "Not valid": "❌ Not valid",
    "Unknown": "⚪ Unknown",
}


def _stretch() -> Dict[str, Any]:
    try:
        from packaging.version import Version
        if Version(st.__version__) >= Version("1.52"):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


# --------------------------------------------------------------------------- #
# Cached Pipeline
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading satellite candidate grid...")
def cached_candidates(path: str, mtime: float) -> pd.DataFrame:
    return load_candidates(path)


@st.cache_data(show_spinner=False)
def cached_targets(path: str, mtime: float) -> Dict[str, Any]:
    return load_targets(path)


@st.cache_data(show_spinner="Computing multi-criteria similarity & physics transfer...")
def run_pipeline(
    csv_path: str,
    csv_mtime: float,
    json_path: str,
    json_mtime: float,
    target_name: str,
    weights_items: Tuple[Tuple[str, float], ...],
    use_filter: bool,
) -> Tuple[Dict[str, int], pd.DataFrame]:
    df = cached_candidates(csv_path, csv_mtime)
    targets = cached_targets(json_path, json_mtime)
    filtered, summary = apply_false_analog_filter(df)
    base = filtered if use_filter else df
    scored = compute_similarity_scores(base, target_name, dict(weights_items), targets=targets)
    return summary, scored


# --------------------------------------------------------------------------- #
# Engineering & Geospatial Helpers
# --------------------------------------------------------------------------- #
def feature_weights_from_groups(group_w: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for group, feats in FEATURE_GROUPS.items():
        for f in feats:
            out[f] = float(group_w[group]) * float(DEFAULT_WEIGHTS[f])
    return out


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def thin_by_distance(df: pd.DataFrame, n: int, min_km: float) -> pd.DataFrame:
    if min_km <= 0:
        return df.head(n)
    keep: List[Any] = []
    kl: List[float] = []
    ko: List[float] = []
    for idx, lat, lon in zip(df.index[:5000], df["lat"].to_numpy()[:5000], df["lon"].to_numpy()[:5000]):
        if kl and haversine_km(lat, lon, np.array(kl), np.array(ko)).min() < min_km:
            continue
        keep.append(idx)
        kl.append(float(lat))
        ko.append(float(lon))
        if len(keep) >= n:
            break
    return df.loc[keep]


def select_pins(scored_valid: pd.DataFrame, top1: bool, n_pins: int, min_km: float) -> pd.DataFrame:
    if top1:
        named = scored_valid[scored_valid["macro_region"].isin(NAMED_REGIONS)]
        return named.groupby("macro_region", sort=False).head(1).sort_values("similarity_score", ascending=False)
    return thin_by_distance(scored_valid, n_pins, min_km)


def best_group(row: pd.Series) -> str:
    sub = {g: row.get(f"subscore_{g}", np.nan) for g in FEATURE_GROUPS}
    sub = {g: v for g, v in sub.items() if pd.notna(v)}
    return max(sub, key=sub.get) if sub else "topography"


def compute_field_logistics(row: pd.Series) -> Dict[str, Any]:
    """Derived aerospace engineering metrics for terrestrial field test campaign planning."""
    elev_m = float(row.get("elevation_m", 0.0))
    p_kpa = 101.325 * np.exp(-elev_m / 8434.5)
    rho_ratio = np.exp(-elev_m / 8434.5)

    mafic = float(row.get("mafic_index", 1.2))
    tri = float(row.get("tri_m", 5.0))
    sinter_temp_c = float(np.clip(1180.0 - 45.0 * (mafic - 1.0), 1040.0, 1220.0))
    excavation_kwh_t = float(np.clip(1.8 + 0.14 * tri, 1.5, 12.0))

    slope = float(row.get("slope_deg", 0.0))
    if slope <= 5.0 and tri <= 8.0:
        trafficability = "Nominal (Class-A Lander & Rover)"
    elif slope <= 12.0 and tri <= 20.0:
        trafficability = "Moderate (4WD Articulated Rover)"
    else:
        trafficability = "Challenging (Hazard-Avoidance Stress Test)"

    return {
        "elevation_m": elev_m,
        "ambient_pressure_kpa": p_kpa,
        "air_density_pct_sl": rho_ratio * 100.0,
        "sinter_temp_c": sinter_temp_c,
        "excavation_kwh_t": excavation_kwh_t,
        "trafficability_class": trafficability,
    }


def fmt(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if isinstance(v, (int, float, np.floating, np.integer)):
        return f"{v:.3g}"
    return str(v)


def worldview_url(lat: float, lon: float) -> str:
    return (
        "https://worldview.earthdata.nasa.gov/"
        f"?v={lon - 0.4:.4f},{lat - 0.3:.4f},{lon + 0.4:.4f},{lat + 0.3:.4f}"
        "&l=MODIS_Terra_CorrectedReflectance_TrueColor,Reference_Labels_15m,Reference_Features_15m,Coastlines_15m"
    )


def quickmap_url(lat: float, lon: float, half: float = 0.5) -> str:
    return (
        "https://quickmap.lroc.asu.edu/"
        f"?extent={lon - half:.4f},{lat - half:.4f},{lon + half:.4f},{lat + half:.4f}&proj=10"
    )


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
def build_map(shown: pd.DataFrame, selected_idx: Any, target_name: str, zoom_to_selected: bool = True) -> folium.Map:
    vmin, vmax = float(shown["similarity_score"].min()), float(shown["similarity_score"].max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    cmap = LinearColormap(
        ["#ef4444", "#f59e0b", "#eab308", "#84cc16", "#10b981"],
        vmin=vmin,
        vmax=vmax,
        caption="Hybrid Similarity Score (0–100)",
    )

    center_lat = float(shown.loc[selected_idx, "lat"]) if selected_idx in shown.index else float(shown["lat"].mean())
    center_lon = float(shown.loc[selected_idx, "lon"]) if selected_idx in shown.index else float(shown["lon"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8 if zoom_to_selected else 3,
        tiles=None,
        control_scale=True,
    )

    # 1. Standard Reference Map (OpenStreetMap)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Reference Map (OSM)",
        overlay=False,
        control=True,
    ).add_to(m)

    # 2. Topographic Relief Basemap (Keyless Esri Shaded Relief)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Shaded Relief",
        name="Topographic Relief (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    # 3. Default High-Res Satellite Layer (Added last so it opens by default!)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    # 4. NASA GIBS MODIS Terra True Color Overlay
    folium.TileLayer(
        tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2026-10-15/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
        attr="NASA GIBS / EOSDIS",
        name="NASA MODIS Terra True Color",
        overlay=True,
        control=True,
        opacity=0.65,
        show=False,
    ).add_to(m)

    target_use = TARGET_USE_CASES.get(target_name, "")
    for rank, (idx, row) in enumerate(shown.iterrows(), start=1):
        score = float(row["similarity_score"])
        use = GROUP_USE_CASES[best_group(row)]
        popup_html = (
            f"<div style='font-family:sans-serif;font-size:12.5px;min-width:250px;line-height:1.4;color:#0f172a'>"
            f"<b>#{rank} &middot; {html.escape(str(row['macro_region']))}</b><br>"
            f"Coordinates: <code>{row['lat']:.4f}°, {row['lon']:.4f}°</code><br>"
            f"Elevation: <b>{row.get('elevation_m', 0):.0f} m</b> | Slope: <b>{row['slope_deg']:.2f}°</b><br>"
            f"Similarity Score: <b>{score:.1f} / 100</b><br>"
            f"<hr style='margin:5px 0'>"
            f"<b>Primary Artemis Test Role:</b><br>{html.escape(use)}<br>"
            f"<span style='color:#475569;font-size:11px'><i>Target ({html.escape(target_name)}):</i> {html.escape(target_use)}</span></div>"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=8,
            color="#0f172a",
            weight=1.2,
            fill=True,
            fill_color=cmap(score),
            fill_opacity=0.92,
            tooltip=f"#{rank} · {row['macro_region']} · Score {score:.1f}",
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(m)

    if selected_idx in shown.index:
        sel = shown.loc[selected_idx]
        folium.CircleMarker(
            location=[float(sel["lat"]), float(sel["lon"])],
            radius=16,
            color="#38bdf8",
            weight=3,
            fill=False,
        ).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    if not zoom_to_selected and len(shown) > 1:
        m.fit_bounds(
            [
                [float(shown["lat"].min()), float(shown["lon"].min())],
                [float(shown["lat"].max()), float(shown["lon"].max())],
            ],
            padding=(35, 35),
        )
    return m


def build_radar(row: pd.Series, params: Dict[str, Any], pool: pd.DataFrame, target_name: str) -> go.Figure:
    labels, cand, ideal, lo_t, hi_t, raw_c, raw_i = [], [], [], [], [], [], []
    for f in FEATURES:
        p = params[f]
        lo = min(float(pool[f].quantile(0.01)), p["min"])
        hi = max(float(pool[f].quantile(0.99)), p["max"])
        span = max(hi - lo, 1e-12)

        def nz(x: float) -> float:
            return float(np.clip((x - lo) / span, 0.0, 1.0))

        labels.append(FEATURE_LABELS[f])
        cand.append(nz(float(row[f])))
        ideal.append(nz(p["ideal"]))
        lo_t.append(nz(p["min"]))
        hi_t.append(nz(p["max"]))
        raw_c.append(float(row[f]))
        raw_i.append(p["ideal"])

    theta = labels + [labels[0]]
    close = lambda v: v + [v[0]]  # noqa: E731
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=close(hi_t),
            theta=theta,
            name="Target Upper Bound",
            mode="lines",
            line=dict(color="rgba(148,163,184,0.7)", dash="dot", width=1),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=close(lo_t),
            theta=theta,
            name="Target Lower Bound",
            mode="lines",
            line=dict(color="rgba(148,163,184,0.7)", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=close(ideal),
            theta=theta,
            name=f"Orbital Ideal ({target_name})",
            fill="toself",
            line=dict(color="#f97316", width=2.2),
            fillcolor="rgba(249,115,22,0.20)",
            text=close([f"{l}: {v:.3g}" for l, v in zip(labels, raw_i)]),
            hoverinfo="text+name",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=close(cand),
            theta=theta,
            name="Selected Earth Analog",
            fill="toself",
            line=dict(color="#38bdf8", width=2.2),
            fillcolor="rgba(56,189,248,0.25)",
            text=close([f"{l}: {v:.3g}" for l, v in zip(labels, raw_c)]),
            hoverinfo="text+name",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(15,23,42,0.45)",
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        ),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=45, r=45, t=25, b=45),
        height=430,
    )
    return fig


def build_regional_comparison_chart(scored_valid: pd.DataFrame) -> go.Figure:
    named = scored_valid[scored_valid["macro_region"].isin(NAMED_REGIONS)]
    champs = named.groupby("macro_region", as_index=False).first().sort_values("similarity_score", ascending=False)
    melt_cols = [f"subscore_{g}" for g in FEATURE_GROUPS]
    melted = champs.melt(
        id_vars=["macro_region", "similarity_score"],
        value_vars=melt_cols,
        var_name="Domain",
        value_name="Subscore",
    )
    melted["Domain"] = melted["Domain"].str.replace("subscore_", "").map(GROUP_LABELS)
    fig = px.bar(
        melted,
        x="macro_region",
        y="Subscore",
        color="Domain",
        barmode="group",
        template="plotly_dark",
        color_discrete_sequence=["#38bdf8", "#f97316", "#10b981", "#a855f7"],
        labels={"macro_region": "Macro-Region Champion", "Subscore": "Domain Similarity Score (0–100)"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=20, r=20, t=20, b=40),
        height=340,
    )
    return fig


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🛰️ NESY")
    st.caption(
        "Multi-Agency Orbital Fusion, Terrestrial Analog Discovery & Engineering Scaling Suite · NASA Space Apps Challenge 2026"
    )

    missing = [p.name for p in (CSV_PATH, JSON_PATH, BASE_DIR / "physics_engine.py") if not p.exists()]
    if missing:
        st.error(f"Missing required files in working directory: {', '.join(missing)}")
        st.stop()

    targets = cached_targets(str(JSON_PATH), JSON_PATH.stat().st_mtime)

    # ---- Sidebar ---------------------------------------------------------- #
    sb = st.sidebar
    sb.header("Mission Parameters")
    target_name = sb.selectbox(
        "Planetary Target Zone",
        list(targets.keys()),
        help="Orbital baseline targets derived from NASA PDS / LRO (LOLA, Diviner, M³, Lunar Prospector) & Mars MOLA/CRISM.",
    )

    sb.markdown("**Domain Weighting Factors**")
    group_w = {g: sb.slider(GROUP_LABELS[g], 0.0, 3.0, 1.0, 0.1, key=f"w_{g}") for g in FEATURE_GROUPS}

    sb.divider()
    sb.markdown("**Screening & Spatial Controls**")
    use_filter = sb.toggle(
        "False-Analog Rejection Filter",
        value=True,
        help="Automatically rejects salt playas/evaporites, mobile sand dunes, vegetated/wet cells, and non-barren landcover.",
    )
    top1 = sb.toggle(
        "Macro-Region Champions (Top 1 / Basin)",
        value=False,
        help="Displays the single highest-scoring 5×5 km cell from each of the 5 geological macro-regions for trade-study comparison.",
    )
    zoom_to_selected = sb.toggle(
        "Terrain Close-Up Zoom (Selected Site)",
        value=True,
        help="Centers and zooms the satellite imagery directly onto the selected terrain patch.",
    )
    n_pins, min_km = 40, 25
    if not top1:
        n_pins = sb.slider("Map Candidate Pins", 5, 250, 40, 5)
        min_km = sb.slider(
            "Min. Spatial Separation (km)",
            0,
            200,
            25,
            5,
            help="Prevents adjacent 5×5 km grid cells from clustering into a single pin.",
        )

    if sum(group_w.values()) <= 0:
        sb.error("At least one domain weight must be greater than 0.0.")
        st.stop()

    weights = feature_weights_from_groups(group_w)
    summary, scored = run_pipeline(
        str(CSV_PATH),
        CSV_PATH.stat().st_mtime,
        str(JSON_PATH),
        JSON_PATH.stat().st_mtime,
        target_name,
        tuple(sorted(weights.items())),
        use_filter,
    )

    with sb.expander("False-Analog Filter Audit Log"):
        for k, v in summary.items():
            st.write(f"`{k}`: **{v:,}**")

    scored_valid = scored.dropna(subset=["similarity_score"])
    if scored_valid.empty:
        st.warning("No valid candidates remain under current filter settings.")
        st.stop()

    shown = select_pins(scored_valid, top1, n_pins, float(min_km))
    if shown.empty:
        st.warning("No candidates found in named macro-regions.")
        st.stop()

    # ---- Target Mission Banner -------------------------------------------- #
    target_data = targets[target_name]
    body = target_data.get("body", "Moon")
    st.markdown(
        f"<div class='mission-banner'><b>Active Target Baseline ({html.escape(body)} — {html.escape(target_name)}):</b> "
        f"{html.escape(TARGET_USE_CASES.get(target_name, ''))}</div>",
        unsafe_allow_html=True,
    )

    # ---- Top KPI Cards ---------------------------------------------------- #
    k1, k2, k3 = st.columns(3)
    k1.metric("Grid Cells Scanned (5×5 km)", f"{summary['total_input']:,}")
    k2.metric(
        "Valid Barren Analogs",
        f"{summary['total_retained']:,}",
        delta=f"-{summary['total_rejected_unique']:,} false analogs rejected",
        delta_color="off",
    )
    k3.metric("Peak Similarity Score", f"{scored_valid['similarity_score'].max():.1f}%")

    if not use_filter:
        st.warning("False-Analog Filter is OFF: scores include unscreened salt flats, dunes, and vegetated cells.")

    # ---- Candidate Selector + Map ----------------------------------------- #
    options = list(shown.index)
    label_map = {
        idx: f"#{rank} · {r['macro_region']} ({r['lat']:.3f}°, {r['lon']:.3f}°) — Score: {r['similarity_score']:.1f}%"
        for rank, (idx, r) in enumerate(shown.iterrows(), start=1)
    }
    prev = st.session_state.get("selected_id")
    current = prev if prev in options else options[0]
    sel_idx = st.selectbox(
        "Select Candidate Site for Engineering Deep-Dive (or click any map marker)",
        options,
        index=options.index(current),
        format_func=lambda i: label_map[i],
    )
    if sel_idx != prev:
        st.session_state["map_epoch"] = st.session_state.get("map_epoch", 0) + 1
        st.session_state["_last_click"] = None
    st.session_state["selected_id"] = sel_idx

    # Every section below is a collapsible block that starts open.
    with st.expander("🌍 Global Candidate Discovery Map", expanded=True):
        fmap = build_map(shown, sel_idx, target_name, zoom_to_selected=zoom_to_selected)
        map_data = st_folium(
            fmap,
            height=500,
            use_container_width=True,
            key=f"map_{st.session_state.get('map_epoch', 0)}",
            returned_objects=["last_object_clicked"],
        )
    click = (map_data or {}).get("last_object_clicked")
    if click and click.get("lat") is not None:
        ckey = (round(float(click["lat"]), 6), round(float(click["lng"]), 6))
        if ckey != st.session_state.get("_last_click"):
            st.session_state["_last_click"] = ckey
            d = (shown["lat"] - float(click["lat"])) ** 2 + (shown["lon"] - float(click["lng"])) ** 2
            if float(d.min()) < 1e-4 and d.idxmin() != sel_idx:
                st.session_state["selected_id"] = d.idxmin()
                st.rerun()

    row = scored.loc[sel_idx]
    render_terrain_twin(row, target_name, body)

    # ---- Candidate Engineering Deep-Dive ---------------------------------- #
    logistics = compute_field_logistics(row)

    with st.expander("🔬 Site Engineering Inspector & Transfer Scaling", expanded=True):
        sc = st.columns(5)
        sc[0].metric("Composite Similarity", f"{row['similarity_score']:.1f}%")
        for col, g in zip(sc[1:], FEATURE_GROUPS):
            col.metric(GROUP_LABELS[g].split(" &")[0], f"{row[f'subscore_{g}']:.1f}%")

    left, right = st.columns([1.08, 1])
    with left:
        with st.expander("(a) Multi-Sensor Physical Fingerprint (Earth vs. Orbital Target)", expanded=True):
            st.plotly_chart(build_radar(row, target_data["parameters"], scored_valid, target_name), **_stretch())
            st.caption(
                "Axes normalized across candidate pool (1st–99th percentile) and target bounds. Hover over vertices for raw physical values."
            )

    with right:
        with st.expander("(b) Engineering Transfer Functions & Field Logistics", expanded=True):
            tf = calculate_transfer_functions(row, target_data)
            g_label = "1/6g" if body == "Moon" else "0.38g"

            m1, m2 = st.columns(2)
            m1.metric(
                f"Bearing Capacity Ratio ({g_label})",
                f"{tf['gravity_load']['bearing_capacity_ratio']:.3f}×",
                help="Ratio of ultimate foundation bearing capacity q_target / q_Earth (Terzaghi/Vesic strip footing).",
            )
            m2.metric(
                "Vacuum Convective Loss Removed",
                f"{tf['vacuum_thermal']['cold_phase']['convective_loss_removed_w_m2']:.1f} W/m²",
                help="Atmospheric convective heat loss present on Earth that vanishes in lunar vacuum.",
            )

            m3, m4 = st.columns(2)
            m3.metric(
                "SPE Regolith Shield Thickness",
                f"{tf['spe_shielding']['required_thickness_cm']:.1f} cm",
                help="Local basaltic regolith overburden needed to attenuate Solar Particle Events below 50 mSv/yr.",
            )
            m4.metric(
                "Site Elevation & Air Density",
                f"{logistics['elevation_m']:.0f} m ({logistics['air_density_pct_sl']:.0f}% SL)",
                help="High-altitude sites naturally reduce ambient atmospheric density and convective noise.",
            )

            st.info(
                f"**ISRU & Mobility Profile:** {logistics['trafficability_class']} · "
                f"Est. Basalt Sintering Temp: **{logistics['sinter_temp_c']:.0f} °C** · "
                f"Excavation Specific Energy: **{logistics['excavation_kwh_t']:.1f} kWh/t**"
            )

            if tf["spe_shielding"]["gcr_exceeds_limit"]:
                st.warning(
                    f"Unshielded GCR dose (~{tf['spe_shielding']['gcr_dose_msv_yr_unshielded']:.0f} mSv/yr) exceeds the "
                    f"{tf['spe_shielding']['dose_limit_msv_yr']:.0f} mSv/yr limit; requires subsurface/lava-tube siting or crew rotation."
                )

            with st.popover("Inspect Raw Transfer Function JSON"):
                st.json(tf)

    # ---- Works vs. Breakdown Matrix --------------------------------------- #
    matrix = generate_breakdown_matrix(row, target_data)
    works = matrix["where_analogy_works"]
    breaks = matrix["where_analogy_breaks_down"]

    with st.expander("(c) Analogy Matrix: Where the Analogy Works vs. Where It Breaks Down", expanded=True):
        t1, t2, t3 = st.tabs(
            [
                "✅ Where the Analogy Works (Target Range Check)",
                "⚠️ Where the Analogy Breaks Down (Engineering Compensation)",
                "📊 Macro-Region Trade-Off Study",
            ]
        )

        with t1:
            rows = []
            for g, v in works.items():
                for feat, d in v["features"].items():
                    rows.append(
                        {
                            "Domain": GROUP_LABELS.get(g, g),
                            "Parameter": FEATURE_LABELS.get(feat, feat),
                            "Earth Candidate": d["candidate_value"],
                            "Target Range": f"{fmt(d['target_range'][0])} – {fmt(d['target_range'][1])}",
                            "Target Ideal": d["target_ideal"],
                            "Status": VERDICT_ICON.get(d["verdict"], d["verdict"]),
                        }
                    )
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                **_stretch(),
                column_config={
                    "Earth Candidate": st.column_config.NumberColumn(format="%.3g"),
                    "Target Ideal": st.column_config.NumberColumn(format="%.3g"),
                },
            )
            st.caption("Valid: the Earth value lies inside the target range (min – max). Not valid: it lies outside.")

        with t2:
            brows = [
                {
                    "Physical Domain": BREAK_LABELS.get(k, k),
                    "Why Earth Analogy Breaks Down": v["limitation"],
                    "Quantified Delta": ", ".join(f"{qk}: {fmt(qv)}" for qk, qv in v["quantified"].items()) or "—",
                    "Required Engineering Compensation": v["engineering_compensation"],
                }
                for k, v in breaks.items()
            ]
            st.table(pd.DataFrame(brows).set_index("Physical Domain"))

        with t3:
            st.markdown("Comparison of the highest-ranking 5×5 km analog site within each geological macro-region:")
            st.plotly_chart(build_regional_comparison_chart(scored_valid), **_stretch())
            named = scored_valid[scored_valid["macro_region"].isin(NAMED_REGIONS)]
            champs_table = (
                named.groupby("macro_region", as_index=False)
                .first()
                .sort_values("similarity_score", ascending=False)[
                    [
                        "macro_region",
                        "lat",
                        "lon",
                        "elevation_m",
                        "similarity_score",
                        "subscore_topography",
                        "subscore_thermal",
                        "subscore_mineralogy_isru",
                        "subscore_volatiles",
                    ]
                ]
            )
            st.dataframe(champs_table, hide_index=True, **_stretch())

            export_cols = [
                "macro_region",
                "lat",
                "lon",
                "elevation_m",
                "slope_deg",
                "tri_m",
                "delta_t_k",
                "mafic_index",
                "clay_hydrated_index",
                "subsurface_ice_moisture",
                "albedo",
                "similarity_score",
            ]
            avail_cols = [c for c in export_cols if c in shown.columns]
            csv_bytes = shown[avail_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Ranked Analog Candidates (CSV)",
                data=csv_bytes,
                file_name=f"nesy_{target_name.lower().replace(' ', '_')}_analogs.csv",
                mime="text/csv",
            )

    # ---- Cross-Agency Orbital Verification Links -------------------------- #
    with st.expander("(d) Cross-Agency Orbital Verification & Data Provenance", expanded=True):
        lat, lon = float(row["lat"]), float(row["lon"])
        l1, l2 = st.columns(2)
        with l1:
            st.link_button(
                f"🌍 Verify Earth Site in NASA Worldview ({lat:.3f}°, {lon:.3f}°)",
                worldview_url(lat, lon),
                **_stretch(),
            )
        with l2:
            if body == "Moon":
                st.link_button(
                    f"🌙 Inspect Lunar Twin in LRO QuickMap ({target_name})",
                    quickmap_url(float(target_data["center_lat"]), float(target_data["center_lon"])),
                    **_stretch(),
                )
            else:
                st.link_button("🔴 Inspect Martian Twin in NASA Mars Trek", "https://trek.nasa.gov/mars/", **_stretch())

    st.divider()
    st.caption(
        "Data Provenance: NASA PDS / LRO (LOLA, Diviner, M³, Lunar Prospector) · NASA Earthdata (NASADEM, MODIS MOD11A1, SMAP SPL4SMGP, Landsat 9) · "
        "ESA Copernicus (WorldCover v200 / Sentinel)."
    )


# Separate pages live in views/; the page menu sits at the top of the sidebar.
st.navigation([
    st.Page(main, title="Analog Suite", icon="🛰️", default=True),
    st.Page(str(BASE_DIR / "views" / "discovery.py"), title="Discovery", icon="🧭", url_path="discovery"),
    st.Page(str(BASE_DIR / "views" / "top.py"), title="Top", icon="🏆", url_path="top"),
]).run()