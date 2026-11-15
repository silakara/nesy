"""Terrain Twin panel (target DEM vs. an Earth site on the same 5 km / 30 m grid), shared by all pages."""

from __future__ import annotations

import html
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import rotate as nd_rotate

from terrain_twin import fetch_earth_dem, hillshade, load_target_dem, summarize


def _stretch() -> Dict[str, Any]:
    try:
        from packaging.version import Version
        if Version(st.__version__) >= Version("1.52"):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


@st.cache_data(show_spinner=False)
def cached_target_dem(target_name: str) -> Any:
    return load_target_dem(target_name)


@st.cache_data(show_spinner="Fetching Earth elevation tiles (AWS Terrain Tiles)...", ttl=24 * 3600)
def cached_earth_dem(lat: float, lon: float) -> Dict[str, Any]:
    return fetch_earth_dem(lat, lon)


def _mark(a: float, b: Optional[float], *, abs_tol: Optional[float] = None, rel_tol: Optional[float] = None) -> str:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "?"
    if abs_tol is not None:
        return "=" if abs(a - b) <= abs_tol else "≠"
    hi = max(abs(a), abs(b), 1e-9)
    return "=" if abs(a - b) / hi <= rel_tol else "≠"


def _tint(shade: np.ndarray, rgb: Tuple[float, float, float]) -> np.ndarray:
    return np.clip(np.stack([shade * c for c in rgb], axis=-1), 0, 255).astype(np.uint8)


def render_terrain_twin(row: pd.Series, target_name: str, body: str, key: str = "twin",
                        rotate_deg: float = 0.0, mirrored: bool = False) -> None:
    """rotate_deg / mirrored turn the Earth plate the way the shape matcher aligned it (display only)."""
    tdem = cached_target_dem(target_name)
    if tdem is None:
        st.info(f"No terrain model is bundled for {target_name}.")
        return

    # Two collapsible blocks that start open: plates + measures, then the 3D relief.
    with st.expander("🌙 Terrain Twin: Target vs. Selected Earth Site", expanded=True):
        st.caption(
            "Both 5 km × 5 km boxes are sampled on the same 30 m grid and measured with the same code "
            "(Horn slope, 8-neighbour roughness, cast-shadow hillshade). Ported from the NESY terrain module."
        )
        polar = abs(float(tdem["meta"]["target"].get("lat", 0))) > 80
        c1, c2 = st.columns(2)
        az = c1.slider("Sun azimuth (°, clockwise from plate top)", 0, 359, 135, 5, key=f"{key}_az")
        el = c2.slider(
            "Sun elevation (°)",
            1, 60, 6 if polar else 25, 1, key=f"{key}_el",
            help="At the lunar south pole the real sun stays around 1.5° above the horizon.",
        )

        lat, lon = float(row["lat"]), float(row["lon"])
        earth, err = None, None
        try:
            earth = cached_earth_dem(round(lat, 5), round(lon, 5))
        except Exception as exc:  # network / tile errors must not break the page
            err = str(exc)

        ts = summarize(tdem["z"])
        es = summarize(earth["z"]) if earth else None
        view = None
        if earth:
            view = earth["z"][:, ::-1] if mirrored else earth["z"]
            if rotate_deg:
                view = nd_rotate(view, rotate_deg, reshape=False, order=1, mode="nearest")
        aligned = bool(rotate_deg or mirrored)
        meta = tdem["meta"]
        tcolor = (1.0, 1.0, 1.0) if body == "Moon" else (1.0, 0.72, 0.52)

        left, right = st.columns(2)
        with left:
            st.markdown(f"**{target_name}** ({body})")
            st.image(_tint(hillshade(tdem["z"], az, el), tcolor), **_stretch())
            st.caption(
                f"{meta.get('source', '')} · {meta.get('product', '')} (native {meta.get('native', '?')}) · "
                f"{meta['target'].get('lat', 0):.2f}°, {meta['target'].get('lon', 0):.2f}°. "
                + ("Plate top = 0° longitude, not north." if polar else "Plate top = north.")
            )
        with right:
            st.markdown(f"**Earth: {html.escape(str(row['macro_region']))}** ({lat:.3f}°, {lon:.3f}°)")
            if earth:
                st.image(_tint(hillshade(view, az, el), (1.0, 0.86, 0.66)), **_stretch())
                orient = (f"rotated {rotate_deg:.0f}°{' and mirrored' if mirrored else ''} to align with the target"
                          if aligned else "plate top = north")
                st.caption(f"{earth['source']} · zoom {earth['zoom']} · {orient}.")
            else:
                st.warning(f"Earth elevation tiles could not be fetched: {err}")

        rows = [
            ("Typical slope (median, °)", ts["slope_median"], es and es["slope_median"], dict(abs_tol=3.0)),
            ("Steep parts (90th pct., °)", ts["slope_p90"], es and es["slope_p90"], dict(abs_tol=3.0)),
            ("Roughness (median step, m)", ts["roughness_median"], es and es["roughness_median"], dict(rel_tol=0.25)),
            ("Relief in box (m)", ts["relief_m"], es and es["relief_m"], dict(rel_tol=0.25)),
        ]
        table = pd.DataFrame(
            [
                {
                    "Measure": name,
                    target_name: round(t, 2),
                    "Earth site": None if e is None else round(e, 2),
                    "Match": _mark(t, e, **tol),
                }
                for name, t, e, tol in rows
            ]
        )
        tcol, hcol = st.columns([1, 1.25])
        with tcol:
            st.dataframe(table, hide_index=True, **_stretch())
            csv_slope = row.get("slope_deg", np.nan)
            st.caption(
                "`=` within tolerance (3° for slopes, 25% for roughness/relief), `≠` outside it, `?` not measured."
                + ("" if pd.isna(csv_slope) else f" Grid-cell CSV slope for this site: {csv_slope:.2f}°.")
            )
        with hcol:
            fig = go.Figure()
            bins = dict(start=0, end=46, size=1)
            fig.add_trace(go.Histogram(x=ts["slopes"], xbins=bins, histnorm="percent", name=target_name,
                                       marker_color="#e2e8f0" if body == "Moon" else "#fb923c", opacity=0.75))
            if es:
                fig.add_trace(go.Histogram(x=es["slopes"], xbins=bins, histnorm="percent", name="Earth site",
                                           marker_color="#38bdf8", opacity=0.6))
            fig.update_layout(
                barmode="overlay", template="plotly_dark", height=260,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.45)",
                xaxis_title="Slope (°)", yaxis_title="% of box", legend=dict(orientation="h", y=1.15),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, key=f"{key}_hist", **_stretch())

    with st.expander("🧊 3D Relief (same vertical scale for both)", expanded=True):
        zs = [tdem["z"]] + ([view] if earth else [])
        rel = max(float(np.nanmax(z) - np.nanmin(z)) for z in zs)
        cols = st.columns(len(zs))
        for col, z, name, scale in zip(cols, zs, [target_name, "Earth site"], ["Greys", "Earth"]):
            ax = np.arange(z.shape[0]) * 30.0
            f3 = go.Figure(go.Surface(z=z - np.nanmin(z), x=ax, y=ax[::-1], colorscale=scale, showscale=False))
            f3.update_layout(
                title=name, template="plotly_dark", height=380, margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(zaxis=dict(range=[0, rel], title="m"), xaxis_title="m", yaxis_title="m",
                           aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35)),
            )
            col.plotly_chart(f3, key=f"{key}_3d_{name}", **_stretch())
