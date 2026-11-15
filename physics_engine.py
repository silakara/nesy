"""
NESY physics engine (NASA Space Apps 2026)
====================================================

Ranks terrestrial analog candidates against lunar / Mars target zones and
quantifies where the Earth analogy breaks down.

Pipeline
--------
1. ``apply_false_analog_filter``   - reject salt playas, mobile dunes, vegetated/wet and non-barren cells.
2. ``compute_similarity_scores``   - weighted Euclidean + cosine hybrid score (0-100), sub-scores and macro-region tags.
3. ``calculate_transfer_functions``- gravity, vacuum-thermal and SPE-shielding engineering corrections.
4. ``generate_breakdown_matrix``   - structured "Analogy Works" vs "Analogy Breaks Down" report.

Inputs
------
* ``earth_candidates.csv``: lat, lon, elevation_m, slope_deg, tri_m, delta_t_k, mafic_index,
  clay_hydrated_index, subsurface_ice_moisture, ndvi, albedo, thermal_inertia_proxy, landcover_class
* ``lunar_targets.json``  : see that file for schema.

Usage
-----
    python physics_engine.py --target "Shackleton Ridge" --top 10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SIGMA_SB = 5.670374419e-8          # Stefan-Boltzmann constant [W m^-2 K^-4]
G_EARTH = 9.80665                  # Earth gravity [m s^-2]

DEFAULT_CANDIDATES_PATH = Path("earth_candidates.csv")
DEFAULT_TARGETS_PATH = Path("lunar_targets.json")

# Features used by the similarity engine (all present in earth_candidates.csv).
FEATURES: List[str] = [
    "slope_deg", "tri_m", "delta_t_k", "mafic_index",
    "clay_hydrated_index", "subsurface_ice_moisture", "albedo", "thermal_inertia_proxy",
]

# Sub-score groupings requested by the project spec.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "topography": ["slope_deg", "tri_m"],
    "thermal": ["delta_t_k", "thermal_inertia_proxy"],
    "mineralogy_isru": ["mafic_index", "clay_hydrated_index", "albedo"],
    "volatiles": ["subsurface_ice_moisture"],
}

# Default feature weights (relative; they are renormalised internally).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "slope_deg": 1.0, "tri_m": 1.0,
    "delta_t_k": 1.0, "thermal_inertia_proxy": 0.8,
    "mafic_index": 1.2, "clay_hydrated_index": 0.8, "albedo": 0.6,
    "subsurface_ice_moisture": 1.5,
}

# Landcover classes accepted as "barren". Supports ESA WorldCover (60 = bare / sparse vegetation),
# MODIS IGBP (16 = barren) and common text labels. Edit if your CSV uses another legend.
BARREN_LANDCOVER_CLASSES = {"60", "16", "barren", "bare", "sparse", "bare_sparse", "bare/sparse vegetation"}

# False-analog thresholds (from project spec).
PLAYA_ALBEDO_MIN, PLAYA_SLOPE_MAX, PLAYA_MAFIC_MAX = 0.38, 1.2, 1.05
DUNE_TRI_MAX, DUNE_MAFIC_MAX = 1.5, 1.0
NDVI_MAX, MOISTURE_MAX = 0.15, 0.18

# Physically possible value ranges (inclusive). Values outside come from sensor artefacts, e.g. Landsat band
# ratios with a near-zero or negative denominator (mafic_index -204 / 302, ndvi -2827 in the GEE world scan).
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "ndvi": (-1.0, 1.0),
    "albedo": (0.0, 1.0),
    "subsurface_ice_moisture": (0.0, 1.0),
    "mafic_index": (1e-6, 10.0),
    "clay_hydrated_index": (1e-6, 10.0),
}

# Scaling uses these candidate percentiles (plus the target range) so a handful of extreme cells cannot
# squeeze every other candidate into a tiny part of the [0, 1] scale.
SCALE_QUANTILES = (0.01, 0.99)

# Macro-region bounding boxes: (name, lat_min, lat_max, lon_min, lon_max). Coarse boxes - refine as needed.
MACRO_REGIONS: List[Tuple[str, float, float, float, float]] = [
    ("Anatolian Volcanic Belt", 36.5, 41.0, 30.0, 44.5),
    ("Harrat Khaybar", 24.5, 26.8, 38.5, 40.5),
    ("Beishan Gobi", 39.2, 42.5, 94.0, 101.5),
    ("Tibet Qaidam", 33.0, 39.2, 85.0, 99.0),
    ("Andean Altiplano", -27.0, -14.0, -71.0, -64.5),
]

# Terrestrial reference environment for transfer functions.
EARTH_ENV = {"mean_surface_temp_k": 288.0, "effective_sky_temp_k": 255.0, "convective_h_w_m2k": 10.0}

# Regolith / soil bulk-density proxy bounds [g cm^-3] driven by mafic_index (felsic -> basaltic).
DENSITY_FELSIC, DENSITY_MAFIC = 1.30, 1.90
MAFIC_FELSIC_REF, MAFIC_SPAN = 1.00, 0.60


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_targets(path: Path | str = DEFAULT_TARGETS_PATH) -> Dict[str, Any]:
    """Load ``lunar_targets.json`` and return the ``targets`` mapping (name -> target dict)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["targets"]


def load_candidates(path: Path | str = DEFAULT_CANDIDATES_PATH) -> pd.DataFrame:
    """Load ``earth_candidates.csv``; validates columns and coerces numerics."""
    df = pd.read_csv(path)
    required = ["lat", "lon", "elevation_m", *FEATURES, "ndvi", "landcover_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"earth_candidates.csv missing columns: {missing}")
    numeric = [c for c in required if c != "landcover_class"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def _get_target(target_name: str, targets: Optional[Mapping[str, Any]] = None,
                targets_path: Path | str = DEFAULT_TARGETS_PATH) -> Dict[str, Any]:
    targets = targets if targets is not None else load_targets(targets_path)
    if target_name not in targets:
        raise KeyError(f"Unknown target '{target_name}'. Available: {list(targets)}")
    return dict(targets[target_name])


# --------------------------------------------------------------------------- #
# 1. False-analog filter
# --------------------------------------------------------------------------- #
def _norm_class(value: Any) -> str:
    """Normalise a landcover label/code to a comparable lowercase string."""
    s = str(value).strip().lower()
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):  # OverflowError: +/-inf (e.g. Earth Engine no-data cells)
        return s


def apply_false_analog_filter(
    df: pd.DataFrame,
    barren_classes: Optional[set] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Reject terrestrial cells that mimic lunar/Mars terrain but differ in process.

    Rules (a cell may trip several; every rule is counted independently):
      * ``salt_playa_evaporite`` : albedo > 0.38 AND slope_deg < 1.2 AND mafic_index < 1.05
      * ``mobile_sand_dune``     : tri_m < 1.5 AND mafic_index < 1.0
      * ``vegetated_or_wet``     : ndvi > 0.15 OR subsurface_ice_moisture > 0.18
      * ``non_barren_landcover`` : landcover_class not in the barren set
      * ``invalid_value``        : a value outside ``VALID_RANGES`` or +/-inf (sensor artefact)
      * ``missing_data``         : NaN in any field needed by the rules

    Returns
    -------
    (filtered_df, summary) where summary holds per-rule counts plus ``total_input``,
    ``total_rejected_unique`` and ``total_retained``.
    """
    barren = {_norm_class(c) for c in (barren_classes or BARREN_LANDCOVER_CLASSES)}
    needed = ["albedo", "slope_deg", "tri_m", "mafic_index", "ndvi", "subsurface_ice_moisture", "landcover_class"]

    missing = df[needed].isna().any(axis=1)
    playa = (df["albedo"] > PLAYA_ALBEDO_MIN) & (df["slope_deg"] < PLAYA_SLOPE_MAX) & (df["mafic_index"] < PLAYA_MAFIC_MAX)
    dune = (df["tri_m"] < DUNE_TRI_MAX) & (df["mafic_index"] < DUNE_MAFIC_MAX)
    wet_veg = (df["ndvi"] > NDVI_MAX) | (df["subsurface_ice_moisture"] > MOISTURE_MAX)
    non_barren = ~df["landcover_class"].map(lambda v: (not pd.isna(v)) and _norm_class(v) in barren)

    invalid = pd.Series(False, index=df.index)
    for col in [c for c in FEATURES if c in df.columns]:
        invalid |= np.isinf(pd.to_numeric(df[col], errors="coerce"))
    for col, (vmin, vmax) in VALID_RANGES.items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            invalid |= (v < vmin) | (v > vmax)

    # Comparisons with NaN evaluate False, so missing rows are handled explicitly above.
    reject = missing | playa | dune | wet_veg | non_barren | invalid
    summary = {
        "total_input": int(len(df)),
        "salt_playa_evaporite": int(playa.sum()),
        "mobile_sand_dune": int(dune.sum()),
        "vegetated_or_wet": int(wet_veg.sum()),
        "non_barren_landcover": int((non_barren & ~df["landcover_class"].isna()).sum()),
        "invalid_value": int(invalid.sum()),
        "missing_data": int(missing.sum()),
        "total_rejected_unique": int(reject.sum()),
        "total_retained": int((~reject).sum()),
    }
    return df.loc[~reject].copy(), summary


# --------------------------------------------------------------------------- #
# 2. Similarity scoring + macro-region tagging
# --------------------------------------------------------------------------- #
def assign_macro_region(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Tag each lat/lon with a macro-region name (``'Other'`` if outside all boxes)."""
    lat_a, lon_a = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    conditions = [(lat_a >= la0) & (lat_a <= la1) & (lon_a >= lo0) & (lon_a <= lo1)
                  for _, la0, la1, lo0, lo1 in MACRO_REGIONS]
    names = [r[0] for r in MACRO_REGIONS]
    return pd.Series(np.select(conditions, names, default="Other"), index=lat.index)


def _normalised_weights(weights: Optional[Mapping[str, float]]) -> pd.Series:
    """Merge user weights over defaults, validate, and normalise to sum 1 over FEATURES."""
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    w = pd.Series({f: float(merged.get(f, 0.0)) for f in FEATURES})
    if (w < 0).any():
        raise ValueError("Weights must be non-negative.")
    if w.sum() == 0:
        raise ValueError("At least one weight must be > 0.")
    return w / w.sum()


def compute_similarity_scores(
    df: pd.DataFrame,
    target_name: str,
    weights_dict: Optional[Mapping[str, float]] = None,
    *,
    targets: Optional[Mapping[str, Any]] = None,
    targets_path: Path | str = DEFAULT_TARGETS_PATH,
    alpha: float = 0.6,
) -> pd.DataFrame:
    """
    Hybrid weighted-Euclidean / cosine similarity (0-100) against a target zone.

    Method
    ------
    * Features are min-max scaled with a ``MinMaxScaler`` fitted on candidates + target
      ideal/min/max so the target lies inside [0, 1] and units are comparable. Candidates are clipped to
      their ``SCALE_QUANTILES`` percentiles for the fit, so rare extreme cells do not compress the scale;
      values beyond the fitted range are clipped to 0 or 1.
    * Euclidean similarity: ``1 - sqrt(sum_i w_i (x_i - t_i)^2)`` (weights sum to 1 -> distance in [0, 1]).
    * Cosine similarity   : cosine of the sqrt(w)-weighted vectors centred at 0.5 (mid-range), mapped to [0, 1].
      Centering avoids the trivial ~1.0 cosine of all-positive scaled vectors.
    * Hybrid = ``alpha * Euclid + (1 - alpha) * Cosine``.
    * Sub-scores use the same weighted Euclidean form restricted to each feature group.

    Scores are relative to the candidate-set scaling; compare scores within one run.

    Returns
    -------
    Copy of ``df`` with: ``similarity_score``, ``score_euclidean``, ``score_cosine``,
    ``subscore_topography``, ``subscore_thermal``, ``subscore_mineralogy_isru``,
    ``subscore_volatiles`` and ``macro_region``; sorted by ``similarity_score`` descending.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    target = _get_target(target_name, targets, targets_path)
    params = target["parameters"]
    w = _normalised_weights(weights_dict)

    out = df.copy()
    X = out[FEATURES].astype(float)
    valid = X.notna().all(axis=1)

    ideal = pd.Series({f: params[f]["ideal"] for f in FEATURES})
    lo = pd.Series({f: params[f]["min"] for f in FEATURES})
    hi = pd.Series({f: params[f]["max"] for f in FEATURES})

    scaler = MinMaxScaler()
    Xfit = X[valid]
    if len(Xfit):
        Xfit = Xfit.clip(Xfit.quantile(SCALE_QUANTILES[0]), Xfit.quantile(SCALE_QUANTILES[1]), axis=1)
    scaler.fit(pd.concat([Xfit, ideal.to_frame().T, lo.to_frame().T, hi.to_frame().T], ignore_index=True))
    Xs = np.clip(scaler.transform(X[valid]), 0.0, 1.0) if valid.any() else np.empty((0, len(FEATURES)))
    ts = np.clip(scaler.transform(ideal.to_frame().T)[0], 0.0, 1.0)

    wv = w.to_numpy()
    diff = Xs - ts

    # Weighted Euclidean similarity
    sim_euc = 1.0 - np.sqrt((wv * diff ** 2).sum(axis=1))

    # Centred, weighted cosine similarity
    a = (Xs - 0.5) * np.sqrt(wv)
    b = (ts - 0.5) * np.sqrt(wv)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-12
    sim_cos = (((a @ b) / denom) + 1.0) / 2.0

    hybrid = alpha * sim_euc + (1.0 - alpha) * sim_cos

    out["score_euclidean"] = np.nan
    out["score_cosine"] = np.nan
    out["similarity_score"] = np.nan
    out.loc[valid, "score_euclidean"] = 100.0 * np.clip(sim_euc, 0, 1)
    out.loc[valid, "score_cosine"] = 100.0 * np.clip(sim_cos, 0, 1)
    out.loc[valid, "similarity_score"] = 100.0 * np.clip(hybrid, 0, 1)

    # Group sub-scores
    idx = {f: i for i, f in enumerate(FEATURES)}
    for group, cols in FEATURE_GROUPS.items():
        ci = [idx[c] for c in cols]
        wg = wv[ci]
        col = f"subscore_{group}"
        out[col] = np.nan
        if wg.sum() > 0 and valid.any():
            wg = wg / wg.sum()
            g = 1.0 - np.sqrt((wg * diff[:, ci] ** 2).sum(axis=1))
            out.loc[valid, col] = 100.0 * np.clip(g, 0, 1)

    out["macro_region"] = assign_macro_region(out["lat"], out["lon"])
    return out.sort_values("similarity_score", ascending=False, na_position="last")


# --------------------------------------------------------------------------- #
# 3. Transfer functions
# --------------------------------------------------------------------------- #
def _density_proxy(mafic_index: float) -> float:
    """Bulk density [g/cm^3] from mafic_index (felsic 1.3 -> basaltic 1.9). Mid value if NaN."""
    if mafic_index is None or (isinstance(mafic_index, float) and math.isnan(mafic_index)):
        frac = 0.5
    else:
        frac = min(max((float(mafic_index) - MAFIC_FELSIC_REF) / MAFIC_SPAN, 0.0), 1.0)
    return DENSITY_FELSIC + (DENSITY_MAFIC - DENSITY_FELSIC) * frac


def _bearing_capacity_kpa(phi_deg: float, c_kpa: float, rho_g_cm3: float, g: float,
                          width_m: float, depth_m: float) -> float:
    """Terzaghi/Vesic strip-footing ultimate bearing capacity [kPa]; gamma = rho * g [kN/m^3]."""
    phi = math.radians(phi_deg)
    gamma = rho_g_cm3 * g                       # t/m^3 * m/s^2 = kN/m^3
    nq = math.exp(math.pi * math.tan(phi)) * math.tan(math.pi / 4 + phi / 2) ** 2
    nc = (nq - 1.0) / math.tan(phi)
    ng = 2.0 * (nq + 1.0) * math.tan(phi)
    return c_kpa * nc + gamma * depth_m * nq + 0.5 * gamma * width_m * ng


def _infinite_slope_fs(phi_deg: float, c_kpa: float, rho_g_cm3: float, g: float,
                       slope_deg: float, depth_m: float = 1.0) -> float:
    """Infinite-slope factor of safety (capped at 99). Cohesion term grows in importance at low g."""
    beta = math.radians(slope_deg)
    if beta <= 1e-6:
        return 99.0
    gamma = rho_g_cm3 * g
    fs = math.tan(math.radians(phi_deg)) / math.tan(beta) + c_kpa / (gamma * depth_m * math.sin(beta) * math.cos(beta))
    return float(min(fs, 99.0))


def _net_thermal_flux(t_skin: float, t_ground: float, t_sky: float, h: float, t_air: Optional[float]) -> Dict[str, float]:
    """
    Net heat loss from an exterior skin [W/m^2] (positive = loss).
    Radiation: view factor 0.5 to sky + 0.5 to ground; convection only if an atmosphere exists (h > 0).
    """
    t_env4 = 0.5 * t_sky ** 4 + 0.5 * t_ground ** 4
    rad = SIGMA_SB * (t_skin ** 4 - t_env4)
    conv = h * (t_skin - t_air) if (h > 0 and t_air is not None) else 0.0
    return {"radiative": rad, "convective": conv, "total": rad + conv}


def calculate_transfer_functions(
    row: pd.Series,
    target_data: Mapping[str, Any],
    *,
    t_skin_k: float = 293.0,
    emissivity: float = 0.9,
    structure_areal_mass_kg_m2: float = 5000.0,
    footing_width_m: float = 2.0,
    footing_depth_m: float = 0.5,
) -> Dict[str, Any]:
    """
    Engineering scaling where the Earth analogy breaks down.

    (a) Gravity load scaling      : dead-load ratio g_t/g_E, Terzaghi bearing capacity & factor of safety
                                    for the *same* soil (density proxy from mafic_index), and infinite-slope FS.
    (b) Vacuum thermal correction : removes Earth convection; compares Stefan-Boltzmann radiative balance
                                    of a ``t_skin_k`` exterior skin on Earth vs the target, cold & hot phases.
    (c) SPE regolith shielding    : D(x) = D0 * exp(-x / lambda); thickness (cm) needed to reach the
                                    dose limit (default 50 mSv/yr) given local density proxy; any atmosphere
                                    column (Mars) is credited. GCR is flagged separately - it is not removable by thin cover.

    Parameters
    ----------
    row : candidate row (needs slope_deg, mafic_index, delta_t_k)
    target_data : one entry of ``lunar_targets.json['targets']``
    """
    env = target_data["environment"]
    params = target_data["parameters"]
    g_t = float(env["gravity_m_s2"])
    mafic = row.get("mafic_index", np.nan)
    slope = float(row.get("slope_deg", 0.0) if not pd.isna(row.get("slope_deg", np.nan)) else 0.0)
    rho = _density_proxy(mafic)
    phi, coh = float(env["friction_angle_deg"]), float(env["regolith_cohesion_kpa"])

    # ---- (a) Gravity load scaling ---------------------------------------- #
    q_e = _bearing_capacity_kpa(phi, coh, rho, G_EARTH, footing_width_m, footing_depth_m)
    q_t = _bearing_capacity_kpa(phi, coh, rho, g_t, footing_width_m, footing_depth_m)
    p_e = structure_areal_mass_kg_m2 * G_EARTH / 1000.0
    p_t = structure_areal_mass_kg_m2 * g_t / 1000.0
    allow_e = q_e / (3.0 * G_EARTH) * 1000.0     # allowable areal mass at FS = 3 [kg/m^2]
    allow_t = q_t / (3.0 * g_t) * 1000.0
    gravity = {
        "g_earth_m_s2": G_EARTH,
        "g_target_m_s2": g_t,
        "gravity_ratio": g_t / G_EARTH,
        "dead_load_ratio": g_t / G_EARTH,
        "density_proxy_g_cm3": rho,
        "bearing_capacity_earth_kpa": q_e,
        "bearing_capacity_target_kpa": q_t,
        "bearing_capacity_ratio": q_t / q_e,
        "applied_pressure_earth_kpa": p_e,
        "applied_pressure_target_kpa": p_t,
        "bearing_fs_earth": q_e / p_e,
        "bearing_fs_target": q_t / p_t,
        "allowable_areal_mass_earth_kg_m2": allow_e,
        "allowable_areal_mass_target_kg_m2": allow_t,
        "mass_allowance_multiplier": allow_t / allow_e,
        "slope_fs_earth": _infinite_slope_fs(phi, coh, rho, G_EARTH, slope),
        "slope_fs_target": _infinite_slope_fs(phi, coh, rho, g_t, slope),
    }

    # ---- (b) Vacuum / thin-atmosphere thermal correction ------------------ #
    dt_e = float(row.get("delta_t_k", np.nan)) if not pd.isna(row.get("delta_t_k", np.nan)) else params["delta_t_k"]["ideal"]
    dt_t = float(params["delta_t_k"]["ideal"])
    tm_e, tm_t = EARTH_ENV["mean_surface_temp_k"], float(env["mean_surface_temp_k"])
    phases: Dict[str, Any] = {}
    for phase, sign in (("cold", -0.5), ("hot", +0.5)):
        tg_e, tg_t = tm_e + sign * dt_e, tm_t + sign * dt_t
        earth = _net_thermal_flux(t_skin_k, tg_e, EARTH_ENV["effective_sky_temp_k"], EARTH_ENV["convective_h_w_m2k"], tg_e)
        targ = _net_thermal_flux(t_skin_k, tg_t, float(env["effective_sky_temp_k"]), float(env["convective_h_w_m2k"]), tg_t)
        phases[phase] = {
            "ground_temp_earth_k": tg_e,
            "ground_temp_target_k": tg_t,
            "net_loss_earth_w_m2": emissivity * earth["radiative"] + earth["convective"],
            "net_loss_target_w_m2": emissivity * targ["radiative"] + targ["convective"],
            "convective_loss_removed_w_m2": earth["convective"] - targ["convective"],
            "radiative_delta_w_m2": emissivity * (targ["radiative"] - earth["radiative"]),
        }
        phases[phase]["net_delta_w_m2"] = phases[phase]["net_loss_target_w_m2"] - phases[phase]["net_loss_earth_w_m2"]
    cold = phases["cold"]
    thermal = {
        "skin_temp_k": t_skin_k,
        "emissivity": emissivity,
        "cold_phase": phases["cold"],
        "hot_phase": phases["hot"],
        "heater_power_scaling": (cold["net_loss_target_w_m2"] / cold["net_loss_earth_w_m2"])
        if cold["net_loss_earth_w_m2"] > 0 else float("nan"),
        "note": "Solar/IR absorption and conduction into regolith are excluded; use as a first-order delta.",
    }

    # ---- (c) SPE regolith shielding thickness ----------------------------- #
    d0 = float(env["spe_design_dose_msv_yr"])
    limit = float(env.get("dose_limit_msv_yr", 50.0))
    lam = float(env["spe_attenuation_length_g_cm2"])
    atm = float(env.get("atmospheric_column_g_cm2", 0.0))
    required = lam * math.log(d0 / limit) if d0 > limit else 0.0
    net_areal = max(0.0, required - atm)
    gcr = float(params["radiation_flux_msv_yr"]["ideal"])
    shielding = {
        "spe_design_dose_msv_yr": d0,
        "dose_limit_msv_yr": limit,
        "required_areal_density_g_cm2": required,
        "atmosphere_credit_g_cm2": min(atm, required),
        "regolith_areal_density_g_cm2": net_areal,
        "local_density_g_cm3": rho,
        "required_thickness_cm": net_areal / rho,
        "required_mass_kg_m2": net_areal * 10.0,
        "gcr_dose_msv_yr_unshielded": gcr,
        "gcr_exceeds_limit": gcr > limit,
        "note": "Thickness addresses SPE only. GCR (hard, secondaries) is not reduced below limit by this cover; "
                "needs mission-duration limits, crew rotation and dose-budget planning.",
    }

    return {"gravity_load": gravity, "vacuum_thermal": thermal, "spe_shielding": shielding}


# --------------------------------------------------------------------------- #
# 4. Breakdown matrix
# --------------------------------------------------------------------------- #
def _range_verdict(value: float, spec: Mapping[str, float]) -> str:
    """'Valid' when the value lies inside the target range [min, max], otherwise 'Not valid'."""
    if pd.isna(value):
        return "Unknown"
    return "Valid" if spec["min"] <= value <= spec["max"] else "Not valid"


def generate_breakdown_matrix(row: pd.Series, target_data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Structured contrast of where the analogy is valid vs where it fails.

    Returns
    -------
    dict with ``where_analogy_works`` (per-feature target-range check: Valid / Not valid),
    ``where_analogy_breaks_down`` (limitation + quantified delta + engineering compensation),
    and a ``summary``.
    """
    params = target_data["parameters"]
    tf = calculate_transfer_functions(row, target_data)
    g, th, sh = tf["gravity_load"], tf["vacuum_thermal"], tf["spe_shielding"]

    works: Dict[str, Any] = {}
    notes = {
        "topography": "Slope/roughness govern trafficability, grading and foundation work independent of atmosphere.",
        "thermal": "Thermal inertia of granular substrate transfers; vacuum lowers real regolith conductivity (see breakdown).",
        "mineralogy_isru": "Basaltic/mafic chemistry and albedo transfer to ISRU (O2 extraction, sintering, feedstock).",
        "volatiles": "Moisture proxy transfers as a detection/extraction test-bed, not as chemistry or phase behaviour.",
    }
    for group, cols in FEATURE_GROUPS.items():
        detail = {}
        for c in cols:
            val = row.get(c, np.nan)
            detail[c] = {
                "candidate_value": None if pd.isna(val) else float(val),
                "target_range": [params[c]["min"], params[c]["max"]],
                "target_ideal": params[c]["ideal"],
                "verdict": _range_verdict(val, params[c]),
            }
        known = [d["verdict"] for d in detail.values() if d["verdict"] != "Unknown"]
        n_ok = sum(v == "Valid" for v in known)
        works[group] = {
            "features_valid": n_ok,
            "features_total": len(known),
            "verdict": "Unknown" if not known else ("Valid" if n_ok == len(known) else "Not valid"),
            "rationale": notes[group],
            "features": detail,
        }

    cold = th["cold_phase"]
    breaks = {
        "gravity": {
            "limitation": f"Dead load is {g['dead_load_ratio']:.2f}x Earth; regolith self-weight confinement drops while cohesion stays constant, "
                          f"so bearing capacity and slope stability do not scale linearly.",
            "quantified": {
                "bearing_capacity_ratio": round(g["bearing_capacity_ratio"], 3),
                "bearing_fs_earth_vs_target": [round(g["bearing_fs_earth"], 2), round(g["bearing_fs_target"], 2)],
                "slope_fs_earth_vs_target": [round(g["slope_fs_earth"], 2), round(g["slope_fs_target"], 2)],
                "mass_allowance_multiplier": round(g["mass_allowance_multiplier"], 2),
            },
            "engineering_compensation": "Validate foundations with 1/6g or 0.38g-scaled models (centrifuge/parabolic flight), re-derive cohesion-dominated "
                                        "capacity, use lightweight/inflatable structures with regolith ballast, and re-check slope cuts at target g.",
        },
        "vacuum_thermal": {
            "limitation": "Earth convection and atmospheric sky temperature are absent; heat exchange is radiation-only, with extreme diurnal swings "
                          "and ~14-day nights (Moon).",
            "quantified": {
                "cold_phase_loss_earth_w_m2": round(cold["net_loss_earth_w_m2"], 1),
                "cold_phase_loss_target_w_m2": round(cold["net_loss_target_w_m2"], 1),
                "convective_loss_removed_w_m2": round(cold["convective_loss_removed_w_m2"], 1),
                "radiative_delta_w_m2": round(cold["radiative_delta_w_m2"], 1),
                "heater_power_scaling": None if pd.isna(th["heater_power_scaling"]) else round(th["heater_power_scaling"], 3),
            },
            "engineering_compensation": "Multilayer insulation, variable-emissivity/louvre radiators, thermal storage (PCM), regolith berms, "
                                        "RHU/fission supply for night; test in thermal-vacuum chambers rather than field sites.",
        },
        "radiation": {
            "limitation": "Earth's magnetosphere and atmosphere (~1000 g/cm^2) shield SPE/GCR; surface sites see none (Moon) or ~16 g/cm^2 (Mars).",
            "quantified": {
                "spe_shield_thickness_cm": round(sh["required_thickness_cm"], 1),
                "spe_shield_mass_kg_m2": round(sh["required_mass_kg_m2"], 0),
                "gcr_dose_msv_yr_unshielded": sh["gcr_dose_msv_yr_unshielded"],
                "gcr_exceeds_limit": sh["gcr_exceeds_limit"],
            },
            "engineering_compensation": "Regolith overburden or water/polyethylene storm shelter for SPE; GCR managed via dose-budgeting, "
                                        "lava-tube/subsurface siting and mission-duration limits.",
        },
        "regolith_and_dust": {
            "limitation": "Lunar regolith is unweathered, sharp, agglutinate/glass-rich and electrostatically charged; Earth soils are chemically/biologically "
                          "weathered, rounded and moisture-bonded.",
            "quantified": {"clay_hydrated_index_candidate": None if pd.isna(row.get("clay_hydrated_index", np.nan)) else float(row["clay_hydrated_index"]),
                           "clay_hydrated_index_target": params["clay_hydrated_index"]["ideal"]},
            "engineering_compensation": "Use lunar simulants (e.g. LHS/JSC-1A class) for excavation, abrasion and sintering tests; add dust mitigation "
                                        "(EDS, seals, airlocks) not covered by field trials.",
        },
        "volatiles_and_chemistry": {
            "limitation": "Earth moisture proxy is liquid/adsorbed water in oxidising soils; lunar volatiles are cold-trapped ice with mixed species "
                          "(H2O, CO2, NH3, H2S) in permanently shadowed regions.",
            "quantified": {"ice_moisture_candidate": None if pd.isna(row.get("subsurface_ice_moisture", np.nan)) else float(row["subsurface_ice_moisture"]),
                           "ice_moisture_target": params["subsurface_ice_moisture"]["ideal"]},
            "engineering_compensation": "Treat as radar/neutron-detection and drilling test-bed only; qualify extraction in cryogenic vacuum chambers.",
        },
        "illumination_and_hazards": {
            "limitation": "Polar sites have grazing sun (1-2 deg), long shadows and PSR cold traps; no weather, but micrometeoroid flux and moonquakes.",
            "quantified": {},
            "engineering_compensation": "Model illumination with LOLA DEMs, plan power via elevated arrays/relay, design for micrometeoroid & seismic loads.",
        },
    }

    n_valid = sum(1 for v in works.values() if v["verdict"] == "Valid")
    return {
        "target": next((k for k in [target_data.get("region")] if k), None),
        "where_analogy_works": works,
        "where_analogy_breaks_down": breaks,
        "transfer_functions": tf,
        "summary": {
            "groups_valid": n_valid,
            "groups_total": len(works),
            "dominant_breakdowns": ["gravity", "vacuum_thermal", "radiation"],
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="NESY physics engine")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS_PATH))
    parser.add_argument("--target", default="Shackleton Ridge")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default="ranked_candidates.csv")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    targets = load_targets(args.targets)

    filtered, summary = apply_false_analog_filter(candidates)
    print("False-analog rejection summary:")
    for k, v in summary.items():
        print(f"  {k:<24}{v}")

    ranked = compute_similarity_scores(filtered, args.target, targets=targets)
    ranked.to_csv(args.out, index=False)
    cols = ["lat", "lon", "macro_region", "similarity_score",
            "subscore_topography", "subscore_thermal", "subscore_mineralogy_isru", "subscore_volatiles"]
    print(f"\nTop {args.top} analogs for {args.target}:")
    print(ranked[cols].head(args.top).round(1).to_string(index=False))

    if not ranked.empty:
        best = ranked.iloc[0]
        matrix = generate_breakdown_matrix(best, targets[args.target])
        print("\nBreakdown matrix (best candidate):")
        print(json.dumps(matrix, indent=2, default=float))


if __name__ == "__main__":
    main()
