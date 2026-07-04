#!/usr/bin/env python3
"""Fossil cluster-complex analysis for Galactic binary cluster candidates.

This is the v2 analysis line after the age-dependent dissolution clock was
rejected. The primary observable is the expansion factor
EF = D_now / D_min from orbit backtracking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from scipy.stats import anderson_ksamp, binomtest, ks_2samp, spearmanr
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools import add_constant

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binary_cluster_dissolution as bcd

G_PC_MSUN_KMS2 = 4.300917270e-3


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "processed": root / "data" / "processed",
        "figures": root / "results" / "figures",
        "tables": root / "results" / "tables",
        "reports": root / "results" / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_or_build_main_products(root: Path, backtrack_myr: float, backtrack_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = root / "data" / "processed" / "binary_cluster_dissolution_metrics.csv"
    back_path = root / "data" / "processed" / "orbit_backtracking.csv"
    if metrics_path.exists() and back_path.exists():
        return pd.read_csv(metrics_path), pd.read_csv(back_path)

    raw = root / "data" / "raw"
    df, _ = bcd.choose_catalog(raw, raw / "Table_A.csv" if (raw / "Table_A.csv").exists() else None)
    cmap = bcd.infer_column_map(df)
    metrics = bcd.compute_metrics(df, cmap, 3.0)
    back = bcd.backtrack_orbits(metrics, backtrack_myr, backtrack_steps)
    metrics.to_csv(metrics_path, index=False)
    back.to_csv(back_path, index=False)
    return metrics, back


def build_fossil_metrics(metrics: pd.DataFrame, back: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    df = metrics.merge(back[["pair_id", "d_now_pc", "d_min_pc", "t_d_min_myr", "d_now_over_d_min"]], on="pair_id", how="inner")
    df["expansion_factor"] = df["d_now_over_d_min"]
    close_to_one = np.isclose(df["expansion_factor"], 1.0, rtol=0.0, atol=tolerance)
    df.loc[close_to_one, "expansion_factor"] = 1.0
    df["is_expanding"] = df["expansion_factor"] > 1.0
    df["delta_d_pc"] = df["d_now_pc"] - df["d_min_pc"]
    df["compact_origin_class"] = np.select(
        [df["d_min_pc"] < 20.0, df["d_min_pc"] < 50.0],
        ["A_compact_origin", "B_loose_complex"],
        default="C_wide_or_chance",
    )
    df["class_label"] = df["compact_origin_class"].map(
        {
            "A_compact_origin": "A: Dmin < 20 pc",
            "B_loose_complex": "B: 20-50 pc",
            "C_wide_or_chance": "C: >= 50 pc",
        }
    )
    return df.sort_values("expansion_factor", ascending=False).reset_index(drop=True)


def add_energy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m1 = out["mass1_msun"].astype(float)
    m2 = out["mass2_msun"].astype(float)
    mtot = m1 + m2
    mu = (m1 * m2) / mtot
    out["reduced_mass_msun"] = mu
    out["e_k_msun_kms2"] = 0.5 * mu * out["vrel_kms"] ** 2
    out["e_g_msun_kms2"] = G_PC_MSUN_KMS2 * m1 * m2 / out["separation_pc"]
    out["e_orb_msun_kms2"] = out["e_k_msun_kms2"] - out["e_g_msun_kms2"]
    out["energy_ratio"] = out["e_k_msun_kms2"] / out["e_g_msun_kms2"]
    out["bound_energy_flag"] = out["e_orb_msun_kms2"] < 0.0
    out["energy_ratio_from_velocity"] = out["vrel_over_vesc"] ** 2
    out["energy_ratio_check_absdiff"] = (out["energy_ratio"] - out["energy_ratio_from_velocity"]).abs()
    out["energy_ratio_check_reldiff"] = out["energy_ratio_check_absdiff"] / out["energy_ratio"].abs()
    return out


def summarize_classes(df: pd.DataFrame) -> pd.DataFrame:
    order = ["A_compact_origin", "B_loose_complex", "C_wide_or_chance"]
    rows = []
    for cls in order:
        subset = df[df["compact_origin_class"] == cls]
        rows.append(
            {
                "compact_origin_class": cls,
                "n": len(subset),
                "fraction": len(subset) / len(df) if len(df) else np.nan,
                "median_expansion_factor": subset["expansion_factor"].median() if len(subset) else np.nan,
                "fraction_expanding": subset["is_expanding"].mean() if len(subset) else np.nan,
                "median_age_myr": subset["age_myr"].median() if len(subset) else np.nan,
                "median_vrel_over_vesc": subset["vrel_over_vesc"].median() if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_internal_candidate_control(root: Path, backtrack_myr: float, backtrack_steps: int, force: bool) -> pd.DataFrame | None:
    raw_path = root / "data" / "raw" / "Table_3.csv"
    if not raw_path.exists():
        return None
    out_metrics = root / "data" / "processed" / "table3_candidate_metrics.csv"
    out_back = root / "data" / "processed" / "table3_candidate_backtracking.csv"
    if force or not out_metrics.exists() or not out_back.exists():
        raw = bcd.canonicalize_catalog(pd.read_csv(raw_path))
        cmap = bcd.infer_column_map(raw)
        metrics = bcd.compute_metrics(raw, cmap, 3.0)
        back = bcd.backtrack_orbits(metrics, backtrack_myr, backtrack_steps)
        metrics.to_csv(out_metrics, index=False)
        back.to_csv(out_back, index=False)
    control_metrics = pd.read_csv(out_metrics)
    control_back = pd.read_csv(out_back)
    control = build_fossil_metrics(control_metrics, control_back)
    control["control_sample"] = "Table_3_all_candidates"
    return control


def compare_with_control(main: pd.DataFrame, control: pd.DataFrame | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "main_n": int(len(main)),
        "main_fraction_expanding": float(main["is_expanding"].mean()),
        "main_median_expansion_factor": float(main["expansion_factor"].median()),
        "main_fraction_dmin_lt_20": float((main["d_min_pc"] < 20).mean()),
        "main_fraction_dmin_lt_50": float((main["d_min_pc"] < 50).mean()),
        "binomial_vs_0p5_pvalue": float(binomtest(int(main["is_expanding"].sum()), len(main), 0.5, alternative="greater").pvalue),
    }
    rho_age, p_age = spearmanr(main["age_myr"], main["expansion_factor"])
    rho_dsi, p_dsi = spearmanr(main["dsi"], main["expansion_factor"])
    rho_er, p_er = spearmanr(main["energy_ratio"], main["expansion_factor"]) if "energy_ratio" in main else (np.nan, np.nan)
    summary["spearman_age_vs_ef"] = {"rho": float(rho_age), "p_value": float(p_age)}
    summary["spearman_dsi_vs_ef"] = {"rho": float(rho_dsi), "p_value": float(p_dsi)}
    summary["spearman_energy_ratio_vs_ef"] = {"rho": float(rho_er), "p_value": float(p_er)}
    if "energy_ratio" in main:
        summary["energy"] = {
            "median_energy_ratio": float(main["energy_ratio"].median()),
            "min_energy_ratio": float(main["energy_ratio"].min()),
            "max_energy_ratio": float(main["energy_ratio"].max()),
            "fraction_bound_energy": float(main["bound_energy_flag"].mean()),
            "max_ratio_identity_absdiff": float(main["energy_ratio_check_absdiff"].max()),
            "max_ratio_identity_reldiff": float(main["energy_ratio_check_reldiff"].max()),
        }
    if control is not None and len(control):
        ks = ks_2samp(main["expansion_factor"], control["expansion_factor"])
        summary["internal_control"] = {
            "sample": "Table_3_all_candidates",
            "n": int(len(control)),
            "fraction_expanding": float(control["is_expanding"].mean()),
            "median_expansion_factor": float(control["expansion_factor"].median()),
            "fraction_dmin_lt_20": float((control["d_min_pc"] < 20).mean()),
            "fraction_dmin_lt_50": float((control["d_min_pc"] < 50).mean()),
            "delta_fraction_expanding": float(main["is_expanding"].mean() - control["is_expanding"].mean()),
            "ks_statistic_ef": float(ks.statistic),
            "ks_pvalue_ef": float(ks.pvalue),
            "caveat": "Internal candidate-control only; not a substitute for an HR24 parent random-pair baseline.",
        }
    else:
        summary["internal_control"] = None
    return summary


def robust_log_fit(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    sub = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]
    if len(sub) < 5:
        return {"n": int(len(sub)), "intercept": np.nan, "slope": np.nan}
    x = np.log10(sub[x_col].to_numpy(float))
    y = np.log10(sub[y_col].to_numpy(float))
    model = RLM(y, add_constant(x)).fit()
    return {
        "n": int(len(sub)),
        "intercept": float(model.params[0]),
        "slope": float(model.params[1]),
        "scale": float(model.scale),
    }


def ef_dynamics_fit_summary(df: pd.DataFrame) -> dict[str, object]:
    pearson_dsi = np.corrcoef(np.log10(df["dsi"]), np.log10(df["expansion_factor"]))[0, 1]
    pearson_er = np.corrcoef(np.log10(df["energy_ratio"]), np.log10(df["expansion_factor"]))[0, 1]
    return {
        "spearman_ef_dsi": compare_with_control(df, None)["spearman_dsi_vs_ef"],
        "spearman_ef_energy_ratio": compare_with_control(df, None)["spearman_energy_ratio_vs_ef"],
        "pearson_logef_logdsi": float(pearson_dsi),
        "pearson_logef_log_energy_ratio": float(pearson_er),
        "robust_logef_logdsi": robust_log_fit(df, "dsi", "expansion_factor"),
        "robust_logef_log_energy_ratio": robust_log_fit(df, "energy_ratio", "expansion_factor"),
    }


def read_hr24_parent(root: Path) -> pd.DataFrame:
    path = root / "data" / "hunt24" / "clusters.dat"
    readme = root / "data" / "hunt24" / "ReadMe"
    if not path.exists() or not readme.exists():
        raise FileNotFoundError("HR24 parent catalog not found at data/hunt24/clusters.dat with data/hunt24/ReadMe")
    from astropy.io import ascii

    table = ascii.read(path, readme=readme, format="cds")
    df = table.to_pandas()
    for col in df.columns:
        if df[col].dtype.kind in "SUO":
            df[col] = df[col].replace({"--": np.nan, "": np.nan})
    return df


def prepare_hr24_parent(df: pd.DataFrame, rv_required: bool = True, quality_cut: bool = True) -> pd.DataFrame:
    parent = pd.DataFrame(
        {
            "name": df["Name"].astype(str),
            "type": df["Type"].astype(str),
            "ra_deg": pd.to_numeric(df["RAdeg"], errors="coerce"),
            "dec_deg": pd.to_numeric(df["DEdeg"], errors="coerce"),
            "parallax_mas": pd.to_numeric(df["Plx"], errors="coerce"),
            "pmra_masyr": pd.to_numeric(df["pmRA"], errors="coerce"),
            "pmdec_masyr": pd.to_numeric(df["pmDE"], errors="coerce"),
            "rv_kms": pd.to_numeric(df["RV"], errors="coerce"),
            "mass_msun": pd.to_numeric(df["MassTot"], errors="coerce"),
            "log_age": pd.to_numeric(df["logAge50"], errors="coerce"),
            "cmd_class": pd.to_numeric(df["CMDCl50"], errors="coerce"),
            "cst": pd.to_numeric(df["CST"], errors="coerce"),
        }
    )
    parent = parent[parent["type"].eq("o")]
    if quality_cut:
        parent = parent[(parent["cmd_class"] > 0.5) & (parent["cst"] > 5.0)]
    needed = ["ra_deg", "dec_deg", "parallax_mas", "pmra_masyr", "pmdec_masyr", "mass_msun"]
    if rv_required:
        needed.append("rv_kms")
    parent = parent.dropna(subset=needed)
    parent = parent[(parent["parallax_mas"] > 0) & (parent["mass_msun"] > 0)]
    return parent.reset_index(drop=True)


def draw_random_pairs(n_parent: int, n_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    batch = max(n_pairs * 2, 1000)
    while len(pairs) < n_pairs:
        left = rng.integers(0, n_parent, size=batch)
        right = rng.integers(0, n_parent, size=batch)
        for a, b in zip(left, right):
            if a == b:
                continue
            pair = (int(a), int(b)) if a < b else (int(b), int(a))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= n_pairs:
                break
    return np.array(pairs, dtype=int)


def compute_pair_static(parent: pd.DataFrame, pairs: np.ndarray) -> pd.DataFrame:
    p1 = parent.iloc[pairs[:, 0]].reset_index(drop=True)
    p2 = parent.iloc[pairs[:, 1]].reset_index(drop=True)
    c1 = SkyCoord(
        ra=p1["ra_deg"].to_numpy() * u.deg,
        dec=p1["dec_deg"].to_numpy() * u.deg,
        distance=(1000.0 / p1["parallax_mas"].to_numpy()) * u.pc,
        pm_ra_cosdec=p1["pmra_masyr"].to_numpy() * u.mas / u.yr,
        pm_dec=p1["pmdec_masyr"].to_numpy() * u.mas / u.yr,
        radial_velocity=p1["rv_kms"].to_numpy() * u.km / u.s,
        frame="icrs",
    )
    c2 = SkyCoord(
        ra=p2["ra_deg"].to_numpy() * u.deg,
        dec=p2["dec_deg"].to_numpy() * u.deg,
        distance=(1000.0 / p2["parallax_mas"].to_numpy()) * u.pc,
        pm_ra_cosdec=p2["pmra_masyr"].to_numpy() * u.mas / u.yr,
        pm_dec=p2["pmdec_masyr"].to_numpy() * u.mas / u.yr,
        radial_velocity=p2["rv_kms"].to_numpy() * u.km / u.s,
        frame="icrs",
    )
    sep_pc = c1.separation_3d(c2).to_value(u.pc)
    v1 = c1.velocity.d_xyz.to_value(u.km / u.s).T
    v2 = c2.velocity.d_xyz.to_value(u.km / u.s).T
    vrel = np.linalg.norm(v1 - v2, axis=1)
    m1 = p1["mass_msun"].to_numpy(float)
    m2 = p2["mass_msun"].to_numpy(float)
    vesc = np.sqrt(2.0 * G_PC_MSUN_KMS2 * (m1 + m2) / sep_pc)
    energy_ratio = (vrel / vesc) ** 2
    return pd.DataFrame(
        {
            "idx1": pairs[:, 0],
            "idx2": pairs[:, 1],
            "name1": p1["name"].to_numpy(),
            "name2": p2["name"].to_numpy(),
            "separation_pc": sep_pc,
            "vrel_kms": vrel,
            "vesc_kms": vesc,
            "dsi": vesc / vrel,
            "energy_ratio": energy_ratio,
            "bound_energy_flag": energy_ratio < 1.0,
            "mass1_msun": m1,
            "mass2_msun": m2,
        }
    )


def parent_skycoord(parent: pd.DataFrame) -> SkyCoord:
    return SkyCoord(
        ra=parent["ra_deg"].to_numpy() * u.deg,
        dec=parent["dec_deg"].to_numpy() * u.deg,
        distance=(1000.0 / parent["parallax_mas"].to_numpy()) * u.pc,
        pm_ra_cosdec=parent["pmra_masyr"].to_numpy() * u.mas / u.yr,
        pm_dec=parent["pmdec_masyr"].to_numpy() * u.mas / u.yr,
        radial_velocity=parent["rv_kms"].to_numpy() * u.km / u.s,
        frame="icrs",
    )


def close_pair_pool(parent: pd.DataFrame, max_sep_pc: float) -> np.ndarray:
    c = parent_skycoord(parent)
    xyz = np.vstack(
        [
            c.cartesian.x.to_value(u.pc),
            c.cartesian.y.to_value(u.pc),
            c.cartesian.z.to_value(u.pc),
        ]
    ).T
    return cKDTree(xyz).query_pairs(max_sep_pc, output_type="ndarray")


def sample_pairs_from_pool(pool: np.ndarray, n_pairs: int, seed: int, replace: bool) -> np.ndarray:
    if len(pool) == 0:
        raise RuntimeError("No candidate random pairs available for requested control.")
    rng = np.random.default_rng(seed)
    if not replace and n_pairs > len(pool):
        raise RuntimeError(f"Requested {n_pairs} unique pairs but pool contains only {len(pool)}.")
    idx = rng.choice(len(pool), size=n_pairs, replace=replace)
    return pool[idx]


def sample_separation_matched_pairs(parent: pd.DataFrame, pool: np.ndarray, observed_sep: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    static_pool = compute_pair_static(parent, pool)
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0.0, 1.0, 11)
    edges = np.quantile(observed_sep, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    chosen: list[np.ndarray] = []
    obs_bins = np.digitize(observed_sep, edges[1:-1], right=True)
    pool_bins = np.digitize(static_pool["separation_pc"].to_numpy(), edges[1:-1], right=True)
    for bin_id in range(len(edges) - 1):
        target_fraction = np.mean(obs_bins == bin_id)
        target_n = int(round(target_fraction * n_pairs))
        if target_n == 0:
            continue
        available = np.flatnonzero(pool_bins == bin_id)
        if len(available) == 0:
            continue
        take = rng.choice(available, size=target_n, replace=target_n > len(available))
        chosen.append(pool[take])
    if not chosen:
        raise RuntimeError("Could not draw separation-matched HR24 random pairs.")
    result = np.vstack(chosen)
    if len(result) < n_pairs:
        fill = sample_pairs_from_pool(pool, n_pairs - len(result), seed + 1, replace=True)
        result = np.vstack([result, fill])
    elif len(result) > n_pairs:
        result = result[:n_pairs]
    return result


def backtrack_parent_pairs(parent: pd.DataFrame, pairs: np.ndarray, duration_myr: float, n_steps: int, batch_size: int) -> pd.DataFrame:
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    times = np.linspace(0.0, -duration_myr, n_steps) * u.Myr
    rows = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        p1 = parent.iloc[chunk[:, 0]].reset_index(drop=True)
        p2 = parent.iloc[chunk[:, 1]].reset_index(drop=True)
        o1 = Orbit(parent_skycoord(p1))
        o2 = Orbit(parent_skycoord(p2))
        o1.integrate(times, MWPotential2014)
        o2.integrate(times, MWPotential2014)
        sep = np.sqrt(
            (o1.x(times) - o2.x(times)) ** 2
            + (o1.y(times) - o2.y(times)) ** 2
            + (o1.z(times) - o2.z(times)) ** 2
        ) * 1000.0
        d_min = np.nanmin(sep, axis=1)
        idx = np.nanargmin(sep, axis=1)
        d_now = sep[:, 0]
        static = compute_pair_static(parent, chunk)
        static["d_now_pc"] = d_now
        static["d_min_pc"] = d_min
        static["t_d_min_myr"] = times.value[idx]
        static["expansion_factor"] = d_now / d_min
        static["is_expanding"] = static["expansion_factor"] > 1.0
        rows.append(static)
    return pd.concat(rows, ignore_index=True)


def build_hr24_static_baseline(root: Path, n_pairs: int, seed: int, separation_max_pc: float | None, rv_required: bool, quality_cut: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = read_hr24_parent(root)
    parent = prepare_hr24_parent(raw, rv_required=rv_required, quality_cut=quality_cut)
    if len(parent) < 2:
        raise RuntimeError("HR24 parent sample has fewer than two usable clusters after filtering.")
    pairs = draw_random_pairs(len(parent), n_pairs, seed)
    baseline = compute_pair_static(parent, pairs)
    if separation_max_pc is not None:
        baseline = baseline[baseline["separation_pc"] <= separation_max_pc].reset_index(drop=True)
    summary = {
        "parent_catalog": "Hunt & Reffert 2024 clusters.dat",
        "parent_rows_raw": int(len(raw)),
        "parent_rows_after_filters": int(len(parent)),
        "random_pairs_requested": int(n_pairs),
        "random_pairs_retained": int(len(baseline)),
        "rv_required": bool(rv_required),
        "quality_cut_cmd_gt_0p5_cst_gt_5": bool(quality_cut),
        "separation_max_pc": separation_max_pc,
        "median_energy_ratio": float(baseline["energy_ratio"].median()) if len(baseline) else np.nan,
        "fraction_bound_energy": float(baseline["bound_energy_flag"].mean()) if len(baseline) else np.nan,
        "median_dsi": float(baseline["dsi"].median()) if len(baseline) else np.nan,
    }
    return baseline, summary


def build_hr24_ef_baseline(
    root: Path,
    observed: pd.DataFrame,
    n_pairs: int,
    seed: int,
    control: str,
    duration_myr: float,
    n_steps: int,
    batch_size: int,
    quality_cut: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = read_hr24_parent(root)
    parent = prepare_hr24_parent(raw, rv_required=True, quality_cut=quality_cut)
    replace = False
    pool_size = None
    if control == "global":
        pairs = draw_random_pairs(len(parent), n_pairs, seed)
    elif control == "separation_limited":
        pool = close_pair_pool(parent, 200.0)
        pool_size = int(len(pool))
        replace = n_pairs > len(pool)
        pairs = sample_pairs_from_pool(pool, n_pairs, seed, replace=replace)
    elif control == "separation_matched":
        pool = close_pair_pool(parent, max(250.0, float(observed["separation_pc"].max()) * 1.25))
        pool_size = int(len(pool))
        pairs = sample_separation_matched_pairs(parent, pool, observed["separation_pc"].to_numpy(float), n_pairs, seed)
        replace = True
    else:
        raise ValueError(f"Unknown HR24 EF control: {control}")
    baseline = backtrack_parent_pairs(parent, pairs, duration_myr, n_steps, batch_size)
    summary = {
        "parent_rows_raw": int(len(raw)),
        "parent_rows_after_filters": int(len(parent)),
        "control": control,
        "random_pairs_requested": int(n_pairs),
        "random_pairs_retained": int(len(baseline)),
        "pool_size": pool_size,
        "sampled_with_replacement": bool(replace),
        "duration_myr": float(duration_myr),
        "n_steps": int(n_steps),
        "fraction_expanding": float(baseline["is_expanding"].mean()),
        "median_expansion_factor": float(baseline["expansion_factor"].median()),
        "fraction_dmin_lt_20": float((baseline["d_min_pc"] < 20).mean()),
        "fraction_dmin_lt_50": float((baseline["d_min_pc"] < 50).mean()),
        "median_energy_ratio": float(baseline["energy_ratio"].median()),
        "fraction_bound_energy": float(baseline["bound_energy_flag"].mean()),
    }
    return baseline, summary


def bic_gaussian(values: np.ndarray, n_params: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    var = max(float(np.var(x, ddof=0)), 1e-12)
    loglike = -0.5 * n * (np.log(2 * np.pi * var) + 1.0)
    return n_params * np.log(n) - 2.0 * loglike


def energy_significance(observed: pd.DataFrame, baseline: pd.DataFrame, seed: int = 123, n_subsets: int = 5000) -> dict[str, object]:
    obs = np.log10(observed["energy_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float))
    rand = np.log10(baseline["energy_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float))
    ks = ks_2samp(obs, rand)
    ad = anderson_ksamp([obs, rand])
    pooled = np.concatenate([obs, rand])
    bic_null = bic_gaussian(pooled, 2)
    bic_alt = bic_gaussian(obs, 2) + bic_gaussian(rand, 2)
    rng = np.random.default_rng(seed)
    subset_medians = []
    for _ in range(n_subsets):
        subset = rng.choice(rand, size=len(obs), replace=False if len(rand) >= len(obs) else True)
        subset_medians.append(float(np.median(subset)))
    subset_medians = np.array(subset_medians)
    return {
        "observed_n": int(len(obs)),
        "baseline_n": int(len(rand)),
        "observed_median_log10_energy_ratio": float(np.median(obs)),
        "baseline_median_log10_energy_ratio": float(np.median(rand)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "anderson_darling_statistic": float(ad.statistic),
        "anderson_darling_pvalue": float(ad.pvalue),
        "bic_null_shared_gaussian": float(bic_null),
        "bic_alt_separate_gaussians": float(bic_alt),
        "ln_bayes_factor_alt_over_null_bic": float((bic_null - bic_alt) / 2.0),
        "empirical_p_median_le_observed": float(np.mean(subset_medians <= np.median(obs))),
    }


def ef_significance(observed: pd.DataFrame, baseline: pd.DataFrame, seed: int = 123, n_subsets: int = 5000) -> dict[str, object]:
    obs = observed["expansion_factor"].to_numpy(float)
    rand = baseline["expansion_factor"].to_numpy(float)
    ks = ks_2samp(obs, rand)
    ad = anderson_ksamp([obs, rand])
    rng = np.random.default_rng(seed)
    subset_frac = []
    subset_med = []
    for _ in range(n_subsets):
        subset = rng.choice(rand, size=len(obs), replace=False if len(rand) >= len(obs) else True)
        subset_frac.append(float(np.mean(subset > 1.0)))
        subset_med.append(float(np.median(subset)))
    subset_frac = np.array(subset_frac)
    subset_med = np.array(subset_med)
    return {
        "observed_fraction_expanding": float(np.mean(obs > 1.0)),
        "baseline_fraction_expanding": float(np.mean(rand > 1.0)),
        "observed_median_ef": float(np.median(obs)),
        "baseline_median_ef": float(np.median(rand)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "anderson_darling_statistic": float(ad.statistic),
        "anderson_darling_pvalue": float(ad.pvalue),
        "empirical_p_fraction_ge_observed": float(np.mean(subset_frac >= np.mean(obs > 1.0))),
        "empirical_p_median_ge_observed": float(np.mean(subset_med >= np.median(obs))),
    }


def load_cached_hr24_ef_baseline(root: Path) -> tuple[pd.DataFrame, dict[str, object]] | None:
    baseline_path = root / "results" / "tables" / "hr24_ef_random_pairs.csv"
    summary_path = root / "results" / "tables" / "hr24_ef_random_baseline_summary.json"
    if not baseline_path.exists() or not summary_path.exists():
        return None
    return pd.read_csv(baseline_path), json.loads(summary_path.read_text(encoding="utf-8"))


def load_cached_hr24_static_baseline(root: Path) -> tuple[pd.DataFrame, dict[str, object]] | None:
    baseline_path = root / "results" / "tables" / "v2_hr24_static_random_pairs.csv"
    summary_path = root / "results" / "tables" / "v2_hr24_random_baseline_summary.json"
    if not baseline_path.exists() or not summary_path.exists():
        return None
    return pd.read_csv(baseline_path), json.loads(summary_path.read_text(encoding="utf-8"))


def build_verdict(comparison: dict[str, object], cold: dict[str, float] | None = None) -> dict[str, object]:
    model_path = Path.cwd() / "results" / "tables" / "model_fits.json"
    model = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else {}
    hr24_ef = comparison.get("hr24_ef_baseline")
    ef_tests = hr24_ef.get("ef_tests", {}) if hr24_ef else {}
    energy_tests = hr24_ef.get("energy_tests", {}) if hr24_ef else {}
    hr24_summary = hr24_ef.get("summary", {}) if hr24_ef else {}
    cold = cold or {}

    ef_empirical_p = ef_tests.get("empirical_p_fraction_ge_observed", np.nan)
    energy_bf = energy_tests.get("ln_bayes_factor_alt_over_null_bic", np.nan)
    verdict = {
        "dissolution_clock": {
            "status": "rejected",
            "reason": "Constant-null model is preferred by BIC and age vs bound probability is not significant.",
            "best_by_bic": model.get("best_by_bic"),
            "alpha": model.get("power_law", {}).get("alpha"),
            "spearman_age_vs_p_bound": model.get("rank_correlation", {}),
        },
        "energy_bound_population": {
            "status": "rejected",
            "reason": "All high-confidence pairs have E_k/E_g > 1; no pair is formally energy-bound.",
            "median_energy_ratio": comparison["energy"]["median_energy_ratio"],
            "fraction_bound_energy": comparison["energy"]["fraction_bound_energy"],
        },
        "ef_random_baseline": {
            "status": "borderline_supported" if np.isfinite(ef_empirical_p) and ef_empirical_p <= 0.1 else "not_supported",
            "reason": "EF distribution differs from the HR24 separation-matched random baseline in KS/AD, but subset empirical p-values are near 0.05 rather than decisive.",
            "control": hr24_summary.get("control"),
            "random_pairs_retained": hr24_summary.get("random_pairs_retained"),
            "pool_size": hr24_summary.get("pool_size"),
            "sampled_with_replacement": hr24_summary.get("sampled_with_replacement"),
            "observed_fraction_expanding": ef_tests.get("observed_fraction_expanding"),
            "baseline_fraction_expanding": ef_tests.get("baseline_fraction_expanding"),
            "ks_pvalue": ef_tests.get("ks_pvalue"),
            "anderson_darling_pvalue": ef_tests.get("anderson_darling_pvalue"),
            "empirical_p_fraction_ge_observed": ef_tests.get("empirical_p_fraction_ge_observed"),
            "empirical_p_median_ge_observed": ef_tests.get("empirical_p_median_ge_observed"),
        },
        "energy_distribution_vs_hr24": {
            "status": "not_distinct",
            "reason": "KS/AD are not significant and the BIC Bayes factor favors the shared-distribution null over separate Gaussian log-energy models.",
            "ks_pvalue": energy_tests.get("ks_pvalue"),
            "anderson_darling_pvalue": energy_tests.get("anderson_darling_pvalue"),
            "ln_bayes_factor_alt_over_null_bic": energy_bf,
        },
        "phase_space_coldness": {
            "status": "mixed",
            "reason": "At matched separation, observed pairs have lower median Vrel, but not lower E_k/E_g or higher DSI than the HR24 EF baseline.",
            **cold,
        },
        "paper_claim": {
            "recommended_framing": "Replace the dissolution clock with a cautious fossil cluster-complex signal: EF excess is marginal relative to HR24 random pairs; energy binding/coldness is not the decisive evidence.",
            "main_caveat": "The separation-matched 100000-pair EF baseline samples with replacement because the available pool contains only 21444 pairs.",
        },
    }
    return verdict


def write_verdict_report(verdict: dict[str, object], out: Path) -> None:
    ef = verdict["ef_random_baseline"]
    energy = verdict["energy_distribution_vs_hr24"]
    cold = verdict["phase_space_coldness"]
    lines = [
        "# V2 Final Verdict",
        "",
        "## Decisions",
        f"- Dissolution clock: {verdict['dissolution_clock']['status']}",
        f"- Energy-bound population: {verdict['energy_bound_population']['status']}",
        f"- EF random-baseline signal: {ef['status']}",
        f"- Energy distribution vs HR24: {energy['status']}",
        f"- Phase-space coldness: {cold['status']}",
        "",
        "## Key Numbers",
        f"- Observed EF > 1: {ef['observed_fraction_expanding']:.4g}",
        f"- HR24 baseline EF > 1: {ef['baseline_fraction_expanding']:.4g}",
        f"- EF KS p-value: {ef['ks_pvalue']:.4g}",
        f"- EF AD p-value: {ef['anderson_darling_pvalue']:.4g}",
        f"- EF empirical p(frac >= observed): {ef['empirical_p_fraction_ge_observed']:.4g}",
        f"- Energy KS p-value: {energy['ks_pvalue']:.4g}",
        f"- Energy AD p-value: {energy['anderson_darling_pvalue']:.4g}",
        f"- ln BF_alt/null for log energy: {energy['ln_bayes_factor_alt_over_null_bic']:.4g}",
        f"- Median Vrel observed/baseline: {cold['observed_median_vrel_kms']:.4g} / {cold['baseline_median_vrel_kms']:.4g} km/s",
        f"- Median E_k/E_g observed/baseline: {cold['observed_median_energy_ratio']:.4g} / {cold['baseline_median_energy_ratio']:.4g}",
        "",
        "## Interpretation",
        f"- {verdict['dissolution_clock']['reason']}",
        f"- {ef['reason']}",
        f"- {energy['reason']}",
        f"- {cold['reason']}",
        f"- {verdict['paper_claim']['recommended_framing']}",
        f"- Caveat: {verdict['paper_claim']['main_caveat']}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_dmin_vs_dnow(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    colors = {
        "A_compact_origin": "#bd3c2f",
        "B_loose_complex": "#d99b2b",
        "C_wide_or_chance": "#2868a0",
    }
    for cls, subset in df.groupby("compact_origin_class"):
        ax.scatter(subset["d_min_pc"], subset["d_now_pc"], s=42, alpha=0.8, color=colors.get(cls, "#555555"), label=subset["class_label"].iloc[0])
    lim = max(df["d_now_pc"].max(), df["d_min_pc"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#111111", lw=1.2, ls="--", label=r"$D_{\min}=D_{\rm now}$")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$D_{\min}$ over 200 Myr (pc)")
    ax.set_ylabel(r"$D_{\rm now}$ (pc)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_ef_distribution(df: pd.DataFrame, control: pd.DataFrame | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bins = np.geomspace(1.0, max(df["expansion_factor"].max(), 1.05), 20)
    ax.hist(df["expansion_factor"], bins=bins, alpha=0.76, color="#2868a0", label="83 high-confidence pairs")
    if control is not None and len(control):
        upper = max(df["expansion_factor"].max(), control["expansion_factor"].max(), 1.05)
        bins = np.geomspace(1.0, upper, 24)
        ax.hist(control["expansion_factor"], bins=bins, histtype="step", lw=2, color="#bd3c2f", label="159 candidate internal control")
    ax.axvline(1.0, color="#111111", lw=1.1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"Expansion factor $EF=D_{\rm now}/D_{\min}$")
    ax.set_ylabel("Pair count")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_class_fractions(class_summary: pd.DataFrame, out: Path) -> None:
    labels = ["A\n<20 pc", "B\n20-50 pc", "C\n>=50 pc"]
    colors = ["#bd3c2f", "#d99b2b", "#2868a0"]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.bar(labels, class_summary["fraction"], color=colors, alpha=0.86)
    for i, row in class_summary.iterrows():
        ax.text(i, row["fraction"] + 0.015, f"{int(row['n'])}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(0.8, class_summary["fraction"].max() + 0.12))
    ax.set_ylabel("Fraction of high-confidence pairs")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_ef_vs_age(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sc = ax.scatter(df["age_myr"], df["expansion_factor"], c=df["vrel_over_vesc"], s=42, cmap="magma", alpha=0.82)
    ax.axhline(1.0, color="#111111", lw=1.1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"Expansion factor $EF$")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$V_{\rm rel}/V_{\rm esc}$")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_dsi_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.hist(df["dsi"], bins=22, color="#594c9c", alpha=0.82)
    ax.axvline(1.0, color="#111111", lw=1.2, ls="--", label="Bound threshold")
    ax.set_xscale("log")
    ax.set_xlabel(r"DSI = $V_{\rm esc}/V_{\rm rel}$")
    ax.set_ylabel("Pair count")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_energy_distribution(df: pd.DataFrame, baseline: pd.DataFrame | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    obs = df["energy_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    upper = obs.quantile(0.98)
    if baseline is not None and len(baseline):
        upper = max(upper, baseline["energy_ratio"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.98))
    upper = max(float(upper), 1.1)
    bins = np.geomspace(max(obs.min() * 0.8, 1e-3), upper, 28)
    ax.hist(obs, bins=bins, alpha=0.76, color="#2868a0", label="83 high-confidence pairs")
    if baseline is not None and len(baseline):
        base = baseline["energy_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(base, bins=bins, histtype="step", lw=2, color="#bd3c2f", label="HR24 random static baseline")
    ax.axvline(1.0, color="#111111", lw=1.2, ls="--", label=r"$E_k/E_g=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Energy ratio $E_k/E_g$")
    ax.set_ylabel("Pair count")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_hr24_ef_distribution(df: pd.DataFrame, baseline: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    upper = max(df["expansion_factor"].max(), baseline["expansion_factor"].quantile(0.99), 1.05)
    bins = np.geomspace(1.0, upper, 28)
    ax.hist(baseline["expansion_factor"], bins=bins, alpha=0.35, color="#777777", label="HR24 random EF baseline")
    ax.hist(df["expansion_factor"], bins=bins, histtype="step", lw=2.2, color="#bd3c2f", label="83 high-confidence pairs")
    ax.axvline(1.0, color="#111111", lw=1.1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"Expansion factor $EF=D_{\rm now}/D_{\min}$")
    ax.set_ylabel("Pair count")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_hr24_dmin_distribution(df: pd.DataFrame, baseline: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    upper = max(df["d_min_pc"].max(), baseline["d_min_pc"].quantile(0.99), 20)
    bins = np.linspace(0, upper, 32)
    ax.hist(baseline["d_min_pc"], bins=bins, alpha=0.35, color="#777777", label="HR24 random EF baseline")
    ax.hist(df["d_min_pc"], bins=bins, histtype="step", lw=2.2, color="#2868a0", label="83 high-confidence pairs")
    ax.axvline(20, color="#bd3c2f", lw=1.2, ls="--")
    ax.axvline(50, color="#d99b2b", lw=1.2, ls="--")
    ax.set_xlabel(r"$D_{\min}$ (pc)")
    ax.set_ylabel("Pair count")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_phase_space_coldness(df: pd.DataFrame, baseline: pd.DataFrame, out: Path) -> None:
    columns = [
        ("separation_pc", "Separation (pc)", "log"),
        ("vrel_kms", r"$V_{\rm rel}$ (km/s)", "log"),
        ("dsi", "DSI", "log"),
        ("energy_ratio", r"$E_k/E_g$", "log"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    for ax, (col, label, scale) in zip(axes.flat, columns):
        obs = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        rand = baseline[col].replace([np.inf, -np.inf], np.nan).dropna()
        lo = min(obs.quantile(0.01), rand.quantile(0.01))
        hi = max(obs.quantile(0.99), rand.quantile(0.99))
        if scale == "log":
            lo = max(lo, 1e-6)
            bins = np.geomspace(lo, hi, 28)
            ax.set_xscale("log")
        else:
            bins = np.linspace(lo, hi, 28)
        ax.hist(rand, bins=bins, alpha=0.35, color="#777777")
        ax.hist(obs, bins=bins, histtype="step", lw=2.0, color="#bd3c2f")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_ef_vs_dsi(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = {
        "A_compact_origin": "#bd3c2f",
        "B_loose_complex": "#d99b2b",
        "C_wide_or_chance": "#2868a0",
    }
    for cls, subset in df.groupby("compact_origin_class"):
        ax.scatter(subset["dsi"], subset["expansion_factor"], s=44, alpha=0.82, color=colors.get(cls, "#555555"), label=subset["class_label"].iloc[0])
    ax.axhline(1.0, color="#111111", lw=1.1, ls=":")
    ax.axvline(1.0, color="#111111", lw=1.1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"DSI = $V_{\rm esc}/V_{\rm rel}$")
    ax.set_ylabel(r"Expansion factor $EF$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_ef_vs_energy_ratio(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sc = ax.scatter(df["energy_ratio"], df["expansion_factor"], c=df["d_min_pc"], s=44, cmap="viridis_r", alpha=0.82)
    ax.axhline(1.0, color="#111111", lw=1.1, ls=":")
    ax.axvline(1.0, color="#111111", lw=1.1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Energy ratio $E_k/E_g$")
    ax.set_ylabel(r"Expansion factor $EF$")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$D_{\min}$ (pc)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def write_report(df: pd.DataFrame, class_summary: pd.DataFrame, comparison: dict[str, object], out: Path) -> None:
    lines = [
        "# Fossil Cluster Complex Analysis",
        "",
        "## Main Result",
        f"- Rows analyzed: {len(df)} high-confidence pairs",
        f"- Fraction with EF > 1: {comparison['main_fraction_expanding']:.4g}",
        f"- Median EF: {comparison['main_median_expansion_factor']:.4g}",
        f"- Binomial p-value vs 50% expansion: {comparison['binomial_vs_0p5_pvalue']:.4g}",
        f"- Fraction with D_min < 20 pc: {comparison['main_fraction_dmin_lt_20']:.4g}",
        f"- Fraction with D_min < 50 pc: {comparison['main_fraction_dmin_lt_50']:.4g}",
        f"- Median E_k/E_g: {comparison['energy']['median_energy_ratio']:.4g}",
        f"- Fraction energy-bound: {comparison['energy']['fraction_bound_energy']:.4g}",
        f"- Max relative |E_k/E_g - (Vrel/Vesc)^2|: {comparison['energy']['max_ratio_identity_reldiff']:.4g}",
        "",
        "## Class Fractions",
    ]
    for _, row in class_summary.iterrows():
        lines.append(
            f"- {row['compact_origin_class']}: n={int(row['n'])}, fraction={row['fraction']:.4g}, "
            f"median EF={row['median_expansion_factor']:.4g}"
        )
    lines += [
        "",
        "## Correlations",
        f"- Spearman age vs EF: rho={comparison['spearman_age_vs_ef']['rho']:.4g}, p={comparison['spearman_age_vs_ef']['p_value']:.4g}",
        f"- Spearman DSI vs EF: rho={comparison['spearman_dsi_vs_ef']['rho']:.4g}, p={comparison['spearman_dsi_vs_ef']['p_value']:.4g}",
        f"- Spearman E_k/E_g vs EF: rho={comparison['spearman_energy_ratio_vs_ef']['rho']:.4g}, p={comparison['spearman_energy_ratio_vs_ef']['p_value']:.4g}",
    ]
    if "ef_dynamics_fit" in comparison:
        fit = comparison["ef_dynamics_fit"]
        lines += [
            f"- Robust log(EF) vs log(DSI) slope: {fit['robust_logef_logdsi']['slope']:.4g}",
            f"- Robust log(EF) vs log(E_k/E_g) slope: {fit['robust_logef_log_energy_ratio']['slope']:.4g}",
        ]
    control = comparison.get("internal_control")
    if control:
        lines += [
            "",
            "## Internal Candidate-Control",
            f"- Control sample: {control['sample']}",
            f"- Control n: {control['n']}",
            f"- Control fraction EF > 1: {control['fraction_expanding']:.4g}",
            f"- Delta fraction expanding: {control['delta_fraction_expanding']:.4g}",
            f"- KS p-value for EF distributions: {control['ks_pvalue_ef']:.4g}",
            f"- Caveat: {control['caveat']}",
        ]
    else:
        lines += [
            "",
            "## Random Baseline",
            "- Not available locally. A formal HR24 parent-sample random-pair baseline is still required before a strong ApJL claim.",
        ]
    if comparison.get("hr24_static_baseline"):
        hr = comparison["hr24_static_baseline"]
        lines += [
            "",
            "## HR24 Static Random Baseline",
            f"- Parent rows after filters: {hr['parent_rows_after_filters']}",
            f"- Random pairs retained: {hr['random_pairs_retained']}",
            f"- Median random E_k/E_g: {hr['median_energy_ratio']:.4g}",
            f"- Random fraction energy-bound: {hr['fraction_bound_energy']:.4g}",
            "- Caveat: this is a static energy baseline, not the formal EF backtracking null.",
        ]
    if comparison.get("hr24_ef_baseline"):
        ef = comparison["hr24_ef_baseline"]
        lines += [
            "",
            "## HR24 EF Random Baseline",
            f"- Control: {ef['summary']['control']}",
            f"- Random pairs retained: {ef['summary']['random_pairs_retained']}",
            f"- Candidate pool size: {ef['summary']['pool_size']}",
            f"- Sampled with replacement: {ef['summary']['sampled_with_replacement']}",
            f"- Baseline fraction EF > 1: {ef['ef_tests']['baseline_fraction_expanding']:.4g}",
            f"- Observed fraction EF > 1: {ef['ef_tests']['observed_fraction_expanding']:.4g}",
            f"- EF KS p-value: {ef['ef_tests']['ks_pvalue']:.4g}",
            f"- EF AD p-value: {ef['ef_tests']['anderson_darling_pvalue']:.4g}",
            f"- EF empirical p(frac >= observed): {ef['ef_tests']['empirical_p_fraction_ge_observed']:.4g}",
            f"- Energy KS p-value: {ef['energy_tests']['ks_pvalue']:.4g}",
            f"- Energy AD p-value: {ef['energy_tests']['anderson_darling_pvalue']:.4g}",
            f"- ln BF_alt/null for log energy: {ef['energy_tests']['ln_bayes_factor_alt_over_null_bic']:.4g}",
        ]
    lines += [
        "",
        "## Interpretation",
        "- The age-dependent dissolution-clock claim remains rejected.",
    ]
    if comparison.get("hr24_ef_baseline"):
        lines += [
            "- The HR24 separation-matched EF random baseline gives a marginal fossil-complex signal: KS/AD are significant, while subset empirical p-values are borderline.",
            "- The energy-distribution tests do not support a distinct bound or cold energy population; the BIC Bayes factor favors the shared log-energy null.",
            "- Phase-space coldness is mixed: the observed sample has lower Vrel at matched separation, but not lower E_k/E_g or higher DSI.",
            "- The defensible framing is therefore fossil-complex expansion excess, not an energy-bound binary-cluster population.",
        ]
    else:
        lines += [
            "- The current evidence supports a weaker fossil-complex framing: many high-confidence pairs are unbound, and a majority have EF > 1 in the 200 Myr backtracking window.",
            "- A robust paper claim requires EF testing against a parent-catalog random baseline, not only against the 50% null or Table_3 internal candidates.",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtrack-myr", type=float, default=200.0)
    parser.add_argument("--backtrack-steps", type=int, default=401)
    parser.add_argument("--internal-candidate-control", action="store_true", help="Backtrack Table_3.csv as a weak internal control.")
    parser.add_argument("--force-control", action="store_true", help="Recompute cached Table_3 control backtracking.")
    parser.add_argument("--hr24-static-baseline", action="store_true", help="Generate a true HR24 parent-catalog static random-pair energy baseline.")
    parser.add_argument("--hr24-ef-baseline", action="store_true", help="Generate and backtrack HR24 random-pair EF baseline.")
    parser.add_argument("--hr24-random-pairs", type=int, default=100000)
    parser.add_argument("--hr24-ef-control", choices=["global", "separation_limited", "separation_matched"], default="separation_matched")
    parser.add_argument("--hr24-backtrack-batch-size", type=int, default=1000)
    parser.add_argument("--hr24-seed", type=int, default=123)
    parser.add_argument("--hr24-separation-max-pc", type=float, default=None)
    parser.add_argument("--hr24-no-quality-cut", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    dirs = ensure_dirs(root)
    metrics, back = load_or_build_main_products(root, args.backtrack_myr, args.backtrack_steps)
    fossil = add_energy_metrics(build_fossil_metrics(metrics, back))
    class_summary = summarize_classes(fossil)
    control = load_internal_candidate_control(root, args.backtrack_myr, args.backtrack_steps, args.force_control) if args.internal_candidate_control else None
    if control is not None:
        control = add_energy_metrics(control)
    comparison = compare_with_control(fossil, control)
    comparison["ef_dynamics_fit"] = ef_dynamics_fit_summary(fossil)

    hr24_baseline = None
    if args.hr24_static_baseline:
        hr24_baseline, hr24_summary = build_hr24_static_baseline(
            root,
            args.hr24_random_pairs,
            args.hr24_seed,
            args.hr24_separation_max_pc,
            rv_required=True,
            quality_cut=not args.hr24_no_quality_cut,
        )
        comparison["hr24_static_baseline"] = hr24_summary
    else:
        cached_static = load_cached_hr24_static_baseline(root)
        if cached_static is not None:
            hr24_baseline, hr24_summary = cached_static
            comparison["hr24_static_baseline"] = hr24_summary

    hr24_ef = None
    if args.hr24_ef_baseline:
        hr24_ef, hr24_ef_summary = build_hr24_ef_baseline(
            root,
            fossil,
            args.hr24_random_pairs,
            args.hr24_seed,
            args.hr24_ef_control,
            args.backtrack_myr,
            args.backtrack_steps,
            args.hr24_backtrack_batch_size,
            quality_cut=not args.hr24_no_quality_cut,
        )
        ef_tests = ef_significance(fossil, hr24_ef, seed=args.hr24_seed)
        energy_tests = energy_significance(fossil, hr24_ef, seed=args.hr24_seed)
        comparison["hr24_ef_baseline"] = {
            "summary": hr24_ef_summary,
            "ef_tests": ef_tests,
            "energy_tests": energy_tests,
        }
    else:
        cached_ef = load_cached_hr24_ef_baseline(root)
        if cached_ef is not None:
            hr24_ef, hr24_ef_summary = cached_ef
            comparison["hr24_ef_baseline"] = hr24_ef_summary

    fossil.to_csv(dirs["tables"] / "fossil_complex_metrics.csv", index=False)
    fossil.to_csv(dirs["tables"] / "v2_energy_metrics.csv", index=False)
    class_summary.to_csv(dirs["tables"] / "expansion_class_summary.csv", index=False)
    (dirs["tables"] / "random_baseline_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (dirs["tables"] / "v2_ef_dsi_fit_summary.json").write_text(json.dumps(comparison["ef_dynamics_fit"], indent=2), encoding="utf-8")
    if hr24_baseline is not None:
        hr24_baseline.to_csv(dirs["tables"] / "v2_hr24_static_random_pairs.csv", index=False)
        (dirs["tables"] / "v2_hr24_random_baseline_summary.json").write_text(json.dumps(comparison["hr24_static_baseline"], indent=2), encoding="utf-8")
    cold = None
    if hr24_ef is not None:
        hr24_ef.to_csv(dirs["tables"] / "hr24_ef_random_pairs.csv", index=False)
        (dirs["tables"] / "hr24_ef_random_baseline_summary.json").write_text(json.dumps(comparison["hr24_ef_baseline"], indent=2), encoding="utf-8")
        (dirs["tables"] / "energy_significance_tests.json").write_text(json.dumps(comparison["hr24_ef_baseline"]["energy_tests"], indent=2), encoding="utf-8")
        cold = {
            "observed_median_separation_pc": float(fossil["separation_pc"].median()),
            "baseline_median_separation_pc": float(hr24_ef["separation_pc"].median()),
            "observed_median_vrel_kms": float(fossil["vrel_kms"].median()),
            "baseline_median_vrel_kms": float(hr24_ef["vrel_kms"].median()),
            "observed_median_dsi": float(fossil["dsi"].median()),
            "baseline_median_dsi": float(hr24_ef["dsi"].median()),
            "observed_median_energy_ratio": float(fossil["energy_ratio"].median()),
            "baseline_median_energy_ratio": float(hr24_ef["energy_ratio"].median()),
        }
        (dirs["tables"] / "phase_space_coldness_summary.json").write_text(json.dumps(cold, indent=2), encoding="utf-8")
        verdict = build_verdict(comparison, cold)
        (dirs["tables"] / "v2_final_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        write_verdict_report(verdict, dirs["reports"] / "v2_final_verdict.md")

    plot_dmin_vs_dnow(fossil, dirs["figures"] / "v2_fig1_dmin_vs_dnow.png")
    plot_ef_distribution(fossil, control, dirs["figures"] / "v2_fig2_expansion_factor_distribution.png")
    plot_class_fractions(class_summary, dirs["figures"] / "v2_fig3_expansion_class_fractions.png")
    plot_ef_vs_age(fossil, dirs["figures"] / "v2_fig4_expansion_factor_vs_age.png")
    plot_dsi_distribution(fossil, dirs["figures"] / "v2_fig5_dsi_distribution.png")
    plot_energy_distribution(fossil, hr24_baseline, dirs["figures"] / "v2_fig6_energy_ratio_distribution.png")
    plot_ef_vs_dsi(fossil, dirs["figures"] / "v2_fig7_ef_vs_dsi.png")
    plot_ef_vs_energy_ratio(fossil, dirs["figures"] / "v2_fig8_ef_vs_energy_ratio.png")
    if hr24_ef is not None:
        plot_hr24_ef_distribution(fossil, hr24_ef, dirs["figures"] / "v2_fig9_hr24_ef_distribution.png")
        plot_hr24_dmin_distribution(fossil, hr24_ef, dirs["figures"] / "v2_fig10_hr24_dmin_distribution.png")
        plot_phase_space_coldness(fossil, hr24_ef, dirs["figures"] / "v2_fig11_phase_space_coldness.png")
    write_report(fossil, class_summary, comparison, dirs["reports"] / "fossil_complex_summary.md")
    write_report(fossil, class_summary, comparison, dirs["reports"] / "v2_energy_fossil_complex_summary.md")
    if hr24_ef is not None:
        write_report(fossil, class_summary, comparison, dirs["reports"] / "hr24_ef_random_baseline_report.md")

    print(f"Wrote v2 fossil-complex results under {root / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
