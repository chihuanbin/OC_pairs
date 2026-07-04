#!/usr/bin/env python3
"""Empirical dissolution-clock analysis for Galactic binary clusters.

This script implements the workflow sketched in ``v0.txt`` for the
Chen et al. (2026) AJ catalog. It intentionally requires a real
machine-readable catalog for science runs; the PDF appendix preview is
not enough to reproduce the 83-pair analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

ZENODO_API = "https://zenodo.org/api/records/19449936"
ZENODO_DOI = "10.5281/zenodo.19449936"


@dataclass(frozen=True)
class ColumnMap:
    pair_id: str
    cluster1: str
    cluster2: str
    log_age1: str
    log_age2: str
    mass1: str
    mass2: str
    separation_pc: str
    vesc_kms: str
    vrel_kms: str
    tf1: str | None = None
    tf2: str | None = None
    ra1_deg: str | None = None
    dec1_deg: str | None = None
    parallax1_mas: str | None = None
    pmra1_masyr: str | None = None
    pmdec1_masyr: str | None = None
    rv1_kms: str | None = None
    ra2_deg: str | None = None
    dec2_deg: str | None = None
    parallax2_mas: str | None = None
    pmra2_masyr: str | None = None
    pmdec2_masyr: str | None = None
    rv2_kms: str | None = None


def norm_col(name: str) -> str:
    value = name.strip().lower()
    value = value.replace("[", "_").replace("]", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "figures": root / "results" / "figures",
        "tables": root / "results" / "tables",
        "reports": root / "results" / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def download_zenodo_files(raw_dir: Path) -> list[Path]:
    """Download all files attached to the Zenodo record.

    The compute environment may not have network access. In that case this
    function raises a clear error instructing the user where to place files.
    """
    try:
        with urllib.request.urlopen(ZENODO_API, timeout=45) as response:
            record = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not reach Zenodo. Download the machine-readable files for "
            f"{ZENODO_DOI} manually and place them in {raw_dir}."
        ) from exc

    downloaded: list[Path] = []
    for item in record.get("files", []):
        key = item.get("key") or item.get("filename")
        links = item.get("links", {})
        url = links.get("self") or links.get("download")
        if not key or not url:
            continue
        out = raw_dir / Path(key).name
        if out.exists() and out.stat().st_size > 0:
            downloaded.append(out)
            continue
        with urllib.request.urlopen(url, timeout=120) as response:
            out.write_bytes(response.read())
        downloaded.append(out)
    if not downloaded:
        raise RuntimeError(f"Zenodo record {ZENODO_DOI} has no downloadable files.")
    return downloaded


def candidate_catalog_files(raw_dir: Path) -> list[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".dat", ".ecsv", ".fits", ".fit", ".xlsx"}
    ignored = {"catalog_schema_template.csv", ".DS_Store"}
    return sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.name not in ignored and p.suffix.lower() in suffixes
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit"}:
        from astropy.table import Table

        return Table.read(path).to_pandas()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".ecsv":
        from astropy.table import Table

        return Table.read(path, format="ascii.ecsv").to_pandas()

    # Try robust text parsing. Machine-readable AJ tables are often fixed-width
    # or whitespace-delimited; CSV/TSV remain preferred when available.
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        return pd.read_fwf(path, comment="#")


def is_long_pair_catalog(df: pd.DataFrame) -> bool:
    cols = {norm_col(c) for c in df.columns}
    if not {"pair", "name", "logage", "mass", "separation_pc", "v_esc_km_s", "v_rel_km_s"}.issubset(cols):
        return False
    counts = df.groupby(next(c for c in df.columns if norm_col(c) == "pair")).size()
    return len(counts) > 0 and counts.min() == 2 and counts.max() == 2


def long_pair_catalog_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Zenodo long format, two cluster rows per pair, to one row per pair."""
    lookup = {norm_col(c): c for c in df.columns}
    pair_col = lookup["pair"]
    name_col = lookup["name"]

    source_cols = {
        "ra": "ra_deg",
        "dec": "dec_deg",
        "parallax": "parallax_mas",
        "pmra": "pmra_masyr",
        "pmdec": "pmdec_masyr",
        "radial_velocity": "rv_kms",
        "logage": "log_age",
        "mass": "mass",
        "tf": "tf",
    }
    pair_level_cols = {
        "separation_pc": "separation_pc",
        "v_esc_km_s": "vesc_kms",
        "v_rel_km_s": "vrel_kms",
        "a_kpc": "a_kpc",
        "e": "eccentricity",
        "p_myr": "period_myr",
        "i_deg": "inclination_deg",
        "status": "status",
        "result": "result",
        "astrometric_cmd": "astrometric_cmd",
        "age_metallicity": "age_metallicity",
        "bound_status": "bound_status_flag",
        "tidal_roche": "tidal_roche_flag",
        "orbit": "orbit_flag",
        "tidal_factor": "tidal_factor_flag",
    }

    rows: list[dict[str, object]] = []
    for pair_id, group in df.groupby(pair_col, sort=True):
        group = group.reset_index(drop=True)
        if len(group) != 2:
            continue
        row: dict[str, object] = {"pair_id": pair_id, "cluster1": group.loc[0, name_col], "cluster2": group.loc[1, name_col]}
        normalized_cols = {norm_col(c): c for c in group.columns}
        for norm, base in source_cols.items():
            col = normalized_cols.get(norm)
            if col is None:
                continue
            row[f"{base}1"] = group.loc[0, col]
            row[f"{base}2"] = group.loc[1, col]
        for norm, out_name in pair_level_cols.items():
            col = normalized_cols.get(norm)
            if col is not None:
                row[out_name] = group.loc[0, col]
        rows.append(row)
    return pd.DataFrame(rows)


def canonicalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    if is_long_pair_catalog(df):
        return long_pair_catalog_to_wide(df)
    return df


def find_col(columns: Iterable[str], patterns: list[str], required: bool = True) -> str | None:
    normalized = {norm_col(c): c for c in columns}
    for pattern in patterns:
        rx = re.compile(pattern)
        for n, original in normalized.items():
            if rx.fullmatch(n) or rx.search(n):
                return original
    if required:
        raise KeyError(f"Could not find a column matching {patterns}. Available: {list(columns)}")
    return None


def infer_column_map(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)
    return ColumnMap(
        pair_id=find_col(cols, [r"pair(_id|_num|_no|_number)?", r"pair"]),
        cluster1=find_col(cols, [r"cluster_?1", r"cluster1", r"name_?1", r"c1"]),
        cluster2=find_col(cols, [r"cluster_?2", r"cluster2", r"name_?2", r"c2"]),
        log_age1=find_col(cols, [r"log_?t_?1", r"log_?age_?1", r"logage_?1", r"age_?1"]),
        log_age2=find_col(cols, [r"log_?t_?2", r"log_?age_?2", r"logage_?2", r"age_?2"]),
        mass1=find_col(cols, [r"m_?1", r"mass_?1", r"m1"]),
        mass2=find_col(cols, [r"m_?2", r"mass_?2", r"m2"]),
        separation_pc=find_col(cols, [r"^s$", r"^sep(aration)?(_?pc)?$", r"^distance_?3d$"]),
        vesc_kms=find_col(cols, [r"v_?esc", r"vesc"]),
        vrel_kms=find_col(cols, [r"v_?rel", r"vrel"]),
        tf1=find_col(cols, [r"tf_?1", r"tf1"], required=False),
        tf2=find_col(cols, [r"tf_?2", r"tf2"], required=False),
        ra1_deg=find_col(cols, [r"ra_?1", r"ra_?deg_?1", r"alpha_?1"], required=False),
        dec1_deg=find_col(cols, [r"dec_?1", r"dec_?deg_?1", r"delta_?1"], required=False),
        parallax1_mas=find_col(cols, [r"parallax_?1", r"parallax_?mas_?1", r"plx_?1", r"varpi_?1"], required=False),
        pmra1_masyr=find_col(cols, [r"pmra_?1", r"pmra_?masyr_?1", r"mu_?alpha_?1", r"mualpha_?1"], required=False),
        pmdec1_masyr=find_col(cols, [r"pmdec_?1", r"pmdec_?masyr_?1", r"mu_?delta_?1", r"mudelta_?1"], required=False),
        rv1_kms=find_col(cols, [r"rv_?1", r"rv_?kms_?1", r"radial_?velocity_?1"], required=False),
        ra2_deg=find_col(cols, [r"ra_?2", r"ra_?deg_?2", r"alpha_?2"], required=False),
        dec2_deg=find_col(cols, [r"dec_?2", r"dec_?deg_?2", r"delta_?2"], required=False),
        parallax2_mas=find_col(cols, [r"parallax_?2", r"parallax_?mas_?2", r"plx_?2", r"varpi_?2"], required=False),
        pmra2_masyr=find_col(cols, [r"pmra_?2", r"pmra_?masyr_?2", r"mu_?alpha_?2", r"mualpha_?2"], required=False),
        pmdec2_masyr=find_col(cols, [r"pmdec_?2", r"pmdec_?masyr_?2", r"mu_?delta_?2", r"mudelta_?2"], required=False),
        rv2_kms=find_col(cols, [r"rv_?2", r"rv_?kms_?2", r"radial_?velocity_?2"], required=False),
    )


def choose_catalog(raw_dir: Path, explicit: Path | None) -> tuple[pd.DataFrame, Path]:
    if explicit:
        return canonicalize_catalog(read_table(explicit)), explicit
    files = candidate_catalog_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No catalog table found in {raw_dir}. Run with --download or place the "
            f"machine-readable table from {ZENODO_DOI} there."
        )
    ranked: list[tuple[int, Path, pd.DataFrame]] = []
    for path in files:
        try:
            df = read_table(path)
        except Exception:
            continue
        score = 0
        names = " ".join(norm_col(c) for c in df.columns)
        for token in ["pair", "cluster", "vrel", "vesc", "log", "mass"]:
            score += token in names
        canonical = canonicalize_catalog(df)
        if len(canonical) == 83:
            score += 1000
        if "result" in {norm_col(c) for c in df.columns}:
            score += 50
        if is_long_pair_catalog(df):
            score += 100
        score += min(len(canonical), 83)
        ranked.append((score, path, canonical))
    if not ranked:
        raise RuntimeError(f"Found files in {raw_dir}, but none could be parsed as a table.")
    ranked.sort(key=lambda item: item[0], reverse=True)
    _, path, df = ranked[0]
    return df, path


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_metrics(df: pd.DataFrame, cmap: ColumnMap, k: float) -> pd.DataFrame:
    out = pd.DataFrame()
    out["pair_id"] = df[cmap.pair_id]
    out["cluster1"] = df[cmap.cluster1].astype(str)
    out["cluster2"] = df[cmap.cluster2].astype(str)
    out["log_age1"] = as_number(df[cmap.log_age1])
    out["log_age2"] = as_number(df[cmap.log_age2])
    out["log_age_mean"] = out[["log_age1", "log_age2"]].mean(axis=1)
    out["age_yr"] = np.power(10.0, out["log_age_mean"])
    out["age_myr"] = out["age_yr"] / 1.0e6
    out["mass1_msun"] = as_number(df[cmap.mass1])
    out["mass2_msun"] = as_number(df[cmap.mass2])
    out["mass_total_msun"] = out["mass1_msun"] + out["mass2_msun"]
    out["separation_pc"] = as_number(df[cmap.separation_pc])
    out["vesc_kms"] = as_number(df[cmap.vesc_kms])
    out["vrel_kms"] = as_number(df[cmap.vrel_kms])
    out["dsi"] = out["vesc_kms"] / out["vrel_kms"]
    out["p_bound"] = 1.0 / (1.0 + np.exp(-k * (out["dsi"] - 1.0)))
    out["p_dis"] = 1.0 - out["p_bound"]
    out["m_dis_msun"] = out["mass_total_msun"] * out["p_dis"]
    out["vrel_over_vesc"] = out["vrel_kms"] / out["vesc_kms"]
    if cmap.tf1 and cmap.tf2:
        out["tf_mean"] = pd.concat([as_number(df[cmap.tf1]), as_number(df[cmap.tf2])], axis=1).mean(axis=1)
    elif cmap.tf1:
        out["tf_mean"] = as_number(df[cmap.tf1])
    elif cmap.tf2:
        out["tf_mean"] = as_number(df[cmap.tf2])
    else:
        out["tf_mean"] = np.nan

    for name in (
        "ra1_deg",
        "dec1_deg",
        "parallax1_mas",
        "pmra1_masyr",
        "pmdec1_masyr",
        "rv1_kms",
        "ra2_deg",
        "dec2_deg",
        "parallax2_mas",
        "pmra2_masyr",
        "pmdec2_masyr",
        "rv2_kms",
    ):
        source = getattr(cmap, name)
        if source:
            out[name] = as_number(df[source])
    required = ["age_myr", "p_bound", "mass_total_msun", "m_dis_msun", "dsi"]
    bad = out[required].isna().any(axis=1)
    if bad.any():
        out = out.loc[~bad].copy()
    return out.sort_values("age_myr").reset_index(drop=True)


def power_law(age_myr: np.ndarray, a: float, alpha: float) -> np.ndarray:
    return a * np.power(age_myr / 100.0, -alpha)


def exponential(age_myr: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-age_myr / tau)


def constant_model(age_myr: np.ndarray, c: float) -> np.ndarray:
    return np.full_like(age_myr, c, dtype=float)


def bic(y: np.ndarray, yhat: np.ndarray, n_params: int) -> float:
    resid = y - yhat
    rss = float(np.sum(resid * resid))
    n = len(y)
    rss = max(rss, 1e-12)
    return n * math.log(rss / n) + n_params * math.log(n)


def fit_models(metrics: pd.DataFrame) -> dict[str, object]:
    fit = metrics[["age_myr", "p_bound"]].dropna()
    x = fit["age_myr"].to_numpy(dtype=float)
    y = fit["p_bound"].to_numpy(dtype=float)
    models: dict[str, object] = {}
    if len(fit) < 5:
        raise RuntimeError("Need at least five valid rows to fit dissolution models.")

    bounds_power = ([0.0, -5.0], [1.5, 5.0])
    popt_power, pcov_power = curve_fit(power_law, x, y, p0=[0.5, 0.5], bounds=bounds_power, maxfev=20000)
    y_power = np.clip(power_law(x, *popt_power), 0.0, 1.0)
    models["power_law"] = {
        "A_at_100Myr": float(popt_power[0]),
        "alpha": float(popt_power[1]),
        "bic": bic(y, y_power, 2),
        "covariance": pcov_power.tolist(),
    }

    tau_min = max(float(np.nanmin(x)) / 100.0, 1e-3)
    tau_max = max(float(np.nanmax(x)) * 100.0, tau_min * 10.0)
    popt_exp, pcov_exp = curve_fit(exponential, x, y, p0=[max(float(np.nanmedian(x)), 1.0)], bounds=([tau_min], [tau_max]), maxfev=20000)
    y_exp = np.clip(exponential(x, *popt_exp), 0.0, 1.0)
    models["exponential"] = {
        "tau_myr": float(popt_exp[0]),
        "bic": bic(y, y_exp, 1),
        "covariance": pcov_exp.tolist(),
    }

    popt_const, pcov_const = curve_fit(constant_model, x, y, p0=[float(np.nanmean(y))], bounds=([0.0], [1.0]))
    y_const = constant_model(x, *popt_const)
    models["constant_null"] = {
        "p_bound": float(popt_const[0]),
        "bic": bic(y, y_const, 1),
        "covariance": pcov_const.tolist(),
    }

    rho, pvalue = spearmanr(x, y)
    models["rank_correlation"] = {"spearman_rho": float(rho), "p_value": float(pvalue)}
    best = min((v["bic"], k) for k, v in models.items() if isinstance(v, dict) and "bic" in v)
    models["best_by_bic"] = best[1]
    return models


def bootstrap_power_law(metrics: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    if n_boot <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    data = metrics[["age_myr", "p_bound"]].dropna().reset_index(drop=True)
    n = len(data)
    for i in range(n_boot):
        sample = data.iloc[rng.integers(0, n, n)]
        try:
            popt, _ = curve_fit(
                power_law,
                sample["age_myr"].to_numpy(float),
                sample["p_bound"].to_numpy(float),
                p0=[0.5, 0.5],
                bounds=([0.0, -5.0], [1.5, 5.0]),
                maxfev=20000,
            )
            rows.append({"iteration": i, "A_at_100Myr": popt[0], "alpha": popt[1]})
        except Exception:
            continue
    return pd.DataFrame(rows)


def age_bins(metrics: pd.DataFrame) -> pd.DataFrame:
    bins = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    df = metrics.copy()
    df["log_age_bin"] = pd.cut(df["log_age_mean"], bins=bins, labels=labels, include_lowest=True)
    grouped = df.groupby("log_age_bin", observed=False)
    return grouped.agg(
        n=("p_bound", "size"),
        log_age_mid=("log_age_mean", "median"),
        age_myr_median=("age_myr", "median"),
        p_bound_mean=("p_bound", "mean"),
        p_bound_std=("p_bound", "std"),
        dsi_median=("dsi", "median"),
        m_dis_sum=("m_dis_msun", "sum"),
    ).reset_index()


def plot_age_binding(metrics: pd.DataFrame, models: dict[str, object], binned: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(metrics["age_myr"], metrics["p_bound"], s=28, alpha=0.72, color="#2868a0", label="Pairs")
    b = binned.dropna(subset=["age_myr_median", "p_bound_mean"])
    ax.errorbar(
        b["age_myr_median"],
        b["p_bound_mean"],
        yerr=b["p_bound_std"].fillna(0.0),
        fmt="o",
        ms=7,
        color="#111111",
        ecolor="#555555",
        capsize=3,
        label="Age-bin mean",
    )
    xs = np.geomspace(metrics["age_myr"].min() * 0.8, metrics["age_myr"].max() * 1.2, 300)
    if "power_law" in models:
        m = models["power_law"]
        ax.plot(xs, np.clip(power_law(xs, m["A_at_100Myr"], m["alpha"]), 0, 1), color="#bd3c2f", lw=2, label="Power law")
    if "exponential" in models:
        m = models["exponential"]
        ax.plot(xs, np.clip(exponential(xs, m["tau_myr"]), 0, 1), color="#3a8f5a", lw=2, ls="--", label="Exponential")
    ax.set_xscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"$P_{\rm bound}$")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_phase(metrics: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    color = metrics["tf_mean"]
    if color.notna().any():
        sc = ax.scatter(metrics["age_myr"], metrics["vrel_over_vesc"], c=color, s=36, cmap="viridis", alpha=0.82)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Mean TF")
    else:
        ax.scatter(metrics["age_myr"], metrics["vrel_over_vesc"], s=36, color="#594c9c", alpha=0.82)
    ax.axhline(1.0, color="#111111", lw=1.2, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"$V_{\rm rel}/V_{\rm esc}$")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_field_mass(metrics: pd.DataFrame, out: Path) -> None:
    df = metrics.sort_values("age_myr").copy()
    df["cum_m_dis_msun"] = df["m_dis_msun"].cumsum()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.step(df["age_myr"], df["cum_m_dis_msun"], where="post", color="#8c4a2f", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"Cumulative $\sum (M_1+M_2)(1-P_{\rm bound})$ ($M_\odot$)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def has_orbit_columns(metrics: pd.DataFrame) -> bool:
    needed = [
        "ra1_deg",
        "dec1_deg",
        "parallax1_mas",
        "pmra1_masyr",
        "pmdec1_masyr",
        "rv1_kms",
        "ra2_deg",
        "dec2_deg",
        "parallax2_mas",
        "pmra2_masyr",
        "pmdec2_masyr",
        "rv2_kms",
    ]
    return all(c in metrics.columns for c in needed)


def backtrack_orbits(metrics: pd.DataFrame, duration_myr: float, n_steps: int) -> pd.DataFrame:
    if not has_orbit_columns(metrics):
        raise RuntimeError("Catalog lacks full astrometric/RV columns needed for galpy backtracking.")
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    times = np.linspace(0.0, -duration_myr, n_steps) * u.Myr
    rows = []
    for _, row in metrics.iterrows():
        if row[
            [
                "ra1_deg",
                "dec1_deg",
                "parallax1_mas",
                "pmra1_masyr",
                "pmdec1_masyr",
                "rv1_kms",
                "ra2_deg",
                "dec2_deg",
                "parallax2_mas",
                "pmra2_masyr",
                "pmdec2_masyr",
                "rv2_kms",
            ]
        ].isna().any():
            continue
        distance1 = (1000.0 / row["parallax1_mas"]) * u.pc
        distance2 = (1000.0 / row["parallax2_mas"]) * u.pc
        c1 = SkyCoord(
            ra=row["ra1_deg"] * u.deg,
            dec=row["dec1_deg"] * u.deg,
            distance=distance1,
            pm_ra_cosdec=row["pmra1_masyr"] * u.mas / u.yr,
            pm_dec=row["pmdec1_masyr"] * u.mas / u.yr,
            radial_velocity=row["rv1_kms"] * u.km / u.s,
            frame="icrs",
        )
        c2 = SkyCoord(
            ra=row["ra2_deg"] * u.deg,
            dec=row["dec2_deg"] * u.deg,
            distance=distance2,
            pm_ra_cosdec=row["pmra2_masyr"] * u.mas / u.yr,
            pm_dec=row["pmdec2_masyr"] * u.mas / u.yr,
            radial_velocity=row["rv2_kms"] * u.km / u.s,
            frame="icrs",
        )
        o1 = Orbit(c1)
        o2 = Orbit(c2)
        o1.integrate(times, MWPotential2014)
        o2.integrate(times, MWPotential2014)
        x1, y1, z1 = o1.x(times), o1.y(times), o1.z(times)
        x2, y2, z2 = o2.x(times), o2.y(times), o2.z(times)
        sep = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) * 1000.0
        idx = int(np.nanargmin(sep))
        rows.append(
            {
                "pair_id": row["pair_id"],
                "cluster1": row["cluster1"],
                "cluster2": row["cluster2"],
                "d_now_pc": float(sep[0]),
                "d_min_pc": float(sep[idx]),
                "t_d_min_myr": float(times[idx].value),
                "d_now_over_d_min": float(sep[0] / sep[idx]) if sep[idx] > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_backtracking(back: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.hist(back["d_min_pc"].dropna(), bins=18, color="#2868a0", alpha=0.78)
    ax.axvline(20.0, color="#bd3c2f", lw=1.6, ls="--", label="20 pc")
    ax.set_xlabel(r"$D_{\min}$ over backtracking window (pc)")
    ax.set_ylabel("Pair count")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(
    metrics: pd.DataFrame,
    models: dict[str, object],
    boot: pd.DataFrame,
    back: pd.DataFrame | None,
    out: Path,
    catalog_path: Path,
    k: float,
) -> None:
    mass_field = float(metrics["m_dis_msun"].sum())
    age_span_myr = float(metrics["age_myr"].max() - metrics["age_myr"].min())
    rate = mass_field / age_span_myr if age_span_myr > 0 else np.nan
    lines = [
        "# Binary Cluster Dissolution Analysis",
        "",
        f"Catalog: `{catalog_path}`",
        f"Rows analyzed: {len(metrics)}",
        f"DSI logistic k: {k}",
        "",
        "## Core Metrics",
        f"- Median DSI: {metrics['dsi'].median():.4g}",
        f"- Median P_bound: {metrics['p_bound'].median():.4g}",
        f"- Sample M_field proxy: {mass_field:.4g} Msun",
        f"- Order-of-magnitude field production proxy: {rate:.4g} Msun/Myr over sample age span",
        "",
        "## Model Fits",
        f"- Best by BIC: {models.get('best_by_bic')}",
        f"- Power-law alpha: {models['power_law']['alpha']:.4g}",
        f"- Power-law BIC: {models['power_law']['bic']:.4g}",
        f"- Exponential tau: {models['exponential']['tau_myr']:.4g} Myr",
        f"- Exponential BIC: {models['exponential']['bic']:.4g}",
        f"- Constant-null BIC: {models['constant_null']['bic']:.4g}",
        f"- Spearman rho(age, P_bound): {models['rank_correlation']['spearman_rho']:.4g}",
        f"- Spearman p-value: {models['rank_correlation']['p_value']:.4g}",
    ]
    if not boot.empty:
        q = boot["alpha"].quantile([0.16, 0.5, 0.84])
        lines += [
            "",
            "## Bootstrap",
            f"- Successful bootstrap fits: {len(boot)}",
            f"- Alpha 16/50/84 percentiles: {q.iloc[0]:.4g}, {q.iloc[1]:.4g}, {q.iloc[2]:.4g}",
        ]
    if back is not None and not back.empty:
        lines += [
            "",
            "## Orbit Backtracking",
            f"- Rows integrated: {len(back)}",
            f"- Median D_min: {back['d_min_pc'].median():.4g} pc",
            f"- Fraction with D_min < D_now: {(back['d_min_pc'] < back['d_now_pc']).mean():.4g}",
            f"- Fraction with D_min < 20 pc: {(back['d_min_pc'] < 20).mean():.4g}",
        ]
    elif back is None:
        lines += [
            "",
            "## Orbit Backtracking",
            "- Not run. Use `--backtrack` with a catalog containing full astrometry and RV columns.",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, help="Machine-readable 83-pair catalog table.")
    parser.add_argument("--download", action="store_true", help=f"Download files from Zenodo DOI {ZENODO_DOI}.")
    parser.add_argument("--k", type=float, default=3.0, help="Logistic steepness for P_bound(DSI).")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations for power-law alpha.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backtrack", action="store_true", help="Run galpy backward orbit integration.")
    parser.add_argument("--backtrack-myr", type=float, default=200.0)
    parser.add_argument("--backtrack-steps", type=int, default=401)
    args = parser.parse_args()

    root = Path.cwd()
    dirs = ensure_dirs(root)

    if args.download:
        download_zenodo_files(dirs["raw"])

    df, catalog_path = choose_catalog(dirs["raw"], args.catalog)
    cmap = infer_column_map(df)
    metrics = compute_metrics(df, cmap, args.k)
    if len(metrics) < 10:
        raise RuntimeError(
            f"Only {len(metrics)} valid rows were parsed from {catalog_path}. "
            "This is probably not the full machine-readable 83-pair catalog."
        )

    models = fit_models(metrics)
    boot = bootstrap_power_law(metrics, args.bootstrap, args.seed)
    binned = age_bins(metrics)

    metrics.to_csv(dirs["processed"] / "binary_cluster_dissolution_metrics.csv", index=False)
    binned.to_csv(dirs["tables"] / "age_bin_summary.csv", index=False)
    (dirs["tables"] / "model_fits.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    if not boot.empty:
        boot.to_csv(dirs["tables"] / "power_law_bootstrap.csv", index=False)

    plot_age_binding(metrics, models, binned, dirs["figures"] / "fig1_age_binding_probability.png")
    plot_phase(metrics, dirs["figures"] / "fig2_dissolution_phase_space.png")
    plot_field_mass(metrics, dirs["figures"] / "fig3_field_mass_proxy.png")

    back = None
    if args.backtrack:
        back = backtrack_orbits(metrics, args.backtrack_myr, args.backtrack_steps)
        back.to_csv(dirs["processed"] / "orbit_backtracking.csv", index=False)
        if not back.empty:
            plot_backtracking(back, dirs["figures"] / "fig4_backtracking_dmin.png")

    write_summary(
        metrics,
        models,
        boot,
        back,
        dirs["reports"] / "dissolution_summary.md",
        catalog_path,
        args.k,
    )
    print(f"Wrote metrics, tables, figures, and summary under {root / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
