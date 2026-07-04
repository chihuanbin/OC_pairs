#!/usr/bin/env python3
"""Supplementary revision diagnostics for the cluster-pair paper.

This script adds the conservative checks requested in ``revise.txt`` without
changing the original analysis pipeline. It writes compact CSV/JSON products
that can be cited directly from the revised manuscript.
"""

from __future__ import annotations

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
from scipy.stats import anderson_ksamp, ks_2samp, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binary_cluster_dissolution as bcd
import fossil_cluster_complex as fcc


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "processed": root / "data" / "processed",
        "figures": root / "A&A_latex",
        "tables": root / "results" / "tables",
        "reports": root / "results" / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_observed(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(root / "data" / "processed" / "binary_cluster_dissolution_metrics.csv")
    back = pd.read_csv(root / "data" / "processed" / "orbit_backtracking.csv")
    fossil = fcc.add_energy_metrics(fcc.build_fossil_metrics(metrics, back))
    fossil["age_diff_myr"] = (10.0 ** fossil["log_age1"] - 10.0 ** fossil["log_age2"]).abs() / 1.0e6
    fossil["age_min_myr"] = np.minimum(10.0 ** fossil["log_age1"], 10.0 ** fossil["log_age2"]) / 1.0e6
    fossil["tmin_lookback_myr"] = fossil["t_d_min_myr"].abs()
    fossil["tmin_within_younger_age"] = fossil["tmin_lookback_myr"] <= fossil["age_min_myr"]
    fossil["candidate_common_origin"] = (
        (fossil["d_min_pc"] < 50.0)
        & (fossil["expansion_factor"] > 1.0)
        & (fossil["age_diff_myr"] <= 50.0)
        & fossil["tmin_within_younger_age"]
    )
    return metrics, back, fossil


def quantile_stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    q16, q50, q84 = np.percentile(arr, [16, 50, 84])
    return {
        "min": float(np.min(arr)),
        "p16": float(q16),
        "median": float(q50),
        "p84": float(q84),
        "max": float(np.max(arr)),
    }


def energy_summary(fossil: pd.DataFrame) -> dict[str, object]:
    eta = fossil["energy_ratio"]
    vratio = fossil["vrel_over_vesc"]
    rows = []
    for mass_factor in [1.0, 3.0, 10.0, 100.0]:
        eta_scaled = eta / mass_factor
        rows.append(
            {
                "mass_factor": mass_factor,
                "min_energy_ratio": float(eta_scaled.min()),
                "median_energy_ratio": float(eta_scaled.median()),
                "bound_count": int((eta_scaled < 1.0).sum()),
                "bound_fraction": float((eta_scaled < 1.0).mean()),
            }
        )
    conservative = pd.DataFrame(rows)
    return {
        "n": int(len(fossil)),
        "energy_ratio": quantile_stats(eta),
        "vrel_over_vesc": quantile_stats(vratio),
        "count_energy_ratio_lt_10": int((eta < 10.0).sum()),
        "count_energy_ratio_lt_100": int((eta < 100.0).sum()),
        "bound_count": int((eta < 1.0).sum()),
        "mass_scaling_tests": conservative.to_dict(orient="records"),
    }


def tmin_summary(fossil: pd.DataFrame) -> dict[str, object]:
    lookback = fossil["tmin_lookback_myr"]
    return {
        "tmin_lookback_myr": quantile_stats(lookback),
        "count_tmin_le_10_myr": int((lookback <= 10.0).sum()),
        "count_tmin_le_50_myr": int((lookback <= 50.0).sum()),
        "count_tmin_le_100_myr": int((lookback <= 100.0).sum()),
        "count_tmin_le_200_myr": int((lookback <= 200.0).sum()),
        "count_tmin_zero": int((lookback == 0.0).sum()),
    }


def age_energy_summary(fossil: pd.DataFrame) -> dict[str, object]:
    pairs = [
        ("age_myr", "energy_ratio", "age_vs_log_energy_ratio"),
        ("age_myr", "dsi", "age_vs_log_dsi"),
        ("age_myr", "expansion_factor", "age_vs_log_ef"),
        ("age_myr", "d_min_pc", "age_vs_log_dmin"),
    ]
    out: dict[str, object] = {}
    for xcol, ycol, name in pairs:
        sub = fossil[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub[(sub[xcol] > 0.0) & (sub[ycol] > 0.0)]
        rho, pval = spearmanr(np.log10(sub[xcol]), np.log10(sub[ycol]))
        out[name] = {"rho": float(rho), "p_value": float(pval), "n": int(len(sub))}
    return out


def observed_vs_hr24_table(fossil: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    quantities = [
        ("Dnow_pc", "separation_pc"),
        ("Vrel_kms", "vrel_kms"),
        ("M1_plus_M2_msun", "mass_total_msun"),
        ("reduced_mass_msun", "reduced_mass_msun"),
        ("Vesc_kms", "vesc_kms"),
        ("DSI", "dsi"),
        ("Ek_over_Eg", "energy_ratio"),
        ("EF", "expansion_factor"),
    ]
    base = baseline.copy()
    base["mass_total_msun"] = base["mass1_msun"] + base["mass2_msun"]
    base["reduced_mass_msun"] = (base["mass1_msun"] * base["mass2_msun"]) / base["mass_total_msun"]
    for label, col in quantities:
        rows.append(
            {
                "quantity": label,
                "observed_median": float(fossil[col].median()),
                "hr24_replacement_median": float(base[col].median()),
                "observed_p16": float(np.percentile(fossil[col], 16)),
                "observed_p84": float(np.percentile(fossil[col], 84)),
                "hr24_replacement_p16": float(np.percentile(base[col], 16)),
                "hr24_replacement_p84": float(np.percentile(base[col], 84)),
            }
        )
    return pd.DataFrame(rows)


def ef_tests(observed: pd.DataFrame, baseline: pd.DataFrame, seed: int = 123, n_boot: int = 10000) -> dict[str, float]:
    obs = observed["expansion_factor"].to_numpy(float)
    base = baseline["expansion_factor"].to_numpy(float)
    ks = ks_2samp(obs, base)
    ad = anderson_ksamp([obs, base])
    rng = np.random.default_rng(seed)
    replace = len(base) < len(obs)
    frac = np.empty(n_boot)
    med = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(base, size=len(obs), replace=replace)
        frac[i] = np.mean(sample > 1.0)
        med[i] = np.median(sample)
    return {
        "observed_fraction_expanding": float(np.mean(obs > 1.0)),
        "baseline_fraction_expanding": float(np.mean(base > 1.0)),
        "observed_median_ef": float(np.median(obs)),
        "baseline_median_ef": float(np.median(base)),
        "ks_pvalue": float(ks.pvalue),
        "anderson_darling_pvalue": float(ad.pvalue),
        "empirical_p_fraction_ge_observed": float(np.mean(frac >= np.mean(obs > 1.0))),
        "empirical_p_median_ge_observed": float(np.mean(med >= np.median(obs))),
        "bootstrap_sampled_with_replacement": bool(replace),
    }


def build_unique_hr24(root: Path, fossil: pd.DataFrame, duration_myr: float, n_steps: int, batch_size: int) -> tuple[pd.DataFrame, dict[str, object]]:
    out_path = root / "results" / "tables" / f"revised_hr24_unique_pairs_{int(duration_myr)}myr.csv"
    summary_path = root / "results" / "tables" / f"revised_hr24_unique_summary_{int(duration_myr)}myr.json"
    if out_path.exists() and summary_path.exists():
        return pd.read_csv(out_path), json.loads(summary_path.read_text(encoding="utf-8"))

    raw = fcc.read_hr24_parent(root)
    parent = fcc.prepare_hr24_parent(raw, rv_required=True, quality_cut=True)
    pool = fcc.close_pair_pool(parent, max(250.0, float(fossil["separation_pc"].max()) * 1.25))
    unique = fcc.backtrack_parent_pairs(parent, pool, duration_myr, n_steps, batch_size)
    unique.to_csv(out_path, index=False)
    summary = {
        "parent_rows_raw": int(len(raw)),
        "parent_rows_after_filters": int(len(parent)),
        "unique_pair_pool_size": int(len(pool)),
        "duration_myr": float(duration_myr),
        "n_steps": int(n_steps),
        "fraction_expanding": float(unique["is_expanding"].mean()),
        "median_expansion_factor": float(unique["expansion_factor"].median()),
        "fraction_dmin_lt_20": float((unique["d_min_pc"] < 20.0).mean()),
        "fraction_dmin_lt_50": float((unique["d_min_pc"] < 50.0).mean()),
        "median_energy_ratio": float(unique["energy_ratio"].median()),
        "median_vrel_kms": float(unique["vrel_kms"].median()),
        "median_vesc_kms": float(unique["vesc_kms"].median()),
        "median_dsi": float(unique["dsi"].median()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return unique, summary


def backtrack_observed_window(root: Path, metrics: pd.DataFrame, duration_myr: float, n_steps: int) -> pd.DataFrame:
    out_path = root / "data" / "processed" / f"orbit_backtracking_{int(duration_myr)}myr.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    back = bcd.backtrack_orbits(metrics, duration_myr, n_steps)
    back.to_csv(out_path, index=False)
    return back


def window_sensitivity(root: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for duration, steps in [(50.0, 101), (100.0, 201), (200.0, 401)]:
        if duration == 200.0:
            back = pd.read_csv(root / "data" / "processed" / "orbit_backtracking.csv")
        else:
            back = backtrack_observed_window(root, metrics, duration, steps)
        fossil = fcc.add_energy_metrics(fcc.build_fossil_metrics(metrics, back))
        rows.append(
            {
                "duration_myr": duration,
                "fraction_expanding": float(fossil["is_expanding"].mean()),
                "median_expansion_factor": float(fossil["expansion_factor"].median()),
                "fraction_dmin_lt_20": float((fossil["d_min_pc"] < 20.0).mean()),
                "fraction_dmin_lt_50": float((fossil["d_min_pc"] < 50.0).mean()),
                "median_tmin_lookback_myr": float(fossil["t_d_min_myr"].abs().median()),
            }
        )
    return pd.DataFrame(rows)


def candidate_table(fossil: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pair_id",
        "cluster1",
        "cluster2",
        "d_now_pc",
        "d_min_pc",
        "tmin_lookback_myr",
        "expansion_factor",
        "vrel_kms",
        "vesc_kms",
        "energy_ratio",
        "age_myr",
        "age_diff_myr",
        "candidate_common_origin",
    ]
    keep = fossil[(fossil["d_min_pc"] < 50.0) & (fossil["expansion_factor"] > 1.0)].copy()
    return keep[cols].sort_values(["candidate_common_origin", "d_min_pc"], ascending=[False, True])


def plot_age_energy(fossil: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sc = ax.scatter(
        fossil["age_myr"],
        fossil["energy_ratio"],
        c=fossil["d_min_pc"],
        s=38,
        cmap="viridis_r",
        alpha=0.84,
    )
    ax.axhline(1.0, color="#111111", lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean age (Myr)")
    ax.set_ylabel(r"$E_{\rm k}/E_{\rm g}$")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$D_{\min}$ over 200 Myr (pc)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def main() -> int:
    root = Path.cwd()
    dirs = ensure_dirs(root)
    metrics, _back, fossil = load_observed(root)
    replacement = pd.read_csv(root / "results" / "tables" / "hr24_ef_random_pairs.csv")

    unique, unique_summary = build_unique_hr24(root, fossil, duration_myr=200.0, n_steps=401, batch_size=1000)
    unique_tests = ef_tests(fossil, unique, seed=321, n_boot=10000)
    repl_tests = ef_tests(fossil, replacement, seed=321, n_boot=10000)

    tables = {
        "energy": energy_summary(fossil),
        "tmin": tmin_summary(fossil),
        "age_energy": age_energy_summary(fossil),
        "hr24_unique_summary": unique_summary,
        "hr24_unique_ef_tests": unique_tests,
        "hr24_replacement_ef_tests": repl_tests,
    }
    (dirs["tables"] / "revised_statistics.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")

    energy_rows = []
    for row in tables["energy"]["mass_scaling_tests"]:
        energy_rows.append(row)
    pd.DataFrame(energy_rows).to_csv(dirs["tables"] / "revised_mass_scaling_tests.csv", index=False)
    observed_vs_hr24_table(fossil, replacement).to_csv(dirs["tables"] / "revised_observed_vs_hr24.csv", index=False)
    window_sensitivity(root, metrics).to_csv(dirs["tables"] / "revised_backtracking_window_sensitivity.csv", index=False)
    candidate_table(fossil).to_csv(dirs["tables"] / "revised_candidate_common_origin_subset.csv", index=False)
    plot_age_energy(fossil, dirs["figures"] / "fig_age_energy.png")

    lines = [
        "# Revised Statistics",
        "",
        f"- Unique HR24 pairs: {unique_summary['unique_pair_pool_size']}",
        f"- Unique HR24 EF>1 fraction: {unique_tests['baseline_fraction_expanding']:.4f}",
        f"- Unique HR24 median EF: {unique_tests['baseline_median_ef']:.4f}",
        f"- Observed-size unique bootstrap p(frac): {unique_tests['empirical_p_fraction_ge_observed']:.4f}",
        f"- Observed-size unique bootstrap p(median): {unique_tests['empirical_p_median_ge_observed']:.4f}",
        f"- Minimum observed Ek/Eg: {tables['energy']['energy_ratio']['min']:.4g}",
        f"- 16/50/84 percentiles Ek/Eg: {tables['energy']['energy_ratio']['p16']:.4g}, {tables['energy']['energy_ratio']['median']:.4g}, {tables['energy']['energy_ratio']['p84']:.4g}",
        f"- Bound count after 10x masses: {energy_rows[2]['bound_count']}",
        f"- Bound count after 100x masses: {energy_rows[3]['bound_count']}",
        f"- Median |tmin|: {tables['tmin']['tmin_lookback_myr']['median']:.2f} Myr",
    ]
    (dirs["reports"] / "revised_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
