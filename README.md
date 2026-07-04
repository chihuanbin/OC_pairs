
# Galactic Cluster-Pair Dynamics from Gaia DR3

This repository contains the analysis scripts and derived data products used for the manuscript:

**The dynamical nature of Galactic cluster-pair candidates from Gaia DR3**  
**Energy diagnostics and tests of the bound-binary interpretation**

The project tests whether high-confidence Galactic cluster-pair candidates can be interpreted as classical bound binary clusters. It combines present-day two-body energy diagnostics, Galactic orbit backtracking, expansion-factor measurements, and random-pair baselines drawn from the Hunt & Reffert (2024, HR24) open-cluster catalogue.

## Scientific Summary

The main conclusion is that the current high-confidence Galactic cluster-pair sample should not be interpreted as a catalogue of classical bound binary clusters.

For the 83 high-confidence cluster-pair candidates analysed here:

- No pair satisfies the two-body binding condition.
- The median energy ratio is \(E_k/E_g = 6568\).
- The minimum energy ratio is \(E_k/E_g = 72.7\), still far above the bound threshold.
- Uniformly increasing all cluster masses by factors of 3 or 10 does not produce any bound pairs.
- Orbit backtracking shows that some pairs were closer in the recent past, but the population-level evidence for excess past compactness is weak.
- A fossil-complex interpretation remains possible for a small subset of unbound systems, but it is not established for the full population.

## Repository Contents

### Main Analysis Scripts

| File | Description |
|---|---|
| `revise_analysis.py` | Main revised analysis script. Computes the primary energy diagnostics, backtracking summaries, HR24 comparisons, and final statistics used in the manuscript. |
| `fossil_cluster_complex.py` | Analysis of possible fossil-complex or common-origin cluster-pair candidates. Produces fossil-complex metrics and candidate subset tables. |
| `binary_cluster_dissolution.py` | Tests related to the binary-cluster dissolution interpretation, including energy ratios, dynamical softness, and age-trend diagnostics. |
| `run_smoke_test.py` | Lightweight smoke test to check that the main derived files and analysis environment are usable. |

### Main Derived Tables

| File | Description |
|---|---|
| `v2_energy_metrics.csv` | Present-day two-body energy diagnostics for the cluster-pair sample, including \(E_k/E_g\), binding status, relative velocity, escape velocity, and related quantities. |
| `fossil_complex_metrics.csv` | Orbit-backtracking and fossil-complex diagnostics, including \(D_{\rm now}\), \(D_{\min}\), expansion factor, and closest-approach time. |
| `revised_candidate_common_origin_subset.csv` | Heuristic subset of possible common-origin candidates satisfying cuts on \(D_{\min}\), expansion factor, age difference, and closest-approach time. |
| `revised_backtracking_window_sensitivity.csv` | Sensitivity of backtracking results to the adopted integration window. |
| `revised_observed_vs_hr24.csv` | Comparison between the observed high-confidence pairs and the HR24 random-pair baseline. |
| `revised_mass_scaling_tests.csv` | Mass-rescaling tests showing how the bound fraction changes when all cluster masses are multiplied by fixed factors. |
| `expansion_class_summary.csv` | Summary of expansion classes based on minimum past separation. |
| `power_law_bootstrap.csv` | Bootstrap output for auxiliary trend-fitting or scaling tests. |
| `hr24_ef_random_pairs.csv` | Random HR24 pair sample used for the auxiliary expansion-factor baseline. |
| `revised_hr24_unique_pairs_200myr.csv` | Unique HR24 separation-limited pair pool with 200 Myr backtracking diagnostics. |
| `v2_hr24_static_random_pairs.csv` | Static HR24 random-pair quantities used for energy and phase-space comparisons. |

### Summary JSON Files

| File | Description |
|---|---|
| `revised_statistics.json` | Primary summary statistics for the revised analysis. |
| `v2_final_verdict.json` | Final machine-readable summary of the main conclusions. |
| `phase_space_coldness_summary.json` | Summary of phase-space coldness comparisons between observed pairs and HR24 baselines. |
| `energy_significance_tests.json` | Statistical tests comparing observed and HR24 energy-ratio distributions. |
| `hr24_ef_random_baseline_summary.json` | Summary of the HR24 expansion-factor replacement baseline. |
| `v2_hr24_random_baseline_summary.json` | Version 2 summary of the HR24 random baseline. |
| `revised_hr24_unique_summary_200myr.json` | Summary statistics for the unique HR24 separation-limited pool integrated over 200 Myr. |
| `v2_ef_dsi_fit_summary.json` | Fit summary for the relation between expansion factor and dynamical softness index. |
| `random_baseline_summary.json` | General random-baseline summary file. |
| `model_fits.json` | Auxiliary model-fit results. |

## Key Quantities

The analysis uses the following main diagnostics.

### Two-body energy ratio

For a pair with cluster masses \(M_1\) and \(M_2\), separation \(D\), and relative speed \(V_{\rm rel}\), the reduced mass is

\[
\mu = \frac{M_1 M_2}{M_1 + M_2}.
\]

The kinetic and gravitational energy scales are

\[
E_k = \frac{1}{2}\mu V_{\rm rel}^2,
\]

\[
E_g = \frac{G M_1 M_2}{D}.
\]

The dimensionless energy ratio is

\[
\eta = \frac{E_k}{E_g}.
\]

A classical two-body bound pair requires

\[
E_k/E_g < 1.
\]

All 83 high-confidence pairs in this analysis have \(E_k/E_g > 1\).

### Dynamical softness index

The dynamical softness index is

\[
{\rm DSI} = \frac{V_{\rm esc}}{V_{\rm rel}},
\]

where

\[
V_{\rm esc} = \left[\frac{2G(M_1+M_2)}{D}\right]^{1/2}.
\]

Bound systems require \(V_{\rm rel} < V_{\rm esc}\), or equivalently DSI \(> 1\).

### Expansion factor

For each pair, the orbit is integrated backward in a smooth Galactic potential. The minimum separation over the integration window is

\[
D_{\min} = \min_t D(t).
\]

The expansion factor is

\[
{\rm EF} = \frac{D_{\rm now}}{D_{\min}}.
\]

Values EF \(> 1\) indicate that the pair reached a smaller separation in the past. EF is used only as a kinematic proximity diagnostic, not as proof of common origin or gravitational binding.

## Main Results

The revised analysis gives the following headline results:

| Diagnostic | Result |
|---|---|
| Number of high-confidence pairs | 83 |
| Bound pairs under two-body criterion | 0 |
| Median \(E_k/E_g\) | 6568 |
| Minimum \(E_k/E_g\) | 72.7 |
| Bound pairs after mass ×3 | 0 |
| Bound pairs after mass ×10 | 0 |
| Fraction with EF \(> 1\) | 61.45% |
| Fraction with \(D_{\min} < 50\) pc | 37.35% |
| Age versus \(\log_{10}(E_k/E_g)\) | \(\rho = -0.107\), \(p = 0.337\) |
| Unique HR24 EF KS test | \(p = 0.205\) |
| Unique HR24 EF Anderson-Darling test | \(p = 0.069\) |
| Observed-size HR24 empirical EF tests | \(p \simeq 0.093-0.104\) |

The HR24 replacement baseline gives smaller formal p-values, but it is treated as auxiliary because it reuses a finite pair pool. The unique HR24 separation-limited pool is used as the main empirical baseline.

## Reproducing the Analysis

A typical workflow is:

```bash
python revise_analysis.py
python fossil_cluster_complex.py
python binary_cluster_dissolution.py
python run_smoke_test.py
```

The exact runtime depends on whether orbit-backtracking products have already been generated.

## Dependencies

The analysis uses standard Python scientific packages, including:

- `numpy`
- `pandas`
- `scipy`
- `astropy`
- `matplotlib`
- `statsmodels`
- `galpy`

The orbit integrations were performed with `galpy` using a smooth Milky Way potential.

## Data Sources

The analysis is based on Gaia DR3-derived open-cluster parameters and cluster-pair candidates. The random-pair baselines are drawn from the Hunt & Reffert open-cluster catalogue:

- Hunt & Reffert (2024), A&A, 686, A42

The high-confidence cluster-pair candidate sample follows the cluster-pair catalogue described in the associated manuscript and related literature.

## Interpretation Notes

The repository distinguishes three levels of interpretation:

1. **Classical bound binary clusters**  
   Ruled out for the 83 high-confidence pairs by present-day two-body energy diagnostics.

2. **Possible fossil-complex remnants**  
   Possible for a small subset of unbound pairs with small \(D_{\min}\), EF \(> 1\), age consistency, and plausible closest-approach times.

3. **Chance or loose phase-space associations**  
   Likely for the remaining systems unless future data provide stronger evidence for common origin.

The orbit-backtracking results should not be interpreted as birth-site reconstructions. They are deterministic integrations in an idealised Galactic potential and do not include full uncertainty propagation, spiral arms, the Galactic bar, molecular-cloud encounters, or detailed cluster disruption physics.

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{Chi2026ClusterPairs,
  title = {The dynamical nature of Galactic cluster-pair candidates from Gaia DR3},
  author = {Chi, Huanbin and Lv, Jinli and Xu, Hong and Wang, Feng},
  journal = {Astronomy & Astrophysics},
  year = {2026},
  note = {submitted}
}
```

## License

Please add the intended license for this repository. For scientific reproducibility, an open-source license such as MIT, BSD-3-Clause, or GPL-3.0 is recommended.

## Contact

For questions about the analysis, please contact:

**Huanbin Chi**  
Yunnan Open University  
Email: chihuanbin@126.com
