"""
Benchmark Delta Pearson Correlation Analysis
=============================================
This script demonstrates the core function used to compute Delta Pearson
correlation at different gene-sensitivity thresholds, as described in
Fig. 2G of the UniCure manuscript.

The key function `calculate_delta_pearson` computes, for each perturbation
sample, the Pearson correlation between predicted and real transcriptomic
changes (deltas) on progressively larger gene subsets ranked by effect size.

Usage in the manuscript:
  - LINCS 2020 (bulk cell-line, top 50 cell lines, n=5173)
  - sci-Plex 3 (single-cell perturbation, 3 cell lines, n=226)
  - Models compared: UniCure, TranSiGen, PRnet
  - TranSiGen and PRnet were trained on the same training set as UniCure
    following the settings reported in the corresponding papers and the
    parameters released in their GitHub repositories, then evaluated on the
    same held-out test sets.

This file documents the metric construction used by the released Fig. 2G
benchmark workflow and provides reusable helper functions for applying the
same Delta Pearson calculation to prediction, observation, and control
matrices.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ===========================================================================
# Core Function
# ===========================================================================

def calculate_delta_pearson(pred, real, ctrl, thresholds=None):
    """
    Compute Delta Pearson correlation at multiple gene-sensitivity thresholds.

    For a single perturbation sample, this function:
      1. Computes the transcriptomic delta: treated - control (for both
         predicted and real values).
      2. Ranks genes by the absolute magnitude of the real delta (i.e.,
         genes most affected by the perturbation rank first).
      3. For each threshold t, takes the top t% of ranked genes and
         computes the Pearson correlation between predicted and real deltas.

    Parameters
    ----------
    pred : array-like, shape (n_genes,)
        Predicted gene expression for the treated sample.
    real : array-like, shape (n_genes,)
        Real (observed) gene expression for the treated sample.
    ctrl : array-like, shape (n_genes,)
        Gene expression of the control (untreated) sample.
    thresholds : list of float, optional
        Fractions of top-ranked genes to evaluate (default: 0.1 to 1.0
        in steps of 0.1).

    Returns
    -------
    dict
        {threshold: delta_pearson_r} for each threshold with >= 10 genes.
        Returns None if any input contains NaN.
    """
    if thresholds is None:
        thresholds = [round(x * 0.1, 1) for x in range(1, 11)]

    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)

    real_delta = real - ctrl
    pred_delta = pred - ctrl

    if np.isnan(real_delta).any() or np.isnan(pred_delta).any():
        return None

    abs_change = np.abs(real_delta)
    sorted_indices = np.argsort(abs_change)[::-1]
    n_genes = len(real_delta)

    results = {}
    for t in thresholds:
        top_n = int(n_genes * t)
        if top_n < 10:
            continue

        idx = sorted_indices[:top_n]
        r_sub = real_delta[idx]
        p_sub = pred_delta[idx]

        if np.std(r_sub) < 1e-9 or np.std(p_sub) < 1e-9:
            results[t] = 0.0
        else:
            results[t], _ = pearsonr(p_sub, r_sub)

    return results


def batch_delta_pearson(pred_matrix, real_matrix, ctrl_matrix,
                        thresholds=None):
    """
    Apply `calculate_delta_pearson` across multiple samples and return a
    tidy DataFrame suitable for plotting.

    Parameters
    ----------
    pred_matrix : np.ndarray, shape (n_samples, n_genes)
    real_matrix : np.ndarray, shape (n_samples, n_genes)
    ctrl_matrix : np.ndarray, shape (n_samples, n_genes)
    thresholds : list of float, optional

    Returns
    -------
    pd.DataFrame
        Columns: Sample_ID, Threshold, Delta_r
    """
    rows = []
    for i in range(len(pred_matrix)):
        res = calculate_delta_pearson(pred_matrix[i], real_matrix[i],
                                      ctrl_matrix[i], thresholds)
        if res is None:
            continue
        for t, r in res.items():
            rows.append({"Sample_ID": i, "Threshold": t, "Delta_r": r})
    return pd.DataFrame(rows)


# ===========================================================================
# Top-K Cell Line Selection
# ===========================================================================

def select_top_k_cell_lines(metric_df, cell_line_col, score_col, k=50):
    """
    Select the top-k cell lines ranked by average perturbation signal quality.

    In the manuscript, we selected the top 50 cell lines based on the average
    Delta Pearson correlation across all perturbation conditions and models,
    as a proxy for perturbation signal quality. This ensures that the
    benchmark evaluation focuses on cell lines with strong, reproducible
    perturbation responses.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Per-sample metrics with cell line identifiers and scores.
    cell_line_col : str
        Column name for cell line identifiers.
    score_col : str
        Column name for the ranking metric (e.g., delta_pearson_r).
    k : int
        Number of top cell lines to select (default: 50).

    Returns
    -------
    list of str
        Top-k cell line names ranked by descending mean score.
    """
    ranked = (metric_df.groupby(cell_line_col)[score_col]
              .mean()
              .sort_values(ascending=False)
              .reset_index())
    return ranked.head(k)[cell_line_col].tolist()


# ===========================================================================
# Demonstration: LINCS 2020 Benchmark
# ===========================================================================

def demo_lincs():
    """
    Demonstrate the benchmark analysis on LINCS 2020 (bulk cell-line dataset).

    Dataset: LINCS 2020
      - 50 cell lines (top 50 selected by signal quality), n = 5173 samples
      - Gene expression in landmark gene space (978 genes)

    Models evaluated under the same train/validation/test split:
      - UniCure
      - TranSiGen
      - PRnet

    The analysis proceeds as follows:
      1. (Optional) Select top-50 cell lines using `select_top_k_cell_lines`.
      2. For each model and sample, compute delta Pearson correlation at
         10 thresholds (10% to 100%) using `batch_delta_pearson`.
      3. Aggregate results into a single DataFrame for plotting.
    """

    # --- Step 1: Top-50 cell line selection ---
    # Load per-sample metrics (e.g., from UniCure's evaluation on all cell lines)
    # metrics = pd.read_csv("lincs_benchmark_metric_unicure.csv")
    # Extract cell line from sample ID (e.g., "ASG001_MCF7" -> "MCF7")
    # metrics["cell_line"] = metrics["new_cid"].str.split("_").str[1]
    # top50 = select_top_k_cell_lines(metrics, "cell_line", "pearson_r", k=50)

    # Or load a pre-computed ranking:
    # ranking = pd.read_csv("cell_line_avg_delta_sign_rank.csv")
    # top50 = ranking.head(50)["cell_line"].tolist()

    # --- Step 2: Compute delta Pearson for each model ---
    # For each model (UniCure, TranSiGen, PRnet), the inputs are:
    #   pred_matrix: shape (n_samples, n_genes) - predicted treated expression
    #   real_matrix: shape (n_samples, n_genes) - real treated expression
    #   ctrl_matrix: shape (n_samples, n_genes) - control expression

    # Example (pseudocode):
    # for model_name, pred, real, ctrl in [("UniCure", ...), ("TranSiGen", ...), ("PRnet", ...)]:
    #     df = batch_delta_pearson(pred, real, ctrl)
    #     df["Model"] = model_name
    #     all_results.append(df)
    # lincs_df = pd.concat(all_results, ignore_index=True)

    pass  # Demonstration only


# ===========================================================================
# Demonstration: sci-Plex 3 Benchmark
# ===========================================================================

def demo_sciplex3():
    """
    Demonstrate the benchmark analysis on sci-Plex 3 (single-cell dataset).

    Dataset: sci-Plex 3
      - 3 cell lines (A549, K562, MCF7), n = 226 samples
      - Single-cell RNA-seq processed to mean expression per condition

    Models evaluated under the same train/validation/test split:
      - UniCure
      - TranSiGen
      - PRnet

    The workflow is identical to LINCS 2020, except no cell-line filtering
    is needed (all 3 cell lines are used).
    """

    # Example (pseudocode):
    # for model_name, pred, real, ctrl in [("UniCure", ...), ("TranSiGen", ...), ("PRnet", ...)]:
    #     df = batch_delta_pearson(pred, real, ctrl)
    #     df["Model"] = model_name
    #     all_results.append(df)
    # sciplex_df = pd.concat(all_results, ignore_index=True)

    pass  # Demonstration only


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("This script documents the core Delta Pearson correlation analysis")
    print("used in Fig. 2G of the UniCure manuscript.")
    print()
    print("Key function: calculate_delta_pearson(pred, real, ctrl)")
    print("See function docstrings for full documentation.")
