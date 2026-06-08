"""
Drug recommendation workflow — UniCure Fig. 5H-K and Fig. S16A-D

This script documents the TCGA drug-ranking workflow and regenerates the
Recommended-versus-Randomized rank-distribution boxplots from released tables.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SCRIPT_DIR)


def find_repo_root():
    current = SCRIPT_DIR
    while True:
        if os.path.exists(os.path.join(current, 'README.md')) and os.path.exists(os.path.join(current, 'model.py')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(os.path.join(MODULE_DIR, '..', '..', '..'))
        current = parent


REPO_ROOT = find_repo_root()
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'raw_data', 'fig5', 'drug_recommendation')
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Upstream drug-ranking workflow with concentration loop
# =============================================================================
# The upstream pipeline generates {cancer}_Recommended_vs_Randomized_ft.csv
# using the pretrained UniCure model (see quick_pred.py for the canonical
# inference interface). Key steps:
#
# import torch
# from utils import load_UniCure_pretrained_model
#
# DOSES = [0.04, 0.12, 0.37, 1.11, 3.33, 10.0]   # uM, six concentration levels
# DRUG_INPUT_DIM = 528                            # model expects 528-dim drug input
#
# model = load_UniCure_pretrained_model(path=r'\path_to\Unicure_best_model.pth')
# model.to(device).eval()
#
# # For each sample, average reversal correlation across all dose levels
# for i in range(sample_emd.shape[0]):
#     delta_cancer = unperturb.iloc[i] - normal.iloc[i]
#     dose_corr_list = []
#
#     for dose in DOSES:
#         # Build drug tensor following quick_pred.py:
#         #   1) take raw Uni-Mol drug embedding
#         #   2) concatenate log10(dose+1) as an extra column
#         #   3) right-pad with the same dose value to reach 528 dims
#         dose_val = np.log10(dose + 1)
#         drug_raw = torch.tensor(lincs_drug_embedding.values).float().to(device)        # (n_drugs, d_emb)
#         dose_col = torch.full((drug_raw.shape[0], 1), dose_val).to(device)
#         drug_combined = torch.cat([drug_raw, dose_col], dim=1)                         # (n_drugs, d_emb + 1)
#         pad_len = DRUG_INPUT_DIM - drug_combined.shape[1]
#         if pad_len > 0:
#             pad = torch.full((drug_combined.shape[0], pad_len), dose_val).to(device)
#             drug_combined = torch.cat([drug_combined, pad], dim=1)                     # (n_drugs, 528)
#
#         cell_tensor = torch.tensor(
#             np.tile(sample_emd.iloc[i].values, (drug_combined.shape[0], 1))).float().to(device)
#
#         with torch.no_grad():
#             pred = model("pertrub_forward", cell_tensor, drug_combined).cpu().numpy()  # (n_drugs, n_genes)
#
#         outputs = pd.DataFrame(pred, index=lincs_drug_embedding.index, columns=gene_list)
#         corr_at_dose = {drug: (outputs.loc[drug] - unperturb.iloc[i]).corr(delta_cancer)
#                         for drug in outputs.index}
#         dose_corr_list.append(pd.Series(corr_at_dose))
#
#     # Average across doses, then rank (ascending: rank 1 = best reversal)
#     avg_corr = pd.concat(dose_corr_list, axis=1).mean(axis=1)
#     correlation_matrix[sample_emd.index[i]] = avg_corr
#
# ranked_matrix = correlation_matrix.rank(ascending=True, method='min', axis=0)
# # Randomized baseline: shuffle ranks within each sample
# shuffled = ranked_matrix.apply(lambda col: col.sample(frac=1).values, axis=0)
# # Extract indicated drugs and save
# plot_data = [{"Drug": d, "Group": "Recommended", "Rank": r}
#              for d in indicated_drugs for r in ranked_matrix.loc[d]]
# plot_data += [{"Drug": d, "Group": "Randomized", "Rank": r}
#               for d in indicated_drugs for r in shuffled.loc[d]]
# pd.DataFrame(plot_data).to_csv('raw_data/Fig5/drug_recommendation/{cancer}_Recommended_vs_Randomized_ft.csv')

# =============================================================================
# Released-table plotting workflow
# =============================================================================

# Figure mapping: cancer -> (figure label, sample count)
FIGURE_MAP = {
    "LUAD": ("fig5h",   57),
    "BRCA": ("fig5i",  112),
    "BLCA": ("fig5j",   19),
    "LUSC": ("fig5k",   49),
    "KIRC": ("figs16a", 72),
    "LIHC": ("figs16b", 50),
    "COAD": ("figs16c", 41),
    "PRAD": ("figs16d", 52),
}

PALETTE = {"Recommended": "#FFACAC", "Randomized": "#3A98B9"}


def plot_cancer(cancer, fig_label, n_samples):
    df = pd.read_csv(os.path.join(RAW_DATA_DIR,
                                  f"{cancer}_Recommended_vs_Randomized_ft.csv"),
                     index_col=0)
    drugs = df[df["Group"] == "Recommended"]["Drug"].unique().tolist()
    n_drugs = len(drugs)

    fig, ax = plt.subplots(figsize=(max(6, n_drugs * 2.2), 5))

    positions = []
    for i, drug in enumerate(drugs):
        rec = df[(df["Drug"] == drug) & (df["Group"] == "Recommended")]["Rank"].values
        rnd = df[(df["Drug"] == drug) & (df["Group"] == "Randomized")]["Rank"].values
        x_rec = i - 0.2
        x_rnd = i + 0.2
        bp_rec = ax.boxplot(rec, positions=[x_rec], widths=0.35,
                            patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor=PALETTE["Recommended"]),
                            medianprops=dict(color="black"),
                            whiskerprops=dict(color="black"),
                            capprops=dict(color="black"),
                            flierprops=dict(marker="o", markersize=2, alpha=0.4,
                                            markerfacecolor=PALETTE["Recommended"]))
        bp_rnd = ax.boxplot(rnd, positions=[x_rnd], widths=0.35,
                            patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor=PALETTE["Randomized"]),
                            medianprops=dict(color="black"),
                            whiskerprops=dict(color="black"),
                            capprops=dict(color="black"),
                            flierprops=dict(marker="o", markersize=2, alpha=0.4,
                                            markerfacecolor=PALETTE["Randomized"]))
        # One-sided Mann-Whitney U test: recommended ranks are expected to be lower than randomized ranks.
        _, p = mannwhitneyu(rec, rnd, alternative="less")
        y_top = df["Rank"].max() * 1.05
        h = df["Rank"].max() * 0.04
        ax.plot([x_rec, x_rec, x_rnd, x_rnd], [y_top, y_top + h, y_top + h, y_top],
                lw=1.2, c="black")
        p_str = f"p={p:.2e}" if p >= 0.0001 else "p<0.0001"
        ax.text((x_rec + x_rnd) / 2, y_top + h * 1.1, p_str,
                ha="center", va="bottom", fontsize=9)
        positions.append(i)

    ax.set_xticks(positions)
    ax.set_xticklabels(drugs, rotation=0, fontsize=11)
    ax.set_ylabel("Rank", fontsize=12)
    ax.set_xlabel(f"Indicated therapies for {cancer} (n={n_samples})", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PALETTE["Recommended"], label="Recommended"),
                       Patch(facecolor=PALETTE["Randomized"], label="Randomized")],
              fontsize=10, frameon=False)
    fig.tight_layout()

    name = f"{fig_label}_{cancer.lower()}_drug_recommendation"
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


for cancer, (fig_label, n) in FIGURE_MAP.items():
    plot_cancer(cancer, fig_label, n)
