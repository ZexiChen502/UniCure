import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# from cuml.manifold import TSNE
from collections import Counter
import colorcet as cc
from sklearn.decomposition import PCA
# import umap
from scipy.interpolate import splprep, splev

# ============================================================
# Global config
# ============================================================
seed = 3
dataset_name = 'lincs2020'
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
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'raw_data', 'fig3', 'moa_dose')
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

node_size = 20

# Cell lines
cell_list = ["A375", "MCF7", "PC3", "A549"]

# Figure mapping for output naming
# A375 -> Fig. 3A (drug), Fig. 3B (dose)
# MCF7 -> Fig. S5A (drug), Fig. S5B (dose)
# PC3  -> Fig. S5C (drug), Fig. S5D (dose)
# A549 -> Fig. S5E (drug), Fig. S5F (dose)

figure_label = {
    "A375": {"drug": "fig3a", "dose": "fig3b"},
    "MCF7": {"drug": "figs5a", "dose": "figs5b"},
    "PC3":  {"drug": "figs5c", "dose": "figs5d"},
    "A549": {"drug": "figs5e", "dose": "figs5f"},
}


# ============================================================
# Upstream coordinate-generation workflow
# ============================================================
# The released MoA/dose plotting tables record representative two-dimensional
# layouts selected from candidate PCA/t-SNE projections of predicted and observed
# expression profiles. The upstream workflow followed the same pattern as the
# Fig. 2 t-SNE coordinate generation: average repeated cell / drug / dose
# conditions, separate metadata from expression values, traverse a PCA/t-SNE
# parameter grid, save candidate coordinate tables, and select representative
# layouts for the released panels.
#
# Example upstream code:
#
# save_dir = os.path.join(r"./result", str(seed), dataset_name)
# predictions_df = pd.read_csv(os.path.join(save_dir, "predictions.csv"))
# real_df = pd.read_csv(os.path.join(save_dir, "real_outputs.csv"))
#
# for source_name, source_df in [("predict", predictions_df), ("real", real_df)]:
#     source_df = source_df.groupby(
#         ["cell_type", "drug_name", "drug_dose"]
#     ).mean().reset_index()
#     source_df["cell_type"] = source_df["cell_type"].str.split("_").str[1]
#
#     for cell_name in cell_list:
#         cell_df = source_df[source_df["cell_type"].isin([cell_name])].copy()
#         drug_counts = Counter(cell_df["drug_name"])
#         top_drugs = pd.DataFrame(
#             drug_counts.items(), columns=["drug_name", "count"]
#         ).sort_values(by="count", ascending=False)["drug_name"].head(35).tolist()
#         filtered_df = cell_df[cell_df["drug_name"].isin(top_drugs)]
#         meta_info = filtered_df[["cell_type", "drug_name", "drug_dose"]].copy()
#         expression_values = filtered_df.iloc[:, 3:]
#
#         for pca_config in PCA_PARAMETER_GRID:
#             pca_input = run_pca_preprocessing(expression_values, pca_config)
#             for tsne_config in TSNE_PARAMETER_GRID:
#                 coords = run_tsne_projection(pca_input, tsne_config)
#                 out = meta_info.copy()
#                 out[["TSNE1", "TSNE2"]] = coords
#                 out.to_csv(
#                     f"{source_name}_{cell_name}_tsne_candidate_{pca_config}_{tsne_config}.csv",
#                     index=False,
#                 )


# ============================================================
# Plotting functions
# ============================================================

def plot_drug_colored(cell_name):
    """
    Plot predicted vs real t-SNE for a given cell line, colored by drug name.
    Left panel: UniCure predicted
    Right panel: experimentally observed (real)
    """
    pred_path = os.path.join(RAW_DATA_DIR, f'{cell_name}_tsne_drug_results.csv')
    real_path = os.path.join(RAW_DATA_DIR, f'real_{cell_name}_tsne_drug_results.csv')

    df_pred = pd.read_csv(pred_path)
    df_real = pd.read_csv(real_path)

    # Build consistent color palette across predicted and real
    all_drugs = sorted(set(df_pred['drug_name'].unique()) | set(df_real['drug_name'].unique()))
    palette = sns.color_palette(cc.glasbey_category10, n_colors=len(all_drugs))
    drug_colors = dict(zip(all_drugs, palette))

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Left: Predicted
    sns.scatterplot(
        data=df_pred,
        x='TSNE1', y='TSNE2',
        hue='drug_name',
        palette=drug_colors,
        s=node_size,
        alpha=1,
        linewidth=0,
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title(f'{cell_name} - Predicted', fontsize=16)
    axes[0].set_xlabel('tSNE 1')
    axes[0].set_ylabel('tSNE 2')

    # Right: Real
    sns.scatterplot(
        data=df_real,
        x='TSNE1', y='TSNE2',
        hue='drug_name',
        palette=drug_colors,
        s=node_size,
        alpha=1,
        linewidth=0,
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title(f'{cell_name} - Real', fontsize=16)
    axes[1].set_xlabel('tSNE 1')
    axes[1].set_ylabel('tSNE 2')

    plt.tight_layout()

    label = figure_label[cell_name]["drug"]
    plt.savefig(os.path.join(OUTPUT_DIR, f'{label}_{cell_name.lower()}_drug.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{label}_{cell_name.lower()}_drug.pdf'), transparent=True)
    plt.close()
    print(f"  Saved {label}_{cell_name.lower()}_drug")


def plot_dose_colored(cell_name):
    """
    Plot predicted vs real t-SNE for a given cell line, colored by drug dose.
    Left panel: UniCure predicted
    Right panel: experimentally observed (real)
    """
    pred_path = os.path.join(RAW_DATA_DIR, f'{cell_name}_tsne_drug_results.csv')
    real_path = os.path.join(RAW_DATA_DIR, f'real_{cell_name}_tsne_drug_results.csv')

    df_pred = pd.read_csv(pred_path)
    df_real = pd.read_csv(real_path)

    # Merge all doses for a shared sorted list
    all_doses = sorted(set(df_pred['drug_dose'].unique()) | set(df_real['drug_dose'].unique()))

    # Use a perceptually uniform sequential colormap
    norm = matplotlib.colors.LogNorm(vmin=min(all_doses), vmax=max(all_doses))
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Left: Predicted
    scatter0 = axes[0].scatter(
        df_pred['TSNE1'], df_pred['TSNE2'],
        c=df_pred['drug_dose'],
        cmap=cmap,
        norm=norm,
        s=node_size,
        alpha=1,
        linewidth=0
    )
    axes[0].set_title(f'{cell_name} - Predicted', fontsize=16)
    axes[0].set_xlabel('tSNE 1')
    axes[0].set_ylabel('tSNE 2')

    # Right: Real
    scatter1 = axes[1].scatter(
        df_real['TSNE1'], df_real['TSNE2'],
        c=df_real['drug_dose'],
        cmap=cmap,
        norm=norm,
        s=node_size,
        alpha=1,
        linewidth=0
    )
    axes[1].set_title(f'{cell_name} - Real', fontsize=16)
    axes[1].set_xlabel('tSNE 1')
    axes[1].set_ylabel('tSNE 2')

    # Shared colorbar
    cbar = fig.colorbar(scatter1, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Drug Dose (uM)', fontsize=12)

    fig.subplots_adjust(right=0.93)

    label = figure_label[cell_name]["dose"]
    plt.savefig(os.path.join(OUTPUT_DIR, f'{label}_{cell_name.lower()}_dose.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{label}_{cell_name.lower()}_dose.pdf'), transparent=True)
    plt.close()
    print(f"  Saved {label}_{cell_name.lower()}_dose")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Fig. 3 MoA / Dose t-SNE Reproducibility Script")
    print("=" * 60)
    print()

    for cell_name in cell_list:
        print(f"Processing {cell_name}...")
        plot_drug_colored(cell_name)
        plot_dose_colored(cell_name)

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
