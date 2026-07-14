import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from cuml.manifold import TSNE
from collections import Counter
import colorcet as cc
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev

# ============================================================
# Configuration
# ============================================================
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
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'raw_data', 'fig2', 'tsne')
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

node_size = 20

# ============================================================
# Fig. 2C - LINCS unperturbed gene expression (cell type colored)
# ============================================================
def plot_fig2c():
    print("Plotting Fig. 2C: LINCS unperturbed gene expression...")
    
    # Upstream coordinate-generation workflow:
    #   1. Load LINCS control profiles and separate metadata from gene-expression values.
    #   2. Traverse a grid of dimensionality-reduction settings, including PCA preprocessing
    #      and t-SNE hyperparameters such as perplexity, learning rate, iteration number,
    #      initialization, and random seed.
    #   3. Save candidate two-dimensional coordinate tables and select the representative
    #      layout used for the released panel.
    #
    # Example upstream code:
    # unperturb_cell = pd.read_parquet("lincs2020_control.parquet")
    # meta_info = unperturb_cell.iloc[:, :4].copy()
    # gene_expressions = unperturb_cell.iloc[:, 4:]
    # for n_pca in [30, 50, 100]:
    #     pca_input = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(gene_expressions))
    #     for perplexity in [30, 50, 80]:
    #         for learning_rate in [200, 500, 1000]:
    #             for seed in [0, 1, 2, 3, 4]:
    #                 tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
    #                             n_iter=2000, init="pca", random_state=seed)
    #                 coords = tsne.fit_transform(pca_input)
    #                 out = meta_info.copy()
    #                 out[["TSNE1", "TSNE2"]] = coords
    #                 out.to_csv(f"lincs_control_tsne_pca{n_pca}_perp{perplexity}_lr{learning_rate}_seed{seed}.csv", index=False)
    # The selected coordinate table is read below to regenerate the figure.
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'lincs_control_tsne.csv'))
    
    cell_type_counts = Counter(df['cell_iname'])
    all_cell_types = sorted(cell_type_counts.keys(), key=lambda x: -cell_type_counts[x])
    palette = sns.color_palette(cc.glasbey_light, n_colors=len(all_cell_types))
    cell_type_colors = dict(zip(all_cell_types, palette))
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=df,
        x='TSNE1',
        y='TSNE2',
        hue='cell_iname',
        palette=cell_type_colors,
        s=node_size,
        alpha=1,
        linewidth=0,
        legend=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2c_lincs_control_tsne.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2c_lincs_control_tsne.pdf'), transparent=True)
    plt.close()
    print("  Saved fig2c_lincs_control_tsne")

# ============================================================
# Fig. 2D - LINCS embedded features after UCE_lora
# ============================================================
def plot_fig2d():
    print("Plotting Fig. 2D: LINCS embedded features (UCE_lora)...")
    
    # Upstream coordinate-generation workflow:
    #   1. Load LINCS UCE-LoRA embedding profiles and retain cell-line labels as metadata.
    #   2. Traverse a grid of PCA/t-SNE settings to generate candidate two-dimensional
    #      embedding layouts.
    #   3. Save candidate coordinate tables and select the representative layout used for
    #      the released panel.
    #
    # Example upstream code:
    # unperturb_emb = pd.read_parquet("lincs2020_uce_lora_emb.parquet").reset_index()
    # unperturb_emb["index"] = unperturb_emb["index"].str.split("_").str[1]
    # meta_info = unperturb_emb.iloc[:, :1].copy()
    # embedding_values = unperturb_emb.iloc[:, 1:]
    # for n_pca in [30, 50, 100]:
    #     pca_input = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(embedding_values))
    #     for perplexity in [30, 50, 80]:
    #         for learning_rate in [200, 500, 1000]:
    #             for seed in [0, 1, 2, 3, 4]:
    #                 tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
    #                             n_iter=2000, init="pca", random_state=seed)
    #                 coords = tsne.fit_transform(pca_input)
    #                 out = meta_info.copy()
    #                 out[["TSNE1", "TSNE2"]] = coords
    #                 out.to_csv(f"lincs_tsne_uce_lora_emb_pca{n_pca}_perp{perplexity}_lr{learning_rate}_seed{seed}.csv", index=False)
    # The selected embedding coordinate table is read below to regenerate the figure.
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'lincs_tsne_uce_lora_emb.csv'))
    
    cell_type_counts = Counter(df['index'])
    all_cell_types = sorted(cell_type_counts.keys(), key=lambda x: -cell_type_counts[x])
    palette = sns.color_palette(cc.glasbey_light, n_colors=len(all_cell_types))
    cell_type_colors = dict(zip(all_cell_types, palette))
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=df,
        x='TSNE1',
        y='TSNE2',
        hue='index',
        palette=cell_type_colors,
        s=node_size,
        alpha=1,
        linewidth=0,
        legend=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2d_lincs_uce_lora_emb_tsne.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2d_lincs_uce_lora_emb_tsne.pdf'), transparent=True)
    plt.close()
    print("  Saved fig2d_lincs_uce_lora_emb_tsne")

# ============================================================
# Fig. 2E - LINCS UniCure-predicted perturbed gene expression
# ============================================================
def plot_fig2e():
    print("Plotting Fig. 2E: LINCS predicted perturbed gene expression...")
    
    # Upstream coordinate-generation workflow:
    #   1. Load UniCure-predicted LINCS perturbed-expression profiles with cell, drug,
    #      and dose metadata.
    #   2. Traverse a grid of PCA/t-SNE settings to generate candidate predicted-profile
    #      coordinate layouts.
    #   3. Save candidate coordinate tables and select the representative layout used for
    #      the released panel.
    #
    # Example upstream code:
    # predictions_df = pd.read_csv("predictions.csv")
    # meta_info = predictions_df[["cell_type", "drug_name", "drug_dose"]].copy()
    # meta_info["cell_type"] = meta_info["cell_type"].str.split("_").str[1]
    # predicted_expression = predictions_df.iloc[:, 3:]
    # for n_pca in [30, 50, 100]:
    #     pca_input = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(predicted_expression))
    #     for perplexity in [30, 50, 80]:
    #         for learning_rate in [200, 500, 1000]:
    #             for seed in [0, 1, 2, 3, 4]:
    #                 tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
    #                             n_iter=2000, init="pca", random_state=seed)
    #                 coords = tsne.fit_transform(pca_input)
    #                 out = meta_info.copy()
    #                 out[["TSNE1", "TSNE2"]] = coords
    #                 out.to_csv(f"lincs_predict_tsne_pca{n_pca}_perp{perplexity}_lr{learning_rate}_seed{seed}.csv", index=False)
    # The selected prediction coordinate table is read below to regenerate the figure.
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'lincs_predict_tsne_result.csv'))
    
    cell_type_counts = Counter(df['cell_type'])
    all_cell_types = sorted(cell_type_counts.keys(), key=lambda x: -cell_type_counts[x])
    palette = sns.color_palette(cc.glasbey_light, n_colors=len(all_cell_types))
    cell_type_colors = dict(zip(all_cell_types, palette))
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=df,
        x='TSNE1',
        y='TSNE2',
        hue='cell_type',
        palette=cell_type_colors,
        s=node_size,
        alpha=1,
        linewidth=0,
        legend=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2e_lincs_predict_tsne.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2e_lincs_predict_tsne.pdf'), transparent=True)
    plt.close()
    print("  Saved fig2e_lincs_predict_tsne")

# ============================================================
# Fig. 2F - sci-Plex3 predicted vs observed (cell type colored)
# ============================================================
def plot_fig2f():
    print("Plotting Fig. 2F: sci-Plex3 predicted vs observed...")
    
    cell_type_colors = {"MCF7": "#FF7517", "K562": "#8FD14F", "A549": "#46C3DB"}
    
    # Upstream coordinate-generation workflow:
    #   1. Load UniCure-predicted sci-Plex3 perturbed-expression profiles with cell,
    #      drug, and dose metadata.
    #   2. Traverse a grid of PCA/t-SNE settings to generate candidate coordinate layouts.
    #   3. Save candidate coordinate tables and select the representative predicted-profile
    #      layout used for the released panel.
    #
    # Example upstream code:
    # predictions_df = pd.read_csv("predictions.csv")
    # meta_info = predictions_df[["cell_type", "drug_name", "drug_dose"]].copy()
    # predicted_expression = predictions_df.iloc[:, 3:]
    # for n_pca in [30, 50, 100]:
    #     pca_input = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(predicted_expression))
    #     for perplexity in [30, 50, 80]:
    #         for learning_rate in [200, 500, 1000]:
    #             for seed in [0, 1, 2, 3, 4]:
    #                 tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
    #                             n_iter=2000, init="pca", random_state=seed)
    #                 coords = tsne.fit_transform(pca_input)
    #                 out = meta_info.copy()
    #                 out[["TSNE1", "TSNE2"]] = coords
    #                 out.to_csv(f"sciplex_predict_tsne_pca{n_pca}_perp{perplexity}_lr{learning_rate}_seed{seed}.csv", index=False)
    # Left: Predicted 
    df_pred = pd.read_csv(os.path.join(RAW_DATA_DIR, 'sciplex_predict_tsne_result.csv'))
    plt.figure(figsize=(20, 14))
    sns.scatterplot(
        data=df_pred,
        x='TSNE1',
        y='TSNE2',
        hue='cell_type',
        palette=cell_type_colors,
        s=node_size,
        alpha=1,
        linewidth=0,
        legend=True
    )
    legend = plt.legend(
        title='Cell Type',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        markerscale=2
    )
    plt.setp(legend.get_texts(), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2f_left_sciplex3_predict_tsne.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2f_left_sciplex3_predict_tsne.pdf'), transparent=True)
    plt.close()
    
    # Upstream coordinate-generation workflow:
    #   1. Load observed sci-Plex3 perturbed-expression profiles with matched metadata.
    #   2. Traverse the corresponding PCA/t-SNE parameter grid to generate candidate
    #      observed-profile coordinate layouts.
    #   3. Save candidate coordinate tables and select the representative observed-profile
    #      layout used for the released panel.
    #
    # Example upstream code:
    # real_df = pd.read_csv("real_outputs.csv")
    # meta_info = real_df[["cell_type", "drug_name", "drug_dose"]].copy()
    # observed_expression = real_df.iloc[:, 3:]
    # for n_pca in [30, 50, 100]:
    #     pca_input = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(observed_expression))
    #     for perplexity in [30, 50, 80]:
    #         for learning_rate in [200, 500, 1000]:
    #             for seed in [0, 1, 2, 3, 4]:
    #                 tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
    #                             n_iter=2000, init="pca", random_state=seed)
    #                 coords = tsne.fit_transform(pca_input)
    #                 out = meta_info.copy()
    #                 out[["TSNE1", "TSNE2"]] = coords
    #                 out.to_csv(f"sciplex_real_tsne_pca{n_pca}_perp{perplexity}_lr{learning_rate}_seed{seed}.csv", index=False)
    # The selected observed-profile coordinate table is read below to regenerate the figure.
    df_real = pd.read_csv(os.path.join(RAW_DATA_DIR, 'sciplex_real_tsne_result.csv'))
    plt.figure(figsize=(20, 14))
    sns.scatterplot(
        data=df_real,
        x='TSNE1',
        y='TSNE2',
        hue='cell_type',
        palette=cell_type_colors,
        s=node_size,
        alpha=1,
        linewidth=0,
        legend=True
    )
    legend = plt.legend(
        title='Cell Type',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        markerscale=2
    )
    plt.setp(legend.get_texts(), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2f_right_sciplex3_real_tsne.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2f_right_sciplex3_real_tsne.pdf'), transparent=True)
    plt.close()
    
    print("  Saved fig2f_left/right_sciplex3_tsne")

# ============================================================
# Fig. S3 A/B/C - sci-Plex3 by cell line (drug colored)
# ============================================================
def plot_fig_s3():
    print("Plotting Fig. S3: sci-Plex3 by cell line (drug colored)...")
    
    # Upstream coordinate-generation workflow:
    #   1. Split predicted and observed sci-Plex3 expression profiles by cell line.
    #   2. For each cell line, traverse a PCA/t-SNE parameter grid and save candidate
    #      predicted and observed coordinate tables.
    #   3. Select the representative layouts used for the released Fig. S5 panels.
    #
    # Example upstream code:
    # predictions_df = pd.read_csv("predictions.csv")
    # real_df = pd.read_csv("real_outputs.csv")
    # for source_name, source_df in [("predict", predictions_df), ("real", real_df)]:
    #     for cell_type, subset in source_df.groupby("cell_type"):
    #         meta_info = subset[["drug_name", "drug_dose", "cell_type"]].copy()
    #         expression_values = subset.iloc[:, 3:]
    #         for n_pca in [30, 50, 100]:
    #             pca_input = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(expression_values))
    #             for perplexity in [10, 20, 30, 50]:
    #                 for learning_rate in [200, 500, 1000]:
    #                     for seed in [0, 1, 2, 3, 4]:
    #                         tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
    #                                     n_iter=2000, init="pca", random_state=seed)
    #                         coords = tsne.fit_transform(pca_input)
    #                         out = meta_info.copy()
    #                         out[["TSNE1", "TSNE2"]] = coords
    #                         out.to_csv(
    #                             f"sciplex_{source_name}_tsne_{cell_type}_pca{n_pca}_perp{perplexity}_lr{learning_rate}_seed{seed}.csv",
    #                             index=False,
    #                         )
    # The released combined coordinate tables are filtered below by cell line for plotting.
    cell_lines = ['A549', 'K562', 'MCF7']
    
    for cell_type in cell_lines:
        print(f"  Processing {cell_type}...")
        
        # Read from combined CSV and filter by cell_type
        df_pred_all = pd.read_csv(os.path.join(RAW_DATA_DIR, 'sciplex_predict_tsne_result.csv'))
        df_pred = df_pred_all[df_pred_all['cell_type'] == cell_type].copy()
        
        all_drugs = sorted(df_pred['drug_name'].unique())
        drug_palette = sns.color_palette(cc.glasbey_light, n_colors=len(all_drugs))
        drug_colors = dict(zip(all_drugs, drug_palette))
        
        fig, axes = plt.subplots(1, 2, figsize=(28, 12))
        
        # Left: Predicted
        sns.scatterplot(
            data=df_pred,
            x='TSNE1',
            y='TSNE2',
            hue='drug_name',
            palette=drug_colors,
            s=node_size,
            linewidth=0,
            alpha=1,
            legend=False,
            ax=axes[0]
        )
        axes[0].set_title(f'{cell_type} - Predicted', fontsize=16)
        axes[0].set_xlabel('tSNE 1')
        axes[0].set_ylabel('tSNE 2')
        
        # Right: Real
        df_real_all = pd.read_csv(os.path.join(RAW_DATA_DIR, 'sciplex_real_tsne_result.csv'))
        df_real = df_real_all[df_real_all['cell_type'] == cell_type].copy()
        sns.scatterplot(
            data=df_real,
            x='TSNE1',
            y='TSNE2',
            hue='drug_name',
            palette=drug_colors,
            s=node_size,
            linewidth=0,
            alpha=1,
            legend=False,
            ax=axes[1]
        )
        axes[1].set_title(f'{cell_type} - Real', fontsize=16)
        axes[1].set_xlabel('tSNE 1')
        axes[1].set_ylabel('tSNE 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'figs3_{cell_type.lower()}_tsne.png'), dpi=300, transparent=True)
        plt.savefig(os.path.join(OUTPUT_DIR, f'figs3_{cell_type.lower()}_tsne.pdf'), transparent=True)
        plt.close()
        print(f"    Saved figs3_{cell_type.lower()}_tsne")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Fig. 2 t-SNE Visualization Reproducibility Script")
    print("=" * 60)
    print()
    
    plot_fig2c()
    plot_fig2d()
    plot_fig2e()
    plot_fig2f()
    plot_fig_s3()
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
