"""
Patient stratification workflow — UniCure Fig. 5B-C and Fig. S14A-B

This script documents UniCure patient-level drug-rank generation, K-means
clustering, and differential-drug analysis logic, and regenerates the PCA and
K-means evaluation panels from released tables.

Run:
    python code/patient_stratification_workflow.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'raw_data', 'fig5', 'patient_stratification')
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Upstream patient drug-rank generation workflow
# =============================================================================
# Generates {cancer}_rankmatrix.csv from TCGA tumor/normal expression and
# LINCS 2020 drug embeddings using the pretrained UniCure model.
#
# Key inputs:
#   — pretrained UniCure weights 
#   — Uni-Mol drug embeddings
#   — UCE cell embeddings
#   — tumor expression
#   — normal expression
#   — clinical TSV filtered to Stage III/IV patients
#
# Core logic:
#
# import torch
# from model import UniCure
# from utils import load_UniCure_pretrained_model
#
# model = load_UniCure_pretrained_model()
# model.load_state_dict(torch.load(r'\path_to\Unicure_best_model.pth', map_location='cpu'), strict=False)
# model.to(device)
#
# for each late-stage sample i:
#     sample_features = sample_emd.iloc[i]          # UCE embedding
#     unperturb = tumor_expr.iloc[i]                # log2(TPM+1) tumor expression
#     normal = normal_expr.iloc[i]                  # log2(TPM+1) normal expression
#     delta_cancer = unperturb - normal             # tumor transcriptomic signature
#
#     # Predict perturbed expression for all drugs simultaneously
#     sample_features_repeated = torch.tensor(
#         np.tile(sample_features.values, (n_drugs, 1))).float().to(device)
#     drug_embed = torch.tensor(lincs_drug_embedding.values).float().to(device)
#     with torch.no_grad():
#         lincs_predict = model(sample_features_repeated, drug_embed)  # (n_drugs, n_genes)
#
#     # Reversal correlation: how well does each drug reverse the tumor signature?
#     for drug, predicted_expr in lincs_predict:
#         delta_drug = predicted_expr - unperturb   # predicted drug effect
#         corr = pearsonr(delta_drug, delta_cancer) # higher = better reversal
#         pearson_corr_reversal[drug] = corr
#
#     correlation_matrix[sample] = pearson_corr_reversal
#
# ranked_matrix = correlation_matrix.rank(ascending=True, method='min', axis=0)
# ranked_matrix.to_csv('raw_data/Fig5/patient_stratification/{cancer}_rankmatrix.csv')
#
# Released tables include BRCA_rankmatrix.csv (4992 drugs × 20 samples)
# and KIRC_rankmatrix.csv (4992 drugs × 36 samples)

# =============================================================================
# Patient clustering and differential-drug table preparation
# =============================================================================
# Reads rank matrix, clusters samples, runs Mann-Whitney U tests between clusters.
#
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from scipy.stats import mannwhitneyu
# from statsmodels.stats.multitest import multipletests
#
# df = pd.read_csv('raw_data/Fig5/patient_stratification/{cancer}_rankmatrix.csv', index_col=0)
# df_log = np.log10(df + 1)
# df_scaled = StandardScaler().fit_transform(df_log.T)  # samples × drugs
#
# # Evaluate k=2..10 and save results to raw_data/Fig5/patient_stratification/{cancer}_kmeans_evaluation_results.csv
# for k in range(2, 11):
#     km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(df_scaled)
#     wcss.append(km.inertia_)
#     silhouette_scores.append(silhouette_score(df_scaled, km.labels_))
#
# # Final clustering with chosen k=2
# final_km = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(df_scaled)
# sample_clusters = pd.DataFrame({'Sample': df_log.columns, 'Cluster': final_km.labels_})
# sample_clusters.to_csv('raw_data/Fig5/patient_stratification/{cancer}_sample_clusters.csv')
#
# # Mann-Whitney U test between clusters for each drug (BH correction)
# for drug in df.index:
#     stat, p = mannwhitneyu(group0_ranks, group1_ranks, alternative='two-sided')
# _, adj_p, _, _ = multipletests(p_values, method='fdr_bh')
# results with log2FC saved to raw_data/Fig5/patient_stratification/{cancer}_diff_drug.csv
#
# # Filter tumor expression to clustered samples
# filtered_tumor.to_csv('raw_data/Fig5/patient_stratification/{cancer}_filtered_tumor.csv')
#
# Outputs: {cancer}_sample_clusters.csv, {cancer}_kmeans_evaluation_results.csv,
#          {cancer}_diff_drug.csv, {cancer}_filtered_tumor.csv

# =============================================================================
# Released-table plotting workflow
# =============================================================================

CLUSTER_COLORS = {0: '#FF7517', 1: '#46C3DB'}


def plot_kmeans_evaluation(cancer, ax_wcss, ax_sil):
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'{cancer}_kmeans_evaluation_results.csv'))
    k = df['k']
    ax_wcss.plot(k, df['WCSS'], marker='o', color='#333333', linewidth=1.5)
    ax_wcss.set_title(f'{cancer} — Elbow Method', fontsize=12)
    ax_wcss.set_xlabel('Number of Clusters (k)')
    ax_wcss.set_ylabel('WCSS (Inertia)')
    ax_wcss.set_xticks(k)
    ax_wcss.grid(False)
    ax_sil.plot(k, df['Silhouette Score'], marker='o', color='#333333', linewidth=1.5)
    ax_sil.set_title(f'{cancer} — Silhouette Score', fontsize=12)
    ax_sil.set_xlabel('Number of Clusters (k)')
    ax_sil.set_ylabel('Average Silhouette Score')
    ax_sil.set_xticks(k)
    ax_sil.grid(False)


def plot_pca_clusters(cancer, ax):
    rank_df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'{cancer}_rankmatrix.csv'), index_col=0)
    clusters_df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'{cancer}_sample_clusters.csv'))

    df_log = np.log10(rank_df + 1)
    df_scaled = StandardScaler().fit_transform(df_log.T)  # samples × drugs

    pca = PCA(n_components=2)
    coords = pca.fit_transform(df_scaled)

    sample_order = rank_df.columns.tolist()
    cluster_map = dict(zip(clusters_df['Sample'], clusters_df['Cluster']))
    labels = [cluster_map[s] for s in sample_order]

    for cl, color in CLUSTER_COLORS.items():
        idx = [i for i, l in enumerate(labels) if l == cl]
        ax.scatter(coords[idx, 0], coords[idx, 1], c=color, label=f'Cluster {cl}',
                   s=60, edgecolors='white', linewidths=0.5)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}% variance)', fontsize=11)
    ax.set_title(f'{cancer} Patient Clusters (k=2)', fontsize=12)
    ax.legend(title='Cluster', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {name}.pdf/.png')


# Fig. S14A — BRCA K-means evaluation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_kmeans_evaluation('BRCA', ax1, ax2)
fig.tight_layout()
save(fig, 'figs14a_brca_kmeans_evaluation')

# Fig. S14B — KIRC K-means evaluation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_kmeans_evaluation('KIRC', ax1, ax2)
fig.tight_layout()
save(fig, 'figs14b_kirc_kmeans_evaluation')

# Fig. 5B — BRCA PCA scatter
fig, ax = plt.subplots(figsize=(6, 5))
plot_pca_clusters('BRCA', ax)
fig.tight_layout()
save(fig, 'fig5b_brca_patient_clusters')

# Fig. 5C — KIRC PCA scatter
fig, ax = plt.subplots(figsize=(6, 5))
plot_pca_clusters('KIRC', ax)
fig.tight_layout()
save(fig, 'fig5c_kirc_patient_clusters')
