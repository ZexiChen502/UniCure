"""
Cell-state-specific response workflow — UniCure Fig. 3D and Fig. S9

This file documents the analysis logic used for the A549 single-cell
cell-state-specific response analysis. It describes how marker-defined
subpopulations were derived and how UniCure-predicted Dasatinib responses were
summarized for the corresponding panels.

Figure mapping:
- Fig. S9A: UMAP of unperturbed A549 cells
- Fig. S9B: marker dotplot for SRC, DDR1, and CAV1
- Fig. 3D / Fig. S9C: Dasatinib perturbation-score boxplot
- Fig. S9D-E: ranked genes driving SRC-dominant and DDR1-dominant responses
"""

# ---------------------------------------------------------------------------
# 1. Load and preprocess unperturbed A549 cells
# ---------------------------------------------------------------------------
# import scanpy as sc
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from model import UniCure
# from utils import load_UniCurePretrainsc
#
# adata = sc.read_h5ad(r"\path_to\sciplex3_prnet.h5ad")
# adata = adata[(adata.obs["drug"] == "CTRL") & (adata.obs["cell"] == "A549")].copy()
# adata.layers["counts"] = adata.X.copy()
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# adata.raw = adata
# sc.pp.scale(adata, max_value=10)
# sc.tl.pca(adata, svd_solver="arpack")
# sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
# sc.tl.umap(adata)
# sc.tl.leiden(adata, resolution=0.5, key_added="leiden_major")
#
# # Fig. S9A
# sc.pl.umap(adata, color=["leiden_major"], show=False)
# plt.savefig("output_plot/figs9a_a549_unperturbed_umap.pdf", bbox_inches="tight")
# plt.close()


# ---------------------------------------------------------------------------
# 2. Identify marker-defined subpopulations
# ---------------------------------------------------------------------------
# mechanism_genes = ["SRC", "DDR1", "CAV1"]
#
# # Fig. S9B
# sc.pl.dotplot(
#     adata,
#     var_names=mechanism_genes,
#     groupby="leiden_major",
#     standard_scale="var",
#     show=False,
# )
# plt.savefig("output_plot/figs9b_src_ddr1_cav1_dotplot.pdf", bbox_inches="tight")
# plt.close()
#
# # Manual mapping from Leiden clusters to biological response groups.
# group_map = {
#     "0": "Negative_Control",
#     "3": "Negative_Control",
#     "1": "SRC-Dominant",
#     "2": "SRC-Dominant",
#     "4": "DDR1-Dominant",
#     "5": "DDR1-Dominant",
# }
# adata.obs["drug_response_group"] = adata.obs["leiden_major"].map(group_map)


# ---------------------------------------------------------------------------
# 3. Predict Dasatinib perturbation with UniCure
# ---------------------------------------------------------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = load_UniCurePretrainsc(
#     path=r"\path_to\sciplex3_best_model.pth",
#     output_size=1923,
# ).to(device)
# model.eval()
#
# cell_emb = pd.read_parquet(r"\path_to\A549_dox_mechanism_filtered_uce_emb.parquet")
# drug_emb = pd.read_parquet(r"\path_to\sciplex3_unimol_emb.parquet").loc["Dasatinib", :].values
#
# cell_tensor = torch.tensor(cell_emb.values, dtype=torch.float32).to(device)
# drug_tensor = torch.tensor(drug_emb, dtype=torch.float32).to(device).unsqueeze(0).repeat(cell_tensor.shape[0], 1)
#
# all_predictions = []
# with torch.no_grad():
#     for start in range(0, cell_tensor.shape[0], 256):
#         output = model(
#             "pertrub_forward",
#             cell_tensor[start:start + 256],
#             drug_tensor[start:start + 256],
#         )
#         if isinstance(output, tuple):
#             output = output[0]
#         all_predictions.append(output.cpu().numpy())
# final_prediction = np.concatenate(all_predictions, axis=0)


# ---------------------------------------------------------------------------
# 4. Compute perturbation score and rank response genes
# ---------------------------------------------------------------------------
# model_gene_names = adata.var_names[:1923]
# pred_adata = sc.AnnData(X=final_prediction, obs=adata.obs.copy())
# pred_adata.var_names = model_gene_names
#
# # Delta = predicted Dasatinib-treated expression - baseline control expression.
# sc.pp.log1p(adata)
# delta_x = pred_adata.X - adata[:, model_gene_names].X
# delta_adata = sc.AnnData(X=delta_x, obs=pred_adata.obs.copy(), var=pred_adata.var.copy())
# delta_adata.obs["perturbation_score"] = np.linalg.norm(delta_adata.X, axis=1)
#
# # Fig. 3D / Fig. S9C
# sns.boxplot(
#     data=delta_adata.obs,
#     x="drug_response_group",
#     y="perturbation_score",
#     order=["Negative_Control", "SRC-Dominant", "DDR1-Dominant"],
# )
# plt.savefig("output_plot/fig3d_a549_dasatinib_perturbation_score.pdf", bbox_inches="tight")
# plt.close()
#
# # Fig. S9D-E
# sc.tl.rank_genes_groups(
#     delta_adata,
#     groupby="drug_response_group",
#     reference="Negative_Control",
#     method="wilcoxon",
# )
# sc.pl.rank_genes_groups(delta_adata, groups=["SRC-Dominant"], n_genes=15, show=False)
# plt.savefig("output_plot/figs9d_src_dominant_ranked_genes.pdf", bbox_inches="tight")
# plt.close()
# sc.pl.rank_genes_groups(delta_adata, groups=["DDR1-Dominant"], n_genes=15, show=False)
# plt.savefig("output_plot/figs9e_ddr1_dominant_ranked_genes.pdf", bbox_inches="tight")
# plt.close()


if __name__ == "__main__":
    print("This file documents the Fig. 3D / Fig. S9 workflow logic.")
    print("It describes the single-cell heterogeneity pipeline and expected outputs.")
