# %%
"""
UniCure Fine-tuning Script

This script performs fine-tuning of the pre-trained UniCure model on specific cancer datasets
(e.g., LUAD, BLCA, TNBC) from the Patient-Derived Tumor Cell (PTC) cohorts. It supports 
evaluating model performance across different training set sizes and random seeds.

Usage:
    Ensure the `model_path` points to a valid pre-trained model weight (e.g., Unicure_best_model.pth).
    Uncomment the corresponding dataset blocks (BLCA or TNBC) if you wish to include them 
    in the fine-tuning process.
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from train import finetune

# ==========================================
# Configuration Parameters
# ==========================================
train_size_list = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]  # Proportions of data used for fine-tuning
seed_list = [1, 2, 3, 4, 5]                        # Random seeds for robustness evaluation

model_path = './result/11/lincs2020/Unicure_best_model.pth' # Path to the pre-trained UniCure weights
device = None                                      # Default device (CUDA if available)
num_epochs = 400                                   # Number of fine-tuning epochs
cancer_name = "LUAD"                               # Target cancer type for saving results


# %%
# ==========================================
# 1. Load LUAD (Lung Adenocarcinoma) Data
# ==========================================
# Load Control (unperturbed) gene expressions
control_df_luad = pd.read_parquet("./data/PTC/LUAD/PTC_luad_control.parquet")
control_luad = control_df_luad.iloc[:, 2:].values.astype(int)
control_luad = np.tile(np.log2(control_luad + 1), (40, 1))

# Load Perturbed (treated) gene expressions
perturbed_df_luad = pd.read_parquet("./data/PTC/LUAD/PTC_luad_perturb.parquet")
perturbed_luad = perturbed_df_luad.iloc[:, 2:].values.astype(int)
perturbed_luad = np.log2(perturbed_luad + 1)

# Load Cell Embeddings (pre-computed by UCE)
sample_embed_df_luad = pd.read_parquet("./data/PTC/LUAD/PTC_luad_control_uce_lora_emb.parquet")
cell_embed_luad = np.tile(sample_embed_df_luad.values, (40, 1))

# Load Drug Embeddings (pre-computed by Uni-Mol)
drug_embed_df_luad = pd.read_csv("./data/PTC/drug_embedding_unimol.csv", index_col=0)
drug_embed_luad = drug_embed_df_luad.loc[perturbed_df_luad['Drug'], :].values

# ==========================================
# 2. Load BLCA (Bladder Cancer) Data (Optional)
# ==========================================
control_df_blca = pd.read_parquet("./data/PTC/BLCA/PTC_blca_control.parquet")
control_blca = control_df_blca.iloc[:, 2:].values.astype(int)
control_blca = np.tile(np.log2(control_blca + 1), (40, 1))

perturbed_df_blca = pd.read_parquet("./data/PTC/BLCA/PTC_blca_perturb.parquet")
perturbed_blca = perturbed_df_blca.iloc[:, 2:].values.astype(int)
perturbed_blca = np.log2(perturbed_blca + 1)

sample_embed_df_blca = pd.read_parquet("./data/PTC/BLCA/PTC_blca_control_uce_lora_emb.parquet")
cell_embed_blca = np.tile(sample_embed_df_blca.values, (40, 1))

drug_embed_df_blca = pd.read_csv("./data/PTC/drug_embedding_unimol.csv", index_col=0)
drug_embed_blca = drug_embed_df_blca.loc[perturbed_df_blca['Drug'], :].values

# ==========================================
# 3. Load TNBC (Triple-Negative Breast Cancer) Data (Optional)
# ==========================================
control_df_tnbc = pd.read_parquet(f"./data/PTC/TNBC/TNBC_control.parquet")
control_tnbc = control_df_tnbc.iloc[:, 2:].values.astype(int)
control_tnbc = np.log2(control_tnbc+1)

perturbed_df_tnbc = pd.read_parquet(f"./data/PTC/TNBC/TNBC_perturb.parquet")
perturbed_tnbc = perturbed_df_tnbc.iloc[:, 2:].values.astype(int)
perturbed_tnbc = np.log2(perturbed_tnbc+1)

sample_embed_df_tnbc = pd.read_parquet(f"./data/PTC/TNBC/TNBC_control_uce_lora_emb.parquet")
cell_embed_tnbc = sample_embed_df_tnbc.iloc[:20, :].values
cell_embed_tnbc = np.tile(cell_embed_tnbc, (4, 1))
cell_embed_tnbc = np.delete(cell_embed_tnbc, 61, axis=0) # Drop specific row to match dimension

# Standardize drug names and load embeddings for TNBC
perturbed_df_tnbc['drug'] = perturbed_df_tnbc['drug'].str.lower()
perturbed_df_tnbc['drug'] = perturbed_df_tnbc['drug'].replace('ptx', 'paclitaxel')
drug_embeddings_tnbc = pd.read_parquet("./data/lincs2020/lincs2020_unimol_emb.parquet")
drug_embed_tnbc = drug_embeddings_tnbc.loc[perturbed_df_tnbc['drug']].values

# %%
# ==========================================
# 4. Data Concatenation
# ==========================================
# Combine the required datasets. Uncomment the corresponding lines 
# if you want to perform cross-cancer fine-tuning.

control = np.concatenate([
    control_luad,
    # control_blca,
    # control_tnbc,
], axis=0)

perturbed = np.concatenate([
    perturbed_luad,
    # perturbed_blca,
    # perturbed_tnbc,
], axis=0)

cell_embed = np.concatenate([
    cell_embed_luad,
    # cell_embed_blca,
    # cell_embed_tnbc,
], axis=0)

drug_embed = np.concatenate([
    drug_embed_luad,
    # drug_embed_blca,
    # drug_embed_tnbc,
], axis=0)


# %%
# ==========================================
# 5. Execute Fine-tuning Process
# ==========================================
# Iterates through different training sizes and seeds to evaluate model robustness

for train_size in tqdm(train_size_list, desc="Outer loop (train_size)"):
    save_dir = f'./result/11/finetune/{cancer_name}/{train_size}'
    os.makedirs(save_dir, exist_ok=True)

    for seed in tqdm(seed_list, desc=f"Inner loop (seed, train_size={train_size})", leave=False):
        finetune(
            model_path=model_path, 
            cell_embed=cell_embed, 
            drug_embed=drug_embed, 
            perturbed=perturbed, 
            control=control, 
            device=device, 
            num_epochs=num_epochs, 
            train_size=train_size, 
            seed=seed,
            save_dir=save_dir
        )


