from preprocessing import generate_esm2_emb
from utils import uce_emb
import pandas as pd

"""
===================================================
Cell Embedding Generation Script
===================================================
This script is responsible for generating cell-level embeddings required for the Phase 2 training.
It involves a two-step process:
  Step 1: Generate protein-level embeddings using ESM2.
  Step 2: Generate cell-level embeddings using UCE (Universal Cell Embedding).

Different strategies are applied based on data modality:
  - Bulk Data (LINCS): Uses the fine-tuned UCE model (trained during Stage 1) to generate embeddings.
  - Single-cell Data (SciPlex): Directly uses the pre-trained (vanilla) UCE model without fine-tuning.
"""

# ===================================================
# 1. LINCS (Bulk RNA-seq): Use Fine-tuned UCE
# ===================================================
# After completing Phase 1 training, use the resulting model weights 
# to generate cell embeddings for Phase 2.

print("Generating ESM2 embeddings for LINCS...")
generate_esm2_emb(control_path='./data/lincs2020/lincs2020_control.parquet',
                  gene_columns_start=4,
                  save_dir='./data/lincs2020/lincs2020_esm2_emb.parquet',
                  dataset_name="lincs")

print("Generating UCE embeddings for LINCS using fine-tuned weights...")
uce_emb(esm2_emb_df_path='./data/lincs2020/lincs2020_esm2_emb.parquet',
        esm2_control_df_path='./data/lincs2020/lincs2020_control.parquet',
        model_path='./result/11/lincs2020/best_unicure_stage_1_model.pth', # Path to fine-tuned weights
        uce_emb_df_path='./data/lincs2020/lincs2020_uce_lora_emb.parquet',
        index_name='new_cid')


# ===================================================
# 2. SciPlex (Single-cell RNA-seq): Use Pre-trained UCE
# ===================================================
# For single-cell data, directly use the vanilla UCE model to generate embeddings.

print("Generating ESM2 embeddings for SciPlex 3...")
generate_esm2_emb(control_path='./data/sciplex/sciplex3_control.parquet',
                  gene_columns_start=3,
                  save_dir='./data/sciplex/sciplex3_esm2_emb.parquet',
                  dataset_name="sciplex")

print("Generating UCE embeddings for SciPlex 3 using pre-trained weights...")
uce_emb(esm2_emb_df_path='./data/sciplex/sciplex3_esm2_emb.parquet',
        esm2_control_df_path='./data/sciplex/sciplex3_control.parquet',
        model_path=None,  # Set to None to use pre-trained UCE without fine-tuning
        uce_emb_df_path='./data/sciplex/sciplex3_uce_lora_emb.parquet',
        index_name='cell')

print("Embedding generation completed successfully.")




