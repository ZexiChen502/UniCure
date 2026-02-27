from preprocessing import generate_esm2_emb
from utils import uce_emb
import pandas as pd

##########  Cell Embedding Generation  ###########
# Step 1: Generate protein-level embeddings using ESM2.
# Step 2: Generate cell-level embeddings using UCE (Universal Cell Embedding).
#   - For Bulk data (LINCS): Use the fine-tuned UCE model from Stage 1 to generate embeddings for Stage 2.
#   - For Single-cell data (SciPlex): Use the pre-trained (non-fine-tuned) UCE model for subsequent tasks.

# --- lincs (Bulk RNA-seq): Use Fine-tuned UCE ---
# 第一阶段微调完成后，使用训练好的模型权重生成细胞嵌入，供第二阶段训练使用
generate_esm2_emb(control_path='./data/lincs2020/lincs2020_control.parquet',
                  gene_columns_start=4,
                  save_dir='./data/lincs2020/lincs2020_esm2_emb.parquet',
                  dataset_name="lincs")

uce_emb(esm2_emb_df_path='./data/lincs2020/lincs2020_esm2_emb.parquet',
        esm2_control_df_path='./data/lincs2020/lincs2020_control.parquet',
        model_path='./result/11/lincs2020/best_unicure_stage_1_model.pth', # Path to fine-tuned weights
        uce_emb_df_path='./data/lincs2020/lincs2020_uce_lora_emb.parquet',
        index_name='new_cid')

# --- sciplex (Single Cell): Use Vanilla/Pre-trained UCE ---
# 单细胞数据直接使用原生（未微调）的UCE模型生成嵌入
generate_esm2_emb(control_path='./data/sciplex/sciplex3_control.parquet',
                  gene_columns_start=3,
                  save_dir='./data/sciplex/sciplex3_esm2_emb.parquet',
                  dataset_name="sciplex")

uce_emb(esm2_emb_df_path='./data/sciplex/sciplex3_esm2_emb.parquet',
        esm2_control_df_path='./data/sciplex/sciplex3_control.parquet',
        model_path=None, # Set to None to use pre-trained UCE without fine-tuning
        uce_emb_df_path='./data/sciplex/sciplex3_uce_lora_emb.parquet',
        index_name='cell')




