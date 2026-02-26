from preprocessing import generate_esm2_emb
from utils import uce_emb
import pandas as pd

##########  For Example  ###########

# lincs
generate_esm2_emb(control_path='./data/lincs2020/lincs2020_control.parquet',
                  gene_columns_start=4,
                  save_dir='./data/lincs2020/lincs2020_esm2_emb.parquet',
                  dataset_name="lincs")

uce_emb(esm2_emb_df_path='./data/lincs2020/lincs2020_esm2_emb.parquet',
        esm2_control_df_path='./data/lincs2020/lincs2020_control.parquet',
        model_path='./result/11/lincs2020/best_unicure_stage_1_model.pth',
        uce_emb_df_path='./data/lincs2020/lincs2020_uce_lora_emb.parquet',
        index_name='new_cid')

# sciplex
generate_esm2_emb(control_path='./data/sciplex/sciplex3_control.parquet',
                  gene_columns_start=3,
                  save_dir='./data/sciplex/sciplex3_esm2_emb.parquet',
                  dataset_name="sciplex")

uce_emb(esm2_emb_df_path='./data/sciplex/sciplex3_esm2_emb.parquet',
        esm2_control_df_path='./data/sciplex/sciplex3_control.parquet',
        model_path=None,
        uce_emb_df_path='./data/sciplex/sciplex3_uce_lora_emb.parquet',
        index_name='cell')




