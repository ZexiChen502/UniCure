# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from train import finetune

train_size_list = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
seed_list = [1, 2, 3, 4, 5]

model_path = './result/11/lincs2020/Unicure_best_model.pth'
device = None
num_epochs = 400
cancer_name = "LUAD"


# %%
# 读取 LUAD 数据
control_df_luad = pd.read_parquet("./data/PTC/LUAD/PTC_luad_control.parquet")
control_luad = control_df_luad.iloc[:, 2:].values.astype(int)
control_luad = np.tile(np.log2(control_luad + 1), (40, 1))

perturbed_df_luad = pd.read_parquet("./data/PTC/LUAD/PTC_luad_perturb.parquet")
perturbed_luad = perturbed_df_luad.iloc[:, 2:].values.astype(int)
perturbed_luad = np.log2(perturbed_luad + 1)

sample_embed_df_luad = pd.read_parquet("./data/PTC/LUAD/PTC_luad_control_uce_lora_emb.parquet")
cell_embed_luad = np.tile(sample_embed_df_luad.values, (40, 1))

drug_embed_df_luad = pd.read_csv("./data/PTC/drug_message_unimol.csv", index_col=0)  # 不动
drug_embed_luad = drug_embed_df_luad.loc[perturbed_df_luad['Drug'], :].values

# 读取 BLCA 数据
control_df_blca = pd.read_parquet("./data/PTC/BLCA/PTC_blca_control.parquet")
control_blca = control_df_blca.iloc[:, 2:].values.astype(int)
control_blca = np.tile(np.log2(control_blca + 1), (40, 1))

perturbed_df_blca = pd.read_parquet("./data/PTC/BLCA/PTC_blca_perturb.parquet")
perturbed_blca = perturbed_df_blca.iloc[:, 2:].values.astype(int)
perturbed_blca = np.log2(perturbed_blca + 1)

sample_embed_df_blca = pd.read_parquet("./data/PTC/BLCA/PTC_blca_control_uce_lora_emb.parquet")
cell_embed_blca = np.tile(sample_embed_df_blca.values, (40, 1))

drug_embed_df_blca = pd.read_csv("./data/PTC/BLCA/drug_message_unimol.csv", index_col=0)  # 不动
drug_embed_blca = drug_embed_df_blca.loc[perturbed_df_blca['Drug'], :].values

# TNBC
control_df_tnbc = pd.read_parquet(f"./data/PTC/TNBC/TNBC_control.parquet")
control_tnbc = control_df_tnbc.iloc[:, 2:].values.astype(int)
control_tnbc = np.log2(control_tnbc+1)

# output
perturbed_df_tnbc = pd.read_parquet(f"./data/PTC/TNBC/TNBC_perturb.parquet")
perturbed_tnbc = perturbed_df_tnbc.iloc[:, 2:].values.astype(int)
perturbed_tnbc = np.log2(perturbed_tnbc+1)

# input
sample_embed_df_tnbc = pd.read_parquet(f"./data/PTC/TNBC/TNBC_control_uce_lora_emb.parquet")
cell_embed_tnbc = sample_embed_df_tnbc.iloc[:20, :].values
cell_embed_tnbc = np.tile(cell_embed_tnbc, (4, 1))
cell_embed_tnbc = np.delete(cell_embed_tnbc, 61, axis=0)

# 不动
perturbed_df_tnbc['drug'] = perturbed_df_tnbc['drug'].str.lower()
perturbed_df_tnbc['drug'] = perturbed_df_tnbc['drug'].replace('ptx', 'paclitaxel')
drug_embeddings_tnbc = pd.read_parquet("./data/lincs2020/lincs2020_unimol_emb.parquet")
drug_embed_tnbc = drug_embeddings_tnbc.loc[perturbed_df_tnbc['drug']].values

#%% 拼接数据
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


#%%
for train_size in tqdm(train_size_list, desc="Outer loop (train_size)"):
    save_dir = f'./result/11/finetune/{cancer_name}/{train_size}'

    for seed in tqdm(seed_list, desc=f"Inner loop (seed, train_size={train_size})",
                     leave=False):

        finetune(model_path, cell_embed, drug_embed, perturbed, control, device, num_epochs, train_size, seed,
                                save_dir)


