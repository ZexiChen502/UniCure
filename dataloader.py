from sklearn.preprocessing import StandardScaler
import os
import scanpy as sc
import torch
from torch.utils.data import DataLoader
import numpy as np
import time

def collate_fn(batch):
    return batch[0]


class CellDrugDataset(torch.utils.data.Dataset):
    def __init__(self, cell_embeddings, drug_embeddings, data_grouped, data_pairs, scaler=False):
        self.cell_embeddings = cell_embeddings
        self.drug_embeddings = drug_embeddings
        self.data_grouped = data_grouped
        self.data_pairs = data_pairs
        self.scaler = scaler

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # Retrieve cell type and drug name
        cell_type, drug_name, drug_dose = self.data_pairs[idx]

        # Retrieve untreated cell representations (multiple)
        cell_embed = self.cell_embeddings.loc[cell_type].values  # Two dimensions
        if len(cell_embed.shape) == 1:  # Check if it's single-dimensional
            cell_embed = cell_embed.reshape(1, -1)

        # Retrieve drug representations (single)
        drug_embed = self.drug_embeddings.loc[drug_name].values  # One dimension
        drug_embed = drug_embed.reshape(1, -1)
        # Calculate log10(drug_dose) and apply to drug_embed
        dose_factor = np.log10(drug_dose + 1)
        drug_embed = drug_embed * dose_factor

        # Retrieve gene expressions after drug treatment (multiple)
        group = self.data_grouped.get_group((cell_type, drug_name, drug_dose))
        gene_exprs_treated = group.iloc[:, 3:].values  # Gene expressions after drug treatment

        # Retrieve untreated gene expressions (multiple)
        group_no_drug = self.data_grouped.get_group((cell_type, 'control', 0))
        gene_exprs_untreated = group_no_drug.iloc[:, 3:].values  # Untreated gene expressions

        # if self.scaler is True:
        #     scaler = StandardScaler()
        #     gene_exprs_treated = scaler.fit_transform(gene_exprs_treated.T).T
        #     gene_exprs_untreated = scaler.fit_transform(gene_exprs_untreated.T).T

        if self.scaler is True:
            # 对每一行（每个细胞）做 log1p 转换（不会改变稀疏性结构）
            gene_exprs_treated = np.log1p(gene_exprs_treated)
            gene_exprs_untreated = np.log1p(gene_exprs_untreated)

        return ((cell_type, drug_name, drug_dose),
                torch.FloatTensor(cell_embed),
                torch.FloatTensor(drug_embed),
                torch.FloatTensor(gene_exprs_treated),
                torch.FloatTensor(gene_exprs_untreated)
                )


class TahoeCellDrugDataset(torch.utils.data.Dataset):
    def __init__(self, data_pairs, cell_embeddings, drug_embeddings, cell_untreated_dict, file_data_dict, scaler=False,
                 global_mean=None,  # 新增参数
                 global_var=None  # 新增参数
                 ):
        self.data_pairs = data_pairs
        self.cell_embeddings = cell_embeddings
        self.drug_embeddings = drug_embeddings
        self.cell_untreated = cell_untreated_dict
        self.file_data_dict = file_data_dict  # 预加载的数据字典
        self.scaler = scaler
        self.global_mean = global_mean  # (n_genes,)
        self.global_var = global_var    # (n_genes,)
        self.epsilon = 1e-8  # 防止除以零

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # total_start = time.time()
        cell_type, drug_name, drug_dose, file_name = self.data_pairs[idx]

        # 细胞嵌入
        cell_embed = self.cell_embeddings.loc[cell_type].values
        if len(cell_embed.shape) == 1:
            cell_embed = cell_embed.reshape(1, -1)

        # 药物嵌入（带剂量调整）
        drug_embed = self.drug_embeddings.loc[drug_name].values.reshape(1, -1)
        drug_embed *= np.log10(drug_dose + 1)

        # 从内存字典直接读取处理后的基因表达
        # t1 = time.time()
        gene_exprs_treated = self.file_data_dict[file_name].toarray()
        # t2 = time.time()
        # print(f"[{idx}] 稀疏转稠密耗时: {t2 - t1:.6f} 秒")

        # 未处理细胞表达（从parquet读取）
        # t3 = time.time()
        gene_exprs_untreated = self.cell_untreated[cell_type]
        # t4 = time.time()
        # print(f"[{idx}] 未处理表达提取耗时: {t4 - t3:.6f} 秒")

        # === 应用全局标准化 ===
        if self.scaler:
            # t5 = time.time()
            # 确保均值和方差与数据维度匹配
            assert self.global_mean.shape[0] == gene_exprs_treated.shape[1], "基因维度不匹配"
            # 处理后的数据标准化
            gene_exprs_treated = (gene_exprs_treated - self.global_mean) / np.sqrt(self.global_var + self.epsilon)
            # 未处理数据应用相同的标准化参数
            gene_exprs_untreated = (gene_exprs_untreated - self.global_mean) / np.sqrt(self.global_var + self.epsilon)
            # t6 = time.time()
            # print(f"[{idx}] 全局标准化耗时: {t6 - t5:.6f} 秒")

        # total_end = time.time()
        # print(f"[{idx}] Dataloader耗时: {total_end - total_start:.6f} 秒\n")

        return (
            (cell_type, drug_name, drug_dose),
            torch.FloatTensor(cell_embed),
            torch.FloatTensor(drug_embed),
            torch.FloatTensor(gene_exprs_treated),
            torch.FloatTensor(gene_exprs_untreated)
        )


class MultiDrugDataset(torch.utils.data.Dataset):
    def __init__(self, cell_embeddings, drug_embeddings, data_grouped, data_pairs, scaler=False):
        self.cell_embeddings = cell_embeddings
        self.drug_embeddings = drug_embeddings
        self.data_grouped = data_grouped
        self.data_pairs = data_pairs
        self.scaler = scaler

    def get_drug_embedding(self, drug_name, drug_dose):
        if drug_name == 'CTRL':
            return np.zeros((1, self.drug_embeddings.shape[1]))
        drug_embed = self.drug_embeddings.loc[drug_name].values.reshape(1, -1)
        dose_factor = np.log10(drug_dose + 1)
        return drug_embed * dose_factor

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose = self.data_pairs[idx]

        cell_embed = self.cell_embeddings.loc[cell_type].values
        # cell_embed = self.cell_embeddings.loc[cell_type].values.mean(axis=0)

        drug1_embed = self.get_drug_embedding(drug1_name, drug1_dose)
        drug2_embed = self.get_drug_embedding(drug2_name, drug2_dose)

        if drug1_name != 'CTRL' and drug2_name != 'CTRL':
            drug_embed = np.concatenate([drug1_embed, drug2_embed], axis=1)
        else:
            drug_embed = drug1_embed if drug1_name != 'CTRL' else drug2_embed

        group = self.data_grouped.get_group((cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose))
        gene_exprs_treated = group.iloc[:, 5:].values

        group_no_drug = self.data_grouped.get_group((cell_type, 'CTRL', 0, 'CTRL', 0))
        gene_exprs_untreated = group_no_drug.iloc[:, 5:].values

        if self.scaler is True:
            scaler = StandardScaler()
            gene_exprs_treated = scaler.fit_transform(gene_exprs_treated.T).T
            gene_exprs_untreated = scaler.fit_transform(gene_exprs_untreated.T).T

        return ((cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose),
                torch.FloatTensor(cell_embed),
                torch.FloatTensor(drug_embed),
                torch.FloatTensor(gene_exprs_treated),
                torch.FloatTensor(gene_exprs_untreated)
                )






