from sklearn.preprocessing import StandardScaler
import os
import scanpy as sc
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm


def collate_fn(batch):
    return batch[0]


def collate_fn_batch(batch):
    """
    batch: List of tuples from __getitem__
    支持 batch_size > 1
    """
    metadata_list = []

    # 用列表收集所有的 Tensor
    all_cell_embeds = []
    all_drug_embeds = []
    all_real_treated = []
    all_real_control = []

    # 记录每个样本包含多少个细胞，用于后续 split 计算 Loss
    treated_lengths = []
    control_lengths = []

    for item in batch:
        meta, cell_emb, drug_emb, real_treated, real_control = item

        metadata_list.append(meta)

        # Real Treated (N, Genes)
        n_treated = real_treated.shape[0]
        treated_lengths.append(n_treated)
        all_real_treated.append(real_treated)

        # Real Control (M, Genes) - M 可能不等于 N
        n_control = real_control.shape[0]
        control_lengths.append(n_control)
        all_real_control.append(real_control)

        # Expand Inputs to match Treated cells count for Forward Pass
        # Cell Emb: (1, dim) -> (N, dim)
        all_cell_embeds.append(cell_emb.repeat(n_treated, 1))

        # Drug Emb: (1, dim) -> (N, dim)
        all_drug_embeds.append(drug_emb.repeat(n_treated, 1))

    # 拼接成大 Tensor (Vectorization)
    batched_cell_embeds = torch.cat(all_cell_embeds, dim=0)  # (Sum_N, dim)
    batched_drug_embeds = torch.cat(all_drug_embeds, dim=0)  # (Sum_N, dim)
    batched_real_treated = torch.cat(all_real_treated, dim=0)
    batched_real_control = torch.cat(all_real_control, dim=0)  # 注意：Control 不参与 Forward，只用于 Loss

    return {
        'metadata': metadata_list,
        'cell_embeds': batched_cell_embeds,
        'drug_embeds': batched_drug_embeds,
        'real_treated': batched_real_treated,
        'real_control': batched_real_control,
        'treated_lengths': treated_lengths,
        'control_lengths': control_lengths
    }


class CellDrugDataset_v2(torch.utils.data.Dataset):
    def __init__(self, cell_embeddings, drug_embeddings, data_grouped, data_pairs,
                 scaler=False, dose_mode='concat', is_null=False, null_seed=None,
                 max_sample_size=600):  # 新增 max_sample_size 参数

        self.scaler = scaler
        self.dose_mode = dose_mode
        self.is_null = is_null
        self.max_sample_size = max_sample_size  # 限制每个组最大细胞数

        # --- 优化 1: 预处理 Embedding 为 Tensor 字典，避免每次 loc 查询 ---
        print("Pre-loading embeddings to memory...", flush=True)
        self.cell_emb_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in
                              zip(cell_embeddings.index, cell_embeddings.values)}
        self.drug_emb_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in
                              zip(drug_embeddings.index, drug_embeddings.values)}

        # --- 优化 2: 预处理所有 Group 数据到内存 List ---
        # 这一步会消耗内存，但速度极快。如果内存不足，可以用 pickle 缓存
        print("Pre-loading grouped data to memory (this may take a minute)...", flush=True)
        self.data_pairs = data_pairs  # List of (cell, drug, dose)

        # 预先构建 data_grouped 的快速索引
        self.cached_groups = {}
        self.cached_controls = {}

        for keys, group in tqdm(data_grouped, desc="Caching groups"):
            # keys is (cell, drug, dose)
            # 存储为 float32 节省一半内存
            exprs = group.iloc[:, 3:].values.astype(np.float32)
            if self.scaler:
                exprs = np.log1p(exprs)
            self.cached_groups[keys] = exprs

        # 单独处理 Control 组 (cell, 'control', 0)
        # 找出所有涉及的 cell
        unique_cells = set(p[0] for p in data_pairs)
        for cell in unique_cells:
            if (cell, 'control', 0) in self.cached_groups:
                self.cached_controls[cell] = self.cached_groups[(cell, 'control', 0)]
            else:
                try:
                    g = data_grouped.get_group((cell, 'control', 0))
                    exprs = g.iloc[:, 3:].values.astype(np.float32)
                    if self.scaler: exprs = np.log1p(exprs)
                    self.cached_controls[cell] = exprs
                except KeyError:
                    pass  # 可能会有缺失，根据具体业务处理

        if self.is_null:
            print("⚠️ WARNING: Null Model Mode Activated!", flush=True)
            rng = np.random.default_rng(null_seed if null_seed is not None else 42)
            self.shuffled_indices = rng.permutation(len(self.data_pairs))
        else:
            self.shuffled_indices = None

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # Input Metadata
        cell_type, drug_name, drug_dose = self.data_pairs[idx]

        # 1. Get Embeddings (Fast Dict Lookup)
        cell_embed = self.cell_emb_dict[cell_type]  # shape: (dim,)
        if cell_embed.dim() == 1: cell_embed = cell_embed.unsqueeze(0)

        drug_embed_raw = self.drug_emb_dict[drug_name].unsqueeze(0)  # shape: (1, dim)

        # Drug Dose Processing
        dose_val = np.log10(drug_dose + 1)

        # 处理 Drug Embedding (PyTorch 操作)
        if self.dose_mode == 'scale':
            drug_embed = drug_embed_raw * dose_val
        elif self.dose_mode == 'concat':
            dose_tensor = torch.tensor([[dose_val]], dtype=torch.float32)
            drug_embed = torch.cat([drug_embed_raw, dose_tensor], dim=1)
            # Padding to 528
            pad_len = 528 - drug_embed.shape[1]
            if pad_len > 0:
                pad_tensor = torch.full((1, pad_len), dose_val, dtype=torch.float32)
                drug_embed = torch.cat([drug_embed, pad_tensor], dim=1)

        # 2. Get Targets (Label)
        if self.is_null:
            target_idx = self.shuffled_indices[idx]
            target_cell, target_drug, target_dose = self.data_pairs[target_idx]
        else:
            target_cell, target_drug, target_dose = cell_type, drug_name, drug_dose

        # Treated
        gene_exprs_treated = self.cached_groups.get((target_cell, target_drug, target_dose))
        if gene_exprs_treated is None:
            # Fallback or Error handling
            raise ValueError(f"Missing group: {target_cell, target_drug, target_dose}")

        # Control
        gene_exprs_untreated = self.cached_controls.get(cell_type)  # Control 对应 Input cell

        def sample_tensor(data_np, n_max):
            n = data_np.shape[0]
            if n > n_max:
                indices = np.random.choice(n, n_max, replace=False)
                return torch.from_numpy(data_np[indices])
            return torch.from_numpy(data_np)

        real_outputs_treated = sample_tensor(gene_exprs_treated, self.max_sample_size)
        unperturb_gexp = sample_tensor(gene_exprs_untreated, self.max_sample_size)

        return (
            (cell_type, drug_name, drug_dose),
            cell_embed,  # (1, dim)
            drug_embed,  # (1, dim)
            real_outputs_treated,  # (N_sample, genes)
            unperturb_gexp  # (N_sample, genes) - 注意这里 Treated 和 Control 样本数可能不同
        )


class CellDrugDataset(torch.utils.data.Dataset):
    def __init__(self, cell_embeddings, drug_embeddings, data_grouped, data_pairs,
                 scaler=False, dose_mode='concat', is_null=False, null_seed=None):
        """
        参数:
            is_null (bool): 如果为 True，则打乱 Input-Output 对应关系（Null Model）
            null_seed (int): Null Model 的随机种子，用于可重复性
        """
        self.cell_embeddings = cell_embeddings
        self.drug_embeddings = drug_embeddings
        self.data_grouped = data_grouped
        self.data_pairs = data_pairs
        self.scaler = scaler
        self.dose_mode = dose_mode
        self.is_null = is_null

        # 如果是 Null Model，生成一个打乱的索引映射
        if self.is_null:
            print("⚠️  WARNING: Null Model Mode Activated! Labels will be shuffled.", flush=True)
            rng = np.random.default_rng(null_seed if null_seed is not None else 42)
            self.shuffled_indices = rng.permutation(len(self.data_pairs))
        else:
            self.shuffled_indices = None

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # ============================================================
        # Part 1: 获取 Input (这部分永远是真实的)
        # ============================================================
        cell_type, drug_name, drug_dose = self.data_pairs[idx]

        # Cell Embedding
        cell_embed = self.cell_embeddings.loc[cell_type].values
        if len(cell_embed.shape) == 1:
            cell_embed = cell_embed.reshape(1, -1)

        # Drug Embedding + Dose Processing
        drug_embed = self.drug_embeddings.loc[drug_name].values.reshape(1, -1)
        dose_val = np.log10(drug_dose + 1)

        if self.dose_mode == 'scale':
            drug_embed = drug_embed * dose_val
        elif self.dose_mode == 'concat':
            dose_arr = np.array([[dose_val]])
            drug_embed = np.concatenate([drug_embed, dose_arr], axis=1)
            # Padding to 528
            current_len = drug_embed.shape[1]
            target_len = 528
            pad_len = target_len - current_len
            if pad_len > 0:
                pad_arr = np.full((1, pad_len), dose_val)
                drug_embed = np.concatenate([drug_embed, pad_arr], axis=1)
        else:
            raise ValueError(f"Unknown dose_mode: {self.dose_mode}")

        # ============================================================
        # Part 2: 获取 Output (Label) - 这里可能被打乱
        # ============================================================
        if self.is_null:
            # Null Model: 使用一个随机的 pair 作为 Label 来源
            target_idx = self.shuffled_indices[idx]
            target_cell, target_drug, target_dose = self.data_pairs[target_idx]
        else:
            # Normal Model: 使用真实的 pair
            target_cell, target_drug, target_dose = cell_type, drug_name, drug_dose

        # 获取处理后的基因表达 (来自 target pair)
        group = self.data_grouped.get_group((target_cell, target_drug, target_dose))
        gene_exprs_treated = group.iloc[:, 3:].values.astype(np.float32)

        # 获取未处理的基因表达 (来自 target cell 的 control)
        group_no_drug = self.data_grouped.get_group((cell_type, 'control', 0))
        gene_exprs_untreated = group_no_drug.iloc[:, 3:].values.astype(np.float32)

        # Scaler (可选)
        if self.scaler:
            gene_exprs_treated = np.log1p(gene_exprs_treated)
            gene_exprs_untreated = np.log1p(gene_exprs_untreated)

        # ============================================================
        # Part 3: 返回 (注意：metadata 还是用原始的 idx 对应的 pair)
        # ============================================================
        return (
            (cell_type, drug_name, drug_dose),  # Metadata (真实的 Input)
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
        # 定义目标维度，参考代码中为 528
        self.target_drug_dim = 528

    def get_drug_embedding(self, drug_name, drug_dose):
        """
        模仿 CellDrugDataset 的处理逻辑：
        1. 获取原始 Embedding
        2. 拼接 Log10 后的剂量
        3. 用剂量值 Padding 到 528 维
        """
        # 计算剂量 (Log10)
        dose_val = np.log10(drug_dose + 1)

        # 如果是 CTRL，生成一个全 0 的向量，维度需要与处理后的药物向量一致 (528)
        if drug_name == 'CTRL':
            return np.zeros((1, self.target_drug_dim))

        # 1. 获取原始 Embedding (Shape: 1 x Original_Dim)
        drug_embed = self.drug_embeddings.loc[drug_name].values.reshape(1, -1)

        # 2. 拼接剂量 (Concat Dose)
        dose_arr = np.array([[dose_val]])
        drug_embed = np.concatenate([drug_embed, dose_arr], axis=1)

        # 3. 补齐到 528 维度 (Padding to 528)
        current_len = drug_embed.shape[1]
        pad_len = self.target_drug_dim - current_len

        if pad_len > 0:
            # 创建填充数组，用 dose_val 填充
            pad_arr = np.full((1, pad_len), dose_val)
            drug_embed = np.concatenate([drug_embed, pad_arr], axis=1)

        # 如果原始维度+1已经超过528，通常需要截断或保持原样，这里假设不会超过
        return drug_embed

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # 这里的解包对应 MultiDrug 的 metadata 结构
        cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose = self.data_pairs[idx]

        # 处理 Cell Embedding
        cell_embed = self.cell_embeddings.loc[cell_type].values
        # 确保维度正确 (防止是 1D 数组)
        if len(cell_embed.shape) == 1:
            cell_embed = cell_embed.reshape(1, -1)

        # 处理 Drug Embedding (使用修改后的 padding 逻辑)
        drug1_embed = self.get_drug_embedding(drug1_name, drug1_dose)
        drug2_embed = self.get_drug_embedding(drug2_name, drug2_dose)

        # 拼接两个药物的向量
        # 注意：如果两个药物都不是 CTRL，现在的拼接维度将是 528 + 528 = 1056
        if drug1_name != 'CTRL' and drug2_name != 'CTRL':
            drug_embed = np.concatenate([drug1_embed, drug2_embed], axis=1)
        else:
            # 如果其中一个是 CTRL，只取非 CTRL 的那个
            drug_embed = drug1_embed if drug1_name != 'CTRL' else drug2_embed

        # 获取基因表达数据
        # MultiDrug 数据通常前5列是 metadata (cell, d1, dose1, d2, dose2)，所以从第 5 列开始取
        group = self.data_grouped.get_group((cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose))
        gene_exprs_treated = group.iloc[:, 5:].values

        group_no_drug = self.data_grouped.get_group((cell_type, 'CTRL', 0, 'CTRL', 0))
        gene_exprs_untreated = group_no_drug.iloc[:, 5:].values

        # 数据标准化 (保持原有逻辑)
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

