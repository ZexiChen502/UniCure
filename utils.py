# %%
import itertools
import os
import pickle
import random
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import *
from mhatten_lora import UCE_MHAttenLoRA


# %% UCE_loRA
def uce_emb(esm2_emb_df_path='./data/lincs2020/lincs2020_esm2_emb.parquet',
            esm2_control_df_path='./data/lincs2020/lincs2020_control.parquet',
            model_path='./result/2/lincs2020/best_original_state_model.pth',
            uce_emb_df_path='./data/lincs2020/lincs2020_uce_lora_emb.parquet',
            index_name="sample",
            gpu_core="cuda:0"):

    device = torch.device(gpu_core if torch.cuda.is_available() else "cpu")

    # 加载数据
    esm2_emb_df = pd.read_parquet(esm2_emb_df_path)
    esm2_control_df = pd.read_parquet(esm2_control_df_path)
    esm2_emb_values = esm2_emb_df.values

    num_features = esm2_emb_values.shape[1]
    src = esm2_emb_values[:, :num_features // 2]
    mask = esm2_emb_values[:, num_features // 2:]

    src_tensor = torch.tensor(src, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    if model_path is None:
        UCE_lora_model = load_uce_pretrained_model(path='./requirement/UCE_pretraining_files/33l_8ep_1024t_1280.torch',
                                                   target_layers=list(range(23, 33)))
        cureall_model = UniCure(UCE_lora_model).to(device)
    else:
        # cureall_model = load_stage1_model(path=model_path)
        cureall_model = load_UniCure_pretrained_model(path=model_path)
        cureall_model = cureall_model.to(device)

    def batch_predict(model, src_data, mask_data, batch_size):

        n_samples = src_data.size(0)
        result_list = []

        with tqdm(total=n_samples, desc="Predicting batches", unit="sample") as pbar:
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                src_batch = src_data[start_idx:end_idx].to(device)
                mask_batch = mask_data[start_idx:end_idx].to(device)

                with torch.no_grad():
                    batch_result = model("generate_emb", src_batch, mask_batch)

                result_list.append(batch_result.detach().cpu())

                pbar.update(end_idx - start_idx)

        return torch.cat(result_list, dim=0)

    batch_size = 128

    uce_emb_tensor = batch_predict(cureall_model, src_tensor, mask_tensor, batch_size=batch_size)

    uce_emb_values = uce_emb_tensor.numpy()
    uce_emb_df = pd.DataFrame(uce_emb_values, index=esm2_control_df.loc[:, index_name])
    uce_emb_df.index.name = "index"
    uce_emb_df.columns = uce_emb_df.columns.map(str)
    print(uce_emb_df.head(5))

    uce_emb_df.to_parquet(uce_emb_df_path)


def load_stage1_model(path, output_size=978, drug_window_size=32, drug_slide_step=16, cell_window_size=32,
                      cell_slide_step=16,
                      hidden_dim=64, dropout_rate=0.0):
    UCE_model = TransformerModel(token_dim=5120, d_model=1280, nhead=20, d_hid=5120,
                                 nlayers=33, dropout=0.05,
                                 output_dim=1280)

    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    UCE_model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    target_layers = list(range(28, 33))
    UCE_MHAttenLoRA.target_layer_indices = target_layers
    UCE_lora_model = UCE_MHAttenLoRA.from_module(UCE_model, rank=32)

    if path is None:
        pretrained_state_dict = torch.load('./requirement/UCE_pretraining_files/33l_8ep_1024t_1280.torch',
                                           map_location="cpu")
        UCE_lora_model.load_state_dict(pretrained_state_dict, strict=False)

    cureall_model = UniCure(output_size=output_size, drug_window_size=drug_window_size, drug_slide_step=drug_slide_step,
                            cell_window_size=cell_window_size, cell_slide_step=cell_slide_step,
                            hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    cureall_model.uce_lora = UCE_lora_model

    if path is not None:
        pretrained_state_dict = torch.load(path, map_location="cpu")
        missing_keys, unexpected_keys = cureall_model.load_state_dict(pretrained_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    # print("merge lora")
    # cureall_model.uce_lora = cureall_model.uce_lora.merge_lora(inplace=False)

    # for param in cureall_model.parameters():
    #     param.requires_grad = False

    # for name, param in cureall_model.named_parameters():
    #     if "key_map" in name or "value_map" in name:
    #         param.requires_grad = True
    #
    # for name, param in cureall_model.named_parameters():
    #     if "fusion_decoder" in name:
    #         param.requires_grad = True

    # for name, param in cureall_model.named_parameters():
    #     if name.startswith(("decoder.0.", "decoder.3.", "decoder.6.")):
    #         param.requires_grad = True

    return cureall_model


def load_cureall_pretrained_model(model, path):
    pretrained_state_dict = torch.load(path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    return model


def load_uce_pretrained_model(path="./model_files/33l_8ep_1024t_1280.torch", target_layers=None):
    # Initialize the modified model
    if target_layers is None:
        target_layers = list(range(33))
    UCE_model = TransformerModel(token_dim=5120, d_model=1280, nhead=20, d_hid=5120,
                                 nlayers=33, dropout=0.05,
                                 output_dim=1280)
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    UCE_model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    # Load the pretrained state_dict
    pretrained_state_dict = torch.load(path, map_location="cpu")
    # Load the state_dict into the model using strict=False to ignore mismatched keys (i.e., LoRA parameters)
    missing_keys, unexpected_keys = UCE_model.load_state_dict(pretrained_state_dict, strict=False)

    # If needed, you can check which parameters were not loaded

    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)
    for param in UCE_model.parameters():
        param.requires_grad = False
    UCE_MHAttenLoRA.target_layer_indices = target_layers
    UCE_lora_model = UCE_MHAttenLoRA.from_module(UCE_model, rank=32)
    return UCE_lora_model


def load_UniCure_pretrained_model(path, output_size=978):
    UCE_model = TransformerModel(token_dim=5120, d_model=1280, nhead=20, d_hid=5120,
                                 nlayers=33, dropout=0.05,
                                 output_dim=1280)
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    UCE_model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    target_layers = list(range(28, 33))
    UCE_MHAttenLoRA.target_layer_indices = target_layers
    UCE_lora_model = UCE_MHAttenLoRA.from_module(UCE_model, rank=32)

    UniCure_model = UniCure(output_size=output_size)
    UniCure_model.uce_lora = UCE_lora_model

    pretrained_state_dict = torch.load(path, map_location="cpu")

    missing_keys, unexpected_keys = UniCure_model.load_state_dict(pretrained_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return UniCure_model


def load_UniCureFTsc(path=None, output_size=1923):
    UniCure_model = UniCure()

    if path is not None:
        pretrained_state_dict = torch.load(path, map_location="cpu")
        UniCure_model.load_state_dict(pretrained_state_dict, strict=False)

    UniCureFTsc_model = UniCureFTsc(UniCure_model, output_size=output_size)

    # for param in UniCureFTsc_model.pretrained_model.parameters():
    #     param.requires_grad = False

    return UniCureFTsc_model


def load_UniCurePretrainsc(path=None, output_size=1923):
    UniCure_model = UniCure()

    UniCureFTsc_model = UniCureFTsc(UniCure_model, output_size=output_size)

    if path is not None:
        pretrained_state_dict = torch.load(path, map_location="cpu")
        UniCureFTsc_model.load_state_dict(pretrained_state_dict, strict=False)

    # for param in UniCureFTsc_model.pretrained_model.parameters():
    #     param.requires_grad = False

    return UniCureFTsc_model


def load_UniCureFTsc4(path=None, output_size=2990):
    UniCure_model = UniCure()

    UniCureFTsc_model = UniCureFTsc(UniCure_model, output_size=1923)

    if path is not None:
        pretrained_state_dict = torch.load(path, map_location="cpu")
        UniCureFTsc_model.load_state_dict(pretrained_state_dict, strict=False)

    UniCureFTsc4_model = UniCureFTsc(UniCureFTsc_model, output_size=output_size)

    # for param in UniCureFTsc_model.pretrained_model.parameters():
    #     param.requires_grad = False

    return UniCureFTsc4_model


def load_UniCurePretrainsc4(path=None, output_size=1923):
    UniCure_model = UniCure()

    UniCureFTsc_model = UniCureFTsc(UniCure_model, output_size=1923)

    UniCureFTsc4_model = UniCureFTsc(UniCureFTsc_model, output_size=output_size)

    if path is not None:
        pretrained_state_dict = torch.load(path, map_location="cpu")
        UniCureFTsc4_model.load_state_dict(pretrained_state_dict, strict=False)

    return UniCureFTsc4_model


def load_UniCureFT(path=None):
    UniCure_model = UniCure()

    if path is not None:
        pretrained_state_dict = torch.load(path, map_location="cpu")
        UniCure_model.load_state_dict(pretrained_state_dict, strict=False)

    UniCureFT_model = UniCureFT(UniCure_model)

    for param in UniCureFT_model.pretrained_model.parameters():
        param.requires_grad = False

    return UniCureFT_model


def get_trainable_parameters(model):
    return filter(lambda p: p.requires_grad, model.parameters())


# %% seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = Falseq


# %% data_pairs_split
def data_pairs_split(cell_types_list, drug_names_list, drug_doses_list, data_grouped, test_size,
                     seed=42, dataset_name='lincs2020'):
    # save_dir = os.path.join(".", "data", "benchmark_data", "random_split")
    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    train_file_path = os.path.join(save_dir, 'data_pairs_train.pkl')
    val_file_path = os.path.join(save_dir, 'data_pairs_val.pkl')
    test_file_path = os.path.join(save_dir, 'data_pairs_test.pkl')

    # Check if the object file exists
    if os.path.exists(train_file_path) and os.path.exists(val_file_path):
        print("Loading data pairs from existing files.", flush=True)
        with open(train_file_path, 'rb') as f:
            data_pairs_train = pickle.load(f)
        with open(val_file_path, 'rb') as f:
            data_pairs_val = pickle.load(f)
        with open(test_file_path, 'rb') as f:
            data_pairs_test = pickle.load(f)
        return data_pairs_train, data_pairs_val, data_pairs_test

    # If the object file does not exist, the following processing is performed
    # Convert data_grouped.groups to sets to improve lookup efficiency
    grouped_keys = set(data_grouped.groups.keys())
    # Use itertools.product to generate all the combinations, simplifying your code
    data_pairs = [
        (cell_type, drug_name, drug_dose)
        for cell_type, drug_name, drug_dose in product(cell_types_list, drug_names_list, drug_doses_list)
        if (cell_type, drug_name, drug_dose) in grouped_keys
    ]

    # Split the data pair into training and validation sets (e.g., 80% training set, 20% validation set)
    print('length of data pairs:', len(data_pairs), flush=True)
    data_pairs_train, data_pairs_val = train_test_split(data_pairs, test_size=test_size, random_state=seed)
    data_pairs_val, data_pairs_test = train_test_split(data_pairs_val, test_size=0.5, random_state=seed)
    os.makedirs(save_dir, exist_ok=True)
    with open(train_file_path, 'wb') as f:
        pickle.dump(data_pairs_train, f)
    with open(val_file_path, 'wb') as f:
        pickle.dump(data_pairs_val, f)
    with open(test_file_path, 'wb') as f:
        pickle.dump(data_pairs_test, f)

    return data_pairs_train, data_pairs_val, data_pairs_test


def multi_data_pairs_split(cell_types_list, drug1_names_list, drug1_doses_list, drug2_names_list,
                           drug2_doses_list, data_grouped, test_size,
                           seed=42, dataset_name='lincs2020'):
    # save_dir = os.path.join(".", "data", "benchmark_data", "random_split")
    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    train_file_path = os.path.join(save_dir, 'data_pairs_train.pkl')
    val_file_path = os.path.join(save_dir, 'data_pairs_val.pkl')
    test_file_path = os.path.join(save_dir, 'data_pairs_test.pkl')

    # Check if the object file exists
    if os.path.exists(train_file_path) and os.path.exists(val_file_path):
        print("Loading data pairs from existing files.", flush=True)
        with open(train_file_path, 'rb') as f:
            data_pairs_train = pickle.load(f)
        with open(val_file_path, 'rb') as f:
            data_pairs_val = pickle.load(f)
        with open(test_file_path, 'rb') as f:
            data_pairs_test = pickle.load(f)
        return data_pairs_train, data_pairs_val, data_pairs_test

    all_combinations = itertools.product(cell_types_list, drug1_names_list, drug1_doses_list, drug2_names_list,
                                         drug2_doses_list)

    data_pairs = [
        (cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose)
        for cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose in all_combinations
        if (cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose) in data_grouped.groups
    ]

    print('length of data pairs:', len(data_pairs))

    data_pairs_train, data_pairs_val = train_test_split(data_pairs, test_size=test_size, random_state=seed)
    data_pairs_val, data_pairs_test = train_test_split(data_pairs_val, test_size=0.5, random_state=seed)
    os.makedirs(save_dir, exist_ok=True)
    with open(train_file_path, 'wb') as f:
        pickle.dump(data_pairs_train, f)
    with open(val_file_path, 'wb') as f:
        pickle.dump(data_pairs_val, f)
    with open(test_file_path, 'wb') as f:
        pickle.dump(data_pairs_test, f)

    return data_pairs_train, data_pairs_val, data_pairs_test


def replace_ctrl_in_index(drug_embeddings_df):
    """
      Replaces 'CTRL' with 'control' in the index labels of a Pandas DataFrame.

      Args:
        drug_embeddings_df: The Pandas DataFrame whose index will be modified.

      Returns:
        The modified Pandas DataFrame.
      """

    # 1. Create a mapping dictionary to map old index values to new index values.
    index_mapping = {index: index.replace('CTRL', 'control') for index in drug_embeddings_df.index}

    # 2. Rename the index, if it has a name
    drug_embeddings_df = drug_embeddings_df.rename_axis(
        index_mapping.get(drug_embeddings_df.index.name, drug_embeddings_df.index.name))

    # 3. Use DataFrame.rename() to replace the index labels using the mapping dictionary.
    drug_embeddings_df = drug_embeddings_df.rename(index=index_mapping)

    return drug_embeddings_df

