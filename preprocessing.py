# %% UCE preprocessing
from torch.utils.data import DataLoader
import scanpy as sc
from dataloader import *
from utils import *


# %%
def sample_cell_sentences(counts, batch_weights,
                          dataset_to_protein_embeddings,  # self.pe_idx_path
                          dataset_to_chroms,  # self.chroms_path
                          dataset_to_starts):  # self.starts_path
    dataset_idxs = dataset_to_protein_embeddings  # get the dataset specific protein embedding idxs
    cell_sentences = torch.zeros((counts.shape[0], 1536))  # init the cell representation as 0s
    mask = torch.zeros((counts.shape[0], 1536))  # start of masking the whole sequence
    chroms = dataset_to_chroms  # get the dataset specific chroms for each gene
    starts = dataset_to_starts  # get the dataset specific genomic start locations for each gene

    longest_seq_len = 0  # we need to keep track of this so we can subset the batch at the end

    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()

        # check NaN
        if np.any(np.isnan(weights)):
            print(c)
            raise ValueError("Weights contain NaN values.")

        # check negative values
        if np.any(weights < 0):
            raise ValueError("Weights contain negative values.")

        # normalize
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            raise ValueError("Sum of weights is zero; cannot normalize.")
        else:
            weights = weights / weights_sum

        # weights = weights / sum(weights)  # RE NORM after mask

        # randomly choose the genes that will make up the sample, weighted by expression, with replacement
        choice_idx = np.random.choice(np.arange(len(weights)),
                                      size=1024, p=weights,
                                      replace=True)
        # choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes
        choosen_chrom = chroms.iloc[choice_idx]
        # order the genes by chromosome
        chrom_sort = np.argsort(choosen_chrom)
        choice_idx = choice_idx[chrom_sort]

        # sort the genes by start
        new_chrom = chroms.iloc[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full(1536, 3)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.
        # Shuffle the chroms now, there's no natural order to chromosomes
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms)  # shuffle

        # This loop is actually just over one cell
        for chrom in uq_chroms:
            # Open Chrom token
            ordered_choice_idx[i] = int(
                chrom) + 143574  # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the genes by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(
                choosen_starts[loc])  # start locations for this chromsome

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = 2  # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = (1536 - i)

        cell_mask = torch.concat((torch.ones(i),
                                  # pay attention to all of these tokens, ignore the rest!
                                  torch.zeros(remainder_len)))

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = 0  # the remainder of the sequence
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)

    cell_sentences_pe = cell_sentences.long()  # token indices

    return cell_sentences_pe, mask


def process_data_to_df(df, gene_columns_start, pe_row_idxs, dataset_chroms, dataset_pos, dataset_name: str):
    """
    Process each row of the DataFrame to generate a new DataFrame,
    where the index is "new_cid", the first 1536 columns are batch_sentences,
    and the last 1536 columns are the mask.

    Args:
    df: pandas DataFrame containing the data.
    pe_row_idxs: Parameter for the sample_cell_sentences function.
    dataset_chroms: Parameter for the sample_cell_sentences function.
    dataset_pos: Parameter for the sample_cell_sentences function.
    start_row: Index of the row to start processing from (default is 0).

    Returns:
    pandas.DataFrame: A DataFrame with index "new_cid",
    first 1536 columns as batch_sentences, and the last 1536 columns as mask.
    """
    df = df.reset_index(drop=True)
    exp = df.iloc[:, gene_columns_start:].astype(float).values
    exp[exp < 0] = 0
    data = []
    index = []
    if dataset_name == "lincs":
        cell_column = "new_cid"
    elif dataset_name == "sciplex":
        cell_column = "cell"
    elif dataset_name == "geo":
        cell_column = "patient"
    elif dataset_name == "tahoe":
        cell_column = "cell_name"
    else:
        cell_column = "sample"

    print("Preprocessing the cell data.")
    for i in tqdm(range(exp.shape[0])):
        cid_name = df.loc[i, cell_column]
        index.append(cid_name)

        counts = exp[i, :]
        counts = torch.tensor(counts).unsqueeze(0)

        if (dataset_name == "lincs") & (dataset_name == "geo"):
            weights = (counts / torch.sum(counts))
        elif dataset_name == "sciplex":
            weights = torch.log1p(counts)
            weights = (weights / torch.sum(weights))
        else:
            # weights = torch.log2(counts+1)
            # weights = (weights / torch.sum(weights))
            weights = (counts / torch.sum(counts))
        batch_sentences, mask = sample_cell_sentences(
            counts,
            weights,
            dataset_to_protein_embeddings=pe_row_idxs,
            dataset_to_chroms=dataset_chroms,
            dataset_to_starts=dataset_pos
        )

        # Concatenate batch_sentences and mask horizontally
        combined = torch.cat((batch_sentences, mask), dim=1).squeeze(0).numpy()
        data.append(combined)

    # Create DataFrame
    result_df = pd.DataFrame(data, index=index)

    return result_df


def generate_esm2_emb(control_path=r"D:\sci_job\CureX\data\lincs2\lincs2020_control.parquet",
                      gene_columns_start=4,  # 要减1
                      save_dir="./data/lincs2020/lincs2020_esm2_emb.parquet",
                      dataset_name: str = "lincs"):

    valid_datasets = {"lincs", "sciplex", "geo", "clinical", "tahoe"}

    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name. Expected one of {valid_datasets}")

    if not os.path.exists(save_dir):
        df = pd.read_parquet(control_path)
        gene_list = df
        # ['SKIC2', 'BLTP2', 'SKIC8', 'DELEC1', 'H2AC25', 'RIGI', 'BMAL2', 'MYCNOS', 'MYL11']
        gene_mapping = {
            "AARS": "AARS1",
            "EPRS": "EPRS1",
            "KIF1BP": "KIFBP",
            "TSTA3": "GFUS",
            "WRB": "GET1",
            "FAM57A": "TLCD3A",
            "HIST2H2BE": "H2BC21",
            "HIST1H2BK": "H2BC12",
            "H2AFV": "H2AZ2",
            "KIAA0355": "GARRE1",
            "FAM155A": "NALF1",
            "TMEM159": "LDAF1",
            'BLTP2': "KIAA0100",
            'SKIC8': "WDR61",
            'H2AC25': "H2AW",
            'RIGI': "DDX58",
            'BMAL2': "ARNTL2",
            'MYL11': "MYLPF"
        }

        gene_list = [gene_mapping.get(gene, gene) for gene in gene_list]
        gene_list = [gene.upper() for gene in gene_list]

        protein_embeddings_paths = {
            'human': './requirement/UCE_pretraining_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt'
        }

        species_to_pe = {
            species: torch.load(pe_dir) for species, pe_dir in protein_embeddings_paths.items()
        }

        species_to_pe = {species: {k.upper(): v for k, v in pe.items()} for species, pe in species_to_pe.items()}
        gene_to_chrom_pos = pd.read_csv("./requirement/UCE_pretraining_files/species_chrom.csv")
        gene_to_chrom_pos["spec_chrom"] = pd.Categorical(
            gene_to_chrom_pos["species"] + "_" + gene_to_chrom_pos["chromosome"])  # add the spec_chrom list
        spec_pe_genes = list(species_to_pe["human"].keys())
        not_in_spec_pe_genes = [gene for gene in gene_list if gene not in spec_pe_genes]
        print(not_in_spec_pe_genes)
        gene_list = [gene for gene in gene_list if gene not in not_in_spec_pe_genes]
        print("drop gene")
        df = df.drop(columns=[gene for gene in not_in_spec_pe_genes if gene in df.columns])
        offset = 13466
        print("pe_idx")
        pe_row_idxs = torch.tensor(
            [spec_pe_genes.index(k.upper()) + offset for k in gene_list]).long()  # self.pe_idx_path
        print("spec_chrom")
        spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == "human"].set_index("gene_symbol")
        print("gene_chrom")
        gene_chrom = spec_chrom.loc[[k.upper() for k in gene_list]]
        print("dataset_chroms")
        dataset_chroms = gene_chrom["spec_chrom"].cat.codes  # self.chroms_path
        dataset_pos = gene_chrom["start"].values  # self.starts_path
        print("process_data_to_df function")
        DF = process_data_to_df(df, gene_columns_start, pe_row_idxs, dataset_chroms, dataset_pos, dataset_name)
        DF.columns = DF.columns.astype(str)
        DF.to_parquet(save_dir)
    else:
        print(f"The file {save_dir} already exists. Skipping generation.")


# def esm2_preprocessing():
#     # lincs_path = './data/lincs2020/'
#     # lincs_folder = os.path.basename(os.path.normpath(lincs_path))
#     # new_string = lincs_folder + "_control.parquet"
#     # new_path = os.path.join(lincs_path, new_string)
#     generate_esm2_emb(control_path='./data/lincs2020/lincs2020_control.parquet',
#                       gene_columns_start=4,
#                       save_dir='./data/lincs2020/lincs2020_esm2_emb.parquet',
#                       dataset_name="lincs")
#
#     # sciplex3
#     generate_esm2_emb(control_path='./data/sciplex3/sciplex3_control.parquet',
#                       gene_columns_start=3,
#                       save_dir='./data/sciplex3/sciplex3_esm2_emb.parquet',
#                       dataset_name="sciplex")
#
#     # sciplex4
#     generate_esm2_emb(control_path='./data/sciplex4/sciplex4_control.parquet',
#                       gene_columns_start=5,
#                       save_dir='./data/sciplex4/sciplex4_esm2_emb.parquet',
#                       dataset_name="sciplex")


def lincs_step2_preprocessing_v2(seed: int):
    print("load data......", flush=True)
    lincs_df = pd.read_parquet('./data/lincs2020/lincs2020_merge_cid.parquet')
    print("Total data:", lincs_df.shape[0], flush=True)

    cid_types_list = list(set(lincs_df['new_cid']))
    print("Total cids:", len(cid_types_list), flush=True)

    cell_types_list = list(set(lincs_df['cell_iname']))
    print("Total cells:", len(cell_types_list), flush=True)

    drug_doses_list = list(set(lincs_df['pert_idose']) - {0})
    drug_names_list = list(set(lincs_df['cmap_name']) - {'control'})
    print("Total drugs:", len(drug_names_list), flush=True)

    drug_embeddings = pd.read_parquet('./data/lincs2020/lincs2020_unimol_emb.parquet')
    missing_drugs = [item for item in drug_names_list if item not in drug_embeddings.index]
    if missing_drugs:
        print("The following drugs do not have unimol_emb:", len(missing_drugs), flush=True)
        drug_names_list = [item for item in drug_names_list if item not in missing_drugs]
    else:
        print("All drugs have unimol_emb.", flush=True)

    cell_embeddings = pd.read_parquet('./data/lincs2020/lincs2020_uce_lora_emb.parquet')
    missing_cells = [item for item in cid_types_list if item not in cell_embeddings.index]
    if missing_cells:
        print("The following cids do not have uce_emb:", len(missing_cells), flush=True)
        cid_types_list = [item for item in cid_types_list if item not in missing_cells]
    else:
        print("All cids have uce_emb.", flush=True)

    print("grouping by cid, drug, and dose.......", flush=True)
    lincs_df = lincs_df.drop(columns=['cell_iname'])
    data_grouped = lincs_df.groupby(['new_cid', 'cmap_name', 'pert_idose'])

    print("spliting train and val datasets.......", flush=True)
    data_pairs_train, data_pairs_val, data_pairs_test = data_pairs_split(cid_types_list, drug_names_list,
                                                                         drug_doses_list,
                                                                         data_grouped,
                                                                         test_size=0.2, seed=seed)
    train_dataset = CellDrugDataset_v2(cell_embeddings, drug_embeddings, data_grouped, data_pairs_train,
                                       max_sample_size=600, dose_mode='concat')
    val_dataset = CellDrugDataset_v2(cell_embeddings, drug_embeddings, data_grouped, data_pairs_val,
                                     max_sample_size=600, dose_mode='concat')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=collate_fn_batch, num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            collate_fn=collate_fn_batch, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def lincs_test_preprocessing(seed: int):
    print("load data......", flush=True)
    lincs_df = pd.read_parquet('./data/lincs2020/lincs2020_merge_cid.parquet')
    print("Total data:", lincs_df.shape[0], flush=True)

    selected_columns = lincs_df.iloc[:, 4:]
    lincs_gene = selected_columns.columns.tolist()

    cid_types_list = list(set(lincs_df['new_cid']))
    print("Total cids:", len(cid_types_list), flush=True)

    cell_types_list = list(set(lincs_df['cell_iname']))
    print("Total cells:", len(cell_types_list), flush=True)

    drug_doses_list = list(set(lincs_df['pert_idose']) - {0})
    drug_names_list = list(set(lincs_df['cmap_name']) - {'control'})
    print("Total drugs:", len(drug_names_list), flush=True)

    drug_embeddings = pd.read_parquet('./data/lincs2020/lincs2020_unimol_emb.parquet')
    missing_drugs = [item for item in drug_names_list if item not in drug_embeddings.index]
    if missing_drugs:
        print("The following drugs do not have unimol_emb:", len(missing_drugs), flush=True)
        drug_names_list = [item for item in drug_names_list if item not in missing_drugs]
    else:
        print("All drugs have unimol_emb.", flush=True)

    cell_embeddings = pd.read_parquet('./data/lincs2020/lincs2020_uce_lora_emb.parquet')
    missing_cells = [item for item in cid_types_list if item not in cell_embeddings.index]
    if missing_cells:
        print("The following cids do not have uce_emb:", len(missing_cells), flush=True)
        cid_types_list = [item for item in cid_types_list if item not in missing_cells]
    else:
        print("All cids have uce_emb.", flush=True)

    print("grouping by cid, drug, and dose.......", flush=True)
    lincs_df = lincs_df.drop(columns=['cell_iname'])
    data_grouped = lincs_df.groupby(['new_cid', 'cmap_name', 'pert_idose'])

    print("spliting train and val datasets.......", flush=True)
    data_pairs_train, data_pairs_val, data_pairs_test = data_pairs_split(cid_types_list, drug_names_list,
                                                                         drug_doses_list,
                                                                         data_grouped,
                                                                         test_size=0.2, seed=seed)
    print(len(data_pairs_train), len(data_pairs_val), len(data_pairs_test))

    test_dataset = CellDrugDataset(cell_embeddings, drug_embeddings, data_grouped, data_pairs_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    return test_loader, lincs_gene


def lincs_test_preprocessing_v2(seed: int, max_size=10):
    print("load data for testing......", flush=True)
    lincs_df = pd.read_parquet('./data/lincs2020/lincs2020_merge_cid.parquet')

    selected_columns = lincs_df.iloc[:, 4:]
    lincs_gene = selected_columns.columns.tolist()

    cid_types_list = list(set(lincs_df['new_cid']))
    drug_doses_list = list(set(lincs_df['pert_idose']) - {0})
    drug_names_list = list(set(lincs_df['cmap_name']) - {'control'})

    drug_embeddings = pd.read_parquet('./data/lincs2020/lincs2020_unimol_emb.parquet')
    missing_drugs = [item for item in drug_names_list if item not in drug_embeddings.index]
    if missing_drugs:
        drug_names_list = [item for item in drug_names_list if item not in missing_drugs]

    cell_embeddings = pd.read_parquet('./data/lincs2020/lincs2020_uce_lora_emb.parquet')
    missing_cells = [item for item in cid_types_list if item not in cell_embeddings.index]
    if missing_cells:
        cid_types_list = [item for item in cid_types_list if item not in missing_cells]

    print("grouping and splitting.......", flush=True)
    lincs_df = lincs_df.drop(columns=['cell_iname'])
    data_grouped = lincs_df.groupby(['new_cid', 'cmap_name', 'pert_idose'])

    _, _, data_pairs_test = data_pairs_split(cid_types_list, drug_names_list,
                                             drug_doses_list,
                                             data_grouped,
                                             test_size=0.2, seed=seed)

    print(f"Test pairs count: {len(data_pairs_test)}")

    test_dataset = CellDrugDataset_v2(cell_embeddings, drug_embeddings, data_grouped, data_pairs_test,
                                      max_sample_size=max_size)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             collate_fn=collate_fn_batch, num_workers=4, pin_memory=True)

    return test_loader, lincs_gene


def sciplex3_preprocessing_v2(seed: int):
    print("load Sciplex3 data......")
    sciplex_df = pd.read_parquet("./data/sciplex/sciplex3.parquet")
    sciplex_df['drug'] = sciplex_df['drug'].replace('CTRL', 'control')
    print("Total data:", sciplex_df.shape[0])
    cell_types_list = list(set(sciplex_df['cell']))
    print("Total cells:", len(cell_types_list))
    drug_doses_list = list(set(sciplex_df['dose']) - {0})
    drug_names_list = list(set(sciplex_df['drug']) - {'control'})
    print("Total drugs:", len(drug_names_list))

    drug_embeddings = pd.read_parquet("./data/sciplex/sciplex3_unimol_emb.parquet")
    drug_embeddings = replace_ctrl_in_index(drug_embeddings)

    missing_drugs = [item for item in drug_names_list if item not in drug_embeddings.index]
    if missing_drugs:
        print("The following drugs do not have unimol_emb:", len(missing_drugs))
        # missing_drugs_df = pd.DataFrame(missing_drugs, columns=['missing_drugs'])
        # missing_drugs_df.to_csv('sciplex_miss_drugs.csv', index=False)
        drug_names_list = [item for item in drug_names_list if item not in missing_drugs]
    else:
        print("All drugs have unimol_emb.")

    cell_embeddings = pd.read_parquet("./data/sciplex/sciplex3_uce_emb.parquet")
    missing_cells = [item for item in cell_types_list if item not in cell_embeddings.index]
    if missing_cells:
        print("The following cells do not have uce_emb:", len(missing_cells))
        # missing_cells_df = pd.DataFrame(missing_cells, columns=['missing_cells'])
        # missing_cells_df.to_csv('sciplex_miss_cells.csv', index=False)
        cell_types_list = [item for item in cell_types_list if item not in missing_cells]
    else:
        print("All cells have uce_emb.")

    print("grouping by cell, drug, and dose.......")
    data_grouped = sciplex_df.groupby(['cell', 'drug', 'dose'])

    print("spliting train and val datasets.......")
    data_pairs_train, data_pairs_val, data_pairs_test = data_pairs_split(cell_types_list, drug_names_list,
                                                                         drug_doses_list,
                                                                         data_grouped, test_size=0.2, seed=seed,
                                                                         dataset_name='sciplex3')

    train_dataset = CellDrugDataset_v2(cell_embeddings, drug_embeddings, data_grouped, data_pairs_train,
                                       max_sample_size=512, scaler=True)
    val_dataset = CellDrugDataset_v2(cell_embeddings, drug_embeddings, data_grouped, data_pairs_val,
                                     max_sample_size=512, scaler=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              collate_fn=collate_fn_batch, num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn_batch, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def sciplex3_test_preprocessing_v2(seed: int, max_size=512):
    print("load Sciplex3 data for testing......", flush=True)
    sciplex_df = pd.read_parquet("./data/sciplex/sciplex3.parquet")

    selected_columns = sciplex_df.iloc[:, 3:]
    sciplex_gene = selected_columns.columns.tolist()

    sciplex_df['drug'] = sciplex_df['drug'].replace('CTRL', 'control')

    cell_types_list = list(set(sciplex_df['cell']))
    drug_doses_list = list(set(sciplex_df['dose']) - {0})
    drug_names_list = list(set(sciplex_df['drug']) - {'control'})

    print(f"Total data: {sciplex_df.shape[0]}, Cells: {len(cell_types_list)}, Drugs: {len(drug_names_list)}",
          flush=True)


    drug_embeddings = pd.read_parquet("./data/sciplex/sciplex3_unimol_emb.parquet")

    drug_embeddings = replace_ctrl_in_index(drug_embeddings)

    missing_drugs = [item for item in drug_names_list if item not in drug_embeddings.index]
    if missing_drugs:
        print(f"Drugs missing embeddings: {len(missing_drugs)}", flush=True)
        drug_names_list = [item for item in drug_names_list if item not in missing_drugs]

    cell_embeddings = pd.read_parquet("./data/sciplex/sciplex3_uce_emb.parquet")
    missing_cells = [item for item in cell_types_list if item not in cell_embeddings.index]
    if missing_cells:
        print(f"Cells missing embeddings: {len(missing_cells)}", flush=True)
        cell_types_list = [item for item in cell_types_list if item not in missing_cells]

    print("grouping and splitting.......", flush=True)

    data_grouped = sciplex_df.groupby(['cell', 'drug', 'dose'])

    _, _, data_pairs_test = data_pairs_split(cell_types_list, drug_names_list,
                                             drug_doses_list,
                                             data_grouped, test_size=0.2, seed=seed,
                                             dataset_name='sciplex3')

    print(f"Test pairs count: {len(data_pairs_test)}", flush=True)

    test_dataset = CellDrugDataset_v2(cell_embeddings, drug_embeddings, data_grouped, data_pairs_test,
                                      max_sample_size=max_size, scaler=True)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_fn_batch, num_workers=4, pin_memory=True)

    return test_loader, sciplex_gene


def sciplex4_preprocessing(seed: int):
    print("load Sciplex4 data......")
    sciplex_df = pd.read_parquet("./data/sciplex/sciplex4.parquet")
    sciplex_df_without_ctrl = sciplex_df[(sciplex_df['drug_1'] != 'CTRL') | (sciplex_df['drug_2'] != 'CTRL')]
    print("Total data:", sciplex_df.shape[0])

    cell_types_list = list(set(sciplex_df_without_ctrl['cell']))
    print("Total cells:", len(cell_types_list))

    drug1_names_list = list(set(sciplex_df_without_ctrl['drug_1']))
    print("Total drugs1:", len(drug1_names_list))

    drug2_names_list = list(set(sciplex_df_without_ctrl['drug_2']))
    print("Total drugs2:", len(drug2_names_list))

    drug1_doses_list = list(set(sciplex_df_without_ctrl['dose_1']))
    drug2_doses_list = list(set(sciplex_df_without_ctrl['dose_2']))

    drug_embeddings = pd.read_parquet("./data/sciplex/sciplex4_unimol_emb.parquet")
    cell_embeddings = pd.read_parquet("./data/sciplex/sciplex4_uce_emb.parquet")

    print("grouping by cell, drug, and dose.......")
    data_grouped = sciplex_df_without_ctrl.groupby(['cell', 'drug_1', 'dose_1', 'drug_2', 'dose_2'])

    print("spliting train and val datasets.......")
    data_pairs_train, data_pairs_val, data_pairs_test = multi_data_pairs_split(cell_types_list, drug1_names_list,
                                                                               drug1_doses_list,
                                                                               drug2_names_list, drug2_doses_list,
                                                                               data_grouped, test_size=0.2, seed=seed,
                                                                               dataset_name='sciplex4')

    data_grouped = sciplex_df.groupby(['cell', 'drug_1', 'dose_1', 'drug_2', 'dose_2'])

    train_dataset = MultiDrugDataset(cell_embeddings, drug_embeddings, data_grouped, data_pairs_train, scaler=True)
    val_dataset = MultiDrugDataset(cell_embeddings, drug_embeddings, data_grouped, data_pairs_val, scaler=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def sciplex4_test_preprocessing(seed: int):
    print("load Sciplex4 data......")
    sciplex_df = pd.read_parquet("./data/sciplex/sciplex4.parquet")

    selected_columns = sciplex_df.iloc[:, 5:]
    sciplex_gene = selected_columns.columns.tolist()

    sciplex_df_without_ctrl = sciplex_df[(sciplex_df['drug_1'] != 'CTRL') | (sciplex_df['drug_2'] != 'CTRL')]
    print("Total data:", sciplex_df.shape[0])

    cell_types_list = list(set(sciplex_df_without_ctrl['cell']))
    print("Total cells:", len(cell_types_list))

    drug1_names_list = list(set(sciplex_df_without_ctrl['drug_1']))
    print("Total drugs1:", len(drug1_names_list))

    drug2_names_list = list(set(sciplex_df_without_ctrl['drug_2']))
    print("Total drugs2:", len(drug2_names_list))

    drug1_doses_list = list(set(sciplex_df_without_ctrl['dose_1']))
    drug2_doses_list = list(set(sciplex_df_without_ctrl['dose_2']))

    drug_embeddings = pd.read_parquet("./data/sciplex/sciplex4_unimol_emb.parquet")
    cell_embeddings = pd.read_parquet("./data/sciplex/sciplex4_uce_emb.parquet")

    print("grouping by cell, drug, and dose.......")
    data_grouped = sciplex_df_without_ctrl.groupby(['cell', 'drug_1', 'dose_1', 'drug_2', 'dose_2'])

    print("spliting train and val datasets.......")
    data_pairs_train, data_pairs_val, data_pairs_test = multi_data_pairs_split(cell_types_list, drug1_names_list,
                                                                               drug1_doses_list,
                                                                               drug2_names_list, drug2_doses_list,
                                                                               data_grouped, test_size=0.2, seed=seed,
                                                                               dataset_name='sciplex4')

    data_grouped = sciplex_df.groupby(['cell', 'drug_1', 'dose_1', 'drug_2', 'dose_2'])

    test_dataset = MultiDrugDataset(cell_embeddings, drug_embeddings, data_grouped, data_pairs_test, scaler=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return test_loader, sciplex_gene

