import pandas as pd
from preprocessing import esm2_preprocessing, lincs_step2_preprocessing, sciplex3_preprocessing, \
    sciplex4_preprocessing, lincs_test_preprocessing, sciplex3_test_preprocessing, sciplex4_test_preprocessing, \
    tahoe_preprocessing, lincs_benchmark_preprocessing, lincs_test_benchmark_preprocessing
from train import train_original_state_model, train_perturbation_model, \
    train_perturbation_model_acc, train_stage1, test_perturbation_model, test_Multiperturbation_model
from dataloader import CellDrugDataset, collate_fn
from utils import load_uce_pretrained_model, get_trainable_parameters, load_cureall_pretrained_model, \
    load_stage1_model, data_pairs_split, uce_emb, load_UniCure_pretrained_model, load_UniCureFTsc, load_UniCureFTtahoe, \
    load_UniCurePretrainsc, load_UniCureFTsc4, load_UniCurePretrainsc4
from model import UniCure, baselineModel
from accelerate import Accelerator
from loss import MMDLoss
from torch.utils.data import DataLoader

# seed = 3
# accelerator = Accelerator()

# if accelerator.is_local_main_process:
#     esm2_preprocessing()
# accelerator.wait_for_everyone()

# Lincs2020


# STEP.1 train_original_state_model
def train_lincs_step1(seed, accelerator):
    esm2_emb_df = pd.read_parquet('./data/lincs2020/lincs2020_esm2_emb.parquet')
    original_exp_df = pd.read_parquet('./data/lincs2020/lincs2020_control.parquet')
    UCE_lora_model = load_uce_pretrained_model(path='./requirement/UCE_pretraining_files/33l_8ep_1024t_1280.torch',
                                               target_layers=list(range(28, 33)))
    cureall_model = load_cureall_pretrained_model(UniCure(), f'./result/{seed}/lincs2020/best_stage_1_model.pth')
    # cureall_model = CureALL()
    cureall_model.uce_lora = UCE_lora_model
    lincs2020_cureall_model = cureall_model

    # check param
    # if accelerator.is_local_main_process:
    #     for name, param in lincs2020_cureall_model.named_parameters():
    #         if not param.requires_grad:
    #             print(f"Frozen: {name}")
    #         else:
    #             print(f"Trainable: {name}")
    # accelerator.wait_for_everyone()

    trainable_parameters = get_trainable_parameters(lincs2020_cureall_model)

    train_original_state_model(model=lincs2020_cureall_model, trainable_parameters=trainable_parameters,
                               esm2_emb=esm2_emb_df, original_exp=original_exp_df,
                               gene_columns_start=4, seed=seed, dataset_name="lincs2020", lr_rate=0.0001, batch_size=64,
                               num_epochs=800, early_stopping_patience=20, accelerator=accelerator)


# STEP.2 train_perturbation_model
def train_lincs_step2(seed):
    train_loader, val_loader = lincs_step2_preprocessing(seed)
    cureall_model = load_stage1_model(path=f'./result/2/lincs2020/best_original_state_model.pth')

    # for name, param in cureall_model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Frozen: {name}")
    #     else:
    #         print(f"Trainable: {name}")

    mmd_loss_fn = MMDLoss(kernel_type='rbf')
    trainable_parameters = get_trainable_parameters(cureall_model)
    train_perturbation_model(cureall_model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5,
                             num_epochs=800, seed=seed)


def train_lincs_benchmark(seed):
    train_loader, val_loader = lincs_benchmark_preprocessing(seed)
    cureall_model = load_stage1_model(path=f'./result/2/lincs2020/best_original_state_model.pth')

    # for name, param in cureall_model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Frozen: {name}")
    #     else:
    #         print(f"Trainable: {name}")

    mmd_loss_fn = MMDLoss(kernel_type='rbf')
    trainable_parameters = get_trainable_parameters(cureall_model)
    train_perturbation_model(cureall_model, mmd_loss_fn, trainable_parameters,
                             train_loader, val_loader, lr_rate=1e-5,
                             num_epochs=800, seed=seed, dataset_name='lincs_benchmark',
                             lambda_val=0.01,
                             max_batch_size=64)


def test_lincs(seed):
    test_loader, lincs_gene = lincs_test_preprocessing(seed)
    unicure_model = load_UniCure_pretrained_model(path=f'./result/{seed}/lincs2020/best_model.pth')
    test_perturbation_model(unicure_model, test_loader, seed, lincs_gene, dataset_name='lincs2020', max_size=10)


def test_lincs_benchmark(seed):
    test_loader, lincs_gene = lincs_test_benchmark_preprocessing(seed)
    unicure_model = load_UniCure_pretrained_model(path=f'./result/6/lincs_benchmark/best_model.pth')
    test_perturbation_model(unicure_model, test_loader, seed, lincs_gene, dataset_name='lincs_benchmark')


# SCiplex 3
def train_sciplex3(seed):
    train_loader, val_loader = sciplex3_preprocessing(seed)

    # cureall_model = load_stage1_model(path=None, output_size=1000, drug_window_size=32, drug_slide_step=16,
    #                                   cell_window_size=32, cell_slide_step=16, hidden_dim=64, dropout_rate=0.3)

    unicure_sciplex3_model = load_UniCureFTsc(path=f'./result/3/lincs2020/best_model.pth', output_size=1923)

    for name, param in unicure_sciplex3_model.named_parameters():
        if not param.requires_grad:
            print(f"Frozen: {name}")
        else:
            print(f"Trainable: {name}")

    # cureall_model = baselineModel()

    mmd_loss_fn = MMDLoss()

    trainable_parameters = get_trainable_parameters(unicure_sciplex3_model)

    train_perturbation_model(unicure_sciplex3_model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5,
                             num_epochs=800, seed=seed, early_stopping_patience=20, dataset_name='sciplex3',
                             lambda_val=0.01,
                             max_batch_size=512)


def test_sciplex3(seed):
    test_loader, sciplex3_gene = sciplex3_test_preprocessing(seed)
    unicure_model = load_UniCurePretrainsc(path=f'./result/{seed}/sciplex3/best_model.pth', output_size=1923)
    test_perturbation_model(unicure_model, test_loader, seed, sciplex3_gene, dataset_name='sciplex3', max_size=512)


# Sciplex 4
def train_sciplex4(seed):
    train_loader, val_loader = sciplex4_preprocessing(seed)

    # cureall_model = load_stage1_model(path=None, output_size=1000, drug_window_size=32, drug_slide_step=16,
    #                                   cell_window_size=32, cell_slide_step=16, hidden_dim=64, dropout_rate=0.3)

    unicure_sciplex4_model = load_UniCureFTsc4(path=f'./result/6/sciplex3/best_model.pth', output_size=1929)

    for name, param in unicure_sciplex4_model.named_parameters():
        if not param.requires_grad:
            print(f"Frozen: {name}")
        else:
            print(f"Trainable: {name}")

    # cureall_model = baselineModel()

    mmd_loss_fn = MMDLoss()

    trainable_parameters = get_trainable_parameters(unicure_sciplex4_model)

    train_perturbation_model(unicure_sciplex4_model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5,
                             num_epochs=800, seed=seed, early_stopping_patience=20, dataset_name='sciplex4',
                             lambda_val=0.01,
                             max_batch_size=512)


def test_sciplex4(seed):
    test_loader, sciplex4_gene = sciplex4_test_preprocessing(seed)
    unicure_model = load_UniCurePretrainsc4(path=f'./result/{seed}/sciplex4/best_model.pth', output_size=1929)
    test_Multiperturbation_model(unicure_model, test_loader, seed, sciplex4_gene, dataset_name='sciplex4', max_size=20)


def train_tahoe(seed):
    train_loader, val_loader = tahoe_preprocessing(seed)

    # cureall_model = load_stage1_model(path=None, output_size=1000, drug_window_size=32, drug_slide_step=16,
    #                                   cell_window_size=32, cell_slide_step=16, hidden_dim=64, dropout_rate=0.3)

    unicure_tahoe_model = load_UniCureFTtahoe(path=f'./result/6/sciplex3/best_model.pth', output_size=2990)

    for name, param in unicure_tahoe_model.named_parameters():
        if not param.requires_grad:
            print(f"Frozen: {name}")
        else:
            print(f"Trainable: {name}")

    # cureall_model = baselineModel()

    mmd_loss_fn = MMDLoss()

    trainable_parameters = get_trainable_parameters(unicure_tahoe_model)

    train_perturbation_model(unicure_tahoe_model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5,
                             num_epochs=800, seed=seed, early_stopping_patience=10, dataset_name='tahoe',
                             lambda_val=0.001,
                             max_batch_size=512)


if __name__ == '__main__':

    # seed = 4
    # accelerator = Accelerator()
    # train_lincs_step2(4)
    # train_lincs_step2(6)

    train_sciplex3(4)

    # train_sciplex4(4)

    # test_sciplex4(6)

    # train_lincs_benchmark(6)

    # test_lincs_benchmark(6)

    # train_tahoe(seed=6)

    # test_lincs(3)

    # test_sciplex3(6)

    # test_sciplex4(6 )







