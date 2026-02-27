from preprocessing import *
from train import *
from utils import *
from model import *
from accelerate import Accelerator
from loss import MMDLoss
import argparse


def train_lincs_step1(seed, accelerator):
    esm2_emb_df = pd.read_parquet('./data/lincs2020/lincs2020_esm2_emb.parquet')
    original_exp_df = pd.read_parquet('./data/lincs2020/lincs2020_control.parquet')
    UCE_lora_model = load_uce_pretrained_model(path='./requirement/UCE_pretraining_files/33l_8ep_1024t_1280.torch',
                                               target_layers=list(range(28, 33)))
    cureall_model = load_cureall_pretrained_model(UniCure(), f'./result/{seed}/lincs2020/best_stage_1_model.pth')

    cureall_model.uce_lora = UCE_lora_model
    lincs2020_cureall_model = cureall_model

    trainable_parameters = get_trainable_parameters(lincs2020_cureall_model)

    train_original_state_model(model=lincs2020_cureall_model, trainable_parameters=trainable_parameters,
                               esm2_emb=esm2_emb_df, original_exp=original_exp_df,
                               gene_columns_start=4, seed=seed, dataset_name="lincs2020", lr_rate=0.0001, batch_size=64,
                               num_epochs=800, early_stopping_patience=20, accelerator=accelerator)


def train_lincs_step2(seed):
    train_loader, val_loader = lincs_step2_preprocessing_v2(seed)

    cureall_model = load_stage1_model(path=f'./result/{seed}/lincs2020/best_stage_1_model.pth')

    mmd_loss_fn = MMDLoss(kernel_type='rbf')
    trainable_parameters = get_trainable_parameters(cureall_model)

    train_perturbation_delta_model_v2(cureall_model, mmd_loss_fn, trainable_parameters, train_loader, val_loader,
                                      lr_rate=1e-5,
                                      num_epochs=400, seed=seed)


def test_lincs(seed):
    test_loader, lincs_gene = lincs_test_preprocessing_v2(seed, max_size=600)
    unicure_model = load_UniCure_pretrained_model(path=f'./result/{seed}/lincs2020/Unicure_best_model.pth')
    test_perturbation_model_v2(unicure_model, test_loader, seed, lincs_gene,
                               dataset_name='lincs2020',
                               subsample_size=10)


def train_sciplex3(seed):
    train_loader, val_loader = sciplex3_preprocessing_v2(seed)

    unicure_sciplex3_model = load_UniCureFTsc(path=f'./result/{seed}/lincs2020/Unicure_best_model.pth',
                                              output_size=1923)

    mmd_loss_fn = MMDLoss(kernel_type='rbf')

    trainable_parameters = get_trainable_parameters(unicure_sciplex3_model)

    train_perturbation_delta_model_v2(unicure_sciplex3_model, mmd_loss_fn,
                                      trainable_parameters, train_loader, val_loader,
                                      lr_rate=1e-5,
                                      num_epochs=400,
                                      seed=seed,
                                      early_stopping_patience=10,
                                      dataset_name='sciplex3')


def test_sciplex3(seed):
    test_loader, sciplex3_gene = sciplex3_test_preprocessing_v2(seed)
    unicure_model = load_UniCurePretrainsc(path=f'./result/{seed}/sciplex3/Unicure_best_model.pth', output_size=1923)
    test_perturbation_model_v2(unicure_model, test_loader, seed, sciplex3_gene,
                               dataset_name='sciplex3',
                               subsample_size=512)


# Sciplex 4
def train_sciplex4(seed):
    train_loader, val_loader = sciplex4_preprocessing(seed)

    unicure_sciplex4_model = load_UniCureFTsc4(path=f'./result/{seed}/sciplex3/Unicure_best_model.pth', output_size=1929)

    mmd_loss_fn = MMDLoss()

    trainable_parameters = get_trainable_parameters(unicure_sciplex4_model)

    train_perturbation_delta_model(unicure_sciplex4_model,
                                   mmd_loss_fn,
                                   trainable_parameters,
                                   train_loader,
                                   val_loader,
                                   lr_rate=1e-5,
                                   num_epochs=400,
                                   seed=seed,
                                   early_stopping_patience=10,
                                   dataset_name='sciplex4',
                                   max_batch_size=512)


def test_sciplex4(seed):
    test_loader, sciplex4_gene = sciplex4_test_preprocessing(seed)
    unicure_model = load_UniCurePretrainsc4(path=f'./result/{seed}/sciplex4/Unicure_best_model.pth', output_size=1929)
    test_Multiperturbation_model(unicure_model, test_loader, seed, sciplex4_gene, dataset_name='sciplex4', max_size=512)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UniCure Pipeline Execution")
    parser.add_argument('--seed', type=int, default=11, help='Random seed')

    # 添加各个阶段的控制开关
    parser.add_argument('--run_lincs1', action='store_true', help='Run LINCS Step 1')
    parser.add_argument('--run_lincs2', action='store_true', help='Run LINCS Step 2 (Train & Test)')
    parser.add_argument('--run_sciplex3', action='store_true', help='Run Sciplex 3 (Train & Test)')
    parser.add_argument('--run_sciplex4', action='store_true', help='Run Sciplex 4 (Train & Test)')
    parser.add_argument('--run_all', action='store_true', help='Run all stages sequentially')

    args = parser.parse_args()
    seed = args.seed

    # 执行逻辑
    if args.run_lincs1 or args.run_all:
        print("=== Starting LINCS Step 1 ===")
        accelerator = Accelerator()
        train_lincs_step1(seed, accelerator)

    if args.run_lincs2 or args.run_all:
        print("=== Starting LINCS Step 2 ===")
        train_lincs_step2(seed)
        test_lincs(seed)

    if args.run_sciplex3 or args.run_all:
        print("=== Starting Sciplex 3 ===")
        train_sciplex3(seed)
        test_sciplex3(seed)

    if args.run_sciplex4 or args.run_all:
        print("=== Starting Sciplex 4 ===")
        train_sciplex4(seed)
        test_sciplex4(seed)

    if not (args.run_lincs1 or args.run_lincs2 or args.run_sciplex3 or args.run_sciplex4 or args.run_all):
        print("No execution flag provided. Please use flags like --run_lincs1 or --run_all.")

