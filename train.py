import os
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from accelerate.utils import gather_object
from tqdm.auto import tqdm
import torch.distributed as dist
from utils import *
from torch import nn
from scipy.stats import spearmanr, pearsonr
from loss import MMDLoss


def train_original_state_model(model, trainable_parameters, esm2_emb: pd.DataFrame, original_exp: pd.DataFrame,
                               gene_columns_start: int, seed: int, dataset_name: str,
                               lr_rate: float, batch_size: int, num_epochs: int, early_stopping_patience: int,
                               accelerator=None):
    # Dataloader
    set_seed(seed)
    input = esm2_emb.values
    output = original_exp.iloc[:, gene_columns_start:].values
    num_features = input.shape[1]
    src = input[:, :num_features // 2]
    mask = input[:, num_features // 2:]

    src_train, src_val, mask_train, mask_val, target_train, target_val = train_test_split(
        src, mask, output, test_size=0.2, random_state=seed
    )

    src_train_tensor = torch.tensor(src_train, dtype=torch.float32)
    mask_train_tensor = torch.tensor(mask_train, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train, dtype=torch.float32)
    src_val_tensor = torch.tensor(src_val, dtype=torch.float32)
    mask_val_tensor = torch.tensor(mask_val, dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val, dtype=torch.float32)

    train_dataset = TensorDataset(src_train_tensor, mask_train_tensor, target_train_tensor)
    val_dataset = TensorDataset(src_val_tensor, mask_val_tensor, target_val_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model
    criterion = nn.MSELoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = nn.DataParallel(model)
    # model.to(device)
    # accelerator = Accelerator()
    device = accelerator.device
    model.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.1)
    model, train_dataloader, val_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer, scheduler)

    train_loss_list = []
    train_r2_list = []
    val_loss_list = []
    val_r2_list = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model = None
    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # train
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_r2 = 0.0
        sample_size = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                    disable=not accelerator.is_local_main_process)
        for batch_idx, (src_batch, mask_batch, targets_batch) in loop:
            src_batch = src_batch.to(device)
            mask_batch = mask_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            outputs = model("baseline", src_batch, mask_batch)
            loss = criterion(outputs.to(torch.float32), targets_batch)
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()

            running_loss += loss.item() * targets_batch.size(0)
            sample_size += targets_batch.size(0)
            outputs_cpu = outputs.detach().float().cpu().numpy()
            targets_cpu = targets_batch.detach().float().cpu().numpy()

            r2 = 0
            for i in range(outputs_cpu.shape[0]):
                r2 += r2_score(targets_cpu[i], outputs_cpu[i])
            avg_batch_r2 = r2 / outputs_cpu.shape[0]
            running_r2 += avg_batch_r2 * targets_batch.size(0)
            loop.set_description(f'Lincs1:Epoch [{epoch + 1}/{num_epochs}] [{batch_idx + 1}/{len(train_dataloader)}]')
            loop.set_postfix(train_loss=loss.item(), r2_score_mean=avg_batch_r2)

        scheduler.step()
        gathered_running_loss = accelerator.gather(torch.tensor(running_loss).to(device))
        gathered_running_r2 = accelerator.gather(torch.tensor(running_r2).to(device))
        gathered_sample_size = accelerator.gather(torch.tensor(sample_size).to(device))
        total_train_loss = torch.sum(gathered_running_loss).item()
        total_train_r2 = torch.sum(gathered_running_r2).item()
        total_sample_size = torch.sum(gathered_sample_size).item()

        avg_train_loss = total_train_loss / total_sample_size
        avg_train_r2 = total_train_r2 / total_sample_size
        train_loss_list.append(avg_train_loss)
        train_r2_list.append(avg_train_r2)
        if accelerator.is_local_main_process:
            print(
                f'Lincs1:Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training R2: {avg_train_r2:.4f}',
                flush=True)
        accelerator.wait_for_everyone()

        # Verification phase
        model.eval()
        val_loss = 0.0
        val_r2 = 0.0
        sample_size = 0
        with torch.no_grad():
            loop_v = tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                          disable=not accelerator.is_local_main_process)
            for batch_idx, (src_batch, mask_batch, targets_batch) in loop_v:
                src_batch = src_batch.to(device)
                mask_batch = mask_batch.to(device)
                targets_batch = targets_batch.to(device)

                outputs = model("baseline", src_batch, mask_batch)

                loss = criterion(outputs.to(torch.float32), targets_batch)

                val_loss += loss.item() * targets_batch.size(0)
                sample_size += targets_batch.size(0)
                outputs_cpu = outputs.detach().float().cpu().numpy()
                targets_cpu = targets_batch.detach().float().cpu().numpy()

                r2 = 0
                for i in range(outputs_cpu.shape[0]):
                    r2 += r2_score(targets_cpu[i], outputs_cpu[i])
                avg_batch_r2 = r2 / outputs_cpu.shape[0]
                val_r2 += avg_batch_r2 * targets_batch.size(0)
                loop_v.set_description(
                    f'Lincs1:Epoch [{epoch + 1}/{num_epochs}], [{batch_idx + 1}/{len(val_dataloader)}]')
                loop_v.set_postfix(val_loss=loss.item(), r2_score_mean=avg_batch_r2)

        gathered_val_loss = accelerator.gather(torch.tensor(val_loss).to(device))
        gathered_val_r2 = accelerator.gather(torch.tensor(val_r2).to(device))
        gathered_sample_size = accelerator.gather(torch.tensor(sample_size).to(device))
        total_val_loss = torch.sum(gathered_val_loss).item()
        total_val_r2 = torch.sum(gathered_val_r2).item()
        total_sample_size = torch.sum(gathered_sample_size).item()

        avg_val_loss = total_val_loss / total_sample_size
        avg_val_r2 = total_val_r2 / total_sample_size
        val_loss_list.append(avg_val_loss)
        val_r2_list.append(avg_val_r2)
        if accelerator.is_local_main_process:
            print(f'Lincs1:Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val R2: {avg_val_r2:.4f}',
                  flush=True)
        accelerator.wait_for_everyone()

        # early_stop
        if accelerator.is_main_process:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                best_model = model
                unwrapped_model = accelerator.unwrap_model(best_model)
                # unwrapped_model.uce_lora = unwrapped_model.uce_lora.merge_lora(inplace=False)
                torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, 'best_unicure_stage_1_model.pth'))
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}, best epoch: {best_epoch}", flush=True)
                accelerator.set_trigger()

        accelerator.wait_for_everyone()  # 所有进程同步
        if accelerator.check_trigger():
            break

    result_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'train_r2': train_r2_list,
        'val_loss': val_loss_list,
        'val_r2': val_r2_list
    })

    if accelerator.is_local_main_process:
        result_df.to_csv(os.path.join(save_dir, 'original_state_model_training_results.csv'), index=False)
        print("Training complete.")
    accelerator.wait_for_everyone()


def train_perturbation_delta_model(model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5,
                                   num_epochs=10,
                                   device=None, seed=48, early_stopping_patience=10, dataset_name='lincs2020',
                                   max_batch_size=64):
    if device is None:
        device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)
    model.to(device)
    mmd_loss_fn.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr_rate)

    # --- [修改点 1] 初始化记录列表 ---
    train_loss_list = []
    train_r2_list = []
    train_delta_mse_list = []  # 新增
    train_delta_pearson_list = []  # 新增

    val_loss_list = []
    val_r2_list = []
    val_delta_mse_list = []  # 新增
    val_delta_pearson_list = []  # 新增

    best_val_loss = float('inf')
    patience_counter = 0

    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_r2 = 0

        # --- [修改点 2] 初始化当前 Epoch 的累加器 ---
        train_delta_mse_sum = 0.0
        train_delta_pearson_sum = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (_, cell_embeds, drug_embed, real_outputs_treated, unperturb_gexp) in loop:
            cell_embeds = cell_embeds.to(device)
            drug_embed = drug_embed.to(device)
            real_outputs_treated = real_outputs_treated.to(device)
            unperturb_gexp = unperturb_gexp.to(device)

            input_size = cell_embeds.size(0)
            batch_size = min(input_size, max_batch_size)

            # Sampling logic
            indices = torch.randperm(input_size)[:batch_size]
            cell_embeds = cell_embeds[indices]
            unperturb_gexp = unperturb_gexp[indices]  # Control batch
            repeated_drug_embed = drug_embed.repeat(batch_size, 1)

            if real_outputs_treated.size(0) >= batch_size:
                cell_indices = torch.randperm(real_outputs_treated.size(0))[:batch_size]
            else:
                cell_indices = torch.randint(0, real_outputs_treated.size(0), (batch_size,))
            real_outputs_treated = real_outputs_treated[cell_indices]  # Real Treated batch

            # Forward propagation
            outputs = model("pertrub_forward", cell_embeds, repeated_drug_embed)

            # Calculate the MMD loss
            loss = mmd_loss_fn(outputs, real_outputs_treated)

            # Calculate the Euclidean distance constraint
            euclidean_distance = torch.norm(outputs - unperturb_gexp, p=2, dim=1).mean()

            # --- [修改点 3] 新增 Delta Metrics 计算模块 (Training) ---
            # 1. 计算均值 (Batch Mean)
            # shape: (Genes,)
            pred_batch_mean = outputs.mean(0)
            real_batch_mean = real_outputs_treated.mean(0)
            control_batch_mean = unperturb_gexp.mean(0)

            # 2. 计算 Delta (Batch Mean - Control Mean)
            pred_delta = pred_batch_mean - control_batch_mean
            real_delta = real_batch_mean - control_batch_mean

            # 3. 计算 Delta MSE
            delta_mse = nn.functional.mse_loss(pred_delta, real_delta).item()

            # 1. Cosine Embedding Loss 或 Pearson Loss
            cosine_sim = nn.functional.cosine_similarity(pred_delta, real_delta, dim=0)
            loss_corr = 1 - cosine_sim

            total_loss = 1 * delta_mse + 0.5 * loss_corr + 0.1 * loss

            # 4. 计算 Delta Pearson Correlation
            # 转为 numpy 计算
            pred_delta_np = pred_delta.detach().cpu().numpy()
            real_delta_np = real_delta.detach().cpu().numpy()

            # 防止标准差为0导致 NaN
            if np.std(pred_delta_np) < 1e-9 or np.std(real_delta_np) < 1e-9:
                delta_pearson = 0.0
            else:
                delta_pearson, _ = pearsonr(pred_delta_np, real_delta_np)
            # -----------------------------------------------------

            # Basic Metrics
            r2 = r2_score(real_outputs_treated.mean(axis=0).detach().cpu().numpy(),
                          outputs.mean(axis=0).detach().cpu().numpy())

            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate
            train_loss += total_loss.item()
            train_r2 += r2
            train_delta_mse_sum += delta_mse
            train_delta_pearson_sum += delta_pearson

            torch.cuda.empty_cache()

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}] [{i}/{len(train_loader)}]')
            # 更新显示的后缀，增加 Delta 指标
            loop.set_postfix(Loss=loss.item(), R2=r2, D_MSE=delta_mse, D_Corr=delta_pearson)

        # Calculate Epoch Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_r2 = train_r2 / len(train_loader)
        avg_train_delta_mse = train_delta_mse_sum / len(train_loader)
        avg_train_delta_pearson = train_delta_pearson_sum / len(train_loader)

        train_loss_list.append(avg_train_loss)
        train_r2_list.append(avg_train_r2)
        train_delta_mse_list.append(avg_train_delta_mse)
        train_delta_pearson_list.append(avg_train_delta_pearson)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Train | Loss: {avg_train_loss:.4f}, R2: {avg_train_r2:.4f}, '
              f'Delta MSE: {avg_train_delta_mse:.4f}, Delta Corr: {avg_train_delta_pearson:.4f}', flush=True)

        # Verification phase
        model.eval()
        val_loss = 0
        val_r2 = 0
        val_delta_mse_sum = 0.0
        val_delta_pearson_sum = 0.0

        with torch.no_grad():
            loop_v = tqdm(enumerate(val_loader), total=len(val_loader))
            # --- [修改点 4] 这里一定要接收 unperturb_gexp_val，之前是 _ ---
            for i, (_, cell_embeds_val, drug_embed_val, real_outputs_treated_val, unperturb_gexp_val) in loop_v:
                cell_embeds_val = cell_embeds_val.to(device)
                drug_embed_val = drug_embed_val.to(device)
                real_outputs_treated_val = real_outputs_treated_val.to(device)
                unperturb_gexp_val = unperturb_gexp_val.to(device)  # 需要放到 device 上

                input_size = cell_embeds_val.size(0)
                batch_size = min(input_size, max_batch_size)

                indices = torch.randperm(input_size)[:batch_size]
                cell_embeds_val = cell_embeds_val[indices]
                # 这里同样需要对 control 进行采样以保持一致
                unperturb_gexp_val = unperturb_gexp_val[indices]

                repeated_drug_embed_val = drug_embed_val.repeat(batch_size, 1)

                if real_outputs_treated_val.size(0) >= batch_size:
                    cell_indices_val = torch.randperm(real_outputs_treated_val.size(0))[:batch_size]
                else:
                    cell_indices_val = torch.randint(0, real_outputs_treated_val.size(0), (batch_size,))
                real_outputs_treated_val = real_outputs_treated_val[cell_indices_val]

                outputs = model("pertrub_forward", cell_embeds_val, repeated_drug_embed_val)

                loss = mmd_loss_fn(outputs, real_outputs_treated_val)
                r2 = r2_score(real_outputs_treated_val.mean(axis=0).detach().cpu().numpy(),
                              outputs.mean(axis=0).detach().cpu().numpy())

                # --- [修改点 5] 新增 Delta Metrics 计算模块 (Validation) ---
                pred_batch_mean_val = outputs.mean(0)
                real_batch_mean_val = real_outputs_treated_val.mean(0)
                control_batch_mean_val = unperturb_gexp_val.mean(0)

                pred_delta_val = pred_batch_mean_val - control_batch_mean_val
                real_delta_val = real_batch_mean_val - control_batch_mean_val

                delta_mse_val = nn.functional.mse_loss(pred_delta_val, real_delta_val).item()

                cosine_sim_val = nn.functional.cosine_similarity(pred_delta_val, real_delta_val, dim=0)
                loss_corr_val = 1 - cosine_sim_val

                total_loss_val = delta_mse_val + 0.5 * loss_corr_val + 0.1 * loss

                pred_delta_val_np = pred_delta_val.detach().cpu().numpy()
                real_delta_val_np = real_delta_val.detach().cpu().numpy()

                if np.std(pred_delta_val_np) < 1e-9 or np.std(real_delta_val_np) < 1e-9:
                    delta_pearson_val = 0.0
                else:
                    delta_pearson_val, _ = pearsonr(pred_delta_val_np, real_delta_val_np)
                # --------------------------------------------------------

                val_loss += total_loss_val.item()
                val_r2 += r2
                val_delta_mse_sum += delta_mse_val
                val_delta_pearson_sum += delta_pearson_val

                loop_v.set_description(f'Epoch [{epoch + 1}/{num_epochs}] Val [{i}/{len(val_loader)}]')
                loop_v.set_postfix(Val_Loss=loss.item(), Val_R2=r2, Val_D_MSE=delta_mse_val)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        avg_val_delta_mse = val_delta_mse_sum / len(val_loader)
        avg_val_delta_pearson = val_delta_pearson_sum / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Val | Loss: {avg_val_loss:.4f}, R2: {avg_val_r2:.4f}, '
              f'Delta MSE: {avg_val_delta_mse:.4f}, Delta Corr: {avg_val_delta_pearson:.4f}', flush=True)

        val_loss_list.append(avg_val_loss)
        val_r2_list.append(avg_val_r2)
        val_delta_mse_list.append(avg_val_delta_mse)
        val_delta_pearson_list.append(avg_val_delta_pearson)

        # Save Best Model Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, 'Unicure_best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"early_stopping, best epoch:{best_epoch}", flush=True)
            break

    # --- [修改点 6] 保存结果到 DataFrame ---
    result_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'train_r2': train_r2_list,
        'train_delta_mse': train_delta_mse_list,
        'train_delta_pearson': train_delta_pearson_list,
        'val_loss': val_loss_list,
        'val_r2': val_r2_list,
        'val_delta_mse': val_delta_mse_list,
        'val_delta_pearson': val_delta_pearson_list
    })

    result_df.to_csv(os.path.join(save_dir, 'training_results.csv'), index=False)


def train_perturbation_delta_model_v2(model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5,
                                      num_epochs=10, device=None, seed=48, early_stopping_patience=10,
                                      dataset_name='lincs2020'):
    """
    注意：max_batch_size 参数现在已经在 Dataset 的 max_sample_size 中控制了。
    这里的 DataLoader batch_size 控制的是一次处理多少个“实验组”。
    """
    if device is None:
        device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    # Utils functions like set_seed need to be available in your scope
    # set_seed(seed)

    model.to(device)
    mmd_loss_fn.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr_rate)

    # Metrics containers
    train_loss_list, train_r2_list, train_delta_mse_list, train_delta_pearson_list = [], [], [], []
    val_loss_list, val_r2_list, val_delta_mse_list, val_delta_pearson_list = [], [], [], []

    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        train_stats = {'loss': 0.0, 'r2': 0.0, 'd_mse': 0.0, 'd_corr': 0.0}

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for batch_idx, batch_data in enumerate(loop):
            # 1. Move to Device
            cell_embeds = batch_data['cell_embeds'].to(device)  # (Total_Cells_in_Batch, Dim)
            drug_embeds = batch_data['drug_embeds'].to(device)
            real_treated = batch_data['real_treated'].to(device)
            real_control = batch_data['real_control'].to(device)

            treated_lens = batch_data['treated_lengths']
            control_lens = batch_data['control_lengths']

            # 2. Vectorized Forward Pass
            # 一次性计算 Batch 内所有组的所有细胞
            outputs = model("pertrub_forward", cell_embeds, drug_embeds)

            # 3. Calculate Loss per Group
            # Split huge tensor back into groups
            outputs_split = torch.split(outputs, treated_lens)
            real_treated_split = torch.split(real_treated, treated_lens)
            real_control_split = torch.split(real_control, control_lens)

            batch_loss_accum = 0.0
            batch_r2_accum = 0.0
            batch_dmse_accum = 0.0
            batch_dcorr_accum = 0.0

            current_batch_size = len(treated_lens)  # How many groups in this batch

            # Loop over groups (e.g., 32 times), purely Tensor operations
            for i in range(current_batch_size):
                pred_g = outputs_split[i]
                real_g = real_treated_split[i]
                ctrl_g = real_control_split[i]

                # Metrics Calculation
                pred_mean = pred_g.mean(0)
                real_mean = real_g.mean(0)
                ctrl_mean = ctrl_g.mean(0)

                pred_delta = pred_mean - ctrl_mean
                real_delta = real_mean - ctrl_mean

                # Losses
                mmd = mmd_loss_fn(pred_g, real_g)
                delta_mse = nn.functional.mse_loss(pred_delta, real_delta)
                cosine_sim = nn.functional.cosine_similarity(pred_delta, real_delta, dim=0)
                loss_corr = 1 - cosine_sim

                loss_group = delta_mse + 0.5 * loss_corr + 0.1 * mmd

                # Accumulate Gradients (Normalize by batch size)
                batch_loss_accum += loss_group

                # Logging Metrics (Detach to CPU for logging)
                with torch.no_grad():
                    # Delta MSE
                    batch_dmse_accum += delta_mse.item()

                    # Pearson
                    pd_np = pred_delta.cpu().numpy()
                    rd_np = real_delta.cpu().numpy()
                    if np.std(pd_np) < 1e-9 or np.std(rd_np) < 1e-9:
                        corr = 0.0
                    else:
                        corr, _ = pearsonr(pd_np, rd_np)
                    batch_dcorr_accum += corr

                    # R2 (on means)
                    r2 = r2_score(real_mean.cpu().numpy(),
                                  pred_mean.cpu().numpy())
                    batch_r2_accum += r2

            # 4. Backward & Opt
            final_batch_loss = batch_loss_accum / current_batch_size

            optimizer.zero_grad()
            final_batch_loss.backward()
            optimizer.step()

            # 5. Update Log Accumulators
            train_stats['loss'] += final_batch_loss.item()
            train_stats['r2'] += batch_r2_accum / current_batch_size
            train_stats['d_mse'] += batch_dmse_accum / current_batch_size
            train_stats['d_corr'] += batch_dcorr_accum / current_batch_size

            if batch_idx % 10 == 0:
                loop.set_postfix(loss=final_batch_loss.item())

        # Average Train Stats
        n_batches = len(train_loader)
        avg_train_loss = train_stats['loss'] / n_batches
        avg_train_r2 = train_stats['r2'] / n_batches
        avg_train_delta_mse = train_stats['d_mse'] / n_batches
        avg_train_delta_pearson = train_stats['d_corr'] / n_batches

        train_loss_list.append(avg_train_loss)
        train_r2_list.append(avg_train_r2)
        train_delta_mse_list.append(avg_train_delta_mse)
        train_delta_pearson_list.append(avg_train_delta_pearson)

        print(f"Epoch {epoch + 1} Train | Loss: {avg_train_loss:.4f} | R2: {avg_train_r2:.4f} | "
              f"DMSE: {avg_train_delta_mse:.4f} | DCorr: {avg_train_delta_pearson:.4f}", flush=True)

        # --- VALIDATION ---
        model.eval()
        val_stats = {'loss': 0.0, 'r2': 0.0, 'd_mse': 0.0, 'd_corr': 0.0}

        with torch.no_grad():
            loop_val = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")

            for batch_data in loop_val:
                cell_embeds = batch_data['cell_embeds'].to(device)
                drug_embeds = batch_data['drug_embeds'].to(device)
                real_treated = batch_data['real_treated'].to(device)
                real_control = batch_data['real_control'].to(device)

                treated_lens = batch_data['treated_lengths']
                control_lens = batch_data['control_lengths']

                outputs = model("pertrub_forward", cell_embeds, drug_embeds)

                outputs_split = torch.split(outputs, treated_lens)
                real_treated_split = torch.split(real_treated, treated_lens)
                real_control_split = torch.split(real_control, control_lens)

                current_batch_size = len(treated_lens)

                for i in range(current_batch_size):
                    pred_g = outputs_split[i]
                    real_g = real_treated_split[i]
                    ctrl_g = real_control_split[i]

                    pred_mean = pred_g.mean(0)
                    real_mean = real_g.mean(0)
                    ctrl_mean = ctrl_g.mean(0)

                    pred_delta = pred_mean - ctrl_mean
                    real_delta = real_mean - ctrl_mean

                    # Losses
                    mmd = mmd_loss_fn(pred_g, real_g)
                    delta_mse = nn.functional.mse_loss(pred_delta, real_delta)
                    cosine_sim = nn.functional.cosine_similarity(pred_delta, real_delta, dim=0)
                    loss_corr = 1 - cosine_sim

                    total_loss = delta_mse + 0.5 * loss_corr + 0.1 * mmd

                    # Accumulate
                    val_stats['loss'] += total_loss.item() / current_batch_size
                    val_stats['d_mse'] += delta_mse.item() / current_batch_size

                    # Pearson
                    pd_np = pred_delta.cpu().numpy()
                    rd_np = real_delta.cpu().numpy()
                    if np.std(pd_np) < 1e-9 or np.std(rd_np) < 1e-9:
                        corr = 0.0
                    else:
                        corr, _ = pearsonr(pd_np, rd_np)
                    val_stats['d_corr'] += corr / current_batch_size

                    # R2
                    r2 = r2_score(real_mean.cpu().numpy(),
                                  pred_mean.cpu().numpy())
                    val_stats['r2'] += r2 / current_batch_size

            # Update Validation Loop Batch Loss (Accumulating across batch)
            # 由于 val_stats 是在每个 item 内部累加除以 batch size 后的值，
            # 这里的 val_stats 实际上记录的是所有 batch 的 loss 总和，
            # 所以在外面除以 len(val_loader) 就是平均值。

        n_val_batches = len(val_loader)
        avg_val_loss = val_stats['loss'] / n_val_batches
        avg_val_r2 = val_stats['r2'] / n_val_batches
        avg_val_delta_mse = val_stats['d_mse'] / n_val_batches
        avg_val_delta_pearson = val_stats['d_corr'] / n_val_batches

        val_loss_list.append(avg_val_loss)
        val_r2_list.append(avg_val_r2)
        val_delta_mse_list.append(avg_val_delta_mse)
        val_delta_pearson_list.append(avg_val_delta_pearson)

        print(f"Epoch {epoch + 1} Val   | Loss: {avg_val_loss:.4f} | R2: {avg_val_r2:.4f} | "
              f"DMSE: {avg_val_delta_mse:.4f} | DCorr: {avg_val_delta_pearson:.4f}", flush=True)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, 'Unicure_best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered. Best epoch: {best_epoch}", flush=True)
            break

    # Save Results
    result_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'train_r2': train_r2_list,
        'train_delta_mse': train_delta_mse_list,
        'train_delta_pearson': train_delta_pearson_list,
        'val_loss': val_loss_list,
        'val_r2': val_r2_list,
        'val_delta_mse': val_delta_mse_list,
        'val_delta_pearson': val_delta_pearson_list
    })
    result_df.to_csv(os.path.join(save_dir, 'training_results.csv'), index=False)


def test_perturbation_model(model, test_loader, seed, gene_list, device=None, dataset_name='lincs2020', max_size=10):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    predictions_list = []
    real_outputs_list = []
    loop = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, ((cell_type, drug_name, drug_dose), cell_embeds, drug_embed, real_outputs_treated, unperturb_gexp) in loop:
        cell_embeds = cell_embeds.to(device)
        drug_embed = drug_embed.to(device)
        real_outputs_treated = real_outputs_treated.to(device)
        # unperturb_gexp = unperturb_gexp.to(device)
        input_size = cell_embeds.size(0)
        # repeated_drug_embed = drug_embed.repeat(input_size, 1)
        batch_size = min(input_size, max_size)
        # indices = torch.randperm(input_size)[:batch_size]
        indices = torch.arange(batch_size)
        cell_embeds = cell_embeds[indices]
        # unperturb_gexp = unperturb_gexp[indices]
        # The drug representation was repeated batch_size times
        repeated_drug_embed = drug_embed.repeat(batch_size, 1)

        if real_outputs_treated.size(0) >= batch_size:
            cell_indices = torch.randperm(real_outputs_treated.size(0))[:batch_size]
        else:
            cell_indices = torch.randint(0, real_outputs_treated.size(0), (batch_size,))
        real_outputs_treated = real_outputs_treated[cell_indices]

        # Forward propagation
        outputs = model("pertrub_forward", cell_embeds, repeated_drug_embed)
        real_outputs_batch = real_outputs_treated.detach().cpu().numpy()

        for j in range(batch_size):
            predictions_list.append([cell_type, drug_name, drug_dose] + outputs[j].tolist())
            real_outputs_list.append([cell_type, drug_name, drug_dose] + real_outputs_batch[j].tolist())

    columns = ['cell_type', 'drug_name', 'drug_dose'] + gene_list
    predictions_df = pd.DataFrame(predictions_list, columns=columns)
    real_outputs_df = pd.DataFrame(real_outputs_list, columns=columns)
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    real_outputs_df.to_csv(os.path.join(save_dir, 'real_outputs.csv'), index=False)


def test_perturbation_model_v2(model, test_loader, seed, gene_list, device=None,
                               dataset_name='lincs2020', subsample_size=10):
    if device is None:
        device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- 容器初始化 ---
    # 1. 用于保存抽样单细胞数据 (Predictions & Real)
    sub_preds, sub_reals, sub_metas = [], [], []

    # 2. 用于保存均值数据 (Mean Predictions, Mean Real, Mean Control)
    mean_preds, mean_reals, mean_controls, mean_metas = [], [], [], []

    # 3. 用于保存评估指标 (Metrics)
    metrics_list = []

    print("Starting Inference with Metrics Calculation...", flush=True)

    with torch.no_grad():
        loop = tqdm(test_loader, total=len(test_loader))

        for batch_data in loop:
            # Move to GPU
            cell_embeds = batch_data['cell_embeds'].to(device)
            drug_embeds = batch_data['drug_embeds'].to(device)
            real_treated = batch_data['real_treated'].to(device)
            real_control = batch_data['real_control'].to(device)  # 需要 Control 来计算 Delta

            metadata_batch = batch_data['metadata']
            treated_lens = batch_data['treated_lengths']
            control_lens = batch_data['control_lengths']

            # Inference (Vectorized)
            outputs = model("pertrub_forward", cell_embeds, drug_embeds)

            # --- Splitting back to Groups ---
            # 为了分别计算每个组的均值和抽样，必须把大 Tensor 切开
            outputs_split = torch.split(outputs, treated_lens)
            real_treated_split = torch.split(real_treated, treated_lens)
            real_control_split = torch.split(real_control, control_lens)  # Control 的数量可能和 Treated 不一样

            # 遍历 Batch 中的每一个实验组
            for i in range(len(metadata_batch)):
                cell_type, drug_name, drug_dose = metadata_batch[i]

                # 获取当前组的 Tensor
                pred_g = outputs_split[i]  # (N_cells, Genes)
                real_g = real_treated_split[i]  # (N_cells, Genes)
                ctrl_g = real_control_split[i]  # (M_cells, Genes)

                curr_n = pred_g.size(0)

                # ===========================
                # A. 抽样保存 (Subsampling)
                # ===========================
                # 随机抽取 subsample_size 个，如果不足则全取
                if curr_n > subsample_size:
                    # 使用 torch.randperm 生成随机索引
                    indices = torch.randperm(curr_n)[:subsample_size]
                    p_sub = pred_g[indices]
                    r_sub = real_g[indices]
                else:
                    p_sub = pred_g
                    r_sub = real_g

                # 转 numpy 存入列表
                sub_preds.append(p_sub.cpu().numpy())
                sub_reals.append(r_sub.cpu().numpy())

                # 扩展 metadata
                actual_sub_n = p_sub.size(0)
                sub_metas.append(np.array([[cell_type, drug_name, drug_dose]] * actual_sub_n))

                # ===========================
                # B. 均值计算 (Means)
                # ===========================
                p_mean = pred_g.mean(dim=0).cpu().numpy()  # (Genes,)
                r_mean = real_g.mean(dim=0).cpu().numpy()
                c_mean = ctrl_g.mean(dim=0).cpu().numpy()

                mean_preds.append(p_mean)
                mean_reals.append(r_mean)
                mean_controls.append(c_mean)
                mean_metas.append([cell_type, drug_name, drug_dose])

                # ===========================
                # C. 指标计算 (Metrics)
                # ===========================
                # 计算 Delta (扰动量)
                delta_pred = p_mean - c_mean
                delta_real = r_mean - c_mean

                # 1. 绝对值相关性 (Pearson, Spearman, R2)
                # 注意：计算两个向量之间的相关性
                if np.std(p_mean) < 1e-9 or np.std(r_mean) < 1e-9:
                    pr, sr, r2 = 0.0, 0.0, 0.0
                else:
                    pr, _ = pearsonr(p_mean, r_mean)
                    sr, _ = spearmanr(p_mean, r_mean)
                    r2 = r2_score(r_mean, p_mean)  # r2_score(y_true, y_pred)

                # 2. Delta 相关性
                if np.std(delta_pred) < 1e-9 or np.std(delta_real) < 1e-9:
                    d_pr, d_sr, d_r2 = 0.0, 0.0, 0.0
                else:
                    d_pr, _ = pearsonr(delta_pred, delta_real)
                    d_sr, _ = spearmanr(delta_pred, delta_real)
                    d_r2 = r2_score(delta_real, delta_pred)

                metrics_list.append([
                    cell_type, drug_name, drug_dose,
                    pr, sr, r2,
                    d_pr, d_sr, d_r2
                ])

    print("Processing results and saving...", flush=True)

    # --- 1. 保存 Subsampled 数据 ---
    full_sub_metas = np.concatenate(sub_metas, axis=0)
    full_sub_preds = np.concatenate(sub_preds, axis=0)
    full_sub_reals = np.concatenate(sub_reals, axis=0)

    df_meta_sub = pd.DataFrame(full_sub_metas, columns=['cell', 'drug', 'dose'])
    df_preds_sub = pd.DataFrame(full_sub_preds, columns=gene_list)
    df_reals_sub = pd.DataFrame(full_sub_reals, columns=gene_list)

    pd.concat([df_meta_sub, df_preds_sub], axis=1).to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    pd.concat([df_meta_sub, df_reals_sub], axis=1).to_csv(os.path.join(save_dir, 'real_outputs.csv'), index=False)

    # --- 2. 保存 Mean 数据 ---
    mean_metas_np = np.array(mean_metas)
    mean_preds_np = np.array(mean_preds)
    mean_reals_np = np.array(mean_reals)
    mean_ctrls_np = np.array(mean_controls)

    df_meta_mean = pd.DataFrame(mean_metas_np, columns=['cell', 'drug', 'dose'])
    df_preds_mean = pd.DataFrame(mean_preds_np, columns=gene_list)
    df_reals_mean = pd.DataFrame(mean_reals_np, columns=gene_list)
    df_ctrls_mean = pd.DataFrame(mean_ctrls_np, columns=gene_list)

    pd.concat([df_meta_mean, df_preds_mean], axis=1).to_csv(os.path.join(save_dir, 'predictions_mean.csv'), index=False)
    pd.concat([df_meta_mean, df_reals_mean], axis=1).to_csv(os.path.join(save_dir, 'real_outputs_mean.csv'),
                                                            index=False)
    pd.concat([df_meta_mean, df_ctrls_mean], axis=1).to_csv(os.path.join(save_dir, 'control_mean.csv'), index=False)

    # --- 3. 保存 Metrics 数据 (如图所示格式) ---
    metrics_columns = ['cell', 'drug', 'dose',
                       'pearson_r', 'spearman_r', 'r2',
                       'delta_pearson_r', 'delta_spearman_r', 'delta_r2']
    df_metrics = pd.DataFrame(metrics_list, columns=metrics_columns)
    df_metrics.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)

    print(f"All done! Results saved to {save_dir}", flush=True)


def test_Multiperturbation_model(model, test_loader, seed, gene_list, device=None, dataset_name='sciplex4',
                                 max_size=512):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    predictions_list = []
    real_outputs_list = []
    loop = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, ((cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose),
            cell_embeds, drug_embed, real_outputs_treated, unperturb_gexp) in loop:
        cell_embeds = cell_embeds.to(device)
        drug_embed = drug_embed.to(device)
        real_outputs_treated = real_outputs_treated.to(device)
        # unperturb_gexp = unperturb_gexp.to(device)
        input_size = cell_embeds.size(0)
        # repeated_drug_embed = drug_embed.repeat(input_size, 1)
        batch_size = min(input_size, max_size)
        # indices = torch.randperm(input_size)[:batch_size]
        indices = torch.arange(batch_size)
        cell_embeds = cell_embeds[indices]
        # unperturb_gexp = unperturb_gexp[indices]
        # The drug representation was repeated batch_size times
        repeated_drug_embed = drug_embed.repeat(batch_size, 1)

        if real_outputs_treated.size(0) >= batch_size:
            cell_indices = torch.randperm(real_outputs_treated.size(0))[:batch_size]
        else:
            cell_indices = torch.randint(0, real_outputs_treated.size(0), (batch_size,))
        real_outputs_treated = real_outputs_treated[cell_indices]

        # Forward propagation
        outputs = model("pertrub_forward", cell_embeds, repeated_drug_embed)
        real_outputs_batch = real_outputs_treated.detach().cpu().numpy()

        for j in range(batch_size):
            predictions_list.append([cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose] + outputs[j].tolist())
            real_outputs_list.append(
                [cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose] + real_outputs_batch[j].tolist())

    columns = ['cell_type', 'drug1_name', 'drug1_dose', 'drug2_name', 'drug2_dose'] + gene_list
    predictions_df = pd.DataFrame(predictions_list, columns=columns)
    real_outputs_df = pd.DataFrame(real_outputs_list, columns=columns)
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    real_outputs_df.to_csv(os.path.join(save_dir, 'real_outputs.csv'), index=False)


def finetune(model_path, cell_embed, drug_embed, perturbed, control, device, num_epochs, train_set_rate, seed,
             save_dir):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 数据准备
    num_samples = cell_embed.shape[0]
    all_indices = np.arange(num_samples)

    cell_embed_train, cell_embed_val, drug_embed_train, drug_embed_val, \
        perturbed_train, perturbed_val, control_train, control_val, \
        train_indices, val_indices = \
        train_test_split(
            cell_embed, drug_embed, perturbed, control, all_indices,
            test_size=(1 - train_set_rate),
            random_state=seed
        )

    # Tensor 转换
    # 封装函数以减少重复代码
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    train_dataset = TensorDataset(to_tensor(cell_embed_train), to_tensor(drug_embed_train),
                                  to_tensor(perturbed_train), to_tensor(control_train))
    val_dataset = TensorDataset(to_tensor(cell_embed_val), to_tensor(drug_embed_val),
                                to_tensor(perturbed_val), to_tensor(control_val))

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    # Load Model
    model = load_UniCureFT(path=model_path)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_mse = nn.MSELoss()  # 基础 MSE Loss

    # --- Metrics 记录列表 ---
    history = {
        'train_total_loss': [],
        'train_mse_loss': [],
        'train_cosine_loss': [],
        'train_abs_pcc': [],
        'train_delta_pcc': [],
        'val_total_loss': [],
        'val_mse_loss': [],
        'val_cosine_loss': [],
        'val_abs_pcc': [],
        'val_delta_pcc': []
    }

    # 保存目录
    save_dir = os.path.join(save_dir, str(seed))
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(train_indices, columns=['index']).to_csv(os.path.join(save_dir, 'train_indices.csv'), index=False)
    pd.DataFrame(val_indices, columns=['index']).to_csv(os.path.join(save_dir, 'val_indices.csv'), index=False)
    print(f"Split indices saved to {save_dir}")

    # Early stopping
    patience = 20
    best_val_score = -float('inf')  # 使用 Delta PCC 作为早停指标可能更好，这里综合考虑
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # ================= TRAIN =================
        model.train()
        epoch_meters = {k: [] for k in ['total_loss', 'mse_loss', 'cosine_loss', 'abs_pcc', 'delta_pcc']}

        for cell_inputs, drug_inputs, targets, control in train_loader:
            cell_inputs, drug_inputs, targets, control = \
                cell_inputs.to(device), drug_inputs.to(device), targets.to(device), control.to(device)

            optimizer.zero_grad()
            outputs = model(cell_inputs, drug_inputs)

            # 1. 计算 MSE Loss
            loss_mse = criterion_mse(outputs, targets)

            # 2. 计算 Cosine Loss (针对 Delta)
            # Delta = 扰动后 - 对照组
            delta_pred = outputs - control
            delta_true = targets - control

            # cosine_similarity 返回形状为 (batch_size,)，值域 [-1, 1]
            # dim=1 表示沿着特征维度计算
            cos_sim = F.cosine_similarity(delta_pred, delta_true, dim=1)
            loss_cosine = 1.0 - cos_sim.mean()

            # 3. 总 Loss 回传
            loss = loss_mse + loss_cosine

            loss.backward()
            optimizer.step()

            # --- 计算指标 (Numpy) ---
            nb_sample = outputs.detach().cpu().numpy()
            y_true = targets.detach().cpu().numpy()
            ctrl_np = control.detach().cpu().numpy()

            delta_sample_np = nb_sample - ctrl_np
            delta_true_np = y_true - ctrl_np

            # 批量计算 Pearson
            batch_abs_pcc = []
            batch_delta_pcc = []

            for i in range(nb_sample.shape[0]):
                # Absolute PCC
                if np.std(nb_sample[i]) > 1e-9 and np.std(y_true[i]) > 1e-9:
                    batch_abs_pcc.append(pearsonr(y_true[i], nb_sample[i])[0])
                else:
                    batch_abs_pcc.append(0.0)

                # Delta PCC (Direction correctness)
                if np.std(delta_sample_np[i]) > 1e-9 and np.std(delta_true_np[i]) > 1e-9:
                    batch_delta_pcc.append(pearsonr(delta_true_np[i], delta_sample_np[i])[0])
                else:
                    batch_delta_pcc.append(0.0)

            # 记录本 batch 数据
            epoch_meters['total_loss'].append(loss.item())
            epoch_meters['mse_loss'].append(loss_mse.item())
            epoch_meters['cosine_loss'].append(loss_cosine.item())
            epoch_meters['abs_pcc'].append(np.mean(batch_abs_pcc))
            epoch_meters['delta_pcc'].append(np.mean(batch_delta_pcc))

        # 汇总 Epoch 训练数据
        for k in ['total_loss', 'mse_loss', 'cosine_loss', 'abs_pcc', 'delta_pcc']:
            history[f'train_{k}'].append(np.mean(epoch_meters[k]))

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train | "
              f"TLoss: {history['train_total_loss'][-1]:.4f} "
              f"(MSE: {history['train_mse_loss'][-1]:.4f}, Cos: {history['train_cosine_loss'][-1]:.4f}) | "
              f"PCC: {history['train_abs_pcc'][-1]:.4f}, D-PCC: {history['train_delta_pcc'][-1]:.4f}", flush=True)

        # ================= VAL =================
        model.eval()
        epoch_meters_val = {k: [] for k in ['total_loss', 'mse_loss', 'cosine_loss', 'abs_pcc', 'delta_pcc']}

        with torch.no_grad():
            for cell_inputs, drug_inputs, targets, control in val_loader:
                cell_inputs, drug_inputs, targets, control = \
                    cell_inputs.to(device), drug_inputs.to(device), targets.to(device), control.to(device)

                outputs = model(cell_inputs, drug_inputs)

                # Loss 计算
                loss_mse = criterion_mse(outputs, targets)

                delta_pred = outputs - control
                delta_true = targets - control
                cos_sim = F.cosine_similarity(delta_pred, delta_true, dim=1)
                loss_cosine = 1.0 - cos_sim.mean()

                loss = loss_mse + loss_cosine

                # Metrics 计算
                nb_sample = outputs.cpu().numpy()
                y_true = targets.cpu().numpy()
                ctrl_np = control.cpu().numpy()
                delta_sample_np = nb_sample - ctrl_np
                delta_true_np = y_true - ctrl_np

                batch_abs_pcc = []
                batch_delta_pcc = []
                for i in range(nb_sample.shape[0]):
                    if np.std(nb_sample[i]) > 1e-9 and np.std(y_true[i]) > 1e-9:
                        batch_abs_pcc.append(pearsonr(y_true[i], nb_sample[i])[0])
                    else:
                        batch_abs_pcc.append(0.0)

                    if np.std(delta_sample_np[i]) > 1e-9 and np.std(delta_true_np[i]) > 1e-9:
                        batch_delta_pcc.append(pearsonr(delta_true_np[i], delta_sample_np[i])[0])
                    else:
                        batch_delta_pcc.append(0.0)

                epoch_meters_val['total_loss'].append(loss.item())
                epoch_meters_val['mse_loss'].append(loss_mse.item())
                epoch_meters_val['cosine_loss'].append(loss_cosine.item())
                epoch_meters_val['abs_pcc'].append(np.mean(batch_abs_pcc))
                epoch_meters_val['delta_pcc'].append(np.mean(batch_delta_pcc))

        # 汇总 Epoch 验证数据
        for k in ['total_loss', 'mse_loss', 'cosine_loss', 'abs_pcc', 'delta_pcc']:
            history[f'val_{k}'].append(np.mean(epoch_meters_val[k]))

        curr_val_delta_pcc = history['val_delta_pcc'][-1]

        print(f"Epoch [{epoch + 1}/{num_epochs}] Val   | "
              f"TLoss: {history['val_total_loss'][-1]:.4f} "
              f"(MSE: {history['val_mse_loss'][-1]:.4f}, Cos: {history['val_cosine_loss'][-1]:.4f}) | "
              f"PCC: {history['val_abs_pcc'][-1]:.4f}, D-PCC: {curr_val_delta_pcc:.4f}", flush=True)

        # Early Stopping: 优先优化 Delta PCC (也可以改成 total_loss)
        if curr_val_delta_pcc > best_val_score:
            best_val_score = curr_val_delta_pcc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # 保存训练记录
    result_df = pd.DataFrame(history)
    result_df.to_csv(os.path.join(save_dir, 'finetune_training_results.csv'), index=False)

    return best_val_score
