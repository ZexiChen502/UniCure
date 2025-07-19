import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from loss import WeightedMSELoss
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from accelerate.utils import gather_object
from tqdm.auto import tqdm
import torch.distributed as dist
from utils import set_seed, load_UniCureFT
from torch import nn
from scipy.stats import spearmanr, pearsonr


def train_stage1(model, trainable_parameters, uce_emb: pd.DataFrame, control_df: pd.DataFrame,
                     gene_columns_start: int, seed: int, dataset_name: str, lr_rate: float, batch_size: int,
                     num_epochs: int, early_stopping_patience: int):
    # Dataloader
    set_seed(seed)
    input = uce_emb.values
    output = control_df.iloc[:, gene_columns_start:].values
    scaler = StandardScaler()
    output = scaler.fit_transform(output.T).T
    input_train, input_val, target_train, target_val = train_test_split(
        input, output, test_size=0.2, random_state=seed
    )
    input_train_tensor = torch.tensor(input_train, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train, dtype=torch.float32)
    input_val_tensor = torch.tensor(input_val, dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val, dtype=torch.float32)

    train_dataset = TensorDataset(input_train_tensor, target_train_tensor)
    val_dataset = TensorDataset(input_val_tensor, target_val_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1)

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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_r2 = 0.0
        sample_size = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (input_batch, targets_batch) in loop:
            input_batch = input_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            # outputs = model("baseline_without_uce", input_batch)
            outputs = model(input_batch)
            # outputs = model(input_batch)
            loss = criterion(outputs, targets_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_batch.size(0)
            sample_size += input_batch.size(0)
            outputs_cpu = outputs.detach().float().cpu().numpy()
            targets_cpu = targets_batch.detach().float().cpu().numpy()

            r2 = 0
            for i in range(outputs_cpu.shape[0]):
                r2 += r2_score(targets_cpu[i], outputs_cpu[i])
            avg_batch_r2 = r2 / outputs_cpu.shape[0]
            running_r2 += avg_batch_r2 * input_batch.size(0)
            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}] [{batch_idx + 1}/{len(train_dataloader)}]')
            loop.set_postfix(train_loss=loss.item(), r2_score_mean=avg_batch_r2)

        scheduler.step()

        avg_train_loss = running_loss / sample_size
        avg_train_r2 = running_r2 / sample_size
        train_loss_list.append(avg_train_loss)
        train_r2_list.append(avg_train_r2)


        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training R2: {avg_train_r2:.4f}',
            flush=True)


        # Verification phase
        model.eval()
        val_loss = 0.0
        val_r2 = 0.0
        sample_size = 0
        with torch.no_grad():
            loop_v = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            for batch_idx, (input_batch, targets_batch) in loop_v:
                input_batch = input_batch.to(device)
                targets_batch = targets_batch.to(device)

                # outputs = model("baseline_without_uce", input_batch)
                outputs = model(input_batch)

                loss = criterion(outputs, targets_batch)

                val_loss += loss.item() * input_batch.size(0)
                sample_size += input_batch.size(0)
                outputs_cpu = outputs.detach().float().cpu().numpy()
                targets_cpu = targets_batch.detach().float().cpu().numpy()

                r2 = 0
                for i in range(outputs_cpu.shape[0]):
                    r2 += r2_score(targets_cpu[i], outputs_cpu[i])
                avg_batch_r2 = r2 / outputs_cpu.shape[0]
                val_r2 += avg_batch_r2 * input_batch.size(0)
                loop_v.set_description(
                    f'Epoch [{epoch + 1}/{num_epochs}], [{batch_idx + 1}/{len(val_dataloader)}]')
                loop_v.set_postfix(val_loss=loss.item(), r2_score_mean=avg_batch_r2)

        avg_val_loss = val_loss / sample_size
        avg_val_r2 = val_r2 / sample_size
        val_loss_list.append(avg_val_loss)
        val_r2_list.append(avg_val_r2)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val R2: {avg_val_r2:.4f}',
              flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_model = model
            torch.save(best_model.state_dict(), os.path.join(save_dir, 'best_stage_1_model.pth'))
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}, best epoch: {best_epoch}", flush=True)
            break

    result_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'train_r2': train_r2_list,
        'val_loss': val_loss_list,
        'val_r2': val_r2_list
    })

    # checkpoint = {
    #     'epoch': best_epoch,
    #     'model_state_dict': unwrapped_model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'scheduler_state_dict': scheduler.state_dict(),
    #     'val_loss': best_val_loss,
    # }
    result_df.to_csv(os.path.join(save_dir, 'best_stage_1_model_training_results.csv'), index=False)
    print("Training complete.")


def train_geo_stage2(model, trainable_parameters, uce_emb: pd.DataFrame, unimol_emb: pd.DataFrame,
                     perturb_df: pd.DataFrame,
                     gene_columns_start: int, seed: int, dataset_name: str, lr_rate: float, batch_size: int,
                     num_epochs: int, early_stopping_patience: int, accelerator=None):
    # Dataloader
    set_seed(seed)
    input_uce = uce_emb.values
    input_unimol = unimol_emb.values
    output = perturb_df.iloc[:, gene_columns_start:].values
    input_uce_train, input_uce_val, input_unimol_train, input_unimol_val, target_train, target_val = train_test_split(
        input_uce, input_unimol, output, test_size=0.2, random_state=seed
    )
    input_uce_train_tensor = torch.tensor(input_uce_train, dtype=torch.float32)
    input_unimol_train_tensor = torch.tensor(input_unimol_train, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train, dtype=torch.float32)
    input_uce_val_tensor = torch.tensor(input_uce_val, dtype=torch.float32)
    input_unimol_val_tensor = torch.tensor(input_unimol_val, dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val, dtype=torch.float32)

    train_dataset = TensorDataset(input_uce_train_tensor, input_unimol_train_tensor, target_train_tensor)
    val_dataset = TensorDataset(input_uce_val_tensor, input_unimol_val_tensor, target_val_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    device = accelerator.device
    model.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 100, 500], gamma=0.1)
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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_r2 = 0.0
        sample_size = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                    disable=not accelerator.is_local_main_process)
        for batch_idx, (input_uce_batch, input_unimol_batch, targets_batch) in loop:
            input_uce_batch = input_uce_batch.to(device)
            input_unimol_batch = input_unimol_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            outputs = model("perturb_forward", input_uce_batch, input_unimol_batch)
            loss = criterion(outputs, targets_batch)
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()

            running_loss += loss.item() * input_uce_batch.size(0)
            sample_size += input_uce_batch.size(0)
            outputs_cpu = outputs.detach().float().cpu().numpy()
            targets_cpu = targets_batch.detach().float().cpu().numpy()

            r2 = 0
            for i in range(outputs_cpu.shape[0]):
                r2 += r2_score(targets_cpu[i], outputs_cpu[i])
            avg_batch_r2 = r2 / outputs_cpu.shape[0]
            running_r2 += avg_batch_r2 * input_uce_batch.size(0)
            loop.set_description(f'GEO2:Epoch [{epoch + 1}/{num_epochs}] [{batch_idx + 1}/{len(train_dataloader)}]')
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
                f'GEO2:Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training R2: {avg_train_r2:.4f}',
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
            for batch_idx, (input_uce_batch, input_unimol_batch, targets_batch) in loop_v:
                input_uce_batch = input_uce_batch.to(device)
                input_unimol_batch = input_unimol_batch.to(device)
                targets_batch = targets_batch.to(device)

                outputs = model("perturb_forward", input_uce_batch, input_unimol_batch)

                loss = criterion(outputs, targets_batch)

                val_loss += loss.item() * input_uce_batch.size(0)
                sample_size += input_uce_batch.size(0)

                outputs_cpu = outputs.detach().float().cpu().numpy()
                targets_cpu = targets_batch.detach().float().cpu().numpy()

                r2 = 0
                for i in range(outputs_cpu.shape[0]):
                    r2 += r2_score(targets_cpu[i], outputs_cpu[i])
                avg_batch_r2 = r2 / outputs_cpu.shape[0]
                val_r2 += avg_batch_r2 * input_uce_batch.size(0)
                loop_v.set_description(
                    f'GEO2:Epoch [{epoch + 1}/{num_epochs}], [{batch_idx + 1}/{len(val_dataloader)}]')
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
            print(f'GEO2:Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val R2: {avg_val_r2:.4f}',
                  flush=True)
        accelerator.wait_for_everyone()

        # early_stop
        if accelerator.is_main_process:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                best_model = model
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
        # best_model.uce_lora = best_model.uce_lora.merge_lora(inplace=False)
        unwrapped_model = accelerator.unwrap_model(best_model)
        torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, 'best_geo_stage_2_model.pth'))
        result_df.to_csv(os.path.join(save_dir, 'best_geo_stage_2_model_training_results.csv'), index=False)
        print("Training complete.")

    accelerator.wait_for_everyone()


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
                torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, 'best_original_state_model.pth'))
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


def train_perturbation_model(model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5, num_epochs=10,
                             device=None, seed=48, early_stopping_patience=20, dataset_name='lincs2020', lambda_val=0.001,
                             max_batch_size=64):

    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)
    model.to(device)
    mmd_loss_fn.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr_rate)
    train_loss_list = []
    train_r2_list = []
    val_loss_list = []
    val_r2_list = []
    best_val_loss = float('inf')
    patience_counter = 0

    save_dir = os.path.join(r'./result', str(seed), dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    best_epoch = 0
    for epoch in range(num_epochs):
        #
        model.train()
        train_loss = 0
        train_r2 = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (_, cell_embeds, drug_embed, real_outputs_treated, unperturb_gexp) in loop:
            # total_start = time.time()

            # t0 = time.time()
            cell_embeds = cell_embeds.to(device)
            drug_embed = drug_embed.to(device)
            real_outputs_treated = real_outputs_treated.to(device)
            unperturb_gexp = unperturb_gexp.to(device)
            # t1 = time.time()

            input_size = cell_embeds.size(0)
            # repeated_drug_embed = drug_embed.repeat(input_size, 1)

            batch_size = min(input_size, max_batch_size)
            indices = torch.randperm(input_size)[:batch_size]
            cell_embeds = cell_embeds[indices]
            unperturb_gexp = unperturb_gexp[indices]
            # The drug representation was repeated batch_size times
            repeated_drug_embed = drug_embed.repeat(batch_size, 1)

            if real_outputs_treated.size(0) >= batch_size:
                cell_indices = torch.randperm(real_outputs_treated.size(0))[:batch_size]
            else:
                cell_indices = torch.randint(0, real_outputs_treated.size(0), (batch_size,))
            real_outputs_treated = real_outputs_treated[cell_indices]
            # t2 = time.time()

            # Forward propagation
            outputs = model("pertrub_forward", cell_embeds, repeated_drug_embed)
            # inputs = torch.cat([cell_embeds, repeated_drug_embed], dim=1)
            # outputs = model(inputs)
            # t3 = time.time()

            # Calculate the MMD loss
            loss = mmd_loss_fn(outputs, real_outputs_treated)

            # Calculate the Euclidean distance constraint
            euclidean_distance = torch.norm(outputs - unperturb_gexp, p=2, dim=1).mean()

            # Combine the losses with a weighting factor (lambda)
            lambda_val = lambda_val  # Adjust this value as needed
            total_loss = loss + lambda_val * euclidean_distance
            # t4 = time.time()

            # loss = get_Wasserstein(outputs, real_outputs_treated_batch)
            r2 = r2_score(real_outputs_treated.mean(axis=0).detach().cpu().numpy(),
                          outputs.mean(axis=0).detach().cpu().numpy())
            # t5 = time.time()

            # Cumulative gradient
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Cumulative index
            train_loss += loss.item()
            train_r2 += r2
            torch.cuda.empty_cache()
            # t6 = time.time()

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}] [{i}/{len(train_loader)}]')
            loop.set_postfix(Loss=loss.item(), r2_score_mean=r2)
            # total_end = time.time()
            #
            # print(f"Step [{i}]:")
            # print(f"  数据转移耗时 (to(device))       : {t1 - t0:.6f} 秒")
            # print(f"  Batch 构造耗时               : {t2 - t1:.6f} 秒")
            # print(f"  模型前向传播耗时            : {t3 - t2:.6f} 秒")
            # print(f"  Loss 计算耗时               : {t4 - t3:.6f} 秒")
            # print(f"  r2_score 计算耗时           : {t5 - t4:.6f} 秒")
            # print(f"  反向传播 + 优化器 step 耗时 : {t6 - t5:.6f} 秒")
            # print(f"  总计耗时                    : {total_end - total_start:.6f} 秒\n")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_r2 = train_r2 / len(train_loader)
        train_loss_list.append(avg_train_loss)
        train_r2_list.append(avg_train_r2)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training R2: {avg_train_r2:.4f}',
              flush=True)

        # Verification phase
        model.eval()
        val_loss = 0
        val_r2 = 0
        with torch.no_grad():
            loop_v = tqdm(enumerate(val_loader), total=len(val_loader))
            for i, (_, cell_embeds_val, drug_embed_val, real_outputs_treated_val, _) in loop_v:
                cell_embeds_val = cell_embeds_val.to(device)
                drug_embed_val = drug_embed_val.to(device)
                real_outputs_treated_val = real_outputs_treated_val.to(device)
                input_size = cell_embeds_val.size(0)
                # repeated_drug_embed_val = drug_embed_val.repeat(input_size, 1)

                batch_size = min(input_size, max_batch_size)
                indices = torch.randperm(input_size)[:batch_size]
                cell_embeds_val = cell_embeds_val[indices]
                # The drug representation was repeated batch_size times
                repeated_drug_embed_val = drug_embed_val.repeat(batch_size, 1)

                if real_outputs_treated_val.size(0) >= batch_size:
                    cell_indices_val = torch.randperm(real_outputs_treated_val.size(0))[:batch_size]
                else:
                    cell_indices_val = torch.randint(0, real_outputs_treated_val.size(0), (batch_size,))
                real_outputs_treated_val = real_outputs_treated_val[cell_indices_val]

                outputs = model("pertrub_forward", cell_embeds_val, repeated_drug_embed_val)
                # inputs = torch.cat([cell_embeds_val, repeated_drug_embed_val], dim=1)
                # outputs = model(inputs)

                loss = mmd_loss_fn(outputs, real_outputs_treated_val)
                # loss = get_Wasserstein(outputs, real_outputs_treated_val_batch)
                r2 = r2_score(real_outputs_treated_val.mean(axis=0).detach().cpu().numpy(),
                              outputs.mean(axis=0).detach().cpu().numpy())
                val_loss += loss.item()
                val_r2 += r2
                loop_v.set_description(f'Epoch [{epoch + 1}/{num_epochs}] [{i}/{len(val_loader)}]')
                loop_v.set_postfix(Val_Loss=loss.item(), Val_r2_mean=r2)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation R2: {avg_val_r2:.4f}',
              flush=True)
        val_loss_list.append(avg_val_loss)
        val_r2_list.append(avg_val_r2)
        #

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"early_stopping, best epoch:{best_epoch}", flush=True)
            break

    result_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'train_r2': train_r2_list,
        'val_loss': val_loss_list,
        'val_r2': val_r2_list
    })

    result_df.to_csv(os.path.join(save_dir, 'training_results.csv'), index=False)


def train_perturbation_model_acc(model, mmd_loss_fn, trainable_parameters, train_loader, val_loader, lr_rate=1e-5, num_epochs=10,
                                seed=48, early_stopping_patience=20, accelerator=None):

    set_seed(seed)
    device = accelerator.device
    model.to(device)
    mmd_loss_fn.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr_rate)

    model, train_loader, val_loader, optimizer = accelerator.prepare(
        model, train_loader, val_loader, optimizer)

    train_loss_list = []
    train_r2_list = []
    val_loss_list = []
    val_r2_list = []
    best_val_loss = float('inf')
    patience_counter = 0

    save_dir = os.path.join(r'./result', str(seed), 'lincs2020')
    os.makedirs(save_dir, exist_ok=True)
    best_epoch = 0

    for epoch in range(num_epochs):
        #
        model.train()
        train_loss = 0
        train_r2 = 0
        train_size = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader),
                    disable=not accelerator.is_local_main_process)

        for i, (_, cell_embeds, drug_embed, real_outputs_treated, unperturb_gexp) in loop:
            cell_embeds = cell_embeds.to(device)
            drug_embed = drug_embed.to(device)
            real_outputs_treated = real_outputs_treated.to(device)
            unperturb_gexp = unperturb_gexp.to(device)
            input_size = cell_embeds.size(0)

            # The drug representation was repeated batch_size times
            repeated_drug_embed = drug_embed.repeat(input_size, 1)

            # Forward propagation
            outputs = model("pertrub_forward", cell_embeds, repeated_drug_embed)

            # Calculate the MMD loss
            loss = mmd_loss_fn(outputs, real_outputs_treated)

            # Calculate the Euclidean distance constraint
            euclidean_distance = torch.norm(outputs - unperturb_gexp, p=2, dim=1).mean()

            # Combine the losses with a weighting factor (lambda)
            lambda_val = 0.001  # Adjust this value as needed
            total_loss = loss + lambda_val * euclidean_distance

            # loss = get_Wasserstein(outputs, real_outputs_treated_batch)
            r2 = r2_score(real_outputs_treated.mean(axis=0).detach().cpu().numpy(),
                          outputs.mean(axis=0).detach().cpu().numpy())

            # Cumulative gradient
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

            # Cumulative index
            train_loss += loss.item()
            train_r2 += r2
            train_size += 1
            torch.cuda.empty_cache()

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}] [{i}/{len(train_loader)}]')
            loop.set_postfix(Loss=loss.item(), r2_score_mean=r2)

        gathered_running_loss = accelerator.gather(torch.tensor(train_loss).to(device))
        gathered_running_r2 = accelerator.gather(torch.tensor(train_r2).to(device))
        gathered_sample_size = accelerator.gather(torch.tensor(train_size).to(device))
        total_train_loss = torch.sum(gathered_running_loss).item()
        total_train_r2 = torch.sum(gathered_running_r2).item()
        total_sample_size = torch.sum(gathered_sample_size).item()

        avg_train_loss = total_train_loss / total_sample_size
        avg_train_r2 = total_train_r2 / total_sample_size
        train_loss_list.append(avg_train_loss)
        train_r2_list.append(avg_train_r2)
        if accelerator.is_local_main_process:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training R2: {avg_train_r2:.4f}',
                flush=True)
        accelerator.wait_for_everyone()

        # avg_train_loss = train_loss / len(train_loader)
        # avg_train_r2 = train_r2 / len(train_loader)
        # train_loss_list.append(avg_train_loss)
        # train_r2_list.append(avg_train_r2)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training R2: {avg_train_r2:.4f}',
        #       flush=True)

        # Verification phase
        model.eval()
        val_loss = 0
        val_r2 = 0
        val_size = 0
        with torch.no_grad():
            loop_v = tqdm(enumerate(val_loader), total=len(val_loader), disable=not accelerator.is_local_main_process)
            for i, (_, cell_embeds, drug_embed, real_outputs_treated) in loop_v:
                cell_embeds = cell_embeds.to(device)
                drug_embed = drug_embed.to(device)
                real_outputs_treated = real_outputs_treated.to(device)
                input_size = cell_embeds.size(0)
                repeated_drug_embed = drug_embed.repeat(input_size, 1)
                outputs = model("pertrub_forward", cell_embeds, repeated_drug_embed)
                loss = mmd_loss_fn(outputs, real_outputs_treated)
                # loss = get_Wasserstein(outputs, real_outputs_treated_batch)
                r2 = r2_score(real_outputs_treated.mean(axis=0).detach().cpu().numpy(),
                              outputs.mean(axis=0).detach().cpu().numpy())
                val_loss += loss.item()
                val_r2 += r2
                val_size += 1
                loop_v.set_description(f'Epoch [{epoch + 1}/{num_epochs}] [{i}/{len(val_loader)}]')
                loop_v.set_postfix(Val_Loss=loss.item(), Val_r2_mean=r2)

        gathered_val_loss = accelerator.gather(torch.tensor(val_loss).to(device))
        gathered_val_r2 = accelerator.gather(torch.tensor(val_r2).to(device))
        gathered_sample_size = accelerator.gather(torch.tensor(val_size).to(device))
        total_val_loss = torch.sum(gathered_val_loss).item()
        total_val_r2 = torch.sum(gathered_val_r2).item()
        total_sample_size = torch.sum(gathered_sample_size).item()

        avg_val_loss = total_val_loss / total_sample_size
        avg_val_r2 = total_val_r2 / total_sample_size
        val_loss_list.append(avg_val_loss)
        val_r2_list.append(avg_val_r2)

        if accelerator.is_local_main_process:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val R2: {avg_val_r2:.4f}',
                  flush=True)
        accelerator.wait_for_everyone()
        #
        if accelerator.is_main_process:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                best_model = model
                unwrapped_model = accelerator.unwrap_model(best_model)
                # unwrapped_model.uce_lora = unwrapped_model.uce_lora.merge_lora(inplace=False)
                torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}, best epoch: {best_epoch}", flush=True)
                accelerator.set_trigger()

        accelerator.wait_for_everyone()  # 所有进程同步
        if accelerator.check_trigger():
            break
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        #     best_epoch = epoch + 1
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        # else:
        #     patience_counter += 1
        #
        # if patience_counter >= early_stopping_patience:
        #     print(f"early_stopping, best epoch:{best_epoch}", flush=True)
        #     break

    result_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'train_r2': train_r2_list,
        'val_loss': val_loss_list,
        'val_r2': val_r2_list
    })

    if accelerator.is_local_main_process:
        result_df.to_csv(os.path.join(save_dir, 'lincs_training_results.csv'), index=False)
    accelerator.wait_for_everyone()


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


def test_Multiperturbation_model(model, test_loader, seed, gene_list, device=None, dataset_name='sciplex4', max_size=512):

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
            real_outputs_list.append([cell_type, drug1_name, drug1_dose, drug2_name, drug2_dose] + real_outputs_batch[j].tolist())

    columns = ['cell_type', 'drug1_name', 'drug1_dose', 'drug2_name', 'drug2_dose'] + gene_list
    predictions_df = pd.DataFrame(predictions_list, columns=columns)
    real_outputs_df = pd.DataFrame(real_outputs_list, columns=columns)
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    real_outputs_df.to_csv(os.path.join(save_dir, 'real_outputs.csv'), index=False)


def finetune(model_path, cell_embed, drug_embed, perturbed, control, device, num_epochs, train_set_rate, seed, save_dir):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    cell_embed_train, cell_embed_val, drug_embed_train, drug_embed_val, \
        perturbed_train, perturbed_val, control_train, control_val = \
        train_test_split(
            cell_embed, drug_embed, perturbed, control,
            test_size=(1 - train_set_rate),
            random_state=seed
        )

    cell_embed_train_tensor = torch.tensor(cell_embed_train, dtype=torch.float32)
    cell_embed_val_tensor = torch.tensor(cell_embed_val, dtype=torch.float32)
    drug_embed_train_tensor = torch.tensor(drug_embed_train, dtype=torch.float32)
    drug_embed_val_tensor = torch.tensor(drug_embed_val, dtype=torch.float32)
    perturbed_train_tensor = torch.tensor(perturbed_train, dtype=torch.float32)
    perturbed_val_tensor = torch.tensor(perturbed_val, dtype=torch.float32)
    control_train_tensor = torch.tensor(control_train, dtype=torch.float32)
    control_val_tensor = torch.tensor(control_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(cell_embed_train_tensor, drug_embed_train_tensor,
                                  perturbed_train_tensor, control_train_tensor)
    val_dataset = TensorDataset(cell_embed_val_tensor, drug_embed_val_tensor,
                                perturbed_val_tensor, control_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    # model
    model = load_UniCureFT(path=model_path)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()

    set_seed(seed)
    model.to(device)
    train_loss_list = []
    train_metrics_list = []
    val_loss_list = []
    val_metrics_list = []

    save_dir = os.path.join(save_dir, str(seed))
    os.makedirs(save_dir, exist_ok=True)

    # Early stopping parameters
    patience = 20  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    best_val_metrics = -float('inf')  # Assuming higher metrics is better (e.g., accuracy, spearman)

    best_model_state = None
    epochs_no_improve = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_metrics = 0
        for cell_inputs, drug_inputs, targets, control in train_loader:
            cell_inputs, drug_inputs, targets, control = \
                cell_inputs.to(device), drug_inputs.to(device), targets.to(device), control.to(device)
            optimizer.zero_grad()

            # outputs = model(inputs)
            outputs = model(cell_inputs, drug_inputs)

            # loss = criterion(outputs, targets, control)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            nb_sample = outputs.detach().cpu().numpy()
            y_true = targets.detach().cpu().numpy()
            num_elements = nb_sample.shape[0]
            metrics_scores = []
            for i in range(num_elements):
                yp_i = nb_sample[i]
                yt_i = y_true[i]
                metrics, _ = pearsonr(yt_i, yp_i)
                metrics_scores.append(metrics)
            metrics_score_mean = np.mean(metrics_scores)
            train_loss += loss.item()
            train_metrics += metrics_score_mean

        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = train_metrics / len(train_loader)
        train_loss_list.append(avg_train_loss)
        train_metrics_list.append(avg_train_metrics)

        # print(
        #     f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training metrics: {avg_train_metrics:.4f}',
        #     flush=True)

        # val
        model.eval()
        val_loss = 0
        val_metrics = 0
        with torch.no_grad():
            for cell_inputs, drug_inputs, targets, control in val_loader:
                cell_inputs, drug_inputs, targets, control = \
                    cell_inputs.to(device), drug_inputs.to(device), targets.to(device), control.to(device)
                # outputs = model(inputs)
                outputs = model(cell_inputs, drug_inputs)
                # loss = criterion(outputs, targets, control)
                loss = criterion(outputs, targets)
                nb_sample = outputs.detach().cpu().numpy()
                y_true = targets.detach().cpu().numpy()
                num_elements = nb_sample.shape[0]
                metrics_scores = []
                for i in range(num_elements):
                    yp_i = nb_sample[i]
                    yt_i = y_true[i]
                    metrics, _ = pearsonr(yt_i, yp_i)
                    metrics_scores.append(metrics)
                metrics_score_mean = np.mean(metrics_scores)
                val_loss += loss.item()
                val_metrics += metrics_score_mean
        avg_val_loss = val_loss / len(val_loader)
        avg_val_metrics = val_metrics / len(val_loader)
        # print(
        #     f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation metrics: {avg_val_metrics:.4f}',
        #     flush=True)
        val_loss_list.append(avg_val_loss)
        val_metrics_list.append(avg_val_metrics)

        # Early Stopping Check (based on validation loss AND metrics)
        if avg_val_loss < best_val_loss or avg_val_metrics > best_val_metrics :  # Consider both loss and metrics
            best_val_loss = min(best_val_loss, avg_val_loss)  # Update best loss if lower
            best_val_metrics = max(best_val_metrics, avg_val_metrics)
            best_model_state = model.state_dict()  # Save the model's state
            epochs_no_improve = 0  # Reset counter
            best_epoch = epoch + 1
            # torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth')) # Save best model

        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            # print(f'Early stopping triggered after epoch {epoch + 1}.  Best epoch: {best_epoch}')
            break  # Stop training

    result_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'train_metrics': train_metrics_list,
        'val_loss': val_loss_list,
        'val_metrics': val_metrics_list
    })

    result_df.to_csv(os.path.join(save_dir, 'finetune_training_results.csv'), index=False)

    return best_val_metrics


