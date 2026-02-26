import torch
import pandas as pd
import numpy as np
from preprocessing import generate_esm2_emb
from utils import uce_emb

# 1. Configuration
seed = 11
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
control_file = './data/test/Test.parquet'  # Should contain ABY001_NCIH596 control data
model_path = f'./result/{seed}/lincs2020/Unicure_best_model.pth'

# 2. Step 1: Generate Cell Embeddings (UCE-LoRA)
# Convert raw control gene expression to biological meaningful embeddings
print("Step 1: Generating Cell Embeddings...")
generate_esm2_emb(
    control_path=control_file,
    gene_columns_start=1,
    save_dir='./data/test/Test_esm2_emb.parquet',
    dataset_name="clinical"
)

uce_emb(
    esm2_emb_df_path='./data/test/Test_esm2_emb.parquet',
    esm2_control_df_path=control_file,
    model_path=model_path,
    uce_emb_df_path='./data/test/Test_uce_lora_emb.parquet',
    index_name='sample'
)

# 3. Step 2: Prepare Input Tensors
print("Step 2: Preparing Input Tensors...")
# Load generated cell embeddings (multiple rows for single cells)
cell_emb_df = pd.read_parquet('./data/test/Test_uce_lora_emb.parquet')
# Extract all cells belonging to this sample
cell_values = cell_emb_df.loc[['ABY001_NCIH596']].values
num_cells = cell_values.shape[0] # This will be 15 in your case
cell_tensor = torch.tensor(cell_values, dtype=torch.float32).to(device)

# Load pre-calculated drug embeddings (Uni-Mol)
drug_emb_df = pd.read_parquet('./data/lincs2020/lincs2020_unimol_emb.parquet')
drug_raw_tensor = torch.tensor(drug_emb_df.loc[['vorinostat']].values, dtype=torch.float32).to(device)

# Process Dose (Log10 scale + Padding to 528)
dose_val = np.log10(2.5 + 1)
dose_tensor = torch.full((1, 1), dose_val, dtype=torch.float32).to(device)
drug_combined = torch.cat([drug_raw_tensor, dose_tensor], dim=1)
pad_len = 528 - drug_combined.shape[1]
if pad_len > 0:
    pad_tensor = torch.full((1, pad_len), dose_val, dtype=torch.float32).to(device)
    drug_combined = torch.cat([drug_combined, pad_tensor], dim=1)

# --- KEY FIX: Expand drug tensor to match the number of cells ---
# drug_combined is [1, 528] -> drug_tensor becomes [15, 528]
drug_tensor = drug_combined.repeat(num_cells, 1)


# 4. Step 3: Load Model and Predict
print("Step 3: Running Inference...")
# Assuming you have the model class imported (e.g., from model import UniCure)
# model = UniCure(...)
# For this example, we assume use of a helper to load
from utils import load_UniCure_pretrained_model

model = load_UniCure_pretrained_model(path=model_path)
model.to(device).eval()

with torch.no_grad():
    # Predict the perturbed gene expression
    prediction = model("pertrub_forward", cell_tensor, drug_tensor)


# 5. Step 4: Save Results
print("Step 4: Saving Predictions...")
gene_list = pd.read_parquet(control_file).columns[1:].tolist()
pred_df = pd.DataFrame(prediction.cpu().numpy(), columns=gene_list)
pred_df.insert(0, 'sample', 'ABY001_NCIH596_vorinostat_2.5')
pred_df.to_csv('./data/test/prediction_result.csv', index=False)

print("Done! Prediction saved to ./prediction_result.csv")


