# UniCure: A Multi-Modal Model for Predicting Personalized Cancer Therapy Response 💊

## Overview 🌟
**UniCure is a multi-modal model integrating omics (UCE) and chemical (Uni-mol) foundation models to predict transcriptomic drug responses across diverse cellular contexts.**

💡 **Key Innovations**  
- **FlexPert**: Sliding-window cross-attention for flexible drug-cell interaction modeling  
- **LoRA-PEFT**: Parameter-efficient tuning preserving pretrained knowledge  
- **MMD loss**: Handles unpaired perturbation data without cell matching    
- **Staged training**: Enhancing computational efficiency and functional modularity

## System Requirements 🛠

### Hardware Requirements
| Use Case              | Minimum Configuration                | Recommended Configuration       |
|-----------------------|--------------------------------------|---------------------------------|
| **Full Reproduction** | 4× NVIDIA GPUs (80GB VRAM each) | 8× A100/H100 80GB |
| **Testing/Inference** | 1× NVIDIA GPU (32GB+ VRAM) | 1× NVIDIA GPU (80GB VRAM) |

### Software Requirements
**OS:** Linux (Ubuntu 22.04 LTS or Rocky Linux 8.6+ recommended)  
**Environment Manager:** Miniconda/Mamba  

#### Installation via Conda:
```
# Base environment (minimal)
conda env create -f environment.yml

# Full environment (with development tools)
conda env create -f environment_full.yml
```

#### Manual Installation Steps (Recommended):
```
1. Install Python 3.10
conda create -n unicure python=3.10
conda activate unicure

2. Install PyTorch (select appropriate CUDA version)
⚠ Check latest at: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. Install Accelerate & DeepSpeed (recommended for reproduction)
pip install accelerate
⚠ Follow configuration (about how to configure DeepSpeed): https://github.com/huggingface/accelerate

4. Install core dependencies
pip install numpy pandas scikit-learn fastparquet tqdm anndata scanpy lora-pytorch scipy

5. Install Uni-Mol (required for testing)
You can create a new conda environment. 
https://github.com/deepmodeling/Uni-Mol
```

## Datasets Requirements 📚

### Step 1: Download Core Folders
We provide two ways to download the required datasets and model weights. Google Drive is recommended for faster and more stable download speeds. If the Google Drive link becomes unavailable, please use the permanent Figshare repository.

**Option A: Download via Google Drive (Recommended)**
Download the following three folders and **overwrite** them into your local UniCure directory:

1. **[data folder](https://drive.google.com/drive/folders/1VPXl8h8iuhr8IdrAmAWQ9ldEHkWRCGWq?usp=drive_link)**  
   - Contains: LINCS, SciPlex and PTC datasets
   - Local path: `your_project_path/UniCure/data/`

2. **[requirement folder](https://drive.google.com/drive/folders/1VPXl8h8iuhr8IdrAmAWQ9ldEHkWRCGWq?usp=drive_link)**  
   - Contains: configuration files
   - Local path: `your_project_path/UniCure/requirement/`

3. **[model weights](https://drive.google.com/drive/folders/1qZ4QwEXST_FcIZDTizMu7aH-OXpdc_CB?usp=drive_link)**  
   - Contains: Pre-trained model weights & Dataset splits (Training, Validation, and Test sets)
   - Local path: `your_project_path/UniCure/result/`

> ⚠️ **Overwrite Notice**: Replace existing directories completely when copying. <br> ⚠️ **Unzip Notice**: Unzip `Unicure_best_model.rar` (if applicable).

**Option B: Download via Figshare (Permanent Archive)**
*(DOI: [10.6084/m9.figshare.32077296](https://doi.org/10.6084/m9.figshare.32077296))*
1. Download the complete archive: **[UniCure.rar](https://figshare.com/articles/dataset/Dataset_and_Model_Weights_for_UniCure_A_Multi-modal_Model_for_Predicting_Personalized_Cancer_Therapy_Response/32077296?file=63934578)**
2. Unzip `UniCure.rar` and copy the `data`, `requirement`, and `result` folders to your local UniCure directory, **overwriting** any existing folders.

### Step 2: Download UCE Pretraining Files
Download these essential files to `requirement/UCE_pretraining_files/`:

| File | Size | Required Path |
|------|------|---------------|
| **[33l_8ep_1024t_1280.torch](https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806?file=43423236)** | 4.2 GB | `requirement/UCE_pretraining_files/` |
| **[all_tokens.torch](https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806?file=43423236)** | 780 MB | `requirement/UCE_pretraining_files/` |
| **[species_chrom.csv](https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806?file=43423236)** | 12 KB | `requirement/UCE_pretraining_files/` |

### Verification Checklist
After downloading, confirm directory structure:
```
UniCure/
├── data/
│   ├── lincs2020/
│ 	├── sciplex/
│	└── PTC/
├── result/
│   └── 11/
│       ├── lincs2020/
│       ├── sciplex3/ 
│	    └── sciplex4/
└── requirement/
    └── UCE_pretraining_files/
	    ├── protein_embeddings/
        ├── 33l_8ep_1024t_1280.torch
        ├── all_tokens.torch
        └── species_chrom.csv
```

## Quick Test :zap:
Run a quick inference test to predict the gene expression response of a specific cell line (e.g., ABY001_NCIH596) treated with a particular drug (e.g., vorinostat).

```bash
python quick_pred.py
```

**Input Data:**
- **Control Data**: Unperturbed (baseline) gene expression profile of the cell line (`./data/test/Test.parquet`)
- **Drug Embeddings**: Pre-calculated Uni-Mol representations for chemical perturbation (`./data/lincs2020/lincs2020_unimol_emb.parquet`)
- **Model Weights**: Pre-trained UniCure weights (`./result/11/lincs2020/Unicure_best_model.pth`)

**Interpretation of Output:**
- **Prediction Result** (`./data/test/prediction_result.csv`): A CSV file containing the condition-specific transcriptomic response. It represents the predicted gene expression profile of the given sample under the specific gene perturbation induced by the target drug and dose.

## Training Reproduction :fire:

Our training pipeline is designed to be progressive. Later stages depend on the pre-trained weights generated by the previous stages (e.g., LINCS Step 1 -> LINCS Step 2 -> Sciplex 3 -> Sciplex 4). 

You can choose to run the entire pipeline at once or execute each stage individually.

### Option 1: Run the Entire Pipeline
To sequentially train and test all stages (LINCS Step 1 & 2, Sciplex 3, and Sciplex 4) with the default seed (`11`), simply run:

```bash
python main.py --run_all
```

### Option 2: Run Step-by-Step (Recommended)
If you want to train a specific stage or debug, you can run the stages individually using the provided flags. 

**1. LINCS Stage 1:**
*(Note: This step requires the UCE pretraining files located in `requirement/UCE_pretraining_files/`. It learns basic cell embeddings by restoring the unperturbed states.)*
```bash
python main.py --run_lincs1
```

**2. LINCS Stage 2 (Train & Test):**
*(Note: Ensure Stage 1 is completed and Cell Embedding is generated by `generate_emb.py` before running this. This step requires the `best_unicure_stage_1_model.pth` generated from Stage 1.)*
```bash
python main.py --run_lincs2
```

**3. Sciplex 3 (Train & Test):**
*(Note: This step fine-tunes the model on the Sciplex 3 dataset. It relies on the pre-trained weights `Unicure_best_model.pth` generated in LINCS Stage 2.)*
```bash
python main.py --run_sciplex3
```

**4. Sciplex 4 (Train & Test):**
*(Note: This step further fine-tunes the model for Sciplex 4. It relies on the weights `Unicure_best_model.pth` produced by Sciplex 3.)*
```bash
python main.py --run_sciplex4
```

### Advanced: Customizing the Random Seed
By default, the pipeline uses `seed=11`. You can easily change the random seed for any execution by adding the `--seed` argument. For example, to run the Sciplex 3 stage with seed `42`:

```bash
python main.py --run_sciplex3 --seed 42
```

## Cell Embedding Generation (After Stage 1 Training) :fire:
```
python generate_emb.py
```

## Fine-tuning Reproduction :fire:
To adapt the pre-trained UniCure model to specific cancer types (e.g., LUAD, BLCA, TNBC) from the Patient-Derived Tumor Cell (PTC) cohorts, you can run the fine-tuning script.

By default, the script fine-tunes on the LUAD dataset across varying training sizes `[0.05, 0.1, 0.2, 0.4, 0.6, 0.8]` and multiple random seeds `[1, 2, 3, 4, 5]`. You can modify the script to include other cancer types by uncommenting the corresponding dataset blocks.

```bash
python finetune.py
```

## Contact 📬
Zexi Chen
📧 Email: jersey8768@outlook.com

## Citation 🧷
doi: https://doi.org/10.1101/2025.06.14.658531


## License 📄
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


