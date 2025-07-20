# UniCure: A Foundation Model for Predicting Personalized Cancer Therapy Response 💊

## Overview 🌟
UniCure is the **first multimodal foundation model** integrating omics (UCE) and chemical (Uni-mol) foundation models to predict transcriptomic drug responses across diverse cellular contexts. Trained on **1.8M+ perturbation profiles** (22k compounds, 166 cell types, 24 tissues).  

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
pip install numpy pandas scikit-learn fastparquet tqdm anndata scanpy lora-pytorch

5. Install Uni-Mol (required for testing)
You can create a new conda environment. 
https://github.com/deepmodeling/Uni-Mol
```

## Datasets Requirements 📚

### Step 1: Download Core Folders
Download and **overwrite** these folders to your local UniCure directories:

1. **[data folder](https://drive.google.com/drive/folders/1VPXl8h8iuhr8IdrAmAWQ9ldEHkWRCGWq?usp=drive_link)**  
   - Contains: LINCS, SciPlex datasets, and PTC
   - Local path: `your_project_path/UniCure/data/`

2. **[requirement folder](https://drive.google.com/drive/folders/1VPXl8h8iuhr8IdrAmAWQ9ldEHkWRCGWq?usp=drive_link)**  
   - Contains: Pre-trained model weights and configuration files
   - Local path: `your_project_path/UniCure/requirement/`

> ⚠️ **Overwrite Notice**: Replace existing directories completely when copying <br> ⚠️ **Overwrite Notice**: Unzip requirement/model_weights/best_model.rar

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
│   └── sciplex/
└── requirement/
    ├── model_weights/
    └── UCE_pretraining_files/
        ├── 33l_8ep_1024t_1280.torch
        ├── all_tokens.torch
        └── species_chrom.csv
```

## Quick Testing :zap:
```
python start_test.py
```
## Reproduction :fire:
```
python main.py
```

## Contact 📬
Zexi Chen
📧 Email: jersey8768@outlook.com

## Citation 🧷
doi: https://doi.org/10.1101/2025.06.14.658531


## License 📄
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

