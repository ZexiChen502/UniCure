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

#### Manual Installation Steps:
```
# 1. Install Python 3.10
conda create -n unicure python=3.10
conda activate unicure

# 2. Install PyTorch (select appropriate CUDA version)
# ⚠ Check latest at: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install Accelerate & DeepSpeed (optional)
pip install accelerate
# ⚠ Follow configuration (about how to configure DeepSpeed): https://github.com/huggingface/accelerate

# 4. Install core dependencies
pip install numpy pandas scikit-learn fastparquet tqdm anndata scanpy lora-pytorch
```

## License 📄
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

