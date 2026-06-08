from pathlib import Path
import pandas as pd

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent


def find_repo_root() -> Path:
    for parent in SCRIPT_DIR.parents:
        if (parent / "README.md").exists() and (parent / "model.py").exists():
            return parent
    return MODULE_DIR.parents[2]


REPO_ROOT = find_repo_root()
RAW_DATA_DIR = REPO_ROOT / "raw_data" / "fig3" / "target_gene"

CANCER_TYPES = ["BRCA", "COAD", "LUAD", "PRAD"]
MEASURE_TYPES = ["absolute", "delta"]

CANCER_CELL_LINES = {
    "BRCA": ["SKBR3", "T47D", "ZR751", "MCF7", "BT474", "BT20", "HS578T", "MDAMB231", "MDAMB468"],
    "COAD": ["HT29", "NCIH508", "SNU407", "HCT116", "GP2D", "CW2", "SNU1040"],
    "LUAD": ["A549", "H1975", "NCIH1975", "NCIH2110", "NCIH2172", "NCIH838", "HCC1588", "HCC515", "NCIH1437"],
    "PRAD": ["PC3", "22RV1", "LNCAP", "VCNP", "C42"],
}

# ============================================================
# Upstream target-gene selection and table-preparation logic
# ============================================================
# This module validates the curated plotting tables in raw_data/:
#   - raw_data/absolute: absolute expression values without averaging repeated
#     profiles. These tables preserve repeated samples under the same
#     cell_type / drug / dose condition and are used for Fig. 3C and Fig. S8A-C.
#   - raw_data/delta: delta expression values after averaging repeated samples
#     within each cell_type / drug / dose condition before subtracting controls.
#     These tables are used for Fig. S8D-G.
#
# The core upstream preparation logic is summarized below to document how the
# released plotting tables were prepared.
#
# # ---- DisGeNET target-gene selection ----
# LUAD_gt = pd.read_csv("data/disgenet/LUAD.tsv", sep="\t")["Gene"]
# PRAD_gt = pd.read_csv("data/disgenet/PRAD.tsv", sep="\t")["Gene"]
# BRCA_gt = pd.read_csv("data/disgenet/BRCA.tsv", sep="\t")["Gene"]
# COAD_gt = pd.read_csv("data/disgenet/COAD.tsv", sep="\t")["Gene"]
#
# # ---- Absolute expression tables ----
# predictions_df = pd.read_csv(os.path.join(save_dir, "predictions.csv"))
# real_outputs_df = pd.read_csv(os.path.join(save_dir, "real_outputs.csv"))
# predictions_df["cell_type"] = predictions_df["cell_type"].str.split("_").str[1]
# real_outputs_df["cell_type"] = real_outputs_df["cell_type"].str.split("_").str[1]
# # For absolute expression, repeated samples are retained and compared directly.
#
# # ---- Delta expression tables ----
# predictions_mean = pd.read_csv(os.path.join(save_dir, "predictions_mean.csv"))
# real_outputs_mean = pd.read_csv(os.path.join(save_dir, "real_outputs_mean.csv"))
# control_mean = pd.read_csv(os.path.join(save_dir, "control_mean.csv"))
# predictions_mean["new_cid"] = predictions_mean["new_cid"].str.split("_").str[1]
# real_outputs_mean["new_cid"] = real_outputs_mean["new_cid"].str.split("_").str[1]
# gene_columns = predictions_mean.columns[3:]
# pred_delta_values = predictions_mean[gene_columns].values - control_mean[gene_columns].values
# real_delta_values = real_outputs_mean[gene_columns].values - control_mean[gene_columns].values
#
# # ---- Cancer-specific filtering ----
# # Select cancer-specific cell lines, intersect DisGeNET genes with LINCS 2020
# # landmark genes, keep the first 15 target genes, and save *_filtered.csv.


def validate_raw_tables() -> None:
    print("=" * 60)
    print("Fig. 3 target-gene raw_data validation")
    print("=" * 60)

    for measure in MEASURE_TYPES:
        print(f"\n[{measure}]")
        for cancer in CANCER_TYPES:
            pred_path = RAW_DATA_DIR / measure / f"{cancer}_pred_filtered.csv"
            real_path = RAW_DATA_DIR / measure / f"{cancer}_real_filtered.csv"

            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            if not real_path.exists():
                raise FileNotFoundError(real_path)

            pred_df = pd.read_csv(pred_path)
            real_df = pd.read_csv(real_path)

            if pred_df.shape != real_df.shape:
                raise ValueError(f"Shape mismatch for {measure} {cancer}: pred={pred_df.shape}, real={real_df.shape}")
            if list(pred_df.columns) != list(real_df.columns):
                raise ValueError(f"Column mismatch for {measure} {cancer}")

            print(f"{cancer}: {pred_df.shape[0]} rows x {pred_df.shape[1]} genes")

    print("=" * 60)
    print("Rawdata validation completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    validate_raw_tables()
