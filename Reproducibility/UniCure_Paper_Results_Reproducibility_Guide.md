# UniCure Reproducibility Guide

This guide describes the figure-level reproducibility package for the UniCure study.

The package is organized around the data-to-result workflow. The reproducibility scope links the implemented source-data workflow in the main repository with immediately runnable figure generation: source datasets and model resources are obtained through the repository README and manuscript data-availability instructions, root-level code records the training, splitting, inference, fine-tuning, and downstream analysis workflows that produce curated result tables, released input files are placed under `./raw_data/`, figure-specific scripts are provided under `./Reproducibility/`, and regenerated panels are written to each module's `output_plot/` directory. Each result section below identifies the scientific purpose of the analysis, the input files used by the released scripts, the code entry points, and the expected outputs.

The figure-level scripts regenerate plotted panels from curated tables, coordinate files, summary statistics, binary R data objects, or experimental source-data records included in the released raw-data folders. 

Recommended execution pattern:

1. Place the downloaded raw-data folders at the repository root under `./raw_data/`.
2. Run the relevant script from the corresponding module under `./Reproducibility/Fig*/case_name/code/`.
3. Inspect regenerated PDF/PNG outputs in `./Reproducibility/Fig*/case_name/output_plot/`.

Python plotting scripts should be run in a Python environment with the dependencies listed in the repository. R plotting scripts should be run with the R packages used by the corresponding module, including `ggplot2`, `dplyr`, `readr`, and module-specific packages where required.

## Directory overview

```text
./Reproducibility/
  Fig2/
    loss/
    metrics/
    tsne/
    benchmark/
  Fig3/
    ...
  Fig4/
    ...
  Fig5/
    ...
  Fig6/
    ...

./raw_data/
  Fig2/
    loss/
    metrics/
    tsne/
    benchmark/
  Fig3/
    ...
  Fig4/
    ...
  Fig5/
    ...
  Fig6/
    ...
```

Each case folder in `./Reproducibility/` contains a `code/` directory,and an `output_plot/` directory. The corresponding `./raw_data/` case folder contains the released inputs required by those scripts.

## Fig. 2: Bulk and single-cell perturbation prediction

Figure 2 evaluates UniCure's training behavior, held-out prediction accuracy, embedding geometry, and benchmark performance across bulk and single-cell perturbation settings. The released Fig. 2 reproducibility modules start from saved training histories, metric summaries, t-SNE coordinate tables, and benchmark tables, then regenerate the figure-level plots.

### Fig. 2A and Fig. S4: training and validation curves

**Scientific purpose.** These panels show UniCure training dynamics across LINCS 2020, UCE-LoRA, sci-Plex3, and sci-Plex4 training stages. Each output figure displays loss and R2 curves so that convergence and validation behavior can be inspected together.

**Input data.** The released training-history tables are stored in:

```text
./raw_data/Fig2/loss/
```

Key files include:

- `lincs_training_results.csv`
- `ucelora_training_results.csv`
- `sciplex3_training_results.csv`
- `sciplex4_training_results.csv`

Each table contains epoch-level `train_loss`, `val_loss`, `train_r2`, and `val_r2` values.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig2/loss/code/plot_loss_r2_curves.py
```

The script reads the four training-history tables, checks the required columns, generates one two-panel plot per dataset, and also exports a combined overview plot. The sci-Plex3 visualization uses the retained epoch range specified in the script so that early-scale differences do not obscure the later training trend.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig2/loss/output_plot/
```

Expected outputs include dataset-specific `*_loss_r2.pdf/.png` files and `all_loss_r2_overview.pdf/.png`.

**Reproducibility notes.** This module begins from training summaries exported by the UniCure training workflow. The LINCS 2020, sci-Plex3, and sci-Plex4 source datasets are part of the repository data release described in `README.md`; after download, they are placed under `./data/` according to the README verification checklist. Root-level `main.py` orchestrates staged training through `train_lincs_step1`, `train_lincs_step2`, `train_sciplex3`, and `train_sciplex4`; `train.py` implements the training and testing loops; `preprocessing.py` builds the data loaders; `utils.py` loads or creates the condition-level split files; and `model.py` defines the UniCure model architecture and default model parameters. The released seed-11 split files under `./result/11/lincs2020/`, `./result/11/sciplex3/`, and `./result/11/sciplex4/` define the train, validation, and test partitions used by the workflow. Training and validation curves are exported from the training/validation partitions, and the Fig. 2 loss module uses the released epoch-level tables to regenerate the plotted training and validation curves.

### Fig. 2B and associated metric panels: held-out evaluation metrics

**Scientific purpose.** These panels summarize prediction accuracy on held-out perturbation conditions using Pearson correlation, Spearman correlation, and R2. The same visualization logic is applied to LINCS 2020, sci-Plex3, and sci-Plex4 summaries.

**Input data.** Plotting-ready metric tables are stored in:

```text
./raw_data/Fig2/metrics/
```

Key files include:

- `lincs_correlation_results_melted.csv`
- `sciplex3_correlation_results_melted.csv`
- `sciplex4_correlation_results_melted.csv`

Each table contains the shared `Metric` and `Value` fields used for plotting.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig2/metrics/code/Metric_on_test_datasets.R
```

The script reads all three metric tables, fixes the metric order as Pearson, Spearman, and R2, and generates both dataset-specific and combined metric panels using half-violin and boxplot overlays.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig2/metrics/output_plot/
```

Expected outputs include `lincs_Metric_on_test_datasets.pdf/.png`, `sciplex3_Metric_on_test_datasets.pdf/.png`, `sciplex4_Metric_on_test_datasets.pdf/.png`, and `all_Metric_on_test_datasets.pdf/.png`.

**Reproducibility notes.** The metric tables summarize held-out comparisons between predicted and observed perturbation profiles. The upstream evaluations are produced by the test routines called from `main.py` (`test_lincs`, `test_sciplex3`, and `test_sciplex4`) using model weights under `./result/11/` and the corresponding `data_pairs_test.pkl` files for LINCS 2020, sci-Plex3, and sci-Plex4. The same source datasets and split files used for training are therefore carried through to held-out test-set evaluation. The released R script standardizes these summaries to a common metric/value format and applies a consistent visualization across datasets.

### Fig. 2C-F and Fig. S5: t-SNE and embedding visualization

**Scientific purpose.** These panels visualize whether UniCure preserves cell identity, resolves batch effects through UCE-LoRA, and recapitulates the global geometry of observed single-cell perturbation profiles.

**Input data.** t-SNE and embedding coordinate tables are stored in:

```text
./raw_data/Fig2/tsne/
```

Key files include:

- `lincs_control_tsne.csv`
- `lincs_tsne_uce_lora_emb.csv`
- `lincs_predict_tsne_result.csv`
- `sciplex_predict_tsne_result.csv`
- `sciplex_real_tsne_result.csv`

The LINCS files support the control, UCE-LoRA embedding, and predicted perturbation panels. The sci-Plex3 files provide paired predicted and observed coordinate tables used for Fig. 2F and the cell-line-specific Fig. S5 panels.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig2/tsne/code/plot_tsne_figures.py
```

The script reads the released coordinate tables and generates panel-specific scatter plots. Fig. S5 is produced by filtering the combined sci-Plex3 predicted and observed coordinate tables into A549, K562, and MCF7 subsets and coloring points by drug identity.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig2/tsne/output_plot/
```

Expected outputs include:

- `fig2c_lincs_control_tsne.pdf/.png`
- `fig2d_lincs_uce_lora_emb_tsne.pdf/.png`
- `fig2e_lincs_predict_tsne.pdf/.png`
- `fig2f_left_sciplex3_predict_tsne.pdf/.png`
- `fig2f_right_sciplex3_real_tsne.pdf/.png`
- `figs5_a549_tsne.pdf/.png`
- `figs5_k562_tsne.pdf/.png`
- `figs5_mcf7_tsne.pdf/.png`

**Reproducibility notes.** Coordinate generation and panel plotting are represented as separate steps in this module. The upstream LINCS and sci-Plex coordinate tables are derived from the same README-described LINCS 2020 and sci-Plex resources and from UniCure predictions on the held-out split files under `./result/11/`. Root-level `generate_emb.py` records the cell-embedding generation step after LINCS stage-1 training, while `main.py`, `preprocessing.py`, `utils.py`, and `model.py` provide the prediction workflow and model definitions used to produce the profiles that are projected for visualization. The released coordinate tables capture the two-dimensional projections used for the figure, and the plotting script reads those coordinates to apply the panel-specific coloring, filtering, and figure export logic.

### Fig. 2G and Fig. S6: benchmark comparison

**Scientific purpose.** These panels compare UniCure with benchmark perturbation-prediction models in bulk and single-cell settings, emphasizing recovery of genes with strong observed perturbation responses.

**Input data.** Benchmark tables are stored in:

```text
./raw_data/Fig2/benchmark/
```

Key plotting files include:

- `lincs_benchmark_rank_threshold_with_delta.csv`
- `sciplex_benchmark_rank_threshold_with_delta.csv`

Additional files in the same folder provide per-sample metric summaries for UniCure, TranSiGen, and PRnet, as well as the LINCS cell-line ranking used for the top-50 benchmark subset.

**Code.** The main plotting entry point is:

```text
./Reproducibility/Fig2/benchmark/code/plot_benchmark_fig2g.R
```

The supporting Python file:

```text
./Reproducibility/Fig2/benchmark/code/calculate_delta_pearson.py
```

documents the Delta Pearson calculation used by the benchmark workflow. The metric ranks genes by observed perturbation magnitude and evaluates prediction quality over progressively larger top-ranked gene subsets.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig2/benchmark/output_plot/
```

Expected outputs include:

- `fig2g_benchmark_lincs.pdf/.png`
- `fig2g_benchmark_sciplex.pdf/.png`
- `fig2g_benchmark_combined.pdf/.png`

**Reproducibility notes.** The benchmark analysis uses perturbation deltas rather than raw expression values alone. LINCS 2020 benchmark training and evaluation use the same repository data release and the seed-11 split files `./result/11/lincs2020/data_pairs_train.pkl`, `./result/11/lincs2020/data_pairs_val.pkl`, and `./result/11/lincs2020/data_pairs_test.pkl`; sci-Plex benchmark evaluation follows the corresponding sci-Plex held-out test-set structure under `./result/11/sciplex3/`. UniCure training and testing are implemented through `main.py`, `train.py`, `preprocessing.py`, `utils.py`, and `model.py`. For the comparator models, TranSiGen and PRnet were trained on the same training and validation partitions and evaluated on the same held-out test partitions, with model hyperparameters following the configurations released in their respective GitHub repositories. For each sample, treatment-minus-control changes are compared between predicted and observed profiles over ranked gene subsets, and the R plotting script summarizes model-level performance curves with uncertainty bands for LINCS 2020 and sci-Plex3.

## Fig. 3: Mechanism, dose, target-gene, cell-state, and combination-response analyses

Figure 3 evaluates whether UniCure predictions preserve biologically interpretable perturbation structure. The released Fig. 3 modules cover mechanism-of-action and dose visualization, disease-relevant target-gene expression, cell-state and pathway-level response signatures, and sci-Plex4 dual-drug combination effects.

### Fig. 3A-B and Fig. S7: mechanism-of-action and dose visualization

**Scientific purpose.** These panels test whether predicted perturbation profiles preserve compound identity, mechanism-of-action structure, and dose-dependent shifts in expression space.

**Input data.** The released coordinate tables are stored in:

```text
./raw_data/Fig3/moa_dose/
```

Key files include predicted and observed t-SNE coordinate tables for A375, MCF7, PC3, and A549:

- `A375_tsne_drug_results.csv` and `real_A375_tsne_drug_results.csv`
- `MCF7_tsne_drug_results.csv` and `real_MCF7_tsne_drug_results.csv`
- `PC3_tsne_drug_results.csv` and `real_PC3_tsne_drug_results.csv`
- `A549_tsne_drug_results.csv` and `real_A549_tsne_drug_results.csv`

Each table contains t-SNE coordinates and metadata for cell type, drug identity, and dose.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig3/moa_dose/code/plot_moa_tsne.py
```

The script reads matched predicted and observed coordinate tables for each cell line, applies consistent color mappings for drug identity or dose, and exports paired predicted/observed panels.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig3/moa_dose/output_plot/
```

Expected outputs include:

- `fig3a_a375_drug.pdf/.png`
- `fig3b_a375_dose.pdf/.png`
- `figs7a_mcf7_drug.pdf/.png`
- `figs7b_mcf7_dose.pdf/.png`
- `figs7c_pc3_drug.pdf/.png`
- `figs7d_pc3_dose.pdf/.png`
- `figs7e_a549_drug.pdf/.png`
- `figs7f_a549_dose.pdf/.png`

**Reproducibility notes.** Predicted and observed profiles are represented as paired coordinate tables with shared metadata fields. The upstream predictions are generated from LINCS 2020 test-set perturbation profiles and UniCure model weights using the same root workflow described for Fig. 2 (`main.py`, `preprocessing.py`, `utils.py`, and `model.py`). The plotting script uses these fields to compare predicted and observed organization by compound identity and dose across the same cell-line contexts.

### Fig. 3C and Fig. S8: target-gene expression

**Scientific purpose.** These panels compare predicted and observed expression for disease-relevant target genes across BRCA, COAD, LUAD, and PRAD, using both absolute expression and treatment-minus-control delta expression.

**Input data.** Target-gene tables are stored in:

```text
./raw_data/Fig3/target_gene/
```

The folder contains two subdirectories:

```text
./raw_data/Fig3/target_gene/absolute/
./raw_data/Fig3/target_gene/delta/
```

For each cancer type, matched predicted and observed tables are provided, for example `BRCA_pred_filtered.csv` and `BRCA_real_filtered.csv`.

**Code.** The data-checking and plotting entry points are:

```text
./Reproducibility/Fig3/target_gene/code/prepare_target_gene_tables.py
./Reproducibility/Fig3/target_gene/code/plot_target_gene_boxplots.R
```

The Python script validates that matched predicted and observed tables have the same shape and gene columns. The R script reshapes matched tables into long format and generates per-gene predicted-versus-observed boxplots.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig3/target_gene/output_plot/
```

Expected outputs include:

- `fig3c_brca_absolute_target_gene.pdf/.png`
- `figs8a_coad_absolute_target_gene.pdf/.png`
- `figs8b_luad_absolute_target_gene.pdf/.png`
- `figs8c_prad_absolute_target_gene.pdf/.png`
- `figs8d_brca_delta_target_gene.pdf/.png`
- `figs8e_coad_delta_target_gene.pdf/.png`
- `figs8f_luad_delta_target_gene.pdf/.png`
- `figs8g_prad_delta_target_gene.pdf/.png`

**Reproducibility notes.** The target-gene workflow links cancer-specific target selection with matched predicted and observed LINCS 2020 test-set expression tables. Cancer-associated target genes are intersected with the LINCS landmark-gene feature space, and UniCure predictions are generated by the root model workflow from the same LINCS source data and split structure used for Fig. 2. Absolute-expression panels retain repeated profiles, while delta-expression panels represent treatment-minus-control changes after condition-level summarization.

### Fig. 3D-F, Fig. S9, and Fig. S10: cell-state and pathway response signatures

**Scientific purpose.** These analyses examine cell-state-specific perturbation responses and pathway-level transcriptional programs associated with PI3K/AKT-pathway drugs.

**Input data.** Cell-state and pathway result tables are stored in:

```text
./raw_data/Fig3/cell_state_specific/
```

Key files include group, differential-expression, and pathway-enrichment tables for Alpelisib and Copanlisib in MCF7, together with related A375 result tables.

**Code.** The relevant scripts are:

```text
./Reproducibility/Fig3/cell_state_specific/code/cell_state_response_workflow.py
./Reproducibility/Fig3/cell_state_specific/code/cell_state_specific_plots.R
```

`cell_state_response_workflow.py` records the A549 single-cell response workflow, including marker-defined subpopulation construction, UniCure-predicted Dasatinib response scoring, and ranked response-gene analysis. `cell_state_specific_plots.R` reads the released pathway-enrichment and differential-expression tables and generates pathway dot plots and volcano plots.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig3/cell_state_specific/output_plot/
```

Expected outputs from the plotting script include:

- `fig3e_alpelisib_MCF7_pathway.pdf/.png`
- `fig3f_copanlisib_MCF7_pathway.pdf/.png`
- `figs10a_alpelisib_MCF7_volcano.pdf/.png`
- `figs10b_copanlisib_MCF7_volcano.pdf/.png`

**Reproducibility notes.** The cell-state workflow connects marker-defined response groups, predicted perturbation effects, differential-expression summaries, and pathway-enrichment results. `cell_state_response_workflow.py` records the source workflow for Fig. 3D and Fig. S9, starting from sci-Plex3 A549 control cells, defining SRC/DDR1/CAV1-associated subpopulations, applying the sci-Plex3-trained UniCure model to Dasatinib prediction, and ranking response genes. The MCF7 pathway and volcano panels use LINCS-derived predicted perturbation profiles and released differential-expression/enrichment summaries. The plotting script uses released enrichment and differential-expression tables to display the pathway and gene-level signatures supporting the corresponding Fig. 3 and supplementary panels.

### Fig. 3G and Fig. S11: sci-Plex4 combination effects

**Scientific purpose.** These panels evaluate UniCure predictions for dual-drug perturbations in sci-Plex4 and compare UniCure combination predictions with an additive single-drug baseline across unseen-combination settings.

**Input data.** Combination-effect inputs are stored in:

```text
./raw_data/Fig3/combination_effects/
```

Key files include:

- `sciplex4_predict_tsne_result.csv`
- `sciplex4_real_tsne_result.csv`
- `sciplex4_unseen_correlations_0.csv`
- `sciplex4_unseen_correlations_1.csv`
- `sciplex4_unseen_correlations_2.csv`
- `sciplex4_unseen_correlations_additive_baseline_0.csv`
- `sciplex4_unseen_correlations_additive_baseline_1.csv`
- `sciplex4_unseen_correlations_additive_baseline_2.csv`

The coordinate tables contain predicted and observed sci-Plex4 profiles with cell type, first-drug, second-drug, dose, and t-SNE coordinate fields. The correlation tables summarize UniCure and additive-baseline performance for 0/2, 1/2, and 2/2 unseen drug-combination settings.

**Code.** The plotting and comparison entry point is:

```text
./Reproducibility/Fig3/combination_effects/code/combination_effects.py
```

The script reads predicted and observed coordinate tables, generates t-SNE panels colored by cell type, drug identity, or dose, and compares UniCure with the additive baseline using one-sided Mann-Whitney U tests across unseen-combination scenarios.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig3/combination_effects/output_plot/
```

Expected outputs include:

- `fig3g_sciplex4_prediction_combination_effects.pdf/.png`
- `figs11a_sciplex4_real_cell_type.pdf/.png`
- `figs11b_sciplex4_real_drug1_name.pdf/.png`
- `figs11c_sciplex4_real_drug1_dose.pdf/.png`
- `figs11d_sciplex4_real_drug2_name.pdf/.png`
- `figs11e_sciplex4_real_drug2_dose.pdf/.png`
- `figs11f_sciplex4_unseen_evaluation.pdf/.png`

**Reproducibility notes.** The combination-effects workflow combines sci-Plex4 coordinate visualization with quantitative unseen-combination evaluation. The sci-Plex4 source data are part of the README-described data release; training and testing use `main.py` functions `train_sciplex4` and `test_sciplex4` with split files under `./result/11/sciplex4/` and the model definitions in `model.py`. Predicted and observed coordinate tables support the t-SNE panels, while the correlation tables support the UniCure-versus-additive-baseline comparison for increasing levels of unseen drug-combination generalization.

## Fig. 4: Patient-derived tumor-cell fine-tuning and evaluation

Figure 4 evaluates UniCure fine-tuning on patient-derived tumor cell profiles and compares fine-tuned UniCure with zero-shot UniCure and external perturbation-prediction baselines. The released Fig. 4 modules cover training-fraction fine-tuning performance, ablation analysis, benchmark comparison, and cross-cancer/real-world evaluation.

### Fig. 4A-G: fine-tuning fraction and ablation analysis

**Scientific purpose.** These panels quantify how UniCure performance changes as the number of patient-derived tumor cell training profiles increases, and assess the contribution of the fine-tuning strategy through ablation comparison.

**Input data.** Fine-tuning and ablation tables are stored in:

```text
./raw_data/Fig4/finetune_ablation/
```

Key files include:

- `LUAD_finetune_results.csv`
- `BLCA_finetune_results.csv`
- `TNBC_finetune_results.csv`
- `LUAD_finetune_result_delta.csv`
- `BLCA_finetune_result_delta.csv`
- `TNBC_finetune_result_delta.csv`
- `LUAD_ablation_fine_tune_result.csv`
- `BLCA_ablation_fine_tune_result.csv`
- `TNBC_ablation_fine_tune_result.csv`
- `ablation_merge.csv`

The fine-tuning tables contain seed-level correlation coefficients across the released training fractions: 5%, 10%, 20%, 40%, 60%, and 80%.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig4/finetune_ablation/code/finetune_ablation.R
```

The script reads cancer-specific absolute and delta fine-tuning tables, reshapes the training-fraction columns into long format, and generates one performance plot per cancer and metric type. It also reads the 5% fine-tuning and ablation tables, calculates Wilcoxon test summaries, and generates the ablation comparison panel.

**Outputs.** Regenerated figures and plotting summaries are written to:

```text
./Reproducibility/Fig4/finetune_ablation/output_plot/
```

Expected figure outputs include:

- `fig4a_luad_finetune_absolute.pdf/.png`
- `fig4b_blca_finetune_absolute.pdf/.png`
- `fig4c_tnbc_finetune_absolute.pdf/.png`
- `fig4d_luad_finetune_delta.pdf/.png`
- `fig4e_blca_finetune_delta.pdf/.png`
- `fig4f_tnbc_finetune_delta.pdf/.png`
- `fig4g_finetune_ablation.pdf/.png`

The script also writes `fig4g_ablation_plotting_data.csv`, `fig4g_ablation_summary_mean_sd.csv`, and `fig4g_ablation_wilcoxon.csv` to document the ablation plotting table and statistical summaries.

**Reproducibility notes.** The fine-tuning tables organize repeated-seed performance by training fraction and cancer type. PTC data are included in the repository data release described in `README.md` before placement under the README verification-checklist paths. Root-level `finetune.py` records the PTC fine-tuning workflow, starting from the pretrained UniCure model and evaluating LUAD, BLCA, and TNBC across the released training fractions and seeds. The R script displays the distribution of correlation coefficients across fractions and summarizes the 5% fine-tuning ablation comparison with mean, standard deviation, and Wilcoxon-test annotations.

### Fig. 4H and Fig. S13A-B: benchmark evaluation

**Scientific purpose.** These panels compare fine-tuned UniCure with zero-shot UniCure, TranSiGen, and PRnet on LUAD, BLCA, and TNBC patient-derived tumor cell benchmark data.

**Input data.** Benchmark tables are stored in:

```text
./raw_data/Fig4/benchmark/
```

Key files include:

- `LUAD_benchmark_data.csv`
- `BLCA_benchmark_data.csv`
- `TNBC_benchmark_data.csv`

Each table contains model labels and correlation coefficients used for the benchmark comparison.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig4/benchmark/code/benchmark.R
```

The script reads each cancer-specific benchmark table, standardizes model labels, writes the plotting data used for each panel, and generates boxplot/jitter summaries with Wilcoxon comparison annotations.

**Outputs.** Regenerated figures and plotting tables are written to:

```text
./Reproducibility/Fig4/benchmark/output_plot/
```

Expected outputs include:

- `fig4h_luad_benchmark.pdf/.png`
- `figs13a_blca_benchmark.pdf/.png`
- `figs13b_tnbc_benchmark.pdf/.png`
- `fig4h_luad_benchmark_plotting_data.csv`
- `figs13a_blca_benchmark_plotting_data.csv`
- `figs13b_tnbc_benchmark_plotting_data.csv`

**Reproducibility notes.** The benchmark workflow compares model-level prediction quality within each patient-derived tumor cell cancer setting. PTC benchmark inputs follow the same downloaded PTC resources used for Fig. 4A-G. The independent real-world clinical transcriptomic profiles used elsewhere in Fig. 4 were collected from GEO according to manuscript Table S24. The plotting script keeps the same model order across panels and applies the corresponding Wilcoxon comparison structure to the released correlation-coefficient tables.

### Fig. 4I: cross-cancer and real-world evaluation

**Scientific purpose.** This panel summarizes UniCure evaluation across LUAD, BLCA, TNBC, and an independent real-world treatment-response collection.

**Input data.** Cross-cancer and real-world evaluation tables are stored in:

```text
./raw_data/Fig4/cross_cancer_real_world/
```

Key files include:

- `UniCure_finetune_LUAD_pearson.csv`
- `UniCure_finetune_BLCA_pearson.csv`
- `UniCure_finetune_TNBC_pearson.csv`
- `real_world_abs_pearson.csv`

Each file provides Pearson correlation values used in the combined evaluation panel.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig4/cross_cancer_real_world/code/cross_cancer_real_world.R
```

The script reads Pearson-correlation values from the cancer-specific and real-world tables, combines them into a single plotting table, and generates the cross-cancer/real-world evaluation boxplot.

**Outputs.** Regenerated figures and plotting data are written to:

```text
./Reproducibility/Fig4/cross_cancer_real_world/output_plot/
```

Expected outputs include:

- `fig4i_cross_cancer_real_world_evaluation.pdf/.png`
- `fig4i_cross_cancer_real_world_plotting_data.csv`

**Reproducibility notes.** This module combines cancer-specific fine-tuned UniCure Pearson correlations with real-world treatment-response correlations into one evaluation panel. Cross-cancer PTC evaluation uses the same README/Figshare-described PTC resource and the `finetune.py` workflow. The real-world clinical evaluation uses paired pre-/post-treatment profiles collected from GEO according to manuscript Table S24 and summarized into the released correlation tables. The plotting table written by the script records the exact grouped values used for the displayed boxplot.

## Fig. 5: Patient stratification, drug recommendation, and survival analysis

Figure 5 applies UniCure to patient-level transcriptomic and clinical contexts. The released Fig. 5 modules cover patient stratification from drug-rank matrices, clinically indicated drug recommendation, and Kaplan-Meier survival analysis from patient-level survival tables.

### Fig. 5B-G and Fig. S14A-B: patient stratification and cluster-associated signatures

**Scientific purpose.** These panels stratify BRCA and KIRC patients using UniCure-derived drug-rank profiles and characterize cluster-associated pathway and drug-response signatures.

**Input data.** Patient stratification inputs are stored in:

```text
./raw_data/Fig5/patient_stratification/
```

Key files include:

- `BRCA_rankmatrix.csv`
- `KIRC_rankmatrix.csv`
- `BRCA_kmeans_evaluation_results.csv`
- `KIRC_kmeans_evaluation_results.csv`
- `BRCA_sample_clusters.csv`
- `KIRC_sample_clusters.csv`
- `BRCA_diff_exp.csv`
- `KIRC_diff_exp.csv`
- `BRCA_pathway_results.csv`
- `KIRC_pathway_results.csv`
- `BRCA_diff_drug.csv`
- `KIRC_diff_drug.csv`
- `BRCA_filtered_tumor.csv`
- `KIRC_filtered_tumor.csv`

**Code.** The relevant entry points are:

```text
./Reproducibility/Fig5/patient_stratification/code/patient_stratification_workflow.py
./Reproducibility/Fig5/patient_stratification/code/patient_stratification_plots.R
```

The Python script reads the BRCA and KIRC drug-rank matrices, applies log10 transformation and standardization, performs PCA for patient visualization, and plots K-means evaluation summaries from the released tables. The R script reads the differential-expression, pathway-enrichment, and differential-drug tables and generates pathway bubble plots and drug lollipop plots.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig5/patient_stratification/output_plot/
```

Expected outputs include:

- `fig5b_brca_patient_clusters.pdf/.png`
- `fig5c_kirc_patient_clusters.pdf/.png`
- `fig5d_brca_pathway_enrichment.pdf/.png`
- `fig5e_brca_diff_drug_lollipop.pdf/.png`
- `fig5f_kirc_pathway_enrichment.pdf/.png`
- `fig5g_kirc_diff_drug_lollipop.pdf/.png`
- `figs14a_brca_kmeans_evaluation.pdf/.png`
- `figs14b_kirc_kmeans_evaluation.pdf/.png`

**Reproducibility notes.** The patient stratification workflow uses UniCure-derived patient-by-drug rank matrices as the central input. TCGA transcriptomic and clinical data were downloaded from UCSC Xena as described in the manuscript, and the released raw-data tables contain the tumor expression, drug-rank, clustering, differential-expression, and pathway-enrichment summaries used for Fig. 5. The Python script displays patient clustering from these matrices, while the R script summarizes pathway enrichment and differential drug ranks associated with the resulting patient groups.

### Fig. 5H-K and Fig. S16A-D: clinically indicated drug recommendation

**Scientific purpose.** These panels compare ranks of clinically indicated therapies against randomized within-patient drug-rank baselines across eight TCGA cancer types.

**Input data.** Drug recommendation tables are stored in:

```text
./raw_data/Fig5/drug_recommendation/
```

Key files include:

- `LUAD_Recommended_vs_Randomized_ft.csv`
- `BRCA_Recommended_vs_Randomized_ft.csv`
- `BLCA_Recommended_vs_Randomized_ft.csv`
- `LUSC_Recommended_vs_Randomized_ft.csv`
- `KIRC_Recommended_vs_Randomized_ft.csv`
- `LIHC_Recommended_vs_Randomized_ft.csv`
- `COAD_Recommended_vs_Randomized_ft.csv`
- `PRAD_Recommended_vs_Randomized_ft.csv`

Each table contains recommended and randomized drug-rank values for the cancer-specific clinically indicated therapies.

**Code.** The plotting entry point is:

```text
./Reproducibility/Fig5/drug_recommendation/code/drug_recommendation_workflow.py
```

The script reads each cancer-specific recommended-versus-randomized table, plots indicated therapy ranks against randomized baselines, and calculates one-sided Mann-Whitney U tests comparing whether recommended ranks are lower than randomized ranks.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig5/drug_recommendation/output_plot/
```

Expected outputs include:

- `fig5h_luad_drug_recommendation.pdf/.png`
- `fig5i_brca_drug_recommendation.pdf/.png`
- `fig5j_blca_drug_recommendation.pdf/.png`
- `fig5k_lusc_drug_recommendation.pdf/.png`
- `figs16a_kirc_drug_recommendation.pdf/.png`
- `figs16b_lihc_drug_recommendation.pdf/.png`
- `figs16c_coad_drug_recommendation.pdf/.png`
- `figs16d_prad_drug_recommendation.pdf/.png`

**Reproducibility notes.** The drug recommendation workflow uses patient-specific UniCure drug ranks and a matched randomized baseline. TCGA expression and clinical therapy annotations are obtained from UCSC Xena as described in the manuscript, then converted into the released recommended-versus-randomized rank tables. For each cancer type, clinically indicated therapies are extracted from the rank table, compared with randomized ranks, and displayed as paired rank distributions.

### Fig. 5L-O and Fig. S16I-K: survival analysis

**Scientific purpose.** These panels evaluate whether UniCure-derived drug-rank groups are associated with patient overall survival across cancer cohorts.

**Input data.** Patient-level survival inputs are stored in:

```text
./raw_data/Fig5/survival_analysis/
```

Key files include:

- `LUAD_combined_survive.csv`
- `BRCA_combined_survive.csv`
- `BLCA_combined_survive.csv`
- `LUSC_combined_survive.csv`
- `COAD_combined_survive.csv`
- `KIRC_combined_survive.csv`
- `LIHC_combined_survive.csv`

Each table contains clinical follow-up/death fields and normalized drug-rank ratio values used for survival grouping; patient identifiers are used for aggregation when repeated patient rows are present. The drug-rank ratio is defined as the raw administered-therapy rank divided by the total number of ranked drugs, so lower values indicate higher-priority UniCure rankings.

**Code.** The relevant scripts are:

```text
./Reproducibility/Fig5/survival_analysis/code/km_survival_plots.R
./Reproducibility/Fig5/survival_analysis/code/deepsurv_contrastive_demo.py
```

`km_survival_plots.R` reads patient-level survival tables, aggregates repeated rows by patient when patient identifiers are present, defines survival time from `days_to_death` or `days_to_last_followup`, assigns events from death status, groups patients by the manuscript top-30% priority rule (`avg_drug_rank <= 0.3` for the High Rank group), calculates log-rank p values, and plots Kaplan-Meier curves. `deepsurv_contrastive_demo.py` records the survival-risk modeling structure, including a UniCure-based survival head, Cox partial-likelihood loss, and administered-versus-random drug contrastive loss.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig5/survival_analysis/output_plot/
```

Expected outputs include:

- `fig5l_luad_km_survival.pdf/.png`
- `fig5m_brca_km_survival.pdf/.png`
- `fig5n_blca_km_survival.pdf/.png`
- `fig5o_lusc_km_survival.pdf/.png`
- `figs16i_coad_km_survival.pdf/.png`
- `figs16j_kirc_km_survival.pdf/.png`
- `figs16k_lihc_km_survival.pdf/.png`

**Reproducibility notes.** The survival workflow starts from patient-level TCGA tables downloaded from UCSC Xena, including survival follow-up, treatment records, and UniCure-derived normalized drug-rank ratios. `deepsurv_contrastive_demo.py` records the survival-risk modeling structure, including a UniCure-based survival head, Cox partial-likelihood loss, and administered-versus-random drug contrastive loss. The R script constructs Kaplan-Meier inputs directly from the released patient-level tables, uses the fixed top-30% priority grouping threshold (`avg_drug_rank <= 0.3`), calculates log-rank p values, and exports one survival panel per cancer type.

## Fig. 6: Natural-product prioritization

These panels prioritize natural product-enriched compounds for TNBC, LUAD, and BLCA using UniCure-predicted perturbation profiles.

### Fig. 6B-C: natural-product candidate scores and reversal heatmaps

**Scientific purpose.** Prioritize natural-product candidates whose predicted perturbation signatures are consistent with reversal of cancer-associated disease signatures, and visualize reversal patterns for top-ranked candidates.

**Input data.** Natural-product inputs are stored in:

```text
./raw_data/Fig6/natural_product/
```

The folder contains computational-prioritization subdirectories and experimental source-data records:

```text
./raw_data/Fig6/natural_product/Figure6B/
./raw_data/Fig6/natural_product/Figure6C/
./raw_data/Fig6/Experiment/
```

Key Fig. 6B files include:

- `MB231_np_cmap.Rdata`
- `A549_np_cmap.Rdata`
- `Blca_5637_np_cmap.Rdata`

Key Fig. 6C files include:

- `MB231_drug_rank.Rdata`
- `A549_drug_rank.Rdata`
- `5637_drug_rank.Rdata`

The Fig. 6B R data objects contain natural-product scoring tables used for top-candidate dot plots. The Fig. 6C R data objects contain drug-by-signature rank matrices used for reversal heatmaps.

**Code.** The plotting entry point is：

```text
./Reproducibility/Fig6/natural_product/code/natural_product_plots.R
```

The script loads the Fig. 6B natural-product CMap-style score tables, selects the top 20 entries for each cancer context, and generates dot plots. It also loads the Fig. 6C reversal-rank matrices and generates heatmaps using the cancer label and drug-candidate labels as axes.

**Outputs.** Regenerated figures are written to:

```text
./Reproducibility/Fig6/natural_product/output_plot/
```

Expected outputs include:

- `fig6b_tnbc_dotplot.pdf/.png`
- `fig6b_luad_dotplot.pdf/.png`
- `fig6b_blca_dotplot.pdf/.png`
- `fig6c_tnbc_reversal_heatmap.pdf/.png`
- `fig6c_luad_reversal_heatmap.pdf/.png`
- `fig6c_blca_reversal_heatmap.pdf/.png`

**Reproducibility notes.** The natural-product workflow uses released R data objects containing candidate scores and reversal-rank matrices. TCGA tumor and adjacent-normal expression data for LUAD and BLCA were downloaded from UCSC Xena, and the TNBC disease signature is derived from the FUSCC-TNBC cohort as described in the manuscript. Disease signatures are intersected with LINCS landmark genes, compared with UniCure-predicted compound perturbation signatures, and summarized in the released R data objects. The R script displays the top-scoring natural products for each cancer context and visualizes the corresponding reversal-score structure across selected candidates. Experimental validation source records for Fig. 6D-G are provided under `./raw_data/Fig6/Experiment/`, including `Figure 6 (D-F-G)-26.5.29-.xlsx` for cell-viability, colony-formation quantification, and PTC viability values, and `Figure_6E_raw_image.pdf` for the representative colony-formation image record.

## Completion checklist

After running any module, regenerated figures should be inspected in the module-specific `output_plot/` folder. The expected folder relationship is:

```text
./raw_data/Fig*/case_name/                  input files
./Reproducibility/Fig*/case_name/code/      executable scripts
./Reproducibility/Fig*/case_name/output_plot/ regenerated figures and plotting tables
```

The guide above covers the released figure-level modules from Fig. 2 through Fig. 6 and maps each module to its inputs, code entry points, and generated outputs.
