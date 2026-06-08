library(ggplot2)
library(dplyr)
library(readr)

module_dir <- if (basename(getwd()) == "code") dirname(getwd()) else getwd()
repo_root <- normalizePath(file.path(module_dir, "..", "..", ".."), winslash = "/", mustWork = TRUE)
raw_dir <- file.path(repo_root, "raw_data", "fig5", "patient_stratification")
out_dir <- file.path(module_dir, "output_plot")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# Upstream differential-expression and pathway-enrichment workflow
# =============================================================================
# This plotting workflow reads {cancer}_diff_exp.csv and
# {cancer}_pathway_results.csv tables from raw_data. The code block below shows
# the analysis logic used to derive those tables.
# library(DESeq2); library(clusterProfiler); library(org.Hs.eg.db)
#
# # DESeq2 on filtered tumor expression vs. cluster assignment
# dds <- DESeqDataSetFromMatrix(countData = mRNA, colData = colData, design = ~index)
# dds <- DESeq(dds)
# result <- data.frame(results(dds)) %>%
#   mutate(group = case_when(
#     log2FoldChange > 1  & padj <= 0.1 ~ "UP",
#     log2FoldChange < -1 & padj <= 0.1 ~ "DOWN",
#     TRUE ~ "NOT_CHANGE"
#   ))
# write.csv(result, "{cancer}_diff_exp.csv")
#
# # Pathway enrichment on UP+DOWN genes
# genes_entrez <- bitr(deg_genes, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
# kegg_result  <- enrichKEGG(gene=genes_entrez$ENTREZID, organism="hsa")
# go_bp        <- enrichGO(gene=genes_entrez$ENTREZID, OrgDb=org.Hs.eg.db, ont="BP", ...)
# go_mf        <- enrichGO(..., ont="MF", ...)
# go_cc        <- enrichGO(..., ont="CC", ...)
# combined_df  <- bind_rows(list(KEGG=kegg_df, GO_BP=go_bp, GO_CC=go_cc, GO_MF=go_mf), .id="category")
# # top 10 by p.adjust saved to {cancer}_pathway_results.csv

# =============================================================================
# Released-table plotting workflow
# =============================================================================

# --- Pathway bubble plots (Fig. 5D: BRCA, Fig. 5F: KIRC) ---

cancer_to_fig <- c(BRCA = "fig5d", KIRC = "fig5f")

for (cancer in names(cancer_to_fig)) {
  df <- read_csv(file.path(raw_dir, paste0(cancer, "_pathway_results.csv")),
                 show_col_types = FALSE)

  df <- df %>%
    arrange(p.adjust) %>%
    slice_head(n = 10) %>%
    mutate(Description = factor(Description, levels = rev(Description)))

  p <- ggplot(df, aes(x = Description, y = `-log10Pvalue`, size = Count, color = p.adjust)) +
    geom_point(alpha = 0.8) +
    scale_size(range = c(3, 10)) +
    scale_color_gradientn(colors = c("#c04851", "#fcc307", "#22a2c3")) +
    coord_flip() +
    labs(x = "", y = "-log10(Adjusted P-value)", size = "Gene Count",
         color = "p.adjust", title = paste(cancer, "Pathway Enrichment")) +
    theme_classic(base_size = 14) +
    theme(
      axis.text.y  = element_text(size = 12, color = "black"),
      axis.text.x  = element_text(color = "black"),
      plot.title   = element_text(hjust = 0.5)
    )

  fig_name <- paste0(cancer_to_fig[[cancer]], "_", tolower(cancer), "_pathway_enrichment")
  ggsave(file.path(out_dir, paste0(fig_name, ".pdf")), plot = p, width = 9, height = 5)
  ggsave(file.path(out_dir, paste0(fig_name, ".png")), plot = p, width = 9, height = 5, dpi = 300)
}

# --- Drug lollipop plots (Fig. 5E: BRCA, Fig. 5G: KIRC) ---
# The diff_drug tables record cluster-level drug-rank differences and adjusted p-values.

target_drugs <- list(
  BRCA = c("clotiapine", "bretazenil", "fenthion", "gavestinel", "wb-4101",
            "mead-ethanolamide", "am-404", "bml-190", "talnetant", "antalarmin", "aprepitant"),
  KIRC = c("tacrolimus", "nadide", "pirfenidone", "ticlopidine", "bromebric-acid",
            "coumaric-acid", "eugenitol", "ppt", "indoximod")
)

cancer_to_fig_drug <- c(BRCA = "fig5e", KIRC = "fig5g")

for (cancer in names(target_drugs)) {
  df <- read_csv(file.path(raw_dir, paste0(cancer, "_diff_drug.csv")),
                 show_col_types = FALSE)

  plot_data <- df %>%
    mutate(Gene_lower = tolower(Feature)) %>%
    filter(Gene_lower %in% target_drugs[[cancer]]) %>%
    arrange(log2FoldChange_1_vs_0) %>%
    mutate(
      Gene  = factor(Feature, levels = Feature),
      Group = factor(ifelse(log2FoldChange_1_vs_0 > 0, "Positive", "Negative"),
                     levels = c("Positive", "Negative"))
    )

  p <- ggplot(plot_data, aes(x = Gene, y = log2FoldChange_1_vs_0)) +
    geom_segment(aes(x = Gene, xend = Gene, y = 0, yend = log2FoldChange_1_vs_0,
                     color = Group), linewidth = 1.2, alpha = 0.8) +
    geom_point(aes(color = Group, size = -log10(adjusted_p_value)), alpha = 1) +
    scale_color_manual(values = c("Positive" = "#46C3DB", "Negative" = "#FF7517")) +
    coord_flip() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.5) +
    labs(x = "Drug Name", y = "Log2 Fold Change", size = "-log10(adj.P)") +
    theme_bw(base_size = 13) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border     = element_rect(colour = "black", fill = NA, linewidth = 1),
      axis.text.y      = element_text(size = 12, color = "black", face = "italic"),
      axis.title       = element_text(size = 12, face = "bold"),
      legend.position  = "right"
    )

  fig_name <- paste0(cancer_to_fig_drug[[cancer]], "_", tolower(cancer), "_diff_drug_lollipop")
  ggsave(file.path(out_dir, paste0(fig_name, ".pdf")), plot = p, width = 6, height = 6)
  ggsave(file.path(out_dir, paste0(fig_name, ".png")), plot = p, width = 6, height = 6, dpi = 300)
}
