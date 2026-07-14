library(ggplot2)
library(dplyr)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) {
  normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
script_dir <- if (grepl("[.]R$", script_path, ignore.case = TRUE)) dirname(script_path) else script_path
module_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = TRUE)
repo_root <- normalizePath(file.path(module_dir, "..", "..", ".."), winslash = "/", mustWork = TRUE)
raw_data_dir <- file.path(repo_root, "raw_data", "fig3", "cell_state_specific")
output_dir <- file.path(module_dir, "output_plot")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Upstream enrichment and differential-expression workflow:
#   1. Compare predicted drug-response profiles with the corresponding reference
#      condition to produce differential-expression tables.
#   2. Use significant response genes for pathway enrichment and record adjusted
#      p-values, gene counts, and pathway descriptions.
#   3. Release the resulting *_pathway_results.csv and *_diff_expr_results.csv
#      tables for regeneration of Fig. 3E-F and Fig. S10A-B.
# This script reads those released tables and regenerates the pathway dot plots
# and differential-expression volcano plots.

plot_pathway <- function(drug_name, cell_line, figure_label) {
  in_file <- file.path(raw_data_dir, paste0(drug_name, "_", cell_line, "_pathway_results.csv"))
  pathway_df <- read.csv(in_file, check.names = FALSE)
  pathway_df <- pathway_df[, names(pathway_df) != "", drop = FALSE]

  top_df <- pathway_df %>%
    mutate(
      Count_numeric = suppressWarnings(as.numeric(Count)),
      Rich_numeric = suppressWarnings(as.numeric(gene_num.rich)),
      Size = dplyr::coalesce(Count_numeric, Rich_numeric),
      PAdjust = suppressWarnings(as.numeric(p.adjust)),
      PlotDescription = paste0(Description, " [", ID, "]")
    ) %>%
    filter(!is.na(PAdjust), !is.na(Size), !is.na(Description), Description != "") %>%
    arrange(PAdjust) %>%
    slice_head(n = 10) %>%
    mutate(PlotDescription = factor(PlotDescription, levels = rev(unique(PlotDescription))))

  p <- ggplot(top_df, aes(x = PlotDescription, y = -log10(PAdjust), size = Size, color = PAdjust)) +
    geom_point(alpha = 0.8) +
    scale_size(range = c(3, 10)) +
    scale_color_gradientn(colors = c("#c04851", "#fcc307", "#22a2c3")) +
    coord_flip() +
    labs(
      title = paste0(drug_name, " ", cell_line, " pathway enrichment"),
      x = "",
      y = "-log10(Adjusted P-value)",
      size = "Gene Count",
      color = "Adjusted P-value"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 12),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color = "black"),
      plot.title = element_text(hjust = 0.5)
    )

  base_name <- paste0(figure_label, "_", drug_name, "_", cell_line, "_pathway")
  ggsave(file.path(output_dir, paste0(base_name, ".pdf")), plot = p, width = 10, height = 8)
  ggsave(file.path(output_dir, paste0(base_name, ".png")), plot = p, width = 10, height = 8, dpi = 300)
  message("Saved ", base_name)
}

plot_volcano <- function(drug_name, cell_line, figure_label) {
  in_file <- file.path(raw_data_dir, paste0(drug_name, "_", cell_line, "_diff_expr_results.csv"))
  diffexp <- read.csv(in_file, check.names = FALSE)

  volcano_data <- data.frame(
    Gene = diffexp$gene,
    LogFC = diffexp$log2FoldChange,
    PValue = diffexp$padj,
    Group = factor(diffexp$group, levels = c("DOWN", "NOT_SIG", "UP"))
  )

  p <- ggplot(volcano_data, aes(x = LogFC, y = -log10(PValue), colour = Group)) +
    scale_color_manual(values = c("DOWN" = "#46C3DB", "NOT_SIG" = "gray", "UP" = "#FF7517"), na.value = "gray") +
    geom_point(alpha = 0.8, size = 1.5) +
    labs(
      title = paste0(drug_name, " ", cell_line, " DEGs"),
      x = "Log2(Fold Change)",
      y = "-log10(P-adjust)",
      color = "Group"
    ) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black", linewidth = 0.2) +
    geom_vline(xintercept = 0.3, linetype = "dashed", color = "black", linewidth = 0.2) +
    geom_vline(xintercept = -0.3, linetype = "dashed", color = "black", linewidth = 0.2) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5),
      legend.position = "right"
    )

  base_name <- paste0(figure_label, "_", drug_name, "_", cell_line, "_volcano")
  ggsave(file.path(output_dir, paste0(base_name, ".pdf")), plot = p, width = 8, height = 6)
  ggsave(file.path(output_dir, paste0(base_name, ".png")), plot = p, width = 8, height = 6, dpi = 300)
  message("Saved ", base_name)
}

plot_pathway("alpelisib", "MCF7", "fig3e")
plot_pathway("copanlisib", "MCF7", "fig3f")
plot_volcano("alpelisib", "MCF7", "figs7a")
plot_volcano("copanlisib", "MCF7", "figs7b")

message("All cell-state-specific plots generated successfully: ", output_dir)
