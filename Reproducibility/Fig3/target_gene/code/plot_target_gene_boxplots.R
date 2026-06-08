library(dplyr)
library(tidyr)
library(ggplot2)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- normalizePath(sub(file_arg, "", args[grep(file_arg, args)]), winslash = "/", mustWork = FALSE)
if (length(script_path) == 0) {
  module_dir <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
} else {
  module_dir <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
}

repo_root <- normalizePath(file.path(module_dir, "..", "..", ".."), winslash = "/", mustWork = TRUE)
raw_data_dir <- file.path(repo_root, "raw_data", "fig3", "target_gene")
output_dir <- file.path(module_dir, "output_plot")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Fig. 3 target-gene plotting workflow:
#   - absolute tables preserve repeated predicted and observed profiles for the
#     selected disease-associated target genes.
#   - delta tables summarize matched cell / drug / dose conditions before
#     calculating drug-control expression changes.
# This script reshapes the released predicted and observed tables into long
# format and regenerates the boxplot-plus-jitter panels.

plot_prediction_vs_real <- function(pred_file, real_file, cancer_type, measure_type, figure_label, out_path) {
  pred_df <- read.csv(pred_file, check.names = FALSE)
  real_df <- read.csv(real_file, check.names = FALSE)

  if (!identical(colnames(pred_df), colnames(real_df))) {
    stop(paste("Predicted and real gene columns differ for", cancer_type, measure_type))
  }

  pred_df$Sample <- paste0("S", seq_len(nrow(pred_df)))
  real_df$Sample <- paste0("S", seq_len(nrow(real_df)))

  pred_long <- pred_df %>%
    pivot_longer(-Sample, names_to = "Gene", values_to = "Expression") %>%
    mutate(Type = "Predicted")

  real_long <- real_df %>%
    pivot_longer(-Sample, names_to = "Gene", values_to = "Expression") %>%
    mutate(Type = "Real")

  combined_df <- bind_rows(pred_long, real_long)
  combined_df$Gene <- factor(combined_df$Gene, levels = colnames(pred_df)[colnames(pred_df) != "Sample"])
  combined_df$Type <- factor(combined_df$Type, levels = c("Predicted", "Real"))

  set.seed(123)
  jitter_data <- combined_df %>%
    group_by(Gene, Type) %>%
    sample_n(size = min(100, n()), replace = FALSE) %>%
    ungroup()

  gene_colors <- c("Predicted" = "#FF7517", "Real" = "#46C3DB")
  y_label <- if (measure_type == "delta") {
    expression(paste(Delta, " Expression (Drug - Control)"))
  } else {
    "Expression"
  }

  p <- ggplot(combined_df, aes(x = Gene, y = Expression, color = Type)) +
    geom_boxplot(
      fill = "white",
      size = 0.3,
      outlier.shape = NA,
      position = position_dodge(width = 0.75)
    ) +
    geom_jitter(
      data = jitter_data,
      aes(color = Type),
      size = 0.1,
      alpha = 1,
      position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.75)
    ) +
    scale_color_manual(values = gene_colors) +
    labs(y = y_label, x = "Target Genes", color = "Type") +
    theme_classic() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "top"
    )

  if (measure_type == "delta") {
    p <- p + coord_cartesian(ylim = c(-2, 2)) + scale_y_continuous(breaks = seq(-2, 2, 1))
  }

  base_name <- paste0(figure_label, "_", tolower(cancer_type), "_", measure_type, "_target_gene")
  ggsave(file.path(out_path, paste0(base_name, ".pdf")), plot = p, width = 10, height = 6)
  ggsave(file.path(out_path, paste0(base_name, ".png")), plot = p, width = 10, height = 6, dpi = 300)
  message("Saved ", base_name, " (n = ", nrow(pred_df), ")")
}

figure_map <- list(
  absolute = c(BRCA = "fig3c", COAD = "figs8a", LUAD = "figs8b", PRAD = "figs8c"),
  delta = c(BRCA = "figs8d", COAD = "figs8e", LUAD = "figs8f", PRAD = "figs8g")
)

for (measure_type in names(figure_map)) {
  for (cancer_type in names(figure_map[[measure_type]])) {
    data_path <- file.path(raw_data_dir, measure_type)
    plot_prediction_vs_real(
      pred_file = file.path(data_path, paste0(cancer_type, "_pred_filtered.csv")),
      real_file = file.path(data_path, paste0(cancer_type, "_real_filtered.csv")),
      cancer_type = cancer_type,
      measure_type = measure_type,
      figure_label = figure_map[[measure_type]][[cancer_type]],
      out_path = output_dir
    )
  }
}

message("All target-gene figures generated successfully: ", output_dir)
