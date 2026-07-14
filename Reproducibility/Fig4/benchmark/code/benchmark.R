library(ggplot2)
library(ggpubr)
library(dplyr)
library(readr)

module_dir <- if (basename(getwd()) == "code") dirname(getwd()) else getwd()
repo_root <- normalizePath(file.path(module_dir, "..", "..", ".."), winslash = "/", mustWork = TRUE)
raw_dir <- file.path(repo_root, "raw_data", "fig4", "benchmark")
out_dir <- file.path(module_dir, "output_plot")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Fig. 4H and Fig. S10A-B benchmark plotting workflow:
#   - LUAD maps to Fig. 4H, BLCA to Fig. S10A, and TNBC to Fig. S10B.
#   - Released benchmark tables compare fine-tuned UniCure, zero-shot UniCure,
#     TranSiGen, and PRnet on the same PTC evaluation setting.
#   - The script writes the plotting data used for each panel and regenerates
#     boxplot-plus-jitter summaries with Wilcoxon-test annotations.

cancer_to_figure <- c(LUAD = "fig4h", BLCA = "figs10a", TNBC = "figs10b")
model_levels <- c("UniCure Finetune", "UniCure", "Transigen", "PRnet")
model_labels <- c(
  "UniCure Finetune" = "Fine-tuned UniCure",
  "UniCure" = "Zero-shot UniCure",
  "Transigen" = "TranSiGen",
  "PRnet" = "PRnet"
)
model_colors <- c(
  "UniCure Finetune" = "#DC0000",
  "UniCure" = "#FF7517",
  "Transigen" = "#8FD14F",
  "PRnet" = "#46C3DB"
)

# Base comparisons evaluate zero-shot UniCure against external baselines;
# top comparisons evaluate fine-tuned UniCure against each comparator.
comparisons_base <- list(c("UniCure", "Transigen"), c("UniCure", "PRnet"))
comparisons_top <- list(
  c("UniCure Finetune", "UniCure"),
  c("UniCure Finetune", "Transigen"),
  c("UniCure Finetune", "PRnet")
)

read_benchmark <- function(cancer) {
  data <- read_csv(file.path(raw_dir, paste0(cancer, "_benchmark_data.csv")), show_col_types = FALSE) %>%
    mutate(
      Model = recode(Model, Transigen = "Transigen", .default = Model),
      Correlation_Coefficient = as.numeric(Correlation_Coefficient)
    )

  data %>%
    mutate(
      Correlation_Coefficient = if_else(Model == "PRnet", Correlation_Coefficient, Correlation_Coefficient),
      Model = factor(Model, levels = model_levels)
    )
}

plot_benchmark <- function(data, cancer) {
  max_y <- max(data$Correlation_Coefficient, na.rm = TRUE)

  ggboxplot(
    data,
    x = "Model",
    y = "Correlation_Coefficient",
    color = "Model",
    fill = "white",
    size = 0.3,
    width = 0.5,
    ylab = "Correlation Coefficient",
    xlab = "Model",
    outlier.shape = NA
  ) +
    geom_jitter(
      aes(color = Model),
      size = 0.7,
      alpha = 0.9,
      position = position_jitterdodge(jitter.width = 0.18, dodge.width = 0.75)
    ) +
    stat_compare_means(
      comparisons = comparisons_base,
      method = "wilcox.test",
      label = "p.signif",
      label.y = c(0.50, 0.58),
      tip.length = 0.01
    ) +
    stat_compare_means(
      comparisons = comparisons_top,
      method = "wilcox.test",
      label = "p.signif",
      label.y = c(max_y + 0.08, max_y + 0.16, max_y + 0.24),
      tip.length = 0.01
    ) +
    scale_color_manual(values = model_colors, labels = model_labels, drop = FALSE) +
    scale_x_discrete(labels = model_labels, drop = FALSE) +
    coord_cartesian(ylim = c(0, max_y + 0.32), clip = "off") +
    labs(title = paste0(cancer, " benchmark")) +
    theme_classic(base_size = 18) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 35, hjust = 1, color = "black"),
      axis.text.y = element_text(color = "black"),
      legend.position = "none"
    )
}

for (cancer in names(cancer_to_figure)) {
  data <- read_benchmark(cancer)
  write_csv(data, file.path(out_dir, paste0(cancer_to_figure[[cancer]], "_", tolower(cancer), "_benchmark_plotting_data.csv")))
  p <- plot_benchmark(data, cancer)
  prefix <- paste0(cancer_to_figure[[cancer]], "_", tolower(cancer), "_benchmark")
  ggsave(file.path(out_dir, paste0(prefix, ".pdf")), plot = p, width = 9, height = 8)
  ggsave(file.path(out_dir, paste0(prefix, ".png")), plot = p, width = 9, height = 8, dpi = 300)
}
