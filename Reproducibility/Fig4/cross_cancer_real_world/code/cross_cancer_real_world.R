library(ggplot2)
library(ggpubr)
library(dplyr)
library(readr)

module_dir <- if (basename(getwd()) == "code") dirname(getwd()) else getwd()
repo_root <- normalizePath(file.path(module_dir, "..", "..", ".."), winslash = "/", mustWork = TRUE)
raw_dir <- file.path(repo_root, "raw_data", "fig4", "cross_cancer_real_world")
out_dir <- file.path(module_dir, "output_plot")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Fig. 4I cross-cancer and real-world evaluation workflow:
#   - LUAD, BLCA, and TNBC tables contain cross-cancer fine-tuning Pearson
#     correlation distributions for the corresponding PTC cancer type.
#   - The real-world table contains correlations from independent clinical
#     transcriptomic profiles.
#   - The script combines these released tables, writes the plotting data, and
#     regenerates the Fig. 4I boxplot-plus-jitter panel.

# Read Pearson values from released CSV files with either named or single-column formats.
read_pearson <- function(filename) {
  df <- read_csv(file.path(raw_dir, filename), show_col_types = FALSE)
  if ("Pearson" %in% names(df)) {
    return(as.numeric(df$Pearson))
  }
  if ("X0" %in% names(df)) {
    return(as.numeric(df$X0))
  }
  as.numeric(df[[1]])
}

data <- bind_rows(
  tibble(Cancer = "LUAD", Correlation_Coefficient = read_pearson("UniCure_finetune_LUAD_pearson.csv")),
  tibble(Cancer = "BLCA", Correlation_Coefficient = read_pearson("UniCure_finetune_BLCA_pearson.csv")),
  tibble(Cancer = "TNBC", Correlation_Coefficient = read_pearson("UniCure_finetune_TNBC_pearson.csv")),
  tibble(Cancer = "real-world", Correlation_Coefficient = read_pearson("real_world_abs_pearson.csv"))
) %>%
  mutate(Cancer = factor(Cancer, levels = c("LUAD", "BLCA", "TNBC", "real-world")))

write_csv(data, file.path(out_dir, "fig4i_cross_cancer_real_world_plotting_data.csv"))

p <- ggboxplot(
  data,
  x = "Cancer",
  y = "Correlation_Coefficient",
  color = "Cancer",
  fill = "white",
  size = 0.6,
  width = 0.5,
  outlier.shape = NA,
  xlab = "Cancer",
  ylab = "Correlation Coefficient"
) +
  geom_jitter(aes(color = Cancer), size = 1.0, alpha = 0.8, width = 0.1) +
  scale_color_manual(values = c("LUAD" = "#ED7D31", "BLCA" = "#8FD14F", "TNBC" = "#DC0000", "real-world" = "#46C3DB")) +
  coord_cartesian(ylim = c(0.15, 1.05)) +
  labs(title = "Cross-Cancer and Real-World Evaluation") +
  theme_classic(base_size = 18) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_text(color = "black"),
    axis.line = element_line(linewidth = 0.6),
    axis.ticks = element_line(linewidth = 0.6),
    legend.position = "top",
    legend.title = element_blank()
  )

ggsave(file.path(out_dir, "fig4i_cross_cancer_real_world_evaluation.pdf"), plot = p, width = 8, height = 6)
ggsave(file.path(out_dir, "fig4i_cross_cancer_real_world_evaluation.png"), plot = p, width = 8, height = 6, dpi = 300)
