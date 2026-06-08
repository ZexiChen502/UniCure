library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)

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
raw_dir <- file.path(repo_root, "raw_data", "fig4", "finetune_ablation")
out_dir <- file.path(module_dir, "output_plot")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Fig. 4A-G fine-tuning and ablation plotting workflow:
#   - Fig. 4A-F summarize LUAD, BLCA, and TNBC fine-tuning performance across
#     released PTC training fractions.
#   - Fig. 4G compares the UniCure model with its ablation variant under the
#     smallest fine-tuning fraction and records Wilcoxon-test annotations.
# The script reads released correlation tables, writes plotting summaries, and
# regenerates the corresponding boxplot, line-summary, and ablation panels.

# Released table columns encode the PTC training fractions used for fine-tuning.
training_levels <- c("0.05", "0.1", "0.2", "0.4", "0.6", "0.8")
training_labels <- c("5%", "10%", "20%", "40%", "60%", "80%")
cancers <- c("LUAD", "BLCA", "TNBC")
fig_labels <- list(
  absolute = c(LUAD = "fig4a", BLCA = "fig4b", TNBC = "fig4c"),
  delta = c(LUAD = "fig4d", BLCA = "fig4e", TNBC = "fig4f")
)

read_training_table <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  missing_cols <- setdiff(c("seed", training_levels), names(df))
  if (length(missing_cols) > 0) {
    stop("Missing columns in ", basename(path), ": ", paste(missing_cols, collapse = ", "))
  }
  df %>%
    pivot_longer(all_of(training_levels), names_to = "TrainingFraction", values_to = "Correlation_Coefficient") %>%
    mutate(
      Training_Set_Size = factor(TrainingFraction, levels = training_levels, labels = training_labels),
      Correlation_Coefficient = as.numeric(Correlation_Coefficient)
    )
}

plot_training_size <- function(df, cancer, metric_label) {
  ggplot(df, aes(x = Training_Set_Size, y = Correlation_Coefficient)) +
    geom_boxplot(outlier.shape = NA, width = 0.55, color = "black", fill = "white") +
    geom_jitter(aes(color = Training_Set_Size), width = 0.12, size = 3, alpha = 0.9, show.legend = FALSE) +
    stat_summary(fun = mean, geom = "line", aes(group = 1), color = "black", linewidth = 0.7, linetype = "dashed") +
    stat_summary(fun = mean, geom = "point", color = "black", size = 2) +
    scale_color_manual(values = c("#DC0000", "#FF7517", "#8FD14F", "#46C3DB", "#7E57C2", "#8A8A8A")) +
    labs(
      title = paste0(cancer, " ", metric_label, " fine-tuning performance"),
      x = "Training Set Size",
      y = "Correlation Coefficient"
    ) +
    theme_classic(base_size = 18) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text = element_text(color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.6),
      axis.ticks = element_line(color = "black", linewidth = 0.6)
    )
}

save_plot <- function(plot, prefix) {
  ggsave(file.path(out_dir, paste0(prefix, ".pdf")), plot = plot, width = 8, height = 6)
  ggsave(file.path(out_dir, paste0(prefix, ".png")), plot = plot, width = 8, height = 6, dpi = 300)
}

for (cancer in cancers) {
  absolute_data <- read_training_table(file.path(raw_dir, paste0(cancer, "_finetune_results.csv")))
  absolute_plot <- plot_training_size(absolute_data, cancer, "absolute")
  save_plot(absolute_plot, paste0(fig_labels$absolute[[cancer]], "_", tolower(cancer), "_finetune_absolute"))

  delta_data <- read_training_table(file.path(raw_dir, paste0(cancer, "_finetune_result_delta.csv")))
  delta_plot <- plot_training_size(delta_data, cancer, "delta")
  save_plot(delta_plot, paste0(fig_labels$delta[[cancer]], "_", tolower(cancer), "_finetune_delta"))
}

significance_label <- function(p_value) {
  case_when(
    p_value < 0.001 ~ "***",
    p_value < 0.01 ~ "**",
    p_value < 0.05 ~ "*",
    TRUE ~ "NS"
  )
}

# Fig. 4G uses the 5% fine-tuning column to compare UniCure and ablation results.
read_pair <- function(cancer) {
  main <- read_csv(file.path(raw_dir, paste0(cancer, "_finetune_results.csv")), show_col_types = FALSE)
  ablation <- read_csv(file.path(raw_dir, paste0(cancer, "_ablation_fine_tune_result.csv")), show_col_types = FALSE)
  bind_rows(
    tibble(CorrCoef = as.numeric(main[["0.05"]]), CancerType = cancer, Treatment = "UniCure"),
    tibble(CorrCoef = as.numeric(ablation[["0.05"]]), CancerType = cancer, Treatment = "UniCure_Ablation")
  )
}

dat <- bind_rows(lapply(cancers, read_pair)) %>%
  mutate(
    CancerType = factor(CancerType, levels = cancers),
    Treatment = factor(Treatment, levels = c("UniCure", "UniCure_Ablation"))
  )

write_csv(dat, file.path(out_dir, "fig4g_ablation_plotting_data.csv"))

summary_data <- dat %>%
  group_by(CancerType, Treatment) %>%
  summarise(Mean = mean(CorrCoef, na.rm = TRUE), SD = sd(CorrCoef, na.rm = TRUE), N = n(), .groups = "drop")

sig_results <- dat %>%
  group_by(CancerType) %>%
  summarise(
    p_value = wilcox.test(CorrCoef[Treatment == "UniCure"], CorrCoef[Treatment == "UniCure_Ablation"])$p.value,
    y_position = max(CorrCoef, na.rm = TRUE) + 0.08,
    .groups = "drop"
  ) %>%
  mutate(sig_label = significance_label(p_value), x = as.numeric(factor(CancerType, levels = cancers)), xmin = x - 0.2, xmax = x + 0.2)

write_csv(summary_data, file.path(out_dir, "fig4g_ablation_summary_mean_sd.csv"))
write_csv(sig_results, file.path(out_dir, "fig4g_ablation_wilcoxon.csv"))

p <- ggplot(summary_data, aes(x = CancerType, y = Mean, fill = Treatment)) +
  geom_col(position = position_dodge(0.8), width = 0.7, alpha = 0.85) +
  geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD), position = position_dodge(0.8), width = 0.2, linewidth = 0.5) +
  geom_point(
    data = dat,
    aes(x = CancerType, y = CorrCoef, color = Treatment, group = Treatment),
    position = position_jitterdodge(jitter.width = 0.08, dodge.width = 0.8),
    size = 2,
    inherit.aes = FALSE
  ) +
  geom_segment(data = sig_results, aes(x = xmin, xend = xmax, y = y_position, yend = y_position), inherit.aes = FALSE) +
  geom_text(data = sig_results, aes(x = x, y = y_position + 0.025, label = sig_label), inherit.aes = FALSE, size = 5) +
  scale_fill_manual(values = c("UniCure" = "#FF7517", "UniCure_Ablation" = "#46C3DB")) +
  scale_color_manual(values = c("UniCure" = "#FF7517", "UniCure_Ablation" = "#46C3DB")) +
  coord_cartesian(ylim = c(0, 1.1), clip = "off") +
  labs(x = "Cancer Type", y = "Correlation Coefficient") +
  theme_classic(base_size = 18) +
  theme(
    legend.position = "top",
    axis.text = element_text(color = "black"),
    axis.line = element_line(color = "black", linewidth = 0.6),
    axis.ticks = element_line(color = "black", linewidth = 0.6)
  )

ggsave(file.path(out_dir, "fig4g_finetune_ablation.pdf"), plot = p, width = 8, height = 6)
ggsave(file.path(out_dir, "fig4g_finetune_ablation.png"), plot = p, width = 8, height = 6, dpi = 300)
message("All Fig. 4 fine-tuning and ablation plots generated successfully: ", out_dir)
