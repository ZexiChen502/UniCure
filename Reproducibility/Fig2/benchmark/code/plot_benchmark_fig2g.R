library(ggplot2)
library(dplyr)
library(patchwork)

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
raw_data_dir <- file.path(repo_root, "raw_data", "fig2", "benchmark")
output_dir <- file.path(module_dir, "output_plot")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

lincs   <- read.csv(file.path(raw_data_dir, "lincs_benchmark_rank_threshold_with_delta.csv"))
sciplex <- read.csv(file.path(raw_data_dir, "sciplex_benchmark_rank_threshold_with_delta.csv"))

lincs$Threshold   <- as.numeric(lincs$Threshold)
sciplex$Threshold <- as.numeric(sciplex$Threshold)

model_order <- c("PRnet", "TranSiGen", "UniCure")
lincs$Model   <- factor(lincs$Model,   levels = model_order)
sciplex$Model <- factor(sciplex$Model, levels = model_order)

summarize_data <- function(df) {
  df %>%
    group_by(Model, Threshold) %>%
    summarise(
      mean_pearson = mean(Delta_r, na.rm = TRUE),
      sd_pearson   = sd(Delta_r, na.rm = TRUE),
      n            = n(),
      se           = sd_pearson / sqrt(n),
      ci_lower     = mean_pearson - 1.96 * se,
      ci_upper     = mean_pearson + 1.96 * se,
      .groups      = "drop"
    )
}

lincs_plot   <- summarize_data(lincs)
sciplex_plot <- summarize_data(sciplex)

my_colors <- c(
  "UniCure"  = "#FF7517",
  "PRnet"    = "#8FD14F",
  "TranSiGen" = "#46C3DB"
)

common_theme <- theme_classic() +
  theme(
    text             = element_text(size = 14, color = "black"),
    axis.text        = element_text(size = 12, color = "black"),
    axis.title       = element_text(size = 14, face = "bold"),
    legend.position  = "top",
    legend.title     = element_blank(),
    legend.text      = element_text(size = 12),
    legend.background = element_rect(fill = "transparent"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5, linetype = "dashed"),
    axis.line        = element_line(linewidth = 0.8),
    plot.title       = element_text(size = 14, face = "bold", hjust = 0.5)
  )

make_panel <- function(plot_data, y_limits = NULL, title = "") {
  p <- ggplot(plot_data, aes(x = Threshold, y = mean_pearson, color = Model, group = Model)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = Model),
                alpha = 0.15, color = NA) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2, shape = 16) +
    scale_color_manual(values = my_colors) +
    scale_fill_manual(values = my_colors) +
    scale_x_continuous(
      name   = "Top % of Genes (Ranked by Sensitivity)",
      breaks = seq(0.1, 1.0, 0.1),
      labels = paste0(seq(10, 100, 10), "%")
    ) +
    scale_y_continuous(name = "Delta Pearson Correlation") +
    ggtitle(title) +
    common_theme

  if (!is.null(y_limits)) {
    p <- p + coord_cartesian(ylim = y_limits)
  }
  p
}

p_left  <- make_panel(lincs_plot,   title = "LINCS 2020 (Bulk, 50 cell lines, n=5173)")
p_right <- make_panel(sciplex_plot, title = "sci-Plex 3 (Single-cell, 3 cell lines, n=226)")

p_combined <- p_left + p_right +
  plot_layout(guides = "collect") &
  theme(legend.position = "top")

ggsave(file.path(output_dir, "fig2g_benchmark_lincs.pdf"),
       plot = p_left, width = 7, height = 5)

ggsave(file.path(output_dir, "fig2g_benchmark_lincs.png"),
       plot = p_left, width = 7, height = 5, dpi = 300)

ggsave(file.path(output_dir, "fig2g_benchmark_sciplex.pdf"),
       plot = p_right, width = 7, height = 5)

ggsave(file.path(output_dir, "fig2g_benchmark_sciplex.png"),
       plot = p_right, width = 7, height = 5, dpi = 300)

ggsave(file.path(output_dir, "fig2g_benchmark_combined.pdf"),
       plot = p_combined, width = 14, height = 5)

ggsave(file.path(output_dir, "fig2g_benchmark_combined.png"),
       plot = p_combined, width = 14, height = 5, dpi = 300)

cat("All benchmark plots saved to:", output_dir, "\n")
cat("\n--- LINCS summary ---\n")
print(lincs_plot)
cat("\n--- sci-Plex summary ---\n")
print(sciplex_plot)
