library(ggplot2)
library(gghalves)
library(grid)

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
  }
  return(normalizePath(getwd(), winslash = "/", mustWork = TRUE))
}

script_path <- get_script_path()
script_dir <- if (grepl("\\.R$", script_path, ignore.case = TRUE)) dirname(script_path) else script_path
module_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = TRUE)
repo_root <- normalizePath(file.path(module_dir, "..", "..", ".."), winslash = "/", mustWork = TRUE)
raw_dir <- file.path(repo_root, "raw_data", "fig2", "metrics")
out_dir <- file.path(module_dir, "output_plot")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

metric_levels <- c("Pearson", "Spearman", "R2")
fill_colors <- c("Pearson" = "#D55E00", "Spearman" = "#56B4E9", "R2" = "#009E73")
line_colors <- c("Pearson" = "#D55E00", "Spearman" = "#0072B2", "R2" = "#009E73")

dataset_map <- list(
  list(file = "lincs_correlation_results_melted.csv", stem = "lincs", title = "Lincs 2020"),
  list(file = "sciplex3_correlation_results_melted.csv", stem = "sciplex3", title = "Sciplex 3"),
  list(file = "sciplex4_correlation_results_melted.csv", stem = "sciplex4", title = "Sciplex 4")
)

prepare_data <- function(file_name, title) {
  data <- read.csv(file.path(raw_dir, file_name), stringsAsFactors = FALSE)
  data <- data[, c("Metric", "Value")]
  data$Metric <- factor(data$Metric, levels = metric_levels)
  data$Dataset <- factor(title, levels = vapply(dataset_map, function(x) x$title, character(1)))
  return(data)
}

base_plot <- function(data) {
  ggplot(data, aes(x = Metric, y = Value, fill = Metric, color = Metric)) +
    scale_fill_manual(values = fill_colors) +
    scale_colour_manual(values = line_colors) +
    geom_half_violin(
      position = position_nudge(x = 0.1, y = 0),
      side = "r",
      adjust = 1.2,
      trim = FALSE,
      color = NA,
      alpha = 0.6
    ) +
    geom_boxplot(
      position = position_nudge(x = -0.2, y = 0),
      outlier.shape = NA,
      width = 0.4,
      alpha = 0.8
    ) +
    theme_light() +
    theme(
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(linetype = "dotted", color = "lightgray", linewidth = 0.5),
      axis.ticks.length = unit(-0.20, "cm"),
      axis.text.x = element_text(margin = margin(0.35, 0.35, 0.35, 0.35, unit = "cm")),
      axis.text.y = element_text(margin = margin(0.35, 0.35, 0.35, 0.35, unit = "cm")),
      text = element_text(size = 18)
    ) +
    scale_y_continuous(breaks = seq(0.0, 1.0, by = 0.2)) +
    coord_cartesian(ylim = c(0.0, 1.0))
}

save_single_plot <- function(data, stem, title) {
  p <- base_plot(data) +
    labs(title = title, x = NULL, y = "Test Metrics")

  ggsave(
    filename = file.path(out_dir, paste0(stem, "_Metric_on_test_datasets.pdf")),
    plot = p,
    width = 6.8,
    height = 5.2
  )
  ggsave(
    filename = file.path(out_dir, paste0(stem, "_Metric_on_test_datasets.png")),
    plot = p,
    width = 6.8,
    height = 5.2,
    dpi = 300
  )
}

all_data <- do.call(
  rbind,
  lapply(dataset_map, function(info) prepare_data(info$file, info$title))
)

for (info in dataset_map) {
  data <- prepare_data(info$file, info$title)
  save_single_plot(data, info$stem, info$title)
}

combined_plot <- base_plot(all_data) +
  facet_wrap(~Dataset, nrow = 1) +
  labs(x = NULL, y = "Test Metrics") +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(size = 18)
  )

ggsave(
  filename = file.path(out_dir, "all_Metric_on_test_datasets.pdf"),
  plot = combined_plot,
  width = 12.8,
  height = 4.4
)
ggsave(
  filename = file.path(out_dir, "all_Metric_on_test_datasets.png"),
  plot = combined_plot,
  width = 12.8,
  height = 4.4,
  dpi = 300
)

readme_path <- file.path(out_dir, "README_metrics_plots.txt")
writeLines(
  c(
    "Generated metric plots for Fig2 metrics panels.",
    "Each panel summarizes Pearson, Spearman, and R2 values using half-violin and boxplot overlays.",
    "",
    "Source raw_data files:",
    "- lincs_correlation_results_melted.csv",
    "- sciplex3_correlation_results_melted.csv",
    "- sciplex4_correlation_results_melted.csv",
    "",
    "Outputs:",
    "- lincs_Metric_on_test_datasets.pdf / .png",
    "- sciplex3_Metric_on_test_datasets.pdf / .png",
    "- sciplex4_Metric_on_test_datasets.pdf / .png",
    "- all_Metric_on_test_datasets.pdf / .png"
  ),
  con = readme_path
)

cat("Metric plots generated successfully in:", out_dir, "\n")
