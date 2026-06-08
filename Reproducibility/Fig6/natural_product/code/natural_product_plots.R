library(ggplot2)
library(dplyr)
library(forcats)
library(RColorBrewer)

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
raw_dir <- file.path(repo_root, "raw_data", "fig6", "natural_product")
out_dir <- file.path(module_dir, "output_plot")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Fig. 6B-C natural-product plotting workflow:
#   - Figure6B Rdata files contain natural-product connectivity-style scores and
#     p-values for TNBC, LUAD, and BLCA candidate compounds.
#   - Figure6C Rdata files contain drug-versus-disease signature reversal-rank
#     matrices for the same cancer contexts.
# This script reads the released Rdata inputs, regenerates top-candidate dot
# plots for Fig. 6B, and regenerates reversal-rank heatmaps for Fig. 6C.

plotcs <- function(data, x = "scaled_score", y = "drug", colby = "pvalue", color = c("red")) {
  score <- data[[x]]
  yvar <- data[[y]]
  pval <- data[[colby]]
  ggplot(data, aes(score, fct_reorder(yvar, -score))) +
    geom_segment(aes(xend = 0, yend = yvar), linetype = 2) +
    geom_point(aes(col = pval), size = 4.5) +
    scale_colour_gradientn(colours = color) +
    labs(x = "Score", y = "Compounds") +
    theme_minimal() +
    theme(panel.background = element_rect(colour = "black", linewidth = 0.5), legend.position = "none")
}

# Fig. 6B: top-20 natural-product score dot plots for each cancer context.
fig6b_map <- list(
  TNBC = list(file = "MB231_np_cmap.Rdata", obj = "MB231_np_cmap", fig = "fig6b_tnbc"),
  LUAD = list(file = "A549_np_cmap.Rdata", obj = "A549_np_cmap", fig = "fig6b_luad"),
  BLCA = list(file = "Blca_5637_np_cmap.Rdata", obj = "Blca_5637_np_cmap", fig = "fig6b_blca")
)

for (cancer in names(fig6b_map)) {
  m <- fig6b_map[[cancer]]
  load(file.path(raw_dir, "figure6b", m$file))
  dat <- get(m$obj)
  top20 <- dat[1:20, ]
  p <- plotcs(top20)
  ggsave(file.path(out_dir, paste0(m$fig, "_dotplot.pdf")), p, width = 5, height = 4)
  ggsave(file.path(out_dir, paste0(m$fig, "_dotplot.png")), p, width = 5, height = 4, dpi = 300)
}

# Fig. 6C: reversal-rank heatmaps comparing cancer signatures with candidate drugs.
heatmap_colors <- colorRampPalette(c("#407EBE", "#FFF5F7", "#D34F41"))(1000)
fig6c_map <- list(
  TNBC = list(file = "MB231_drug_rank.Rdata", cancer_label = "TNBC", fig = "fig6c_tnbc"),
  LUAD = list(file = "A549_drug_rank.Rdata", cancer_label = "LUAD", fig = "fig6c_luad"),
  BLCA = list(file = "5637_drug_rank.Rdata", cancer_label = "BLCA", fig = "fig6c_blca")
)

plot_heatmap <- function(out_file, axis_labels) {
  par(mar = c(13, 6, 2, 0.5))
  image(t(drug_dz_signature_rank2), col = heatmap_colors, axes = FALSE)
  axis(1, at = seq(0, 1, length.out = ncol(drug_dz_signature_rank2)), labels = FALSE)
  text(
    x = seq(0, 1, length.out = ncol(drug_dz_signature_rank2)),
    y = -0.05,
    labels = axis_labels,
    srt = 45,
    pos = 2,
    offset = 0.05,
    xpd = TRUE,
    cex = 1.2
  )
}

for (cancer in names(fig6c_map)) {
  m <- fig6c_map[[cancer]]
  load(file.path(raw_dir, "figure6c", m$file))
  drug_names <- colnames(drug_dz_signature_rank2)[-1]
  axis_labels <- c(m$cancer_label, as.character(drug_names))

  pdf(file.path(out_dir, paste0(m$fig, "_reversal_heatmap.pdf")))
  plot_heatmap(file.path(out_dir, paste0(m$fig, "_reversal_heatmap.pdf")), axis_labels)
  dev.off()

  png(file.path(out_dir, paste0(m$fig, "_reversal_heatmap.png")), width = 800, height = 600, res = 150)
  plot_heatmap(file.path(out_dir, paste0(m$fig, "_reversal_heatmap.png")), axis_labels)
  dev.off()
}

message("All Fig. 6 natural-product plots generated successfully: ", out_dir)
