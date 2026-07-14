library(ggplot2)
library(dplyr)
library(readr)
library(survival)

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(script_arg) > 0) dirname(normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/")) else getwd()
module_dir <- if (basename(script_dir) == "code") dirname(script_dir) else script_dir
repo_root <- normalizePath(file.path(module_dir, "..", "..", ".."), winslash = "/", mustWork = TRUE)
raw_dir <- file.path(repo_root, "raw_data", "Fig5", "survival_analysis")
out_dir <- file.path(module_dir, "output_plot")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Fig. 5L-O and Fig. S14I-K Kaplan-Meier plotting workflow:
#   - Read released patient-level survival and drug-rank tables for each cancer type.
#   - Treat each row as one sample in the survival analysis.
#   - Use days_to_death when available, otherwise days_to_last_followup, as survival time.
#   - Treat observed death records as events and last-follow-up records as censored observations.
#   - drug_rank is normalized as raw_rank divided by the total number of ranked drugs; lower values indicate higher priority.
#   - Apply the manuscript top-30% grouping rule and calculate log-rank p-values with survdiff.

cancer_map <- list(
  LUAD = list(fig = "fig5l"),
  BRCA = list(fig = "fig5m"),
  BLCA = list(fig = "fig5n"),
  LUSC = list(fig = "fig5o"),
  COAD = list(fig = "figs14i"),
  KIRC = list(fig = "figs14j"),
  LIHC = list(fig = "figs14k")
)

COLORS <- c("High Rank" = "#FF7517", "Low Rank" = "#46C3DB")

for (cancer in names(cancer_map)) {
  raw_df <- read_csv(file.path(raw_dir, paste0(cancer, "_combined_survive.csv")),
                     show_col_types = FALSE) %>%
    mutate(
      days_to_death         = as.numeric(days_to_death),
      days_to_last_followup = as.numeric(days_to_last_followup),
      drug_rank             = as.numeric(drug_rank)
    )

  df <- raw_df %>%
    transmute(
      avg_drug_rank         = drug_rank,
      days_to_death         = days_to_death,
      days_to_last_followup = days_to_last_followup
    )

  df <- df %>%
    mutate(
      days_to_death         = ifelse(is.infinite(days_to_death), NA, days_to_death),
      days_to_last_followup = ifelse(is.infinite(days_to_last_followup), NA, days_to_last_followup),
      time   = ifelse(!is.na(days_to_death), days_to_death, days_to_last_followup),
      status = ifelse(!is.na(days_to_death), 1L, 0L),
      # drug_rank is a normalized rank ratio (raw rank / total ranked drugs); lower values indicate higher priority.
      # Manuscript grouping rule: patients in the top 30% priority group (avg_drug_rank <= 0.3) are labeled High Rank.
      group  = factor(ifelse(avg_drug_rank <= 0.3, "High Rank", "Low Rank"),
                      levels = c("High Rank", "Low Rank"))
    ) %>%
    filter(!is.na(time), time > 0, !is.na(avg_drug_rank))

  sd    <- survdiff(Surv(time, status) ~ group, data = df)
  p_val <- 1 - pchisq(sd$chisq, df = 1)
  p_label <- if (p_val < 0.001) "p < 0.001" else sprintf("p = %.3f", p_val)

  km    <- survfit(Surv(time, status) ~ group, data = df)
  km_df <- data.frame(
    time  = km$time,
    surv  = km$surv,
    upper = km$upper,
    lower = km$lower,
    group = rep(sub("group=", "", names(km$strata)), km$strata)
  )

  p <- ggplot(km_df, aes(x = time, y = surv, color = group, fill = group)) +
    geom_step(linewidth = 0.8) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
    scale_color_manual(values = COLORS) +
    scale_fill_manual(values = COLORS) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = "Time (Days)", y = "Overall Survival Probability",
         title = paste(cancer, "Kaplan-Meier Survival"),
         color = NULL, fill = NULL) +
    annotate("text", x = Inf, y = Inf, label = p_label,
             hjust = 1.1, vjust = 1.5, size = 4) +
    theme_classic(base_size = 13) +
    theme(legend.position = "top", axis.text = element_text(color = "black"))

  name <- paste0(cancer_map[[cancer]]$fig, "_", tolower(cancer), "_km_survival")
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = 6, height = 5)
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = 6, height = 5, dpi = 300)
  cat(sprintf("%-4s  p=%.4f  saved %s\n", cancer, p_val, name))
}
