#!/usr/bin/env Rscript

# ============================================================
# Reproducibility statistics - LaTeX summary table by method
# Reads the global TSV files written by stats_figures.R
# (label 0 is already excluded in those files)
# ============================================================

# ------------------------------------------------------------
# 0. Libraries, logging, Snakemake objects
# ------------------------------------------------------------

suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(tibble)
    library(knitr)
})

log_con <- file(snakemake@log[[1]], open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")

cat("============================================================\n")
cat("Reproducibility statistics: summary table\n")
cat("============================================================\n\n")

input  <- snakemake@input
output <- snakemake@output

# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

#' Read a TSV file; subject/method cast to character.
read_tsv_safe <- function(file) {
    cat("Reading:", file, "\n")
    read_tsv(file, show_col_types = FALSE, progress = FALSE) %>%
        mutate(across(any_of(c("subject", "method")), as.character))
}

#' Split the descriptive table into TEST and RETEST sides, with
#' suffixed column names (_test / _retest) ready to be joined.
#' @return list(test = tibble, retest = tibble)
split_descriptive <- function(descriptive) {
    side_table <- function(acq, suffix) {
        descriptive %>%
            filter(test == acq) %>%
            transmute(
                method, subject, side,
                "label_HCP_{suffix}"   := basin,
                "IG_{suffix}"          := IG,
                "area_mesh_{suffix}"   := area_mesh,
                "volume_mesh_{suffix}" := volume_mesh,
                "nb_feuilles_{suffix}" := n_leaves_basin,
                "length_{suffix}"      := length_basin,
                "area_basin_{suffix}"  := area_basin,
                "n_cc_basin_{suffix}"  := n_cc_basin
            )
    }

    list(
        test   = side_table("HCP_test", "test"),
        retest = side_table("HCP_retest", "retest")
    )
}

#' Join Hausdorff + matching + descriptive (test / retest) tables.
#' Assumption: mean_nn is the former mean_dist.
build_all_dist <- function(hausdorff, matching, desc_split) {
    hausdorff %>%
        inner_join(
            matching,
            by = c("subject", "side", "label_HCP_test", "label_HCP_retest")
        ) %>%
        left_join(
            desc_split$test,
            by = c("method", "subject", "side", "label_HCP_test")
        ) %>%
        left_join(
            desc_split$retest,
            by = c("method", "subject", "side", "label_HCP_retest")
        ) %>%
        mutate(mean_dist = mean_nn)
}

# ------------------------------------------------------------
# 2. Load global tables and rebuild all_dist
# ------------------------------------------------------------

matching    <- read_tsv_safe(input$matching)
hausdorff   <- read_tsv_safe(input$hausdorff)
descriptive <- read_tsv_safe(input$desc)

all_dist <- build_all_dist(hausdorff, matching, split_descriptive(descriptive))

cat("\n--- all_dist check ---\n")
cat("Rows:", nrow(all_dist), "\n")
cat("Methods:", n_distinct(all_dist$method), "\n")

cat("\nMissing values:\n")
print(
    all_dist %>%
        summarise(
            mean_dist_NA     = sum(is.na(mean_dist)),
            area_test_NA     = sum(is.na(area_basin_test)),
            area_retest_NA   = sum(is.na(area_basin_retest)),
            length_test_NA   = sum(is.na(length_test)),
            length_retest_NA = sum(is.na(length_retest))
        )
)

cat("\nRows per method:\n")
print(count(all_dist, method))

# ------------------------------------------------------------
# 3. Per method / subject / side summary (pooled test + retest)
# ------------------------------------------------------------

all_dist_resume <- all_dist %>%
    group_by(method, subject, side) %>%
    summarise(
        nb_cc_moyen = mean(c(n_cc_basin_test, n_cc_basin_retest), na.rm = TRUE),
        length_moyenne = mean(c(length_test, length_retest), na.rm = TRUE),
        nb_feuilles_moyen = mean(c(nb_feuilles_test, nb_feuilles_retest),
                                 na.rm = TRUE),
        # share of basin area with no distance (mean_dist is NA)
        aire_non_couverte_pourc =
            sum(area_basin_test[is.na(mean_dist)] +
                    area_basin_retest[is.na(mean_dist)], na.rm = TRUE) /
            sum(area_basin_test + area_basin_retest, na.rm = TRUE) * 100,
        nb_bassins = n(),
        nb_bassins_mean_dist_NA = sum(is.na(mean_dist)),
        .groups = "drop"
    )

# ------------------------------------------------------------
# 4. Summary by method (distances only for overlap >= 0.7)
# ------------------------------------------------------------

dist_stats <- all_dist %>%
    filter(overlap >= 0.7) %>%
    group_by(method) %>%
    summarise(
        mean_dist_mean   = mean(mean_dist, na.rm = TRUE),
        mean_dist_median = median(mean_dist, na.rm = TRUE),
        mean_dist_Q1     = quantile(mean_dist, 0.25, na.rm = TRUE),
        mean_dist_Q3     = quantile(mean_dist, 0.75, na.rm = TRUE),
        mean_dist_IQR    = IQR(mean_dist, na.rm = TRUE),
        mean_dist_range  = diff(range(mean_dist, na.rm = TRUE)),
        mean_dist_var    = var(mean_dist, na.rm = TRUE),
        mean_dist_sd     = sd(mean_dist, na.rm = TRUE),
        .groups = "drop"
    )

morpho_stats <- all_dist_resume %>%
    group_by(method) %>%
    summarise(
        across(
            c(nb_cc_moyen, length_moyenne, nb_feuilles_moyen,
              aire_non_couverte_pourc),
            ~ mean(.x, na.rm = TRUE)
        ),
        .groups = "drop"
    )

tableau_methodes <- left_join(dist_stats, morpho_stats, by = "method")

# ------------------------------------------------------------
# 5. LaTeX export
# ------------------------------------------------------------

writeLines(
    kable(
        tableau_methodes,
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Summary statistics by method",
        label = "tab:summary_methods"
    ),
    output$tab
)

cat("\nSummary table exported to:", output$tab, "\n")

sink(type = "message")
sink()
close(log_con)