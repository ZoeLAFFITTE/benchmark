#!/usr/bin/env Rscript

# ============================================================
# Reproducibility statistics - figures and tests
#
# 1. Concatenate the per-subject TSV files (label 0 excluded)
#    and export the three global TSV files
# 2. Basin / population figure (IG and cube root of volume)
# 3. Single-subject and population figures
# 4. Subject-level correlations, Friedman test, Wilcoxon post-hoc
# ============================================================

# ------------------------------------------------------------
# 0. Libraries, logging, Snakemake objects
# ------------------------------------------------------------

suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(readr)
    library(purrr)
    library(tibble)
    library(ggplot2)
    library(effectsize)
})

log_con <- file(snakemake@log[[1]], open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")

cat("============================================================\n")
cat("Reproducibility statistics: figures and tests\n")
cat("============================================================\n\n")

inputs <- snakemake@input
params <- snakemake@params
output <- snakemake@output

# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

#' Read and row-bind several TSV files; subject/method cast to character.
read_tsv_safe <- function(files) {
    files <- unlist(files)
    if (length(files) == 0) {
        return(tibble())
    }

    map_dfr(files, function(file) {
        cat("Reading:", file, "\n")
        read_tsv(file, show_col_types = FALSE, progress = FALSE)
    }) %>%
        mutate(across(any_of(c("subject", "method")), as.character))
}

#' Drop every row whose basin label is "0" (in any label column found).
exclude_label_zero <- function(df) {
    label_columns <- intersect(
        c("basin", "basin_test", "basin_retest",
          "label_HCP_test", "label_HCP_retest", "label"),
        names(df)
    )

    for (col in label_columns) {
        df <- df %>%
            filter(is.na(.data[[col]]) | as.character(.data[[col]]) != "0")
    }

    df
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

#' Mean number of leaves and sqrt(area) per matched basin
#' (averaged over test/retest, then over rows of the same basin).
#' @param df table with nb_feuilles_*/area_basin_* (test, retest)
#'   columns and label_HCP_test
basin_means <- function(df) {
    df %>%
        mutate(
            nb_feuilles = (nb_feuilles_test + nb_feuilles_retest) / 2,
            area_basin  = (area_basin_test + area_basin_retest) / 2
        ) %>%
        group_by(method, subject, basin_test = label_HCP_test) %>%
        summarise(
            nb_feuilles = mean(nb_feuilles, na.rm = TRUE),
            area_basin  = mean(area_basin, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        mutate(x = sqrt(area_basin))
}

#' First non-NA value of a vector (NA_real_ if none).
first_non_na <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) NA_real_ else first(x)
}

#' Pearson correlation between `x` and `nb_feuilles` for one group.
#' Returns NA values when the correlation is not computable.
cor_safe <- function(d) {
    sx <- sd(d$x, na.rm = TRUE)
    sy <- sd(d$nb_feuilles, na.rm = TRUE)

    if (nrow(d) < 3 || !is.finite(sx) || !is.finite(sy) ||
        sx == 0 || sy == 0) {
        return(tibble(
            n = nrow(d), r = NA_real_, p = NA_real_,
            ci_low = NA_real_, ci_high = NA_real_
        ))
    }

    ct <- cor.test(d$x, d$nb_feuilles, method = "pearson")

    tibble(
        n = nrow(d),
        r = unname(ct$estimate),
        p = ct$p.value,
        ci_low = ct$conf.int[1],
        ci_high = ct$conf.int[2]
    )
}

#' Correlation table per method x facet, with significance stars.
#' @param facet name of the facet column (string)
compute_cor <- function(df, facet) {
    df %>%
        group_by(across(all_of(c("method", facet)))) %>%
        group_modify(~ cor_safe(.x)) %>%
        ungroup() %>%
        mutate(
            significance = as.character(cut(
                p,
                breaks = c(0, 0.001, 0.01, 0.05, 1),
                labels = c("***", "**", "*", "ns"),
                include.lowest = TRUE
            ))
        )
}

#' Methods in the configured order, followed by any newly detected ones.
ordered_methods <- function(descriptive, method_names) {
    detected <- unique(as.character(descriptive$method))
    detected <- detected[!is.na(detected) & detected != ""]

    methods <- c(
        method_names[method_names %in% detected],
        setdiff(detected, method_names)
    )

    if (length(methods) == 0) {
        stop("No methods found in descriptive TSV files.")
    }
    methods
}

#' Named colour vector, cycling over red/yellow/blue/green.
method_colors <- function(methods) {
    setNames(
        rep(c("red", "yellow", "blue", "green"), length.out = length(methods)),
        methods
    )
}

#' Upper/lower bound of regression CI and data, per facet x method
#' (used to place annotations above the data).
ci_bounds <- function(df, facet) {
    df %>%
        group_by(across(all_of(c(facet, "method")))) %>%
        group_modify(~ {
            d <- .x
            sx <- sd(d$x, na.rm = TRUE)

            if (nrow(d) < 3 || !is.finite(sx) || sx == 0) {
                return(tibble(
                    upper = max(d$nb_feuilles),
                    lower = min(d$nb_feuilles)
                ))
            }

            fit <- lm(nb_feuilles ~ x, data = d)
            nd <- data.frame(
                x = seq(min(d$x), max(d$x), length.out = 100)
            )
            pr <- predict(fit, nd, interval = "confidence", level = 0.95)

            tibble(
                upper = max(pr[, "upr"], d$nb_feuilles),
                lower = min(pr[, "lwr"], d$nb_feuilles)
            )
        }) %>%
        ungroup()
}

#' Positions of annotation texts and of their background box.
#'
#' @param df_plot   plotting data (x, nb_feuilles, method factor, facet)
#' @param df_cor    compute_cor() output + an `annotation` column
#' @param facet     facet column name
#' @param global_y  TRUE: y range shared by all facets; FALSE: per facet
#' @param box_right function(xmax, xr) giving the right edge of the box
#' @return list(annot = tibble, box = tibble)
annotation_layout <- function(df_plot, df_cor, facet, global_y, box_right) {

    n_meth <- nlevels(df_plot$method)

    # y range (CI + data) used to stack the annotations
    ci <- ci_bounds(df_plot, facet)
    if (!global_y) {
        ci <- group_by(ci, across(all_of(facet)))
    }

    y_ranges <- ci %>%
        mutate(
            ytop = max(upper),
            ybot = min(lower),
            yr = ytop - ybot,
            yr = if_else(is.finite(yr) & yr != 0, yr, pmax(abs(ytop), 1))
        ) %>%
        ungroup() %>%
        distinct(across(all_of(facet)), ytop, yr)

    # x range per facet
    x_ranges <- df_plot %>%
        group_by(across(all_of(facet))) %>%
        summarise(xmin = min(x), xmax = max(x), .groups = "drop") %>%
        mutate(xr = xmax - xmin, x0 = xmin + 0.04 * xr)

    annot <- df_cor %>%
        select(method, all_of(facet), annotation) %>%
        left_join(y_ranges, by = facet) %>%
        left_join(x_ranges, by = facet) %>%
        mutate(
            y = ytop + yr * (
                PAD_ANNOT + (n_meth - as.integer(method)) * STEP_ANNOT
            )
        )

    box <- annot %>%
        group_by(across(all_of(facet))) %>%
        summarise(
            box_xmin = first(xmin) + 0.01 * first(xr),
            box_xmax = box_right(first(xmax), first(xr)),
            box_ymin = min(y) - 0.55 * STEP_ANNOT * first(yr),
            box_ymax = max(y) + 0.55 * STEP_ANNOT * first(yr),
            .groups = "drop"
        )

    list(annot = annot, box = box)
}

#' Scatter + linear fit + correlation annotations, one colour per method.
#' @param layout      output of annotation_layout()
#' @param facet_layer a ggplot2 facet specification
scatter_with_annotations <- function(df, layout, facet_layer,
                                     cols, xlab, ylab,
                                     scale_x, scale_y) {
    ggplot(df, aes(x = x, y = nb_feuilles, color = method, fill = method)) +
        geom_point(size = 2, alpha = 0.9) +
        geom_smooth(
            method = "lm", formula = y ~ x, se = TRUE,
            level = 0.95, linewidth = 0.9, alpha = 0.15
        ) +
        geom_blank(
            data = layout$box, aes(x = box_xmin, y = box_ymax),
            inherit.aes = FALSE
        ) +
        geom_rect(
            data = layout$box,
            aes(xmin = box_xmin, xmax = box_xmax,
                ymin = box_ymin, ymax = box_ymax),
            inherit.aes = FALSE,
            fill = alpha("white", 0.6), color = "white", linewidth = 0.3
        ) +
        geom_text(
            data = layout$annot,
            aes(x = x0, y = y, label = annotation, color = method),
            inherit.aes = FALSE, hjust = 0, vjust = 0.5,
            size = 4, lineheight = 1, show.legend = FALSE
        ) +
        facet_layer +
        scale_x +
        scale_y +
        scale_color_manual(values = cols, name = "Method", drop = FALSE) +
        scale_fill_manual(values = cols, name = "Method", drop = FALSE) +
        labs(x = xlab, y = ylab) +
        theme_perso
}

# Plot constants
STEP_ANNOT <- 0.09   # vertical spacing between annotations
PAD_ANNOT  <- 0.04   # gap between data and first annotation

theme_perso <- theme_grey(base_size = 12) +
    theme(
        aspect.ratio = 1,
        strip.background = element_rect(fill = "grey30", color = "grey30"),
        strip.text = element_text(size = 12, face = "bold", color = "white"),
        legend.title = element_text(size = 11),
        legend.text = element_text(size = 10),
        legend.position = "bottom",
        panel.spacing = unit(0.8, "lines"),
        plot.margin = margin(5, 5, 5, 5)
    )

# ============================================================
# 2. Read, clean, export global TSV files
# ============================================================

cat("Matching files:", length(unlist(inputs$match_tsv)), "\n")
cat("Hausdorff files:", length(unlist(inputs$hausdorff_tsv)), "\n")
cat("Descriptive files:", length(unlist(inputs$desc_tsv)), "\n\n")

matching    <- exclude_label_zero(read_tsv_safe(inputs$match_tsv))
hausdorff   <- exclude_label_zero(read_tsv_safe(inputs$hausdorff_tsv))
descriptive <- exclude_label_zero(read_tsv_safe(inputs$desc_tsv))

cat("\nLabel 0 exclusion completed.\n")
cat("Matching rows:", nrow(matching), "\n")
cat("Hausdorff rows:", nrow(hausdorff), "\n")
cat("Descriptive rows:", nrow(descriptive), "\n")

write_tsv(matching,    output$matching)
write_tsv(hausdorff,   output$hausdorff)
write_tsv(descriptive, output$desc)
cat("Global TSV files exported.\n")

# Test / retest tables and joined distance table
desc_split <- split_descriptive(descriptive)
all_dist   <- build_all_dist(hausdorff, matching, desc_split)

# ============================================================
# 3. Methods and colours
# ============================================================

methods <- ordered_methods(
    descriptive,
    as.character(unlist(params$method_names))
)
cols_methods <- method_colors(methods)

cat("\nMethods detected:", paste(methods, collapse = ", "), "\n")

# ============================================================
# 4. Basin / population figure
# ============================================================

cat("\nPreparing basin / population figure...\n")

# Global measurements per method / subject / side / acquisition
df_global <- descriptive %>%
    group_by(method, subject, side, test) %>%
    summarise(
        nb_feuilles_total = sum(n_leaves_basin, na.rm = TRUE),
        IG = first_non_na(IG),
        cbrt_volume_mesh = first_non_na(cbrt_volume_mesh),
        .groups = "drop"
    )

# Per-subject means (shared by this figure and the population figure)
df_subject <- df_global %>%
    group_by(method, subject) %>%
    summarise(
        nb_feuilles = mean(nb_feuilles_total, na.rm = TRUE),
        IG = mean(IG, na.rm = TRUE),
        racine_volume = mean(cbrt_volume_mesh, na.rm = TRUE),
        .groups = "drop"
    )

df_both <- df_subject %>%
    pivot_longer(
        cols = c(IG, racine_volume),
        names_to = "variable_x",
        values_to = "x"
    ) %>%
    filter(is.finite(x), is.finite(nb_feuilles)) %>%
    mutate(
        variable_x = factor(
            variable_x,
            levels = c("IG", "racine_volume"),
            labels = c("Mean IG", "Cube Root of Mean Volume")
        ),
        method = factor(as.character(method), levels = methods)
    ) %>%
    filter(!is.na(method))

if (nrow(df_both) == 0) {
    stop("No valid data available for basin / population figure.")
}

cor_both <- compute_cor(df_both, "variable_x") %>%
    mutate(
        R2 = r^2,
        effect_size = ifelse(
            is.na(r), NA_character_,
            as.character(interpret_r(r, rules = "cohen1988"))
        ),
        annotation = ifelse(
            is.na(r),
            paste0("n=", n, ", correlation not calculable"),
            paste0(
                "n=", n, ", r=", sprintf("%.2f", r),
                " (", effect_size, ")",
                ", R²=", sprintf("%.2f", R2), " ", significance
            )
        )
    )

layout_both <- annotation_layout(
    df_both, cor_both,
    facet = "variable_x",
    global_y = TRUE,
    box_right = function(xmax, xr) xmax - 0.015 * xmax
)

p_both <- scatter_with_annotations(
    df_both, layout_both,
    facet_layer = facet_wrap(~ variable_x, ncol = 2, scales = "free_x"),
    cols = cols_methods,
    xlab = NULL,
    ylab = "Mean number of leaves per subject",
    scale_x = scale_x_continuous(),
    scale_y = scale_y_continuous(expand = expansion(mult = c(0.05, 0.01)))
)

ggsave(
    filename = output$figure_bassin_pop_both,
    plot = p_both, width = 8, height = 6, dpi = 300
)
cat("figure_bassin_pop_both exported.\n")

# ============================================================
# 5. Single-subject (one per subject) and population figures
# ============================================================

cat("\nPreparing single-subject / population figures...\n")

subjects_all <- as.character(unlist(params$subjects))
single_files <- unlist(output$figure_single)

if (length(subjects_all) == 0) {
    stop("No subject provided through params$subjects.")
}
if (length(subjects_all) != length(single_files)) {
    stop("params$subjects (", length(subjects_all), ") and output$figure_single (",
         length(single_files), ") must have the same length.")
}

# Sanity check: each output file name must contain its subject id
bad <- !mapply(function(s, f) grepl(s, basename(f), fixed = TRUE),
               subjects_all, single_files)
if (any(bad)) {
    stop("Subject / output file mismatch for: ",
         paste(subjects_all[bad], collapse = ", "))
}

lab_pop <- "Population"

# Population data (unchanged)
df_pop <- df_subject %>%
    transmute(method, x = IG, nb_feuilles, panel = lab_pop)

#' Data for the single-subject panel of one subject.
single_subject_data <- function(subj, lab) {
    matching %>%
        filter(subject == subj) %>%
        select(subject, side, label_HCP_test, label_HCP_retest) %>%
        distinct() %>%
        inner_join(desc_split$test,
                   by = c("subject", "side", "label_HCP_test")) %>%
        inner_join(desc_split$retest,
                   by = c("method", "subject", "side", "label_HCP_retest")) %>%
        basin_means() %>%
        transmute(method, x, nb_feuilles, panel = lab)
}

#' Build one panel figure, or an empty placeholder if there is no data.
make_panel <- function(d, xlab, ylab, empty_label, empty_theme) {

    d <- d %>%
        filter(is.finite(x), is.finite(nb_feuilles)) %>%
        mutate(method = factor(as.character(method), levels = methods)) %>%
        filter(!is.na(method))

    if (nrow(d) == 0) {
        cat("WARNING:", empty_label, "- exporting an empty figure.\n")
        return(
            ggplot() +
                annotate("text", x = 0, y = 0, label = empty_label, size = 4) +
                labs(x = xlab, y = ylab) +
                theme_minimal() +
                empty_theme
        )
    }

    d$panel <- factor(d$panel)

    cor_p <- compute_cor(d, "panel") %>%
        mutate(
            annotation = ifelse(
                is.na(r),
                paste0("n=", n, " - Correlation not calculable"),
                paste0(
                    "r=", sprintf("%.2f", r), " ", significance,
                    ", CI95%=[", sprintf("%.2f", ci_low), "; ",
                    sprintf("%.2f", ci_high), "]"
                )
            )
        )

    layout <- annotation_layout(
        d, cor_p,
        facet = "panel",
        global_y = FALSE,
        box_right = function(xmax, xr) xmax - 0.2 * xr
    )

    scatter_with_annotations(
        d, layout,
        facet_layer = facet_wrap(~ panel),
        cols = cols_methods,
        xlab = xlab, ylab = ylab,
        scale_x = scale_x_continuous(expand = expansion(mult = c(0.01, 0.01))),
        scale_y = scale_y_continuous(expand = expansion(mult = c(0, 0)))
    )
}

# One single-subject figure per subject
for (i in seq_along(subjects_all)) {
    subj <- subjects_all[i]
    cat("Selected subject:", subj, "->", single_files[i], "\n")

    if (!subj %in% unique(as.character(matching$subject))) {
        warning("Subject ", subj, " is not present in matching files. ",
                "Its single-subject figure will be empty.")
    }

    lab_single <- paste0("Single subject: ", subj)
    d_single <- single_subject_data(subj, lab_single)
    cat("  Single subject data:", nrow(d_single), "\n")

    p_single <- make_panel(
        d_single,
        xlab = "Square root of basin area",
        ylab = "Mean number of leaves per basin",
        empty_label = paste("No data available for subject", subj),
        empty_theme = theme(aspect.ratio = 1,
                            plot.margin = margin(0, 0, 0, 0, unit = "pt"))
    )

    ggsave(single_files[i], p_single,
           width = 4.3, height = 5, units = "in", dpi = 600, bg = "white")
}

# Population figure
p_pop <- make_panel(
    df_pop,
    xlab = "Mean IG",
    ylab = "Mean number of leaves per subject",
    empty_label = "No population data available",
    empty_theme = theme()
)

ggsave(output$figure_population, p_pop,
       width = 4.3, height = 5, units = "in", dpi = 600, bg = "white")

cat("Single-subject and population figures exported.\n")

# ============================================================
# 6. Subject-level correlations, Friedman, Wilcoxon post-hoc
# ============================================================

cat("\nPreparing subject-level correlations...\n")

df_all_basins <- basin_means(all_dist) %>%
    filter(is.finite(x), is.finite(nb_feuilles))

cat("Rows in df_all_basins:", nrow(df_all_basins), "\n")

r_subject <- df_all_basins %>%
    group_by(method, subject) %>%
    group_modify(~ cor_safe(.x)) %>%
    ungroup() %>%
    filter(is.finite(r))

print(r_subject)

if (nrow(r_subject) == 0) {
    stop(
        "No valid subject-level correlations could be calculated. ",
        "Check the number of basins and the variability of the data."
    )
}

methods_comparison <- methods[methods %in% unique(as.character(r_subject$method))]
cat("Methods compared:", paste(methods_comparison, collapse = ", "), "\n")

if (length(methods_comparison) < 2) {
    stop("At least two methods with valid subject-level correlations ",
         "are required for the Friedman test.")
}

# Paired data: one row per subject, one column per method
r_wide <- r_subject %>%
    mutate(method = as.character(method), subject = as.character(subject)) %>%
    select(subject, method, r) %>%
    pivot_wider(names_from = method, values_from = r) %>%
    drop_na(all_of(methods_comparison))

cat("Subjects with complete data:", nrow(r_wide), "\n")

if (nrow(r_wide) < 2) {
    stop("The Friedman and paired Wilcoxon tests require at least two ",
         "subjects with valid correlations for every compared method. ",
         "Currently available: ", nrow(r_wide), " complete subject(s).")
}

# Friedman test
friedman_result <- friedman.test(
    as.matrix(r_wide[, methods_comparison, drop = FALSE])
)
print(friedman_result)

write_tsv(
    tibble(
        test = "Friedman",
        n_subjects = nrow(r_wide),
        n_methods = length(methods_comparison),
        methods = paste(methods_comparison, collapse = ";"),
        statistic = unname(friedman_result$statistic),
        df = unname(friedman_result$parameter),
        p_value = friedman_result$p.value
    ),
    output$friedman
)

# Pairwise Wilcoxon signed-rank tests, Holm correction
posthoc <- combn(methods_comparison, 2, simplify = FALSE) %>%
    map_dfr(function(pair) {
        wt <- wilcox.test(
            r_wide[[pair[1]]], r_wide[[pair[2]]],
            paired = TRUE, exact = FALSE
        )
        tibble(
            method_1 = pair[1],
            method_2 = pair[2],
            n_subjects = nrow(r_wide),
            wilcox_V = unname(wt$statistic),
            p_value = wt$p.value
        )
    }) %>%
    mutate(p_holm = p.adjust(p_value, method = "holm"))

print(posthoc)
write_tsv(posthoc, output$posthoc)

# ------------------------------------------------------------
# 7. Close logs
# ------------------------------------------------------------

cat("\nReproducibility statistics (figures and tests) completed successfully.\n")

sink(type = "message")
sink()
close(log_con)