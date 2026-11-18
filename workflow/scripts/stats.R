
#!/usr/bin/env Rscript

# ============================================================
# Reproducibility statistics
# ============================================================

# 1. Logging
log_con <- file(snakemake@log[[1]], open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")

cat("============================================================\n")
cat("Reproducibility statistics\n")
cat("============================================================\n\n")

# 2. Libraries
suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(tidyr)
    library(readr)
    library(purrr)
    library(tibble)
    library(stringr)
    library(forcats)
    library(effectsize)
})

# 3. Snakemake inputs and outputs
inputs <- snakemake@input
params <- snakemake@params
output <- snakemake@output

match_files <- unlist(inputs$match_tsv)
hausdorff_files <- unlist(inputs$hausdorff_tsv)
desc_files <- unlist(inputs$desc_tsv)

cat("Number of matching files:", length(match_files), "\n")
cat("Number of Hausdorff files:", length(hausdorff_files), "\n")
cat("Number of descriptive files:", length(desc_files), "\n\n")


# ============================================================
# 4. Read and concatenate TSV files
#    Exclude basin label 0 globally
# ============================================================

read_tsv_safe <- function(files) {
    if (length(files) == 0) {
        return(tibble())
    }

    map_dfr(files, function(file) {
        cat("Reading:", file, "\n")
        read_tsv(file, show_col_types = FALSE, progress = FALSE)
    })
}

cat("Concatenating matching files...\n")
matching <- read_tsv_safe(match_files)

cat("Concatenating Hausdorff files...\n")
hausdorff <- read_tsv_safe(hausdorff_files)

cat("Concatenating descriptive files...\n")
descriptive <- read_tsv_safe(desc_files)

# ------------------------------------------------------------
# Exclude label 0 from all datasets
# ------------------------------------------------------------

exclude_label_zero <- function(df) {

    label_columns <- intersect(
        c(
            "basin",
            "basin_test",
            "basin_retest",
            "label_HCP_test",
            "label_HCP_retest",
            "label"
        ),
        names(df)
    )

    for (col in label_columns) {
        df <- df %>%
            filter(
                is.na(.data[[col]]) |
                as.character(.data[[col]]) != "0"
            )
    }

    df
}

matching <- exclude_label_zero(matching)
hausdorff <- exclude_label_zero(hausdorff)
descriptive <- exclude_label_zero(descriptive)

cat("\nLabel 0 exclusion completed.\n")
cat("Matching rows:", nrow(matching), "\n")
cat("Hausdorff rows:", nrow(hausdorff), "\n")
cat("Descriptive rows:", nrow(descriptive), "\n")

# Standardize identifiers
if ("subject" %in% names(matching)) {
    matching <- matching %>%
        mutate(subject = as.character(subject))
}

if ("subject" %in% names(descriptive)) {
    descriptive <- descriptive %>%
        mutate(
            subject = as.character(subject),
            method = as.character(method)
        )
}

# ============================================================
# 5. Export global TSV files
# ============================================================

cat("Exporting global TSV files...\n")

write_tsv(matching, output$matching)
write_tsv(hausdorff, output$hausdorff)
write_tsv(descriptive, output$desc)

cat("\nGlobal TSV concatenation completed successfully\n")

# ============================================================
# 6. Global methods and colors
# ============================================================

# Methods detected in descriptive TSV files
methods_detected <- unique(as.character(descriptive$method))
methods_detected <- methods_detected[
    !is.na(methods_detected) & methods_detected != ""
]

# Methods ordered according to plot_isbi
method_names <- as.character(unlist(params$method_names))

# Keep the configured order, then append any new methods
# not yet listed in METHOD_NAMES
methods <- c(
    method_names[method_names %in% methods_detected],
    setdiff(methods_detected, method_names)
)

if (length(methods) == 0) {
    stop("No methods found in descriptive TSV files.")
}

# Same colors and same cyclic assignment as plot_isbi
colors <- c("red", "yellow", "blue", "green")

cols_methods <- setNames(
    rep(colors, length.out = length(methods)),
    methods
)

cat("\nMethods detected:\n")
print(methods)

cat("\nColors assigned:\n")
print(cols_methods)

# ============================================================
# 7. Helper functions
# ============================================================

first_non_na <- function(x) {
    x <- x[!is.na(x)]

    if (length(x) == 0) {
        return(NA_real_)
    }

    first(x)
}

cor_safe <- function(d) {

    sx <- sd(d$x, na.rm = TRUE)
    sy <- sd(d$nb_feuilles, na.rm = TRUE)

    if (
        nrow(d) < 3 ||
        !is.finite(sx) ||
        !is.finite(sy) ||
        sx == 0 ||
        sy == 0
    ) {
        return(tibble(
            n = nrow(d),
            r = NA_real_,
            p = NA_real_,
            ci_low = NA_real_,
            ci_high = NA_real_
        ))
    }

    ct <- cor.test(
        d$x,
        d$nb_feuilles,
        method = "pearson"
    )

    tibble(
        n = nrow(d),
        r = unname(ct$estimate),
        p = ct$p.value,
        ci_low = ct$conf.int[1],
        ci_high = ct$conf.int[2]
    )
}

# ============================================================
# 8. Basin / population figure
# ============================================================

cat("\nPreparing basin / population figure...\n")

# ------------------------------------------------------------
# 8.1. Global measurements per method / subject / side / test
# ------------------------------------------------------------

df_global <- descriptive %>%
    group_by(method, subject, side, test) %>%
    summarise(
        nb_feuilles_total = sum(n_leaves_basin, na.rm = TRUE),
        IG = first_non_na(IG),
        cbrt_volume_mesh = first_non_na(cbrt_volume_mesh),
        .groups = "drop"
    )

# ------------------------------------------------------------
# 8.2. Mean measurements per method / subject
# ------------------------------------------------------------

df_moyennes <- df_global %>%
    group_by(method, subject) %>%
    summarise(
        nb_feuilles = mean(nb_feuilles_total, na.rm = TRUE),
        IG = mean(IG, na.rm = TRUE),
        racine_volume = mean(cbrt_volume_mesh, na.rm = TRUE),
        .groups = "drop"
    )

# ------------------------------------------------------------
# 8.3. Plot configuration
# ------------------------------------------------------------

step_annot <- 0.09
pad_annot <- 0.04
n_meth <- length(methods)

theme_perso <- theme_grey(base_size = 12) +
    theme(
        aspect.ratio = 1,
        strip.background = element_rect(
            fill = "grey30",
            color = "grey30"
        ),
        strip.text = element_text(
            size = 12,
            face = "bold",
            color = "white"
        ),
        legend.title = element_text(size = 11),
        legend.text = element_text(size = 10),
        legend.position = "bottom",
        panel.spacing = unit(0.8, "lines"),
        plot.margin = margin(5, 5, 5, 5)
    )

# ------------------------------------------------------------
# 8.4. Prepare plotting data
# ------------------------------------------------------------

df_plot <- df_moyennes %>%
    pivot_longer(
        cols = c(IG, racine_volume),
        names_to = "variable_x",
        values_to = "x"
    ) %>%
    filter(
        is.finite(x),
        is.finite(nb_feuilles)
    ) %>%
    mutate(
        variable_x = factor(
            variable_x,
            levels = c("IG", "racine_volume"),
            labels = c(
                "Mean IG",
                "Cube Root of Mean Volume"
            )
        ),
        method = factor(
            as.character(method),
            levels = methods
        )
    ) %>%
    filter(!is.na(method))

if (nrow(df_plot) == 0) {
    stop("No valid data available for basin / population figure.")
}

# ------------------------------------------------------------
# 8.5. Pearson correlations
# ------------------------------------------------------------

df_cor <- df_plot %>%
    group_by(method, variable_x) %>%
    group_modify(~ cor_safe(.x)) %>%
    ungroup() %>%
    mutate(
        R2 = r^2,

        significance = as.character(cut(
            p,
            breaks = c(0, 0.001, 0.01, 0.05, 1),
            labels = c("***", "**", "*", "ns"),
            include.lowest = TRUE
        )),

        effect_size = ifelse(
            is.na(r),
            NA_character_,
            as.character(interpret_r(r, rules = "cohen1988"))
        ),

        annotation = ifelse(
            is.na(r),
            paste0("n=", n, ", correlation not calculable"),
            paste0(
                "n=", n,
                ", r=", sprintf("%.2f", r),
                " (", effect_size, ")",
                ", R²=", sprintf("%.2f", R2),
                " ", significance
            )
        )
    )

# ------------------------------------------------------------
# 8.6. Confidence intervals for annotation positioning
# ------------------------------------------------------------

df_ci <- df_plot %>%
    group_by(variable_x, method) %>%
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
            x = seq(
                min(d$x),
                max(d$x),
                length.out = 100
            )
        )

        pr <- predict(
            fit,
            nd,
            interval = "confidence",
            level = 0.95
        )

        tibble(
            upper = max(pr[, "upr"], d$nb_feuilles),
            lower = min(pr[, "lwr"], d$nb_feuilles)
        )

    }) %>%
    ungroup()

ytop <- max(df_ci$upper)
ybot <- min(df_ci$lower)
yr <- ytop - ybot

if (!is.finite(yr) || yr == 0) {
    yr <- max(abs(ytop), 1)
}

# ------------------------------------------------------------
# 8.7. X-axis ranges
# ------------------------------------------------------------

df_xr <- df_plot %>%
    group_by(variable_x) %>%
    summarise(
        xmin = min(x),
        xmax = max(x),
        xr = xmax - xmin,
        .groups = "drop"
    ) %>%
    mutate(
        x0 = xmin + 0.04 * xr
    )

# ------------------------------------------------------------
# 8.8. Annotation positions
# ------------------------------------------------------------

df_annot <- df_cor %>%
    select(method, variable_x, annotation) %>%
    left_join(df_xr, by = "variable_x") %>%
    mutate(
        y = ytop + yr * (
            pad_annot +
            (n_meth - as.integer(method)) * step_annot
        )
    )

df_box <- df_annot %>%
    group_by(variable_x) %>%
    summarise(
        box_xmin = first(xmin) + 0.01 * first(xr),
        box_xmax = first(xmax) - 0.015 * first(xmax),
        box_ymin = min(y) - 0.55 * step_annot * yr,
        box_ymax = max(y) + 0.55 * step_annot * yr,
        .groups = "drop"
    )

# ------------------------------------------------------------
# 8.9. Final plot
# ------------------------------------------------------------

p_comparaison <- ggplot(
    df_plot,
    aes(
        x = x,
        y = nb_feuilles,
        color = method,
        fill = method
    )
) +
    geom_point(
        size = 2,
        alpha = 0.9
    ) +
    geom_smooth(
        method = "lm",
        formula = y ~ x,
        se = TRUE,
        level = 0.95,
        linewidth = 0.9,
        alpha = 0.15
    ) +
    geom_blank(
        data = df_box,
        aes(
            x = box_xmin,
            y = box_ymax
        ),
        inherit.aes = FALSE
    ) +
    geom_rect(
        data = df_box,
        aes(
            xmin = box_xmin,
            xmax = box_xmax,
            ymin = box_ymin,
            ymax = box_ymax
        ),
        inherit.aes = FALSE,
        fill = alpha("white", 0.6),
        color = "white",
        linewidth = 0.3
    ) +
    geom_text(
        data = df_annot,
        aes(
            x = x0,
            y = y,
            label = annotation,
            color = method
        ),
        inherit.aes = FALSE,
        hjust = 0,
        vjust = 0.5,
        size = 4,
        lineheight = 1,
        show.legend = FALSE
    ) +
    facet_wrap(
        ~ variable_x,
        ncol = 2,
        scales = "free_x"
    ) +
    scale_y_continuous(
        expand = expansion(mult = c(0.05, 0.01))
    ) +
    scale_color_manual(
        values = cols_methods,
        name = "Method",
        drop = FALSE
    ) +
    scale_fill_manual(
        values = cols_methods,
        name = "Method",
        drop = FALSE
    ) +
    labs(
        x = NULL,
        y = "Mean number of leaves per subject"
    ) +
    theme_perso

# ------------------------------------------------------------
# 8.10. Export figure
# ------------------------------------------------------------

cat("Exporting figure_bassin_pop_both.svg...\n")

ggsave(
    filename = output$figure_bassin_pop_both,
    plot = p_comparaison,
    width = 8,
    height = 6,
    dpi = 300
)

cat("Figure exported successfully.\n")

# ============================================================
# 9. Single subject / Population figures
# ============================================================

cat("\nPreparing single-subject / population figures...\n")

# ------------------------------------------------------------
# 9.1. Select subject and define panel labels
# ------------------------------------------------------------

sujet_choisi <- as.character(params$subject)

if (
    length(sujet_choisi) != 1 ||
    is.na(sujet_choisi) ||
    sujet_choisi == ""
) {
    stop("A valid subject must be provided through params$subject.")
}

cat("Selected subject:", sujet_choisi, "\n")

lab_single <- paste0("Single subject: ", sujet_choisi)
lab_pop <- "Population"

available_subjects <- unique(as.character(matching$subject))

if (!sujet_choisi %in% available_subjects) {
    warning(
        "Selected subject ",
        sujet_choisi,
        " is not present in matching files. ",
        "The single-subject figure will be empty."
    )
}

# ------------------------------------------------------------
# 9.2. Single subject: match test and retest basins
# ------------------------------------------------------------

if (!is.na(sujet_choisi)) {

    subject_matching <- matching %>%
        filter(subject == sujet_choisi) %>%
        select(
            subject,
            side,
            label_HCP_test,
            label_HCP_retest
        ) %>%
        distinct()

} else {

    subject_matching <- tibble(
        subject = character(),
        side = character(),
        label_HCP_test = character(),
        label_HCP_retest = character()
    )
}

# Descriptive measurements for each acquisition

desc_test <- descriptive %>%
    filter(test == "HCP_test") %>%
    transmute(
        method,
        subject,
        side,
        label_HCP_test = basin,
        leaves_test = n_leaves_basin,
        area_test = area_basin
    )

desc_retest <- descriptive %>%
    filter(test == "HCP_retest") %>%
    transmute(
        method,
        subject,
        side,
        label_HCP_retest = basin,
        leaves_retest = n_leaves_basin,
        area_retest = area_basin
    )

# Join the two acquisitions using HCP matching labels

df_single <- subject_matching %>%
    inner_join(
        desc_test,
        by = c(
            "subject",
            "side",
            "label_HCP_test"
        )
    ) %>%
    inner_join(
        desc_retest,
        by = c(
            "method",
            "subject",
            "side",
            "label_HCP_retest"
        )
    ) %>%
    mutate(
        nb_feuilles = (
            leaves_test + leaves_retest
        ) / 2,

        area_basin = (
            area_test + area_retest
        ) / 2
    ) %>%
    group_by(
        method,
        subject,
        basin_test = label_HCP_test
    ) %>%
    summarise(
        nb_feuilles = mean(
            nb_feuilles,
            na.rm = TRUE
        ),
        area_basin = mean(
            area_basin,
            na.rm = TRUE
        ),
        .groups = "drop"
    ) %>%
    transmute(
        method,
        x = sqrt(area_basin),
        nb_feuilles,
        panel = lab_single
    )

# ------------------------------------------------------------
# 9.3. Population data
# ------------------------------------------------------------

df_pop <- df_global %>%
    group_by(method, subject) %>%
    summarise(
        nb_feuilles = mean(
            nb_feuilles_total,
            na.rm = TRUE
        ),
        IG = mean(
            IG,
            na.rm = TRUE
        ),
        .groups = "drop"
    ) %>%
    transmute(
        method,
        x = IG,
        nb_feuilles,
        panel = lab_pop
    )

# ------------------------------------------------------------
# 9.4. Combine and prepare plotting data
# ------------------------------------------------------------

df_plot <- bind_rows(
    df_single,
    df_pop
) %>%
    filter(
        is.finite(x),
        is.finite(nb_feuilles)
    ) %>%
    mutate(
        method = factor(
            as.character(method),
            levels = methods
        ),
        panel = factor(
            panel,
            levels = c(
                lab_single,
                lab_pop
            )
        )
    ) %>%
    filter(!is.na(method))

cat("\nSingle subject data:", nrow(df_single), "\n")
cat("Population data:", nrow(df_pop), "\n")
cat("Combined plotting data:", nrow(df_plot), "\n")

cat(
    "Methods in plotting data:",
    paste(unique(as.character(df_plot$method)), collapse = ", "),
    "\n"
)

cat(
    "Panels in plotting data:",
    paste(unique(as.character(df_plot$panel)), collapse = ", "),
    "\n"
)

# ------------------------------------------------------------
# 9.5. Pearson correlations
# ------------------------------------------------------------

if (nrow(df_plot) > 0) {

    df_cor <- df_plot %>%
        group_by(method, panel) %>%
        group_modify(~ cor_safe(.x)) %>%
        ungroup() %>%
        mutate(
            significance = as.character(cut(
                p,
                breaks = c(0, 0.001, 0.01, 0.05, 1),
                labels = c("***", "**", "*", "ns"),
                include.lowest = TRUE
            )),

            annotation = ifelse(
                is.na(r),
                paste0(
                    "n=", n,
                    " - Correlation not calculable"
                ),
                paste0(
                    "r=", sprintf("%.2f", r),
                    " ", significance,
                    ", CI95%=[",
                    sprintf("%.2f", ci_low),
                    "; ",
                    sprintf("%.2f", ci_high),
                    "]"
                )
            )
        )

} else {

    df_cor <- tibble(
        method = factor(levels = methods),
        panel = character(),
        n = integer(),
        r = double(),
        p = double(),
        ci_low = double(),
        ci_high = double(),
        significance = character(),
        annotation = character()
    )
}

# ------------------------------------------------------------
# 9.6. Confidence intervals for annotation positioning
# ------------------------------------------------------------

if (nrow(df_plot) > 0) {

    df_ci <- df_plot %>%
        group_by(panel, method) %>%
        group_modify(~ {

            d <- .x

            sx <- sd(d$x, na.rm = TRUE)

            if (
                nrow(d) < 3 ||
                !is.finite(sx) ||
                sx == 0
            ) {
                return(tibble(
                    upper = max(d$nb_feuilles),
                    lower = min(d$nb_feuilles)
                ))
            }

            fit <- lm(
                nb_feuilles ~ x,
                data = d
            )

            nd <- data.frame(
                x = seq(
                    min(d$x),
                    max(d$x),
                    length.out = 100
                )
            )

            pr <- predict(
                fit,
                nd,
                interval = "confidence",
                level = 0.95
            )

            tibble(
                upper = max(
                    pr[, "upr"],
                    d$nb_feuilles
                ),
                lower = min(
                    pr[, "lwr"],
                    d$nb_feuilles
                )
            )

        }) %>%
        ungroup()

} else {

    df_ci <- tibble(
        panel = character(),
        method = factor(levels = methods),
        upper = double(),
        lower = double()
    )
}

# ------------------------------------------------------------
# 9.7. X-axis ranges
# ------------------------------------------------------------

df_xr <- df_plot %>%
    group_by(panel) %>%
    summarise(
        xmin = min(x),
        xmax = max(x),
        xr = xmax - xmin,
        .groups = "drop"
    ) %>%
    mutate(
        x0 = xmin + 0.04 * xr
    )

# ------------------------------------------------------------
# 9.8. Annotation positions
# ------------------------------------------------------------

if (nrow(df_ci) > 0) {

    df_panel_range <- df_ci %>%
        group_by(panel) %>%
        summarise(
            ytop = max(upper),
            ybot = min(lower),
            .groups = "drop"
        ) %>%
        mutate(
            yr = ytop - ybot,
            yr = ifelse(yr == 0, pmax(abs(ytop), 1), yr)
        )

    df_annot <- df_ci %>%
        left_join(
            df_panel_range,
            by = "panel"
        ) %>%
        left_join(
            df_xr,
            by = "panel"
        ) %>%
        left_join(
            df_cor %>%
                select(method, panel, annotation),
            by = c("method", "panel")
        ) %>%
        mutate(
            y = ytop + yr * (
                pad_annot +
                (n_meth - as.integer(method)) * step_annot
            )
        )

    df_box <- df_annot %>%
        group_by(panel) %>%
        summarise(
            box_xmin = first(xmin) + 0.01 * first(xr),
            box_xmax = first(xmax) - 0.2 * first(xr),
            box_ymin = min(y) -
                0.55 * step_annot * first(yr),
            box_ymax = max(y) +
                0.55 * step_annot * first(yr),
            .groups = "drop"
        )

} else {

    df_annot <- tibble()
    df_box <- tibble()
}

# ------------------------------------------------------------
# 9.9. Plot function
# ------------------------------------------------------------

make_panel <- function(panel_name, xlab, ylab) {

    plot_data <- df_plot %>%
        filter(panel == panel_name)

    if (nrow(plot_data) == 0) {
        stop(
            paste(
                "No data available for panel:",
                panel_name
            )
        )
    }

    ggplot(
        plot_data,
        aes(
            x = x,
            y = nb_feuilles,
            color = method,
            fill = method
        )
    ) +
        geom_point(
            size = 2,
            alpha = 0.9
        ) +
        geom_smooth(
            method = "lm",
            formula = y ~ x,
            se = TRUE,
            level = 0.95,
            linewidth = 0.9,
            alpha = 0.15
        ) +
        geom_blank(
            data = filter(
                df_box,
                panel == panel_name
            ),
            aes(
                x = box_xmin,
                y = box_ymax
            ),
            inherit.aes = FALSE
        ) +
        geom_rect(
            data = filter(
                df_box,
                panel == panel_name
            ),
            aes(
                xmin = box_xmin,
                xmax = box_xmax,
                ymin = box_ymin,
                ymax = box_ymax
            ),
            inherit.aes = FALSE,
            fill = alpha("white", 0.6),
            color = "white",
            linewidth = 0.3
        ) +
        geom_text(
            data = filter(
                df_annot,
                panel == panel_name
            ),
            aes(
                x = x0,
                y = y,
                label = annotation,
                color = method
            ),
            inherit.aes = FALSE,
            hjust = 0,
            vjust = 0.5,
            size = 4,
            lineheight = 1,
            show.legend = FALSE
        ) +
        facet_wrap(~panel) +
        scale_x_continuous(
            expand = expansion(
                mult = c(0.01, 0.01)
            )
        ) +
        scale_y_continuous(
            expand = expansion(
                mult = c(0, 0)
            )
        ) +
        scale_color_manual(
            values = cols_methods,
            name = "Method",
            drop = FALSE
        ) +
        scale_fill_manual(
            values = cols_methods,
            name = "Method",
            drop = FALSE
        ) +
        labs(
            x = xlab,
            y = ylab
        ) +
        theme_perso
}

# ------------------------------------------------------------
# 9.10. Generate and export figures
# ------------------------------------------------------------

cat("\nExporting single-subject and population figures...\n")

# Single-subject figure

if (lab_single %in% as.character(df_plot$panel)) {

    d <- make_panel(
        lab_single,
        xlab = "Square root of basin area",
        ylab = "Mean number of leaves per basin"
    )

} else {

    cat(
        "WARNING: No data available for subject",
        sujet_choisi,
        "- exporting an empty figure.\n"
    )

    d <- ggplot() +
        annotate(
            "text",
            x = 0,
            y = 0,
            label = paste(
                "No data available for subject",
                sujet_choisi
            ),
            size = 4
        ) +
        labs(
            x = "Square root of basin area",
            y = "Mean number of leaves per basin"
        ) +
        theme_minimal() +
        theme(
            aspect.ratio = 1,
            plot.margin = margin(0, 0, 0, 0, unit = "pt")
        )
}

ggsave(
    filename = output$figure_single,
    plot = d,
    width = 4.3,
    height = 5,
    units = "in",
    dpi = 600,
    bg = "white"
)

# Population figure

if (lab_pop %in% as.character(df_plot$panel)) {

    c <- make_panel(
        lab_pop,
        xlab = "Mean IG",
        ylab = "Mean number of leaves per subject"
    )

} else {

    cat(
        "WARNING: No population data available - exporting an empty figure.\n"
    )

    c <- ggplot() +
        annotate(
            "text",
            x = 0,
            y = 0,
            label = "No population data available",
            size = 4
        ) +
        labs(
            x = "Mean IG",
            y = "Mean number of leaves per subject"
        ) +
        theme_minimal()
}

ggsave(
    filename = output$figure_population,
    plot = c,
    width = 4.3,
    height = 5,
    units = "in",
    dpi = 600,
    bg = "white"
)

cat("Both figures exported successfully.\n")

# ============================================================
# 10. Close logs
# ============================================================

sink(type = "message")
sink()
close(log_con)

cat("Reproducibility statistics completed successfully.\n")
