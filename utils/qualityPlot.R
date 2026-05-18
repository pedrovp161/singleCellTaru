#####

preprocessing_quality_metrics <- function(
    seurat_list,
    title = "Quality Metrics: Aggregate (Top) and Individual (Bottom)",
    xlabel = "Filtering Stages",
    ylabel_top = "Total Sum",
    ylabel_bottom = "Spots per Sample",
    x_labels = c("Raw", "Counts & Genes", "MT"),
    font_size = 16,
    colors = NULL
) {

  library(Seurat)
  library(dplyr)
  library(tidyr)
  library(ggplot2)

  # -----------------------------
  # helper: MAD outlier
  # -----------------------------
  is_outlier <- function(x, k = 4, method = "both") {

    x <- as.numeric(x)

    if (all(is.na(x))) return(rep(FALSE, length(x)))

    med <- median(x, na.rm = TRUE)
    dev <- x - med
    mad <- median(abs(dev), na.rm = TRUE)

    if (is.na(mad) || mad == 0) return(rep(FALSE, length(x)))

    thr <- k * mad

    if (method == "high") return(dev > thr)
    if (method == "low")  return(dev < -thr)
    abs(dev) > thr
  }

  # -----------------------------
  # 1. extract metrics per sample
  # -----------------------------
  df_list <- lapply(names(seurat_list), function(sample) {

    obj <- seurat_list[[sample]]
    meta <- obj[[]]

    data.frame(
      sample = sample,
      Raw = ncol(obj),

      CountsGenes = sum(
        !(
          is_outlier(meta$nCount_RNA, k = 4, method = "low") |
          is_outlier(meta$nFeature_RNA, k = 4, method = "low")
        )
      ),

      MT = if ("percent.mt" %in% colnames(meta)) {
        sum(!is_outlier(meta$percent.mt, k = 4, method = "high"))
      } else {
        seurat_list$CTH92_ML[["percent.mt"]] <- PercentageFeatureSet(
        seurat_list$CTH92_ML,
        pattern = "^MT-"
        )
        sum(!is_outlier(meta$percent.mt, k = 4, method = "high"))
      }
    )
  })

  df <- bind_rows(df_list)

  # -----------------------------
  # 2. reshape (long format)
  # -----------------------------
  df_long <- df %>%
    pivot_longer(cols = c("Raw", "CountsGenes", "MT"),
                 names_to = "step",
                 values_to = "cells")

  df_long$step <- factor(df_long$step, levels = c("Raw", "CountsGenes", "MT"))

  # -----------------------------
  # 3. aggregate (top plot)
  # -----------------------------
  df_sum <- df_long %>%
    group_by(step) %>%
    summarise(total = sum(cells, na.rm = TRUE), .groups = "drop")

  # -----------------------------
  # colors
  # -----------------------------
  if (is.null(colors)) {
    colors <- scales::hue_pal()(length(unique(df_long$sample)))
  }

  # -----------------------------
  # 4. plot
  # -----------------------------
  p <- ggplot() +

    # TOP: aggregate
    geom_line(data = df_sum,
              aes(x = step, y = total, group = 1),
              color = "darkred", linewidth = 1.2) +
    geom_point(data = df_sum,
               aes(x = step, y = total),
               color = "darkred", size = 3) +
    geom_text(data = df_sum,
              aes(x = step, y = total, label = round(total)),
              vjust = -0.5, color = "darkred", fontface = "bold") +

    # BOTTOM: per sample
    geom_line(data = df_long,
              aes(x = step, y = cells, group = sample, color = sample),
              alpha = 0.6) +
    geom_point(data = df_long,
               aes(x = step, y = cells, color = sample),
               alpha = 0.6) +

    labs(
      title = title,
      x = xlabel,
      y = ylabel_bottom
    ) +

    scale_color_manual(values = colors) +

    theme_minimal(base_size = font_size) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.title = element_blank(),
      legend.position = "right"
    )

  print(p)

  return(df)
}

df <- preprocessing_quality_metrics(
  seurat_list,
  title = "QC per sample (MAD k = 4)"
)