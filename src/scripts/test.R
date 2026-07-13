setwd("/data04/projects04/MarianaBoroni/spatial_ovary/bin/Pedro/Scripts/deseq")

# # diretorio da nova lib 24-06-2024
# libdir <- "/scr/R/Rpackages/4.4.1"
### diretorio da nova lib no data04
libdir <- "/data04/tools/R/Rpackages/4.4.1"

#
## set local directory for R packages
.libPaths(libdir)
.libPaths()
#
# set CRAN Mirror https://cran-r.c3sl.ufpr.br/
options(repos = c("CRAN" = "https://cran-r.c3sl.ufpr.br/"))

my_path <- "/data04/projects04/MarianaBoroni/spatial_ovary/data/Pedro_data/"

# Carregar pacotes necessários
library(Seurat)
library(DESeq2)
library(dplyr)
library(tidyr)
library(tibble)

packages <- c(
  "Seurat", "dplyr", "tidyr", "purrr", "broom", "emmeans",
  "ggplot2", "forcats", "stringr", "scales",
  "cowplot", "tibble"
)

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(broom)
  library(emmeans)
  library(ggplot2)
  library(forcats)
  library(stringr)
  library(scales)
  library(cowplot)
  library(tibble)
})

# 🔹 1. Lendo obj seurat
seurat_obj <- readRDS("/data04/projects04/MarianaBoroni/spatial_ovary/data/Pedro_data/adata_harmony_subset_T_no8_.RDS")  # Se necessário carregar o objeto

meta <- read.csv("metadata_updated.csv", row.names = 1)

# garantir mesma ordem
meta <- meta[colnames(seurat_obj), ]

# checar alinhamento
all(rownames(meta) == colnames(seurat_obj))
seurat_obj$BroadCelltypes <- meta$BroadCelltypes
seurat_obj$Celltypes <- meta$Celltypes

sample_col <- "library_id"      # unidade amostral (réplica biológica / paciente)
group_col  <- "classificacao"   # coluna com MB / PB / controle / NP

# ---- composição ---------------------------------------------------------

pseudocount          <- 1e-6
min_cells_per_sample <- 30      # amostras com menos células que isso são descartadas da composição

levels(seurat_obj@meta.data$classificacao) <- c(
  levels(seurat_obj@meta.data$classificacao),
  "NP"
)

seurat_obj@meta.data$classificacao[
  grepl("^HNP", seurat_obj@meta.data$library_id)
] <- "NP"

# ordem desejada dos grupos (níveis ausentes no objeto são descartados automaticamente)
group_levels <- c("controle", "PB", "MB", "NP")

group_labels <- c(
  "controle" = "Controle",
  "PB"       = "Paucibacilar (PB)",
  "MB"       = "Multibacilar (MB)",
  "NP"       = "Neural pura (NP)"
)

group_palette <- c(
  "controle" = "#B3CDE3",
  "PB"       = "#FBB4AE",
  "MB"       = "#E15759",
  "NP"       = "#8C6BB1"
)

# ---- tamanho das letras (ajuste tudo aqui) ------------------------------

font_axis_title   <- 15   # títulos dos eixos (x/y)
font_axis_text    <- 13   # números/nomes dos eixos
font_axis_text_x  <- 11   # texto do eixo x quando há muitas categorias (ex: heatmap)
font_legend_title <- 14   # título da legenda
font_legend_text  <- 12   # itens da legenda
font_strip        <- 14   # títulos de facetas/painéis
font_plot_title   <- 16   # título do gráfico
font_sig_label    <- 6    # tamanho do "*" / "." de significância no heatmap

# ============================================================
# Carregar objeto e preparar metadata
# ============================================================

meta <- seurat_obj@meta.data

stopifnot(all(c(sample_col, group_col, "BroadCelltypes", "Celltypes") %in% colnames(meta)))

present_levels <- group_levels[group_levels %in% unique(as.character(meta[[group_col]]))]
group_palette  <- group_palette[present_levels]
group_labels   <- group_labels[present_levels]

meta[[group_col]]  <- factor(as.character(meta[[group_col]]), levels = present_levels)
meta[[sample_col]] <- as.character(meta[[sample_col]])
meta <- meta[!is.na(meta[[group_col]]), ]

# ============================================================
# Tema base (usa os tamanhos definidos no bloco de parâmetros)
# ============================================================

base_theme <- theme_classic(base_size = font_axis_text) +
  theme(
    axis.title    = element_text(size = font_axis_title, color = "black"),
    axis.text     = element_text(size = font_axis_text, color = "black"),
    legend.title  = element_text(size = font_legend_title, face = "bold"),
    legend.text   = element_text(size = font_legend_text),
    strip.text    = element_text(size = font_strip, face = "bold"),
    plot.title    = element_text(size = font_plot_title, face = "bold")
  )

# ============================================================
# Composição por amostra (proporção de células por tipo)
# ============================================================

build_composition <- function(meta, cell_col, sample_col, group_col, min_cells = 0) {
  
  sample_group <- unique(meta[, c(sample_col, group_col)])
  colnames(sample_group) <- c("sample", "group")
  
  counts <- meta %>%
    group_by(.data[[sample_col]], .data[[cell_col]]) %>%
    summarise(n = n(), .groups = "drop")
  colnames(counts)[1:2] <- c("sample", "cell_type")
  
  wide <- counts %>%
    pivot_wider(names_from = cell_type, values_from = n, values_fill = 0)
  
  cell_cols <- setdiff(colnames(wide), "sample")
  
  totals <- rowSums(wide[, cell_cols])
  keep <- totals >= min_cells
  if (any(!keep)) {
    message(
      "Removendo amostras com menos de ", min_cells, " células: ",
      paste(wide$sample[!keep], collapse = ", ")
    )
  }
  wide <- wide[keep, ]
  
  prop <- wide
  prop[, cell_cols] <- prop[, cell_cols] / rowSums(prop[, cell_cols])
  
  list(
    counts     = wide,
    proportion = left_join(prop, sample_group, by = "sample"),
    cell_cols  = cell_cols
  )
}

clr_transform <- function(mat, pseudo = 1e-6) {
  mat <- as.matrix(mat) + pseudo
  gm <- exp(rowMeans(log(mat)))
  as.data.frame(log(mat / gm))
}

# ============================================================
# Modelo: ANOVA (CLR ~ classificação) + todas as comparações
# pareadas entre os grupos (MB vs PB, MB vs controle, PB vs NP, ...)
# ============================================================

run_group_comparisons <- function(comp, group_col = "group") {
  
  cell_cols <- comp$cell_cols
  prop_df   <- comp$proportion
  
  clr_mat <- clr_transform(prop_df[, cell_cols], pseudo = pseudocount)
  colnames(clr_mat) <- cell_cols
  
  clr_df <- bind_cols(prop_df[, c("sample", group_col)], clr_mat)
  
  run_one <- function(cell) {
    form <- as.formula(paste0("`", cell, "` ~ ", group_col))
    fit  <- lm(form, data = clr_df)
    
    emm <- emmeans(fit, specs = group_col)
    ctr <- pairs(emm, adjust = "none")
    
    broom::tidy(ctr, conf.int = TRUE) %>%
      transmute(
        cell_population = cell,
        comparison       = contrast,
        estimate         = estimate,
        std.error        = std.error,
        conf.low         = conf.low,
        conf.high        = conf.high,
        p.value          = p.value
      )
  }
  
  map_dfr(cell_cols, run_one) %>%
    mutate(
      FDR = p.adjust(p.value, method = "BH"),
      sig_label = case_when(
        FDR < 0.05     ~ "*",
        p.value < 0.05 ~ ".",
        TRUE           ~ ""
      )
    )
}

# ============================================================
# Rodar para BroadCelltypes e Celltypes
# ============================================================

comp_broad <- build_composition(meta, "BroadCelltypes", sample_col, group_col, min_cells_per_sample)
comp_fine  <- build_composition(meta, "Celltypes",      sample_col, group_col, min_cells_per_sample)

results_broad <- run_group_comparisons(comp_broad)
results_fine  <- run_group_comparisons(comp_fine)

write.csv(results_broad, "BroadCelltypes_classificacao_comparisons.csv", row.names = FALSE)
write.csv(results_fine,  "Celltypes_classificacao_comparisons.csv",      row.names = FALSE)

# ============================================================
# Painel de composição: barras empilhadas por amostra + barras
# médias por grupo (classificação)
# Inclui legenda de tipo celular E legenda de classificação (grupo)
# ============================================================

make_composition_panel <- function(comp, group_palette, group_labels, cell_palette = NULL,
                                   sample_col = "sample", group_col = "group") {
  
  cell_cols <- comp$cell_cols
  prop_df   <- comp$proportion
  
  if (is.null(cell_palette)) {
    cell_palette <- setNames(scales::hue_pal()(length(cell_cols)), cell_cols)
  }
  
  hc  <- hclust(as.dist(1 - cor(t(as.matrix(prop_df[, cell_cols])), method = "pearson")), method = "ward.D2")
  ord <- hc$order
  prop_df[[sample_col]] <- factor(prop_df[[sample_col]], levels = prop_df[[sample_col]][ord])
  
  # --- tira empilhada com legenda de classificação (para extrair depois) ---
  ann_legend_src <- ggplot(prop_df, aes(x = .data[[sample_col]], y = 1, fill = .data[[group_col]])) +
    geom_tile() +
    scale_fill_manual(values = group_palette, labels = group_labels, drop = FALSE, name = "Classificação") +
    base_theme +
    theme(legend.position = "right")
  
  legend_group <- cowplot::get_legend(ann_legend_src)
  
  ann <- ann_legend_src +
    theme_void() +
    theme(legend.position = "none", plot.margin = margin(0, 2, 0, 2))
  
  long <- prop_df %>%
    pivot_longer(all_of(cell_cols), names_to = "cell_type", values_to = "fraction")
  
  bar <- ggplot(long, aes(x = .data[[sample_col]], y = fraction, fill = cell_type)) +
    geom_col(width = 1) +
    scale_fill_manual(values = cell_palette, name = "Tipo celular") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(labels = percent_format(), expand = c(0, 0)) +
    labs(x = NULL, y = "Fração de células") +
    base_theme +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      legend.position = "none"
    )
  
  legend_cells <- cowplot::get_legend(bar + theme(legend.position = "right"))
  
  mean_df <- prop_df %>%
    group_by(.data[[group_col]]) %>%
    summarise(across(all_of(cell_cols), mean), .groups = "drop") %>%
    pivot_longer(all_of(cell_cols), names_to = "cell_type", values_to = "fraction")
  
  mean_bar <- ggplot(mean_df, aes(x = .data[[group_col]], y = fraction, fill = cell_type)) +
    geom_col(width = 0.7, color = "white", linewidth = 0.15) +
    scale_fill_manual(values = cell_palette, guide = "none") +
    scale_x_discrete(labels = group_labels) +
    scale_y_continuous(labels = percent_format(), expand = c(0, 0)) +
    labs(x = NULL, y = "Fração média") +
    base_theme +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  legends <- cowplot::plot_grid(legend_group, legend_cells, ncol = 1)
  
  plot_area <- cowplot::plot_grid(
    cowplot::plot_grid(ann, bar, ncol = 1, align = "v", rel_heights = c(0.06, 1)),
    mean_bar,
    ncol = 2, rel_widths = c(2.2, 1)
  )
  
  cowplot::plot_grid(plot_area, legends, ncol = 2, rel_widths = c(4, 1))
}

panel_broad <- make_composition_panel(comp_broad, group_palette, group_labels)
panel_fine  <- make_composition_panel(comp_fine,  group_palette, group_labels)

# ============================================================
# Heatmap de comparações pareadas (efeito CLR + significância)
# ============================================================

make_comparison_heatmap <- function(results, title = NULL) {
  
  results <- results %>%
    mutate(
      comparison = as.character(comparison),
      cell_population = factor(cell_population, levels = rev(sort(unique(cell_population))))
    )
  
  ggplot(results, aes(x = comparison, y = cell_population, fill = estimate)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = sig_label), fontface = "bold", size = font_sig_label) +
    scale_fill_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B",
      midpoint = 0, name = "Efeito\n(CLR)"
    ) +
    labs(x = NULL, y = NULL, title = title) +
    base_theme +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = font_axis_text_x),
      legend.position = "right"
    )
}

heatmap_broad <- make_comparison_heatmap(results_broad, "BroadCelltypes")
heatmap_fine  <- make_comparison_heatmap(results_fine,  "Celltypes")

# ============================================================
# Salvar
# ============================================================

ggsave("Composition_BroadCelltypes_classificacao.png", panel_broad,
       width = 12, height = 5.5, dpi = 400, bg = "white")
ggsave("Composition_Celltypes_classificacao.png", panel_fine,
       width = 12, height = 7, dpi = 400, bg = "white")

ggsave("Comparisons_BroadCelltypes_classificacao.png", heatmap_broad,
       width = 8, height = 5.5, dpi = 400, bg = "white")
ggsave("Comparisons_Celltypes_classificacao.png", heatmap_fine,
       width = 9, height = 9, dpi = 400, bg = "white")

(print(panel_broad) | print(panel_fine)) / (print(heatmap_broad) | print(heatmap_fine))

