# Composição celular por biblioteca: proporção → CLR → lm → BH → forest plots.
#
# Adaptado, sem mudar o método, de dois scripts do projeto espacial:
#   A  /mnt/SATA/spatialPaper/src/integration/composition_condition_by_resolution.R
#      (painéis A–E: composição com dendrograma pvclust, média por grupo, forests C/D/E)
#   B  /mnt/SATA/spatialPaper/src/integration/composition_classificacao_forest.R
#      (forests de classificação par a par e do estímulo dentro de cada grupo)
# As funções vêm desses scripts; os blocos novos ou alterados estão marcados com [P15].
# O que mudou e por quê está em raw/analises_previas/P15_composicao.py, seção "Adaptação".
#
# Uso: Rscript --vanilla composicao_forest.R <dir_entrada> <dir_figuras> <dir_tabelas>
#   <dir_entrada>/obs.csv: uma linha por célula (library_id, patient, classificacao,
#   estimulo, atlas_L2, atlas_L1), escrito pelo P15 só com as células CD4⁺ com rótulo do atlas.

args <- commandArgs(trailingOnly = TRUE)
dir_in  <- args[1]
fig_dir <- args[2]
tab_dir <- args[3]
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(broom)
  library(emmeans)
  library(ggplot2)
  # [P15] sem library(forcats): carregado nas referências, mas nenhuma função dele é usada
  library(stringr)
  library(scales)
  library(cowplot)
  library(tibble)
  library(pvclust)
})

# ============================================================
# 1. Parâmetros
# ============================================================

meta <- read.csv(file.path(dir_in, "obs.csv"), colClasses = "character", encoding = "UTF-8")

sample_col    <- "library_id"     # unidade amostral (biblioteca)
patient_col   <- "patient"        # [P15] cada paciente tem uma biblioteca NS e uma ML
condition_col <- "estimulo"       # "nao estimulado" / "estimulado"  (A: condition)
group_col     <- "classificacao"  # controle / PB / MB / NP          (A: response; B: classificacao)

# >>> granularidades (B roda uma figura para cada) <<<
granularities <- c("atlas_L2", "atlas_L1")

# [P15] ordem dos tipos = a do atlas (Yasumizu 2024), em vez da alfabética
cell_levels <- list(
  atlas_L2 = c("Tnaive", "TnaiveAct", "TnaiveMX1", "TnaiveSOX4", "TcmTh0", "TcmTh0Act", "TcmTh17",
               "TcmTh2", "TcmTfh", "TcmPHLDA3", "TemTh1", "TemTh117", "TemTh1pre", "TemTph",
               "TemraTh1", "TregNaive", "TregAct", "TregEff"),
  atlas_L1 = c("Tnaive", "Tcm", "Tem", "Temra", "Treg")
)
# [P15] <dir_entrada>/niveis.tsv, se existir, troca as granularidades: uma linha por coluna,
# "coluna<TAB>tipo1<TAB>tipo2...", na ordem das figuras (ex.: anotacao e grupo do P09)
if (file.exists(file.path(dir_in, "niveis.tsv"))) {
  linhas <- strsplit(readLines(file.path(dir_in, "niveis.tsv"), encoding = "UTF-8"), "\t")
  cell_levels   <- setNames(lapply(linhas, `[`, -1), vapply(linhas, `[`, "", 1))
  granularities <- names(cell_levels)
}

# ---- condição (painel E) ----
# referência = não estimulado  => efeito CLR positivo = tipo enriquecido no estimulado (ML)
condition_ref     <- "nao estimulado"
condition_stim    <- "estimulado"
condition_labels  <- c("nao estimulado" = "Não estimulado (NS)", "estimulado" = "Estimulado (ML)")
# [P15] cores do estímulo usadas em todo o Reseq (no B, #4C78A8/#E15759, que colidia com o MB)
condition_palette <- c("nao estimulado" = "#2a78d6", "estimulado" = "#eb6834")

# ---- classificação (B) ----
group_levels  <- c("controle", "PB", "MB", "NP")
group_labels  <- c("controle" = "Controle", "PB" = "Paucibacilar (PB)",
                   "MB" = "Multibacilar (MB)", "NP" = "Neural pura (NP)")
group_palette <- c("controle" = "#B3CDE3", "PB" = "#FBB4AE", "MB" = "#E15759", "NP" = "#8C6BB1")

# ---- painéis C e D (A: FB × UB dentro de cada condição) ----
# [P15] PB × controle dentro de NS e dentro de ML; referência = controle
#   => efeito CLR positivo = tipo enriquecido no PB. MB e NP têm 1 biblioteca por estímulo.
resp_levels <- c("controle", "PB")
resp_ref    <- "controle"

# ---- cores dos tipos (B: 13 originais + 8 de alto contraste) ----
NICHES_COLORS <- c(
  "#cd3f35", "#62b74f", "#9e5ecf", "#c54eae", "#538342", "#676bc6",
  "#db9448", "#40bbce", "#1d19e6", "#e7b4e9", "#4ebba0", "#ca6f87",
  "#615963",
  "#eacb2a", "#7d431c", "#144633", "#ff7a45", "#b6d354", "#421045",
  "#8bd0eb", "#82293b"
)

# ---- composição ----
pseudocount          <- 1e-6
min_cells_per_sample <- 30   # amostras com menos células que isso são descartadas

# ---- clusterização hierárquica das amostras (painel A) ----
clust_use_clr    <- TRUE
clust_dist       <- "correlation"
clust_linkage    <- "ward.D2"
clust_nboot      <- 1000
clust_alpha      <- 0.95
clust_seed       <- 1234
clust_au_min_lab <- 0.70

CLUSTER_COLORS <- c('#1B9E77', '#D95F02', '#7570B3', '#E7298A',
                    '#66A61E', '#E6AB02', '#A6761D', '#666666')

# ---- tamanho das letras ----
font_axis_title   <- 14
font_axis_text    <- 12
font_legend_title <- 13
font_legend_text  <- 11
font_strip        <- 13
font_plot_title   <- 14

NS_LABEL <- "Não significativo"   # [P15] rótulos em português (no original, "Not significant")

# ============================================================
# 2. Metadata
# ============================================================

stopifnot(all(c(sample_col, patient_col, condition_col, group_col, granularities) %in% colnames(meta)))
meta <- meta[!is.na(meta[[condition_col]]) & !is.na(meta[[group_col]]), ]
present_levels <- group_levels[group_levels %in% unique(meta[[group_col]])]
group_palette  <- group_palette[present_levels]
group_labels   <- group_labels[present_levels]
est_order      <- c(condition_ref, condition_stim)
sample_covars  <- c(condition_col, group_col, patient_col)

# [P15] ordem dos tipos pelo atlas; tipos fora da lista vão para o fim, em ordem alfabética
order_cells <- function(x, gran) {
  u <- unique(as.character(x))
  c(intersect(cell_levels[[gran]], u), sort(setdiff(u, cell_levels[[gran]])))
}

# [P15] números com vírgula decimal nas figuras
fmt_num <- function(x, d = 2) vapply(x, function(v) {
  if (is.na(v)) return("")
  format(signif(v, d), decimal.mark = ",", scientific = abs(v) < 1e-3 && v != 0, trim = TRUE)
}, character(1))

# ============================================================
# 3. Tema base
# ============================================================

base_theme <- theme_classic(base_size = font_axis_text) +
  theme(
    axis.title    = element_text(size = font_axis_title, color = "black"),
    axis.text     = element_text(size = font_axis_text, color = "black"),
    legend.title  = element_text(size = font_legend_title, face = "bold"),
    legend.text   = element_text(size = font_legend_text),
    strip.text    = element_text(size = font_strip, face = "bold"),
    plot.title    = element_text(size = font_plot_title, face = "bold"),
    plot.subtitle = element_text(size = font_legend_text, color = "grey30")
  )

# ============================================================
# 4. Composição por amostra (proporção de células por tipo)
# ============================================================

build_composition <- function(meta, cell_col, sample_col, sample_covars, min_cells = 0, gran = cell_col) {

  sample_meta <- unique(meta[, c(sample_col, sample_covars)])
  colnames(sample_meta)[1] <- "sample"
  stopifnot(!anyDuplicated(sample_meta$sample))   # covariáveis constantes dentro da amostra

  counts <- meta %>%
    mutate(!!cell_col := as.character(.data[[cell_col]])) %>%
    group_by(.data[[sample_col]], .data[[cell_col]]) %>%
    summarise(n = n(), .groups = "drop")
  colnames(counts)[1:2] <- c("sample", "cell_type")

  wide <- counts %>%
    pivot_wider(names_from = cell_type, values_from = n, values_fill = 0)

  cell_cols <- order_cells(setdiff(colnames(wide), "sample"), gran)

  totals <- rowSums(wide[, cell_cols])
  keep <- totals >= min_cells
  if (any(!keep)) {
    message("Removendo amostras com menos de ", min_cells, " células: ",
            paste(wide$sample[!keep], collapse = ", "))
  }
  wide <- wide[keep, ]

  prop <- wide
  prop[, cell_cols] <- prop[, cell_cols] / rowSums(prop[, cell_cols])

  list(
    counts     = wide,
    proportion = left_join(prop, sample_meta, by = "sample"),
    cell_cols  = cell_cols
  )
}

clr_transform <- function(mat, pseudo = 1e-6) {
  mat <- as.matrix(mat) + pseudo
  gm <- exp(rowMeans(log(mat)))
  as.data.frame(log(mat / gm))
}

# [P15] CLR das amostras `samples`. "proporcao" = o dos scripts R (proporção + 1e-6);
# "contagem" = o de src/forest_composicao/composicao.py, (k + 0,5) / (N + 0,5 K), só na sensibilidade.
clr_of <- function(comp, samples, pseudo_mode = "proporcao") {
  cc <- comp$cell_cols
  if (pseudo_mode == "contagem") {
    k <- as.matrix(comp$counts[match(samples, comp$counts$sample), cc])
    p <- (k + 0.5) / (rowSums(k) + 0.5 * ncol(k))
    out <- as.data.frame(log(p / exp(rowMeans(log(p)))))
  } else {
    out <- clr_transform(comp$proportion[match(samples, comp$proportion$sample), cc], pseudo = pseudocount)
  }
  colnames(out) <- cc
  out
}

# ============================================================
# 5. Modelo (A): lm(CLR ~ variável [+ covariáveis]) + IC por tipo
#    - compare_col : variável comparada
#    - subset_col/subset_val : restringe as amostras (ex.: PB × controle só nas NS)
#    - [P15] keep_levels : níveis de compare_col mantidos (ex.: controle e PB)
#    - [P15] covars : covariáveis do modelo (o paciente, no E pareado)
# ============================================================

run_composition_lm <- function(comp, compare_col, ref_level = NULL,
                               subset_col = NULL, subset_val = NULL,
                               keep_levels = NULL, covars = NULL, pseudo_mode = "proporcao") {

  cell_cols <- comp$cell_cols
  df <- comp$proportion

  if (!is.null(subset_col)) {
    df <- df[as.character(df[[subset_col]]) == subset_val, , drop = FALSE]
  }
  if (!is.null(keep_levels)) df <- df[as.character(df[[compare_col]]) %in% keep_levels, , drop = FALSE]
  df <- df[!is.na(df[[compare_col]]), , drop = FALSE]

  clr_mat <- clr_of(comp, df$sample, pseudo_mode)

  clr_df <- bind_cols(df[, unique(c("sample", compare_col, covars))], clr_mat)
  clr_df[[compare_col]] <- factor(as.character(clr_df[[compare_col]]))
  if (!is.null(ref_level) && ref_level %in% levels(clr_df[[compare_col]])) {
    clr_df[[compare_col]] <- relevel(clr_df[[compare_col]], ref = ref_level)
  }
  ref <- levels(clr_df[[compare_col]])[1]

  # só testa tipos com variabilidade (evita tipo ausente em todo o subset)
  v <- vapply(cell_cols, function(c) stats::var(clr_df[[c]]), numeric(1))
  testable <- cell_cols[!is.na(v) & v > 0]

  rhs <- paste(c(compare_col, covars), collapse = " + ")
  run_one <- function(cell) {
    form <- as.formula(paste0("`", cell, "` ~ ", rhs))
    fit  <- lm(form, data = clr_df)
    broom::tidy(fit, conf.int = TRUE) %>%
      filter(str_detect(term, paste0("^", compare_col))) %>%
      transmute(
        cell_population = cell,
        level_vs_ref = str_remove(term, paste0("^", compare_col)),
        reference    = ref,
        estimate, std.error, conf.low, conf.high, p.value,
        df_resid = fit$df.residual
      )
  }

  tested <- map_dfr(testable, run_one) %>%
    mutate(FDR = p.adjust(p.value, method = "BH"))

  skipped <- setdiff(cell_cols, testable)
  if (length(skipped) > 0) {
    tested <- bind_rows(tested, tibble(
      cell_population = skipped, level_vs_ref = NA_character_, reference = ref,
      estimate = NA_real_, std.error = NA_real_, conf.low = NA_real_,
      conf.high = NA_real_, p.value = NA_real_, FDR = NA_real_
    ))
  }

  tested %>%
    mutate(
      direction = ifelse(is.na(estimate), NA_character_,
                         ifelse(estimate > 0, level_vs_ref, reference)),
      sig_shape = case_when(
        is.na(p.value) ~ NS_LABEL,
        FDR < 0.05     ~ "FDR < 0.05",
        p.value < 0.05 ~ "p < 0.05",
        TRUE           ~ NS_LABEL
      ),
      plot_color = ifelse(sig_shape == NS_LABEL | is.na(direction), NS_LABEL, direction),
      n_amostras = nrow(clr_df),
      modelo = paste0("CLR ~ ", rhs),
      pseudo = pseudo_mode
    )
}

# n por nível (para o subtítulo dos forest plots)
n_label <- function(comp, compare_col, subset_col = NULL, subset_val = NULL, keep_levels = NULL) {
  df <- comp$proportion
  if (!is.null(subset_col)) df <- df[as.character(df[[subset_col]]) == subset_val, ]
  if (!is.null(keep_levels)) df <- df[as.character(df[[compare_col]]) %in% keep_levels, ]
  tb <- table(as.character(df[[compare_col]]))
  paste0("n bibliotecas: ", paste(paste0(names(tb), "=", as.integer(tb)), collapse = ", "))
}

# ============================================================
# 5b. Modelo (B): lm(CLR ~ variável) + comparações pareadas (emmeans)
#     FDR (BH) por comparação (entre tipos).
# ============================================================

run_pairwise <- function(comp, compare_col, group_order = NULL,
                         subset_col = NULL, subset_val = NULL, min_per_level = 2,
                         pseudo_mode = "proporcao") {

  cell_cols <- comp$cell_cols
  df <- comp$proportion

  if (!is.null(subset_col)) df <- df[as.character(df[[subset_col]]) == subset_val, , drop = FALSE]
  df <- df[!is.na(df[[compare_col]]), , drop = FALSE]

  lv <- table(as.character(df[[compare_col]]))
  df <- df[as.character(df[[compare_col]]) %in% names(lv)[lv >= min_per_level], , drop = FALSE]
  if (length(unique(df[[compare_col]])) < 2) return(NULL)

  clr_mat <- clr_of(comp, df$sample, pseudo_mode)

  clr_df <- bind_cols(df[, c("sample", compare_col)], clr_mat)
  ord <- if (is.null(group_order)) sort(unique(as.character(clr_df[[compare_col]])))
         else intersect(group_order, unique(as.character(clr_df[[compare_col]])))
  clr_df[[compare_col]] <- factor(as.character(clr_df[[compare_col]]), levels = ord)

  v <- vapply(cell_cols, function(c) stats::var(clr_df[[c]]), numeric(1))
  testable <- cell_cols[!is.na(v) & v > 0]
  if (length(testable) == 0) return(NULL)

  run_one <- function(cell) {
    form <- as.formula(paste0("`", cell, "` ~ ", compare_col))
    fit  <- lm(form, data = clr_df)
    emm  <- emmeans(fit, specs = compare_col)
    ctr  <- pairs(emm, adjust = "none")
    broom::tidy(ctr, conf.int = TRUE) %>%
      transmute(
        cell_population = cell,
        comparison = as.character(contrast),
        estimate, std.error, conf.low, conf.high, p.value
      )
  }

  suppressMessages(map_dfr(testable, run_one)) %>%
    separate(comparison, into = c("g1", "g2"), sep = " - ", remove = FALSE) %>%
    group_by(comparison) %>%
    mutate(FDR = p.adjust(p.value, method = "BH")) %>%
    ungroup() %>%
    mutate(
      direction = ifelse(estimate > 0, g1, g2),
      sig_shape = case_when(
        FDR < 0.05     ~ "FDR < 0.05",
        p.value < 0.05 ~ "p < 0.05",
        TRUE           ~ NS_LABEL
      ),
      plot_color = ifelse(sig_shape == NS_LABEL, NS_LABEL, direction)
    )
}

level_counts <- function(comp, col, subset_col = NULL, subset_val = NULL) {
  df <- comp$proportion
  if (!is.null(subset_col)) df <- df[as.character(df[[subset_col]]) == subset_val, ]
  table(as.character(df[[col]]))
}

n_subtitle <- function(counts) paste0("n bibliotecas: ", paste(paste0(names(counts), "=", as.integer(counts)), collapse = ", "))

# ============================================================
# 6. Clusterização hierárquica não supervisionada com bootstrap (pvclust)
# ============================================================

cluster_samples <- function(comp, sample_col = "sample",
                            use_clr  = clust_use_clr,
                            dist_method   = clust_dist,
                            hclust_method = clust_linkage,
                            nboot = clust_nboot, alpha = clust_alpha,
                            seed  = clust_seed) {

  df  <- comp$proportion
  mat <- as.matrix(df[, comp$cell_cols])
  rownames(mat) <- as.character(df[[sample_col]])
  if (use_clr) mat <- as.matrix(clr_transform(mat, pseudo = pseudocount))

  X <- t(mat)

  message("Clusterização hierárquica: ", ncol(X), " amostras x ", nrow(X),
          " tipos | dist = ", dist_method, " | linkage = ", hclust_method,
          " | CLR = ", use_clr, " | nboot = ", nboot)
  if (nrow(X) < 10) {
    warning("Poucos tipos (", nrow(X), ") para reamostragem bootstrap; ",
            "os valores de AU serão pouco informativos.")
  }

  set.seed(seed)
  pv <- tryCatch(
    pvclust::pvclust(X, method.dist = dist_method, method.hclust = hclust_method,
                     nboot = nboot, quiet = TRUE),
    error = function(e) {
      warning("pvclust falhou (", conditionMessage(e),
              "); usando hclust simples, sem valores de bootstrap.")
      NULL
    }
  )

  if (!is.null(pv)) {
    hc    <- pv$hclust
    edges <- pv$edges
    au    <- edges$au
    bp    <- edges$bp
    picked <- tryCatch(pvclust::pvpick(pv, alpha = alpha, pv = "au", type = "geq")$clusters,
                       error = function(e) list())
    picked <- Filter(function(z) length(z) > 0, picked)
  } else {
    d  <- if (identical(dist_method, "correlation")) as.dist(1 - cor(X, method = "pearson"))
    else dist(t(X), method = dist_method)
    hc <- hclust(d, method = hclust_method)
    au <- bp <- rep(NA_real_, length(hc$height))
    picked <- list()
  }

  memb <- setNames(rep("Não suportado", ncol(X)), colnames(X))
  if (length(picked) > 0) {
    for (i in seq_along(picked)) memb[picked[[i]]] <- paste0("Cluster ", i)
  }
  cluster_levels <- c(if (length(picked) > 0) paste0("Cluster ", seq_along(picked)),
                      "Não suportado")

  list(
    hc       = hc,
    au       = au,
    bp       = bp,
    pvclust  = pv,
    nboot    = if (is.null(pv)) NA_integer_ else nboot,
    alpha    = alpha,
    order    = if (!is.null(hc$labels)) hc$labels[hc$order] else as.character(df[[sample_col]])[hc$order],
    clusters = tibble(sample = names(memb), sample_cluster = unname(memb)),
    levels   = cluster_levels
  )
}

cluster_annotation <- function(clust, col = "sample_cluster") {
  lev <- clust$levels
  k   <- sum(lev != "Não suportado")
  cols <- setNames(rep("grey85", length(lev)), lev)
  if (k > 0) {
    pal <- CLUSTER_COLORS
    if (k > length(pal)) pal <- c(pal, scales::hue_pal()(k - length(pal)))
    cols[seq_len(k)] <- pal[seq_len(k)]
  }
  list(col = col, palette = cols, labels = setNames(lev, lev),
       title = paste0("Cluster (AU>=", round(clust$alpha * 100), "%)"))
}

hclust_layout <- function(hc) {
  n   <- length(hc$order)
  hgt <- hc$height
  pos <- numeric(max(n - 1, 1))
  leaf_x <- match(seq_len(n), hc$order)
  xpos <- function(k) if (k < 0) leaf_x[-k] else pos[k]
  ypos <- function(k) if (k < 0) 0 else hgt[k]

  segs <- vector("list", n - 1)
  for (i in seq_len(n - 1)) {
    a <- hc$merge[i, 1]; b <- hc$merge[i, 2]
    xa <- xpos(a); xb <- xpos(b)
    ya <- ypos(a); yb <- ypos(b)
    pos[i] <- (xa + xb) / 2
    segs[[i]] <- data.frame(x    = c(xa, xa, xb),
                            xend = c(xa, xb, xb),
                            y    = c(ya, hgt[i], hgt[i]),
                            yend = c(hgt[i], hgt[i], yb))
  }
  list(segments = do.call(rbind, segs),
       nodes    = data.frame(node = seq_len(n - 1), x = pos, y = hgt))
}

make_dendrogram_plot <- function(clust, n_samples,
                                 au_label_min = clust_au_min_lab,
                                 au_threshold = clust_alpha) {

  lay <- hclust_layout(clust$hc)

  p <- ggplot() +
    geom_segment(data = lay$segments,
                 aes(x = x, xend = xend, y = y, yend = yend),
                 linewidth = 0.35, colour = "grey25", lineend = "round") +
    scale_x_continuous(limits = c(0.5, n_samples + 0.5), expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
    theme_void() +
    theme(plot.margin = margin(2, 2, 0, 2),
          plot.title  = element_text(size = font_legend_text - 1, colour = "grey30", hjust = 1))

  if (all(is.na(clust$au))) return(p)

  nodes <- cbind(lay$nodes, au = clust$au[lay$nodes$node], bp = clust$bp[lay$nodes$node])
  lab   <- nodes[is.finite(nodes$au) & nodes$au >= au_label_min, , drop = FALSE]

  p <- p + labs(title = paste0("AU (%) — bootstrap pvclust, n = ", clust$nboot,
                               "; vermelho: AU >= ", round(au_threshold * 100), "%"))
  if (nrow(lab) > 0) {
    p <- p + geom_text(data = lab, aes(x = x, y = y, label = round(au * 100)),
                       vjust = -0.30, hjust = 0.5, size = 2.1,
                       colour = ifelse(lab$au >= au_threshold, "#B2182B", "grey40"))
  }
  p
}

# ============================================================
# 6b. Faixa de anotação (estímulo / classificação / cluster) para o painel A
# ============================================================

get_legend_grob <- function(p) {
  if ("get_plot_component" %in% getNamespaceExports("cowplot")) {
    g <- tryCatch(cowplot::get_plot_component(p, "guide-box-right"), error = function(e) NULL)
    if (!is.null(g) && !inherits(g, "zeroGrob")) return(g)
  }
  suppressWarnings(cowplot::get_legend(p))
}

grob_height_cm <- function(g, default = 1) {
  if (is.null(g)) return(default)
  opened <- FALSE
  if (is.null(grDevices::dev.list())) { grDevices::pdf(NULL); opened <- TRUE }
  on.exit(if (opened) grDevices::dev.off(), add = TRUE)
  h <- tryCatch(grid::convertHeight(grid::grobHeight(g), "cm", valueOnly = TRUE),
                error = function(e) NA_real_)
  if (!is.finite(h) || h <= 0) default else h
}

stack_legends <- function(legends) {
  legends <- Filter(Negate(is.null), legends)
  hs <- vapply(legends, grob_height_cm, numeric(1))
  cowplot::plot_grid(plotlist = legends, ncol = 1, rel_heights = hs)
}

make_ann_strip <- function(df, sample_col, col, palette, labels, title) {
  d <- df
  lv <- union(names(palette), unique(as.character(d[[col]])))
  d[[col]] <- factor(as.character(d[[col]]), levels = lv)

  src <- ggplot(d, aes(x = .data[[sample_col]], y = 1, fill = .data[[col]])) +
    geom_tile() +
    scale_fill_manual(values = palette, labels = labels, drop = FALSE, name = title,
                      na.value = "white") +
    base_theme +
    theme(legend.position      = "right",
          legend.justification = "left",
          legend.margin        = margin(2, 2, 2, 2),
          legend.key.size      = grid::unit(0.42, "cm"))
  list(
    strip  = src + theme_void() + theme(legend.position = "none", plot.margin = margin(0, 2, 0, 2)),
    legend = get_legend_grob(src)
  )
}

# ============================================================
# 7. Painel A: dendrograma (bootstrap) + anotações + composição empilhada
# ============================================================

make_composition_stacked <- function(comp, niche_palette, annotations, clust,
                                     sample_col = "sample", panel_label = NULL,
                                     show_dendrogram = TRUE) {

  cell_cols <- comp$cell_cols
  df <- comp$proportion

  df[[sample_col]] <- factor(as.character(df[[sample_col]]), levels = clust$order)
  n_samples <- nlevels(df[[sample_col]])

  strips      <- lapply(annotations, function(a) make_ann_strip(df, sample_col, a$col, a$palette, a$labels, a$title))
  strip_plots <- lapply(strips, `[[`, "strip")
  strip_legs  <- lapply(strips, `[[`, "legend")

  long <- df %>%
    pivot_longer(all_of(cell_cols), names_to = "niche", values_to = "fraction") %>%
    mutate(niche = factor(niche, levels = cell_cols))

  niche_ncol <- max(1, ceiling(length(cell_cols) / 8))

  # [P15] com 14 amostras o nome da biblioteca cabe no eixo x (no original, oculto: 28 amostras)
  bar <- ggplot(long, aes(x = .data[[sample_col]], y = fraction, fill = niche)) +
    geom_col(width = 1) +
    scale_fill_manual(values = niche_palette, name = "Tipo celular") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(labels = percent_format(decimal.mark = ","), expand = c(0, 0)) +
    labs(x = NULL, y = "Fração de células") +
    base_theme +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = font_axis_text - 2),
          legend.position = "none")

  niche_leg <- get_legend_grob(
    bar + guides(fill = guide_legend(ncol = niche_ncol)) +
      theme(legend.position      = "right",
            legend.justification = "left",
            legend.margin        = margin(2, 2, 2, 2),
            legend.key.size      = grid::unit(0.42, "cm"))
  )

  dendro <- if (show_dendrogram) list(make_dendrogram_plot(clust, n_samples)) else list()

  n_ann <- length(strip_plots)
  main <- cowplot::plot_grid(plotlist = c(dendro, strip_plots, list(bar)),
                             ncol = 1, align = "v", axis = "lr",
                             rel_heights = c(rep(0.55, length(dendro)),
                                             rep(0.07, n_ann), 1.25))

  legs <- stack_legends(c(strip_legs, list(niche_leg)))

  panel <- cowplot::plot_grid(main, legs, ncol = 2,
                              rel_widths = c(4, 0.85 + 0.45 * niche_ncol))

  if (is.null(panel_label)) return(panel)
  cowplot::ggdraw(panel) +
    cowplot::draw_label(panel_label, x = 0, y = 1, hjust = 0, vjust = 1, fontface = "bold", size = 16)
}

# ============================================================
# 8. Painel B: composição média por estímulo × classificação
# ============================================================

make_composition_mean <- function(comp, niche_palette, condition_col, response_col,
                                  condition_labels, response_labels, panel_label = NULL) {

  cell_cols <- comp$cell_cols
  df <- comp$proportion

  mean_df <- df %>%
    group_by(.data[[condition_col]], .data[[response_col]]) %>%
    summarise(across(all_of(cell_cols), mean), .groups = "drop") %>%
    pivot_longer(all_of(cell_cols), names_to = "niche", values_to = "fraction") %>%
    mutate(niche = factor(niche, levels = cell_cols),
           !!response_col := factor(.data[[response_col]], levels = names(response_labels)),
           !!condition_col := factor(.data[[condition_col]], levels = names(condition_labels)))

  p <- ggplot(mean_df, aes(x = .data[[response_col]], y = fraction, fill = niche)) +
    geom_col(width = 0.85, color = "white", linewidth = 0.1) +
    facet_grid(cols = vars(.data[[condition_col]]), labeller = as_labeller(condition_labels)) +
    scale_fill_manual(values = niche_palette, guide = "none") +
    scale_x_discrete(labels = response_labels) +
    scale_y_continuous(labels = percent_format(decimal.mark = ","), expand = c(0, 0)) +
    labs(x = NULL, y = "Fração média") +
    base_theme +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  if (is.null(panel_label)) return(p)
  cowplot::ggdraw(p) +
    cowplot::draw_label(panel_label, x = 0, y = 1, hjust = 0, vjust = 1, fontface = "bold", size = 16)
}

# ============================================================
# 9. Forest plot: effect size + IC por tipo
#    cor = grupo enriquecido ; forma = nível de significância
# ============================================================

make_forest <- function(results, cluster_order, panel_label = NULL, x_axis_label,
                        color_values, color_labels, direction_title = "Enriquecido em",
                        plot_title = NULL, plot_subtitle = NULL) {

  plot_df <- results %>%
    mutate(
      cell_population = factor(cell_population, levels = rev(cluster_order)),
      stat_label = ifelse(is.na(p.value), "",
                          paste0("p=", fmt_num(p.value), "; FDR=", fmt_num(FDR)))
    )

  # [P15] contraste sem IC (1 biblioteca por lado: resíduo com 0 graus de liberdade)
  # usa a faixa das estimativas; no original o range de um vetor vazio quebrava o eixo
  finite_ci <- c(plot_df$conf.low, plot_df$conf.high)
  finite_ci <- finite_ci[is.finite(finite_ci)]
  if (length(finite_ci) < 2) finite_ci <- c(plot_df$estimate[is.finite(plot_df$estimate)], 0)
  x_range <- diff(range(finite_ci))
  if (!is.finite(x_range) || x_range == 0) x_range <- 1
  x_min   <- min(finite_ci) - 0.08 * x_range
  x_label <- max(finite_ci) + 0.15 * x_range
  x_limit <- max(finite_ci) + 0.85 * x_range

  p <- ggplot(plot_df,
              aes(x = estimate, y = cell_population,
                  xmin = conf.low, xmax = conf.high,
                  color = plot_color, shape = sig_shape)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.35, color = "grey55") +
    geom_errorbarh(height = 0.16, linewidth = 0.6, na.rm = TRUE) +
    geom_point(size = 2.6, na.rm = TRUE) +
    geom_text(aes(x = x_label, y = cell_population, label = stat_label),
              inherit.aes = FALSE, hjust = 0, size = 2.6, color = "black") +
    scale_color_manual(values = color_values, labels = color_labels,
                       breaks = names(color_values), drop = FALSE) +
    scale_shape_manual(values = setNames(c(8, 18, 16), c("FDR < 0.05", "p < 0.05", NS_LABEL)),
                       breaks = c("FDR < 0.05", "p < 0.05", NS_LABEL), drop = FALSE) +
    labs(x = x_axis_label, y = "Tipo celular", color = direction_title, shape = "Significância",
         title = plot_title, subtitle = plot_subtitle) +
    coord_cartesian(xlim = c(x_min, x_limit), clip = "off") +
    guides(color = guide_legend(order = 1, override.aes = list(shape = 16, size = 2.6)),
           shape = guide_legend(order = 2, override.aes = list(color = "black", size = 2.6))) +
    base_theme +
    theme(legend.position = "bottom", legend.box = "vertical",
          plot.margin = margin(6, 80, 6, 6))

  if (is.null(panel_label)) return(p)
  cowplot::ggdraw(p) +
    cowplot::draw_label(panel_label, x = 0, y = 1, hjust = 0, vjust = 1, fontface = "bold", size = 16)
}

# ============================================================
# 10. Rodar para cada granularidade
# ============================================================

resp_labels_cd  <- group_labels[resp_levels]
resp_cols  <- c(group_palette[resp_levels], setNames("grey75", NS_LABEL))
resp_labs  <- c(resp_labels_cd,             setNames(NS_LABEL, NS_LABEL))
cond_cols  <- c(condition_palette,          setNames("grey75", NS_LABEL))
cond_labs  <- c(condition_labels,           setNames(NS_LABEL, NS_LABEL))
x_lab_resp <- paste0("Efeito (CLR, vs ", group_labels[[resp_ref]], ")")
x_lab_cond <- paste0("Efeito (CLR, vs ", condition_labels[[condition_ref]], ")")
lab_ns <- condition_labels[[condition_ref]]; lab_ml <- condition_labels[[condition_stim]]

pairs_list <- combn(present_levels, 2, simplify = FALSE)   # controle-PB, controle-MB, ...

for (gran in granularities) {

  message("\n==== Granularidade: ", gran, " ====")
  m <- meta[!is.na(meta[[gran]]) & meta[[gran]] != "", ]

  comp <- build_composition(m, gran, sample_col, sample_covars, min_cells_per_sample, gran = gran)
  comp$proportion[[group_col]] <- factor(as.character(comp$proportion[[group_col]]), levels = present_levels)

  n_niches   <- length(comp$cell_cols)
  niche_cols <- NICHES_COLORS
  if (n_niches > length(niche_cols)) {
    warning("Mais tipos (", n_niches, ") que cores; completando com cores automáticas.")
    niche_cols <- c(niche_cols, scales::hue_pal()(n_niches - length(niche_cols)))
  }
  niche_palette <- setNames(niche_cols[seq_len(n_niches)], comp$cell_cols)
  n_lib <- nrow(comp$proportion); n_pac <- length(unique(comp$proportion[[patient_col]]))

  # --- tabelas de composição ---
  write.csv(comp$counts, file.path(tab_dir, paste0("composicao_", gran, "_contagens.csv")), row.names = FALSE)
  write.csv(comp$proportion, file.path(tab_dir, paste0("composicao_", gran, "_proporcoes.csv")), row.names = FALSE)

  # ==========================================================
  # FIGURA ABCDE (script A)
  # ==========================================================

  resC <- run_composition_lm(comp, group_col, ref_level = resp_ref, keep_levels = resp_levels,
                             subset_col = condition_col, subset_val = condition_ref)
  resD <- run_composition_lm(comp, group_col, ref_level = resp_ref, keep_levels = resp_levels,
                             subset_col = condition_col, subset_val = condition_stim)
  resE <- run_composition_lm(comp, condition_col, ref_level = condition_ref)
  # [P15] E2: o mesmo contraste com o paciente no modelo (as 14 bibliotecas são 7 pares)
  resE2 <- run_composition_lm(comp, condition_col, ref_level = condition_ref, covars = patient_col)

  resC$comparison  <- "C: PB vs Controle | NS"
  resD$comparison  <- "D: PB vs Controle | ML"
  resE$comparison  <- "E: ML vs NS"
  resE2$comparison <- "E2: ML vs NS, pareado por paciente"

  clust <- cluster_samples(comp)
  comp$proportion$sample_cluster <- clust$clusters$sample_cluster[
    match(as.character(comp$proportion$sample), clust$clusters$sample)
  ]
  n_supported <- sum(clust$levels != "Não suportado")
  message(">>> Clusters de amostras com AU >= ", round(clust_alpha * 100), "%: ", n_supported)

  annotations <- list(
    list(col = condition_col, palette = condition_palette, labels = condition_labels, title = "Estímulo"),
    list(col = group_col,     palette = group_palette,     labels = group_labels,     title = "Classificação")
  )
  if (n_supported > 0) annotations <- c(annotations, list(cluster_annotation(clust)))

  panel_A <- make_composition_stacked(comp, niche_palette, annotations, clust, panel_label = "A")
  panel_B <- make_composition_mean(comp, niche_palette, condition_col, group_col,
                                   condition_labels, group_labels, panel_label = "B")

  panel_C <- make_forest(resC, comp$cell_cols, panel_label = "C",
                         plot_title = paste0("PB vs Controle — ", lab_ns),
                         plot_subtitle = n_label(comp, group_col, condition_col, condition_ref, resp_levels),
                         x_axis_label = x_lab_resp, color_values = resp_cols, color_labels = resp_labs)
  panel_D <- make_forest(resD, comp$cell_cols, panel_label = "D",
                         plot_title = paste0("PB vs Controle — ", lab_ml),
                         plot_subtitle = n_label(comp, group_col, condition_col, condition_stim, resp_levels),
                         x_axis_label = x_lab_resp, color_values = resp_cols, color_labels = resp_labs)
  panel_E <- make_forest(resE, comp$cell_cols, panel_label = "E",
                         plot_title = "Estimulado vs Não estimulado",
                         plot_subtitle = n_label(comp, condition_col),
                         x_axis_label = x_lab_cond, color_values = cond_cols, color_labels = cond_labs)
  panel_E2 <- make_forest(resE2, comp$cell_cols, panel_label = "E2",
                          plot_title = "Estimulado vs Não estimulado — pareado",
                          plot_subtitle = paste0(n_pac, " pares NS/ML; paciente no modelo"),
                          x_axis_label = x_lab_cond, color_values = cond_cols, color_labels = cond_labs)

  all_results <- bind_rows(resC, resD, resE, resE2) %>%
    dplyr::select(comparison, modelo, n_amostras, df_resid, cell_population, direction, estimate,
                  std.error, conf.low, conf.high, p.value, FDR, sig_shape)
  write.csv(all_results, file.path(tab_dir, paste0("Comparisons_", gran, "_estimulo_classificacao.csv")),
            row.names = FALSE)

  cluster_table <- comp$proportion %>%
    dplyr::select(sample, all_of(c(condition_col, group_col, patient_col)), sample_cluster) %>%
    dplyr::mutate(order_in_dendrogram = match(as.character(sample), clust$order)) %>%
    dplyr::arrange(order_in_dendrogram)
  write.csv(cluster_table, file.path(tab_dir, paste0("SampleClusters_", gran, ".csv")), row.names = FALSE)

  edges_table <- tibble(
    edge   = seq_along(clust$au),
    height = clust$hc$height,
    AU     = clust$au,
    BP     = clust$bp,
    members = vapply(seq_along(clust$au), function(i) {
      leaves <- function(k) if (k < 0) -k else c(leaves(clust$hc$merge[k, 1]),
                                                 leaves(clust$hc$merge[k, 2]))
      idx <- c(leaves(clust$hc$merge[i, 1]), leaves(clust$hc$merge[i, 2]))
      paste(clust$hc$labels[sort(idx)], collapse = ";")
    }, character(1))
  )
  write.csv(edges_table, file.path(tab_dir, paste0("SampleClusters_", gran, "_bootstrap_edges.csv")),
            row.names = FALSE)

  f <- function(nome) file.path(fig_dir, paste0(nome, "_", gran, ".png"))
  ggsave(f("PanelA_composicao"), panel_A, width = 12, height = 7, dpi = 400, bg = "white")
  ggsave(f("PanelB_media"),      panel_B, width = 7.5, height = 5, dpi = 400, bg = "white")
  ggsave(f("PanelC_NS_PBvsControle"), panel_C, width = 6.5, height = 7, dpi = 400, bg = "white")
  ggsave(f("PanelD_ML_PBvsControle"), panel_D, width = 6.5, height = 7, dpi = 400, bg = "white")
  ggsave(f("PanelE_NS_vs_ML"),        panel_E, width = 6.5, height = 7, dpi = 400, bg = "white")
  ggsave(f("PanelE2_NS_vs_ML_pareado"), panel_E2, width = 6.5, height = 7, dpi = 400, bg = "white")

  top_row    <- cowplot::plot_grid(panel_A, panel_B, ncol = 2, rel_widths = c(3.2, 1.6))
  forest_row <- cowplot::plot_grid(panel_C, panel_D, panel_E, panel_E2, ncol = 4, align = "h", axis = "tb")
  title <- cowplot::ggdraw() + cowplot::draw_label(
    paste0("Composição por biblioteca — ", gran, " · ", n_lib, " bibliotecas, ", n_pac, " pacientes · ",
           "CLR (pseudocontagem 1e-6) + lm, BH entre ", length(comp$cell_cols), " tipos"),
    x = 0.01, hjust = 0, fontface = "bold", size = 15)
  final_plot <- cowplot::plot_grid(title, top_row, forest_row, ncol = 1, rel_heights = c(0.06, 1.2, 1.25))
  ggsave(f("Figure_ABCDE"), final_plot, width = 25, height = 15, dpi = 300, bg = "white")

  # ==========================================================
  # FIGURA de forests por classificação (script B)
  #   (a) classificação par a par, todas as bibliotecas
  #   (b) estimulado vs não estimulado dentro de cada grupo
  # ==========================================================

  grp_n <- level_counts(comp, group_col)
  lt <- 0
  next_letter <- function() { lt <<- lt + 1; LETTERS[lt] }
  all_forests <- list()
  res_b <- list()

  res_class <- run_pairwise(comp, group_col, group_order = present_levels)
  res_class$analysis <- "classificacao"
  res_b[["classificacao"]] <- res_class

  for (i in seq_along(pairs_list)) {
    a <- pairs_list[[i]][1]; b <- pairs_list[[i]][2]
    sub <- res_class %>% filter((g1 == a & g2 == b) | (g1 == b & g2 == a))
    all_forests[[length(all_forests) + 1]] <- make_forest(
      sub, comp$cell_cols,
      panel_label   = next_letter(),
      plot_title    = paste0(group_labels[[a]], " vs ", group_labels[[b]]),
      plot_subtitle = paste0("n bibliotecas: ", a, "=", grp_n[[a]], ", ", b, "=", grp_n[[b]]),
      x_axis_label  = paste0("Efeito (CLR, ", a, " \u2212 ", b, ")"),   # [P15] sentido do contraste no eixo
      color_values  = c(group_palette[c(a, b)], setNames("grey75", NS_LABEL)),
      color_labels  = c(group_labels[c(a, b)], setNames(NS_LABEL, NS_LABEL))
    )
  }

  for (g in present_levels) {
    # min_per_level = 1: permite comparar com uma amostra por lado (MB, NP). O IC
    # fica indefinido e o resultado é só descritivo (sinalizado no subtítulo).
    r_g <- run_pairwise(comp, condition_col, group_order = est_order,
                        subset_col = group_col, subset_val = g, min_per_level = 1)
    if (is.null(r_g)) { message("Estímulo em ", g, ": dados insuficientes, pulado."); next }
    r_g$analysis <- paste0("estimulo | ", g)
    res_b[[paste0("estimulo_", g)]] <- r_g

    counts_g   <- level_counts(comp, condition_col, group_col, g)
    low_n_flag <- if (any(counts_g < 2)) "  [1/lado: descritivo]" else ""

    all_forests[[length(all_forests) + 1]] <- make_forest(
      r_g, comp$cell_cols,
      panel_label   = next_letter(),
      plot_title    = paste0("Estim. vs Não estim. — ", group_labels[[g]]),
      plot_subtitle = paste0(n_subtitle(counts_g), low_n_flag),
      x_axis_label  = "Efeito (CLR, NS \u2212 ML)",   # [P15] emmeans: 1º nível − 2º (> 0 = maior em NS)
      color_values  = cond_cols, color_labels = cond_labs
    )
  }

  nrows <- ceiling(length(all_forests) / 3)
  fig_forests <- cowplot::plot_grid(plotlist = all_forests, ncol = 3, align = "hv", axis = "tblr")
  ggsave(f("Figure_forests_classificacao"), fig_forests,
         width = 20, height = 6.5 * nrows, dpi = 300, bg = "white", limitsize = FALSE)

  all_out <- bind_rows(res_b) %>%
    mutate(granularity = gran) %>%
    dplyr::select(granularity, analysis, comparison, cell_population, direction,
                  estimate, std.error, conf.low, conf.high, p.value, FDR, sig_shape)
  write.csv(all_out, file.path(tab_dir, paste0("Comparisons_", gran, ".csv")), row.names = FALSE)

  # ==========================================================
  # [P15] Sensibilidade: CLR com pseudocontagem na escala de contagem
  #       (src/forest_composicao/composicao.py) nos contrastes C, D, E e E2
  # ==========================================================

  sens <- list(
    C  = run_composition_lm(comp, group_col, ref_level = resp_ref, keep_levels = resp_levels,
                            subset_col = condition_col, subset_val = condition_ref, pseudo_mode = "contagem"),
    D  = run_composition_lm(comp, group_col, ref_level = resp_ref, keep_levels = resp_levels,
                            subset_col = condition_col, subset_val = condition_stim, pseudo_mode = "contagem"),
    E  = run_composition_lm(comp, condition_col, ref_level = condition_ref, pseudo_mode = "contagem"),
    E2 = run_composition_lm(comp, condition_col, ref_level = condition_ref, covars = patient_col,
                            pseudo_mode = "contagem")
  )
  base <- list(C = resC, D = resD, E = resE, E2 = resE2)
  sens_tab <- map_dfr(names(base), function(k) {
    full_join(
      base[[k]] %>% dplyr::select(cell_population, estimate_1e6 = estimate, p_1e6 = p.value, FDR_1e6 = FDR),
      sens[[k]] %>% dplyr::select(cell_population, estimate_contagem = estimate, p_contagem = p.value,
                                  FDR_contagem = FDR),
      by = "cell_population") %>% mutate(contraste = k, .before = 1)
  })
  write.csv(sens_tab, file.path(tab_dir, paste0("sensibilidade_pseudocontagem_", gran, ".csv")), row.names = FALSE)

  report_sig <- function(r, nome) {
    s <- r %>% dplyr::filter(sig_shape != NS_LABEL) %>% dplyr::arrange(FDR, p.value)
    message(">>> ", gran, " | ", nome, ": ", if (nrow(s) == 0) "nenhum tipo com p < 0,05"
            else paste0(s$cell_population, " (p=", fmt_num(s$p.value), ", FDR=", fmt_num(s$FDR), ")",
                        collapse = "; "))
  }
  report_sig(resC, "C"); report_sig(resD, "D"); report_sig(resE, "E"); report_sig(resE2, "E2")
}

message("\nOK: figuras em ", fig_dir, "; tabelas em ", tab_dir)
