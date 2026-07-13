############################################################
# Análise de vias: Estimulado vs Não estimulado
# dentro de cada subgrupo de classificação (MB, PB, controle, NP)
#
# MÉTODO: pseudobulk (soma de contagens cruas por amostra) + DESeq2
# ----------------------------------------------------------------
# Por que pseudobulk em vez de teste célula-a-célula (Wilcoxon)?
# Células do mesmo paciente NÃO são réplicas independentes entre si
# (pseudorreplicação) -> um teste célula-a-célula infla o N e gera
# p-valores artificialmente pequenos. Somando as contagens por amostra
# (library_id), cada PACIENTE/amostra vira uma réplica real, e o
# DESeq2 fornece uma estatística (Wald "stat") já adequada como
# ranking para o fgsea.
#
# LÓGICA do fgsea (conforme discutido): o fgsea compara duas condições
# através de UM vetor de ranking (aqui: stat do DESeq2, estimulado vs
# não estimulado). Rodamos esse contraste separadamente PARA CADA
# subgrupo de classificação e juntamos os resultados com uma coluna
# "classificacao".
#
# COLEÇÕES DE VIAS: Hallmark, Reactome e KEGG são rodadas para cada
# subgrupo, reaproveitando o mesmo ranking (stat do DESeq2) -- só o
# fgseaMultilevel() é repetido por coleção, o DESeq2 roda uma vez só.
#
# UP vs DOWN (fgsea): a seleção de vias para o dot plot agora garante
# representação de vias UP (NES>0) E DOWN (NES<0) por subgrupo. Antes
# a seleção era só pelo menor padj, então se as vias UP fossem
# sistematicamente mais significativas, elas dominavam a lista e as
# DOWN desapareciam do gráfico -- mesmo existindo em fgsea_all.
#
# ORA (Over-Representation Analysis): além do fgsea (que usa o
# ranking completo), rodamos também ORA (fgsea::fora(), teste
# hipergeométrico) para cada subgrupo, separando genes UP e DOWN
# (definidos por padj/log2FC do DESeq2) contra um fundo (universe) =
# todos os genes testados. ORA e fgsea são complementares: o fgsea
# usa toda a distribuição do ranking; o ORA usa só os genes que
# passaram um corte de significância, mas já separa UP/DOWN de forma
# natural (duas listas de genes distintas), então não sofre do
# mesmo desbalanceamento.
############################################################

setwd("/data04/projects04/MarianaBoroni/spatial_ovary/bin/Pedro/Scripts/clear_path_analysis")

# setup env and repository
libdir = "~/spatial_ovary/bin/Pedro/4.4.1_adapted/"

.libPaths(libdir)

.libPaths()

options(repos = c("CRAN" = "https://repo-ciencia.inca.gov.br/repository/inca-r-group"))

data_path <- "/data04/projects04/MarianaBoroni/spatial_ovary/data/Pedro_data"
obj_path <- file.path(data_path, "adata_harmony_subset_T_no8_.RDS")

# Set seed
set.seed(42)

packages <- c(
  "Seurat", "dplyr", "tidyr", "purrr", "tibble", "stringr",
  "ggplot2", "forcats", "scales", "Matrix"
)
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!requireNamespace("DESeq2", quietly = TRUE))  BiocManager::install("DESeq2", update = FALSE, ask = FALSE)
if (!requireNamespace("fgsea", quietly = TRUE))   BiocManager::install("fgsea", update = FALSE, ask = FALSE)
if (!requireNamespace("msigdbr", quietly = TRUE)) install.packages("msigdbr")

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(ggplot2)
  library(forcats)
  library(scales)
  library(Matrix)
  library(DESeq2)
  library(fgsea)
  library(msigdbr)
})

############################################################
##                                                        ##
##            >>>  PARÂMETROS AJUSTÁVEIS  <<<             ##
##   (tudo que você normalmente precisa mudar está aqui)  ##
##                                                        ##
############################################################

# ---- arquivo e colunas ---------------------------------------------------

group_col  <- "classificacao"   # MB / PB / controle / NP -> um fgsea por nível
cond_col   <- "estimulo"        # "estimulado" vs "nao estimulado"
sample_col <- "library_id"      # amostra pseudobulk = 1 paciente + 1 condição

group_levels <- c("controle", "PB", "MB", "NP")           # ordem desejada (ausentes são ignorados)
cond_levels  <- c("nao estimulado", "estimulado")          # referência primeiro

# ---- onde estão as contagens cruas ----------------------------------------
# seurat_obj@assays$counts@data
counts_assay <- "counts"
counts_layer <- "data"    # nome da camada/slot onde estão as contagens cruas

# ---- filtros de qualidade ---------------------------------------------------

# Com o modelo combinado, nenhum subgrupo é mais descartado por ter poucas
# amostras (a dispersão é estimada com TODAS as amostras juntas). Esse
# parâmetro agora só dispara um AVISO no console quando um subgrupo tem
# menos réplicas que isso por condição (ex: MB/NP com 1 paciente cada) --
# útil para saber quais resultados interpretar com mais cautela.
min_samples_per_condition <- 2
min_counts_gene            <- 10   # gene é mantido se a soma de contagens (todas as amostras) >= isso

# ---- ORA: definição de gene "DE" (up/down) por subgrupo -------------------
# Um gene entra na lista UP/DOWN do ORA se, no contraste DESeq2 daquele
# subgrupo, padj < padj_cutoff_ora E log2FoldChange > lfc_cutoff_ora (UP)
# ou < -lfc_cutoff_ora (DOWN). lfc_cutoff_ora = 0 -> qualquer sinal conta.
padj_cutoff_ora <- 0.05
lfc_cutoff_ora  <- 0

# ---- gene sets (msigdbr) ---------------------------------------------------
# Cada entrada da lista é uma coleção de vias que será rodada separadamente
# (mesmo ranking para o fgsea, mesmos genes DE para o ORA). Para
# adicionar/remover uma coleção, edite esta lista. category/subcategory
# seguem a nomenclatura do pacote msigdbr -- se algum nome mudar de versão
# para versão, rode msigdbr::msigdbr_collections() para conferir os nomes
# válidos no seu ambiente e ajuste subcategory abaixo.

species <- "Homo sapiens"

gene_set_collections <- list(
  Hallmark = list(category = "H",  subcategory = NULL),
  Reactome = list(category = "C2", subcategory = "CP:REACTOME"),
  KEGG     = list(category = "C2", subcategory = "CP:KEGG")
)

# prefixo removido do nome da via ao exibir no gráfico, por coleção
geneset_prefix <- c(
  Hallmark = "^HALLMARK_",
  Reactome = "^REACTOME_",
  KEGG     = "^KEGG_"
)

fgsea_minSize <- 15
fgsea_maxSize <- 500
fgsea_seed    <- 42

# ---- mapeamento de gene ID -> símbolo --------------------------------------
# Se os rownames() da matriz de contagens forem Ensembl IDs em vez de
# símbolos de gene, mude para TRUE (usa a coluna gene_symbol do assay "RNA").
map_gene_ids <- FALSE

# ---- visualização -----------------------------------------------------------

padj_cutoff    <- 0.05   # ponto (fgsea) ou via (ora) considerada significativa -- usado para marcar/ordenar
top_n_pathways <- 30     # numero maximo de vias distintas mostradas no dot plot (uniao entre grupos)

# no fgsea, cada classificacao contribui com ate N vias UP e N vias DOWN
# (em vez de só "as N com menor padj", que tendia a mostrar só UP)
top_n_per_group_per_direction <- 8

font_axis_title   <- 14
font_axis_text    <- 12
font_axis_text_y  <- 10   # nomes de vias costumam ser longos -> um pouco menor
font_legend_title <- 13
font_legend_text  <- 11
font_plot_title   <- 15

# ============================================================
# Carregar objeto e preparar
# ============================================================

seurat_obj <- readRDS(obj_path)

levels(seurat_obj@meta.data$classificacao) <- c(
  levels(seurat_obj@meta.data$classificacao),
  "NP"
)

seurat_obj@meta.data$classificacao[
  grepl("^HNP", seurat_obj@meta.data$library_id)
] <- "NP"

stopifnot(all(c(group_col, cond_col, sample_col) %in% colnames(seurat_obj@meta.data)))

get_counts_matrix <- function(obj, assay, layer) {
  tryCatch(
    GetAssayData(obj, assay = assay, layer = layer),
    error = function(e) GetAssayData(obj, assay = assay, slot = layer)
  )
}

counts_mat <- get_counts_matrix(seurat_obj, counts_assay, counts_layer)
stopifnot(ncol(counts_mat) == ncol(seurat_obj))

meta <- as.data.frame(seurat_obj@meta.data)

# Sanitiza eventuais colunas do tipo "lista" herdadas do objeto Seurat.
# Isso é comum em objetos que passaram por conversões (ex: scanpy -> Seurat)
# e é a causa mais provável de erros como "Argument 'x' is not a vector: list"
# em qualquer verbo do dplyr (count, distinct, etc.) mais adiante.
list_cols <- names(meta)[vapply(meta, is.list, logical(1))]
if (length(list_cols) > 0) {
  message("Convertendo colunas do tipo lista para character: ", paste(list_cols, collapse = ", "))
  for (col in list_cols) {
    meta[[col]] <- vapply(meta[[col]], function(x) paste(as.character(x), collapse = "; "), character(1))
  }
}

meta[[group_col]]  <- factor(as.character(meta[[group_col]]),
                             levels = group_levels[group_levels %in% unique(as.character(meta[[group_col]]))])
meta[[cond_col]]   <- factor(as.character(meta[[cond_col]]),
                             levels = cond_levels[cond_levels %in% unique(as.character(meta[[cond_col]]))])
meta[[sample_col]] <- as.character(meta[[sample_col]])

present_groups <- levels(meta[[group_col]])
present_conds  <- levels(meta[[cond_col]])
stopifnot(length(present_conds) == 2)

cat("\n[Diagnóstico] Amostras (library_id) por classificacao x estimulo:\n")
diag_tab <- unique(data.frame(
  classificacao = as.character(meta[[group_col]]),
  library_id    = as.character(meta[[sample_col]]),
  estimulo      = as.character(meta[[cond_col]]),
  stringsAsFactors = FALSE
))
print(table(diag_tab$classificacao, diag_tab$estimulo))

# ---- mapear ENSG -> símbolo (opcional) -------------------------------------

if (map_gene_ids) {
  feat_meta <- seurat_obj[["RNA"]]@meta.data
  gene_map <- setNames(feat_meta$gene_symbol, rownames(feat_meta))
  gene_map[is.na(gene_map) | gene_map == ""] <- names(gene_map)[is.na(gene_map) | gene_map == ""]
  rownames(counts_mat) <- unname(gene_map[rownames(counts_mat)])
  counts_mat <- counts_mat[!is.na(rownames(counts_mat)) & !duplicated(rownames(counts_mat)), ]
}

# ============================================================
# Gene sets (msigdbr) -- uma lista de coleções (Hallmark, Reactome, KEGG)
# ============================================================

pathways_by_collection <- map(gene_set_collections, function(coll) {
  pathway_df <- msigdbr(species = species, category = coll$category, subcategory = coll$subcategory)
  split(pathway_df$gene_symbol, pathway_df$gs_name)
})

cat("\n[Diagnóstico] Coleções de vias carregadas (n. de vias por coleção):\n")
print(sapply(pathways_by_collection, length))

# ============================================================
# Tema base
# ============================================================

base_theme <- theme_classic(base_size = font_axis_text) +
  theme(
    axis.title   = element_text(size = font_axis_title, color = "black"),
    axis.text.x  = element_text(size = font_axis_text, color = "black"),
    axis.text.y  = element_text(size = font_axis_text_y, color = "black"),
    legend.title = element_text(size = font_legend_title, face = "bold"),
    legend.text  = element_text(size = font_legend_text),
    plot.title   = element_text(size = font_plot_title, face = "bold")
  )

# ============================================================
# Função: pseudobulk (soma de contagens) por library_id
# ============================================================

build_pseudobulk <- function(counts_mat, meta, cells, sample_col) {
  sub_counts <- counts_mat[, cells, drop = FALSE]
  samples <- meta[cells, sample_col]
  
  agg <- sapply(split(seq_along(samples), samples), function(idx) {
    Matrix::rowSums(sub_counts[, idx, drop = FALSE])
  })
  as.matrix(agg)
}

# ============================================================
# Modelo ÚNICO combinado (todas as amostras, todos os subgrupos)
# ============================================================
#
# Por que um modelo único em vez de um DESeq2 separado por subgrupo?
# Alguns subgrupos (ex: MB, NP) podem ter só 1 paciente = 1 amostra
# por condição. Um DESeq2 rodado ISOLADO dentro desse subgrupo não
# consegue estimar dispersão com n=1 por condição, e por isso ele
# era descartado silenciosamente antes. Ajustando UM modelo com
# TODAS as amostras juntas (todos os subgrupos), o DESeq2 estima a
# dispersão usando o conjunto completo (bem mais estável) e usa essa
# dispersão moderada para testar cada subgrupo -- inclusive os que
# têm só 1 réplica por condição. Essa é a abordagem recomendada pelo
# próprio DESeq2 para desenhos desbalanceados/com poucas réplicas.
#
# ATENÇÃO: um contraste com n=1 por condição continua tendo poder
# estatístico baixo (intervalo de confiança largo) -- a correção
# evita que o grupo suma da análise, mas não cria replicatas que
# não existem. Resultados de grupos com poucas amostras (reportado
# em n_estimulado/n_nao_estimulado no resultado final) devem ser
# interpretados com cautela.
# ============================================================

cells_all <- rownames(meta)
pb_counts_all <- build_pseudobulk(counts_mat, meta, cells_all, sample_col)

coldata_all <- meta[cells_all, c(sample_col, group_col, cond_col), drop = FALSE]
coldata_all <- coldata_all[!duplicated(coldata_all[[sample_col]]), , drop = FALSE]
coldata_all[[sample_col]] <- as.character(coldata_all[[sample_col]])
rownames(coldata_all) <- coldata_all[[sample_col]]
coldata_all <- coldata_all[colnames(pb_counts_all), , drop = FALSE]
coldata_all[[group_col]] <- factor(as.character(coldata_all[[group_col]]), levels = present_groups)
coldata_all[[cond_col]]  <- factor(as.character(coldata_all[[cond_col]]),  levels = present_conds)
coldata_all$group_cond   <- factor(paste(coldata_all[[group_col]], coldata_all[[cond_col]], sep = "__"))

cat("\n[Diagnóstico] Amostras (library_id) por classificacao x estimulo (modelo combinado):\n")
print(table(coldata_all[[group_col]], coldata_all[[cond_col]]))

# filtro de genes de baixa expressão (usando todas as amostras)
keep <- rowSums(pb_counts_all) >= min_counts_gene
pb_counts_all <- pb_counts_all[keep, , drop = FALSE]

dds <- DESeqDataSetFromMatrix(countData = pb_counts_all, colData = coldata_all, design = ~ group_cond)
dds <- DESeq(dds, quiet = TRUE)

# ============================================================
# Função: extrai o contraste estimulado vs não (dentro de 1 subgrupo)
# do modelo combinado, e roda fgsea E ora PARA CADA coleção de vias
# (Hallmark, Reactome, KEGG), reaproveitando o mesmo contraste DESeq2
# ============================================================

run_pathway_analysis_for_group <- function(group_level) {
  
  lvl_estim  <- paste(group_level, present_conds[2], sep = "__")  # estimulado
  lvl_naoest <- paste(group_level, present_conds[1], sep = "__")  # nao estimulado
  
  n_estim  <- sum(coldata_all$group_cond == lvl_estim)
  n_naoest <- sum(coldata_all$group_cond == lvl_naoest)
  
  if (n_estim == 0 || n_naoest == 0) {
    message("Pulando '", group_level, "': nao existe amostra em uma das condicoes.")
    return(NULL)
  }
  if (n_estim < min_samples_per_condition || n_naoest < min_samples_per_condition) {
    message("Aviso: '", group_level, "' tem poucas replicatas (estimulado=", n_estim,
            ", nao_estimulado=", n_naoest, "). Contraste calculado mesmo assim, ",
            "usando a dispersao estimada com TODAS as amostras -- mas com poder estatistico baixo.")
  }
  
  res <- results(dds, contrast = c("group_cond", lvl_estim, lvl_naoest)) %>%
    as.data.frame() %>%
    rownames_to_column("gene") %>%
    filter(!is.na(stat))
  
  # ---------------- fgsea (ranking completo) ----------------
  
  stat_vec <- setNames(res$stat, res$gene)
  stat_vec <- sort(stat_vec, decreasing = TRUE)
  
  fgsea_res_group <- map_dfr(names(pathways_by_collection), function(coll_name) {
    set.seed(fgsea_seed)
    fgseaMultilevel(
      pathways = pathways_by_collection[[coll_name]],
      stats    = stat_vec,
      minSize  = fgsea_minSize,
      maxSize  = fgsea_maxSize
    ) %>%
      mutate(
        classificacao      = group_level,
        geneset_collection = coll_name,
        n_estimulado       = n_estim,
        n_nao_estimulado   = n_naoest
      ) %>%
      arrange(padj)
  })
  
  # ---------------- ORA (genes DE fixos, hipergeométrico) ----------------
  
  universe  <- res$gene
  genes_up   <- res %>% filter(padj < padj_cutoff_ora,  log2FoldChange >  lfc_cutoff_ora) %>% pull(gene)
  genes_down <- res %>% filter(padj < padj_cutoff_ora,  log2FoldChange < -lfc_cutoff_ora) %>% pull(gene)
  
  message("'", group_level, "': ", length(genes_up), " genes UP, ", length(genes_down),
          " genes DOWN (padj_ora<", padj_cutoff_ora, "), universo=", length(universe), " genes.")
  
  run_ora_direction <- function(gene_list, direction_label) {
    if (length(gene_list) == 0) return(NULL)
    map_dfr(names(pathways_by_collection), function(coll_name) {
      fora(
        pathways = pathways_by_collection[[coll_name]],
        genes    = gene_list,
        universe = universe,
        minSize  = fgsea_minSize,
        maxSize  = fgsea_maxSize
      ) %>%
        as.data.frame() %>%
        mutate(
          classificacao      = group_level,
          geneset_collection = coll_name,
          direction          = direction_label,
          n_de_genes         = length(gene_list),
          n_universe         = length(universe),
          n_estimulado       = n_estim,
          n_nao_estimulado   = n_naoest
        ) %>%
        arrange(padj)
    })
  }
  
  ora_res_group <- bind_rows(
    run_ora_direction(genes_up,   "up"),
    run_ora_direction(genes_down, "down")
  )
  
  list(fgsea = fgsea_res_group, ora = ora_res_group)
}

# ============================================================
# Rodar para todos os subgrupos presentes
# ============================================================

results_list <- map(present_groups, run_pathway_analysis_for_group)
names(results_list) <- present_groups
results_list <- compact(results_list)   # remove subgrupos sem nenhuma amostra em alguma condicao

cat("\n[Diagnóstico] Grupos analisados com sucesso:\n")
print(names(results_list))

fgsea_all <- bind_rows(map(results_list, "fgsea"))
ora_all   <- bind_rows(map(results_list, "ora"))

cat("\n[Diagnóstico] fgsea: resumo por classificacao x coleção (padj minimo, n. de vias sig. e N de réplicas):\n")
print(
  fgsea_all %>%
    group_by(classificacao, geneset_collection) %>%
    summarise(
      menor_padj       = min(padj, na.rm = TRUE),
      n_sig_up         = sum(padj < padj_cutoff & NES > 0, na.rm = TRUE),
      n_sig_down       = sum(padj < padj_cutoff & NES < 0, na.rm = TRUE),
      n_total          = n(),
      n_estimulado     = dplyr::first(n_estimulado),
      n_nao_estimulado = dplyr::first(n_nao_estimulado),
      .groups = "drop"
    )
)

cat("\n[Diagnóstico] ORA: resumo por classificacao x coleção x direção (padj minimo e n. de vias sig.):\n")
print(
  ora_all %>%
    group_by(classificacao, geneset_collection, direction) %>%
    summarise(
      n_de_genes  = dplyr::first(n_de_genes),
      menor_padj  = min(padj, na.rm = TRUE),
      n_sig       = sum(padj < padj_cutoff, na.rm = TRUE),
      n_total     = n(),
      .groups = "drop"
    )
)

# ============================================================
# Salvar CSVs
# ============================================================

fgsea_all_csv <- fgsea_all %>%
  mutate(leadingEdge = map_chr(leadingEdge, ~ paste(.x, collapse = "/")))
write.csv(fgsea_all_csv, "fgsea_estimulo_por_classificacao_todas_colecoes.csv", row.names = FALSE)

walk(names(pathways_by_collection), function(coll_name) {
  sub_csv <- fgsea_all_csv %>% filter(geneset_collection == coll_name)
  write.csv(sub_csv, paste0("fgsea_estimulo_por_classificacao_", coll_name, ".csv"), row.names = FALSE)
})

ora_all_csv <- ora_all %>%
  mutate(overlapGenes = map_chr(overlapGenes, ~ paste(.x, collapse = "/")))
write.csv(ora_all_csv, "ora_estimulo_por_classificacao_todas_colecoes.csv", row.names = FALSE)

walk(names(pathways_by_collection), function(coll_name) {
  sub_csv <- ora_all_csv %>% filter(geneset_collection == coll_name)
  write.csv(sub_csv, paste0("ora_estimulo_por_classificacao_", coll_name, ".csv"), row.names = FALSE)
})

# ============================================================
# Dot plot -- FGSEA: NES por subgrupo, vias UP e DOWN balanceadas
# -- um gráfico separado POR COLEÇÃO (Hallmark, Reactome, KEGG)
# ============================================================

make_fgsea_dot_plot <- function(fgsea_coll, coll_name) {
  
  # seleciona até N vias UP (NES>0) e N vias DOWN (NES<0) por classificacao,
  # em vez de só "as N com menor padj" (que tende a favorecer só UP)
  top_pathways <- fgsea_coll %>%
    mutate(direction = ifelse(NES >= 0, "up", "down")) %>%
    group_by(classificacao, direction) %>%
    slice_min(padj, n = top_n_per_group_per_direction, with_ties = FALSE) %>%
    ungroup() %>%
    distinct(pathway) %>%
    pull(pathway)
  
  if (length(top_pathways) > top_n_pathways) {
    top_pathways <- fgsea_coll %>%
      filter(pathway %in% top_pathways) %>%
      group_by(pathway) %>%
      summarise(min_padj = min(padj, na.rm = TRUE), .groups = "drop") %>%
      slice_min(min_padj, n = top_n_pathways) %>%
      pull(pathway)
  }
  
  if (length(top_pathways) == 0) {
    message("Nenhuma via para plotar (fgsea) na coleção '", coll_name, "'. Pulando gráfico.")
    return(NULL)
  }
  
  prefix <- geneset_prefix[[coll_name]]
  if (is.null(prefix)) prefix <- "^"
  
  plot_df <- fgsea_coll %>%
    filter(pathway %in% top_pathways) %>%
    mutate(
      classificacao = factor(classificacao, levels = present_groups),
      pathway_label = pathway %>% str_remove(prefix) %>% str_replace_all("_", " ") %>% str_to_sentence(),
      sig = ifelse(padj < padj_cutoff, "sig", "ns"),
      x_label = paste0(classificacao, "\n(n=", n_nao_estimulado, "/", n_estimulado, ")")
    )
  
  x_label_order <- plot_df %>% distinct(classificacao, x_label) %>% arrange(classificacao) %>% pull(x_label)
  plot_df$x_label <- factor(plot_df$x_label, levels = x_label_order)
  
  pathway_order <- plot_df %>%
    group_by(pathway_label) %>%
    summarise(mean_nes = mean(NES, na.rm = TRUE), .groups = "drop") %>%
    arrange(mean_nes) %>%
    pull(pathway_label)
  plot_df$pathway_label <- factor(plot_df$pathway_label, levels = pathway_order)
  
  p_dot <- ggplot(plot_df, aes(x = x_label, y = pathway_label,
                               size = -log10(pmax(padj, 1e-10)), color = NES)) +
    geom_point(aes(shape = sig, stroke = ifelse(sig == "sig", 1, 0.3))) +
    scale_shape_manual(values = c(sig = 16, ns = 1), guide = "none") +
    scale_color_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0, name = "NES") +
    scale_size_continuous(name = "-log10(padj)", range = c(1, 8)) +
    labs(
      x = NULL, y = NULL,
      title = paste0("fgsea (", coll_name, "): estimulado vs não estimulado, por classificação"),
      subtitle = paste0(
        "modelo combinado (1 DESeq2, todas as amostras)  |  ",
        "até ", top_n_per_group_per_direction, " vias UP + ", top_n_per_group_per_direction, " DOWN por grupo  |  ",
        "círculo cheio = padj < ", padj_cutoff, "  |  eixo x: n(não estimulado/estimulado)"
      )
    ) +
    base_theme
  
  ggsave(
    paste0("fgsea_estimulo_por_classificacao_", coll_name, "_dotplot.png"), p_dot,
    width = 9, height = 0.35 * length(top_pathways) + 3, dpi = 400, bg = "white"
  )
  p_dot
}

fgsea_plots <- map(names(pathways_by_collection), function(coll_name) {
  make_fgsea_dot_plot(fgsea_all %>% filter(geneset_collection == coll_name), coll_name)
})
names(fgsea_plots) <- names(pathways_by_collection)
iwalk(fgsea_plots, function(p, coll_name) if (!is.null(p)) print(p))

# ============================================================
# Dot plot -- ORA: vias UP e DOWN (já naturalmente separadas)
# -- um gráfico separado POR COLEÇÃO
# ============================================================

make_ora_dot_plot <- function(ora_coll, coll_name) {
  
  top_pathways <- ora_coll %>%
    group_by(classificacao, direction) %>%
    slice_min(padj, n = top_n_per_group_per_direction, with_ties = FALSE) %>%
    ungroup() %>%
    distinct(pathway) %>%
    pull(pathway)
  
  if (length(top_pathways) > top_n_pathways) {
    top_pathways <- ora_coll %>%
      filter(pathway %in% top_pathways) %>%
      group_by(pathway) %>%
      summarise(min_padj = min(padj, na.rm = TRUE), .groups = "drop") %>%
      slice_min(min_padj, n = top_n_pathways) %>%
      pull(pathway)
  }
  
  if (length(top_pathways) == 0) {
    message("Nenhuma via para plotar (ORA) na coleção '", coll_name, "'. Pulando gráfico.")
    return(NULL)
  }
  
  prefix <- geneset_prefix[[coll_name]]
  if (is.null(prefix)) prefix <- "^"
  
  plot_df <- ora_coll %>%
    filter(pathway %in% top_pathways) %>%
    mutate(
      classificacao = factor(classificacao, levels = present_groups),
      pathway_label = pathway %>% str_remove(prefix) %>% str_replace_all("_", " ") %>% str_to_sentence(),
      sig = ifelse(padj < padj_cutoff, "sig", "ns"),
      x_label = paste0(classificacao, " (", direction, ")\nn_DE=", n_de_genes)
    )
  
  x_label_order <- plot_df %>%
    distinct(classificacao, direction, x_label) %>%
    arrange(classificacao, desc(direction)) %>%
    pull(x_label)
  plot_df$x_label <- factor(plot_df$x_label, levels = x_label_order)
  
  pathway_order <- plot_df %>%
    group_by(pathway_label) %>%
    summarise(mean_ratio = mean(overlap / size, na.rm = TRUE), .groups = "drop") %>%
    arrange(mean_ratio) %>%
    pull(pathway_label)
  plot_df$pathway_label <- factor(plot_df$pathway_label, levels = pathway_order)
  
  p_dot <- ggplot(plot_df, aes(x = x_label, y = pathway_label,
                               size = -log10(pmax(padj, 1e-10)), color = direction)) +
    geom_point(aes(shape = sig, stroke = ifelse(sig == "sig", 1, 0.3))) +
    scale_shape_manual(values = c(sig = 16, ns = 1), guide = "none") +
    scale_color_manual(values = c(up = "#B2182B", down = "#2166AC"), name = "Direção") +
    scale_size_continuous(name = "-log10(padj)", range = c(1, 8)) +
    labs(
      x = NULL, y = NULL,
      title = paste0("ORA (", coll_name, "): genes DE (estimulado vs não), por classificação"),
      subtitle = paste0(
        "genes DE: padj_ora < ", padj_cutoff_ora, "  |  círculo cheio = padj (enriquecimento) < ", padj_cutoff
      )
    ) +
    base_theme
  
  ggsave(
    paste0("ora_estimulo_por_classificacao_", coll_name, "_dotplot.png"), p_dot,
    width = 9.5, height = 0.35 * length(top_pathways) + 3, dpi = 400, bg = "white"
  )
  p_dot
}

ora_plots <- map(names(pathways_by_collection), function(coll_name) {
  make_ora_dot_plot(ora_all %>% filter(geneset_collection == coll_name), coll_name)
})

names(ora_plots) <- names(pathways_by_collection)
iwalk(ora_plots, function(p, coll_name) if (!is.null(p)) print(p))

library(patchwork)
library(purrr)

combined_plots <- map(names(pathways_by_collection), function(coll_name) {
  
  p_fgsea <- fgsea_plots[[coll_name]]
  p_ora   <- ora_plots[[coll_name]]
  
  # trata casos onde algum é NULL
  if (is.null(p_fgsea) & is.null(p_ora)) return(NULL)
  if (is.null(p_fgsea)) return(p_ora)
  if (is.null(p_ora))   return(p_fgsea)
  
  # empilha: FGSEA em cima, ORA embaixo
  p_fgsea / p_ora
})

names(combined_plots) <- names(pathways_by_collection)

# remove NULL
combined_plots <- combined_plots[!sapply(combined_plots, is.null)]

print(combined_plots)