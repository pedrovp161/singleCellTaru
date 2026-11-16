# GSEA por cluster (Reactome, KEGG e Hallmark).
#
# Cópia dos chunks de /mnt/SATA/analiseR/GSEA.rmd (projeto espacial), na mesma
# ordem e com o mesmo método. O texto fora dos chunks virou comentário. As
# alterações estão marcadas com [P16]; o porquê está em raw/analises_previas/P16_gsea_vias.py.
#
# Uso: Rscript --vanilla gsea_vias.R <dir_entrada> <dir_figuras> <dir_tabelas> \
#        <coluna_cluster> <clusterA,clusterB> <n_workers>
args <- commandArgs(trailingOnly = TRUE)
dir_in      <- args[1]                         # counts.h5 + obs.csv escritos pelo P16
fig_dir     <- args[2]
tab_dir     <- args[3]
cluster_col <- args[4]                         # [P16] no espacial, leiden_0_8
clusters_comparar_arg <- strsplit(args[5], ",")[[1]]
n_workers   <- as.integer(args[6])
modo_score  <- if (length(args) >= 7) args[7] else "ref"   # [P16] ref | sinal_log10p | log2fc (sensibilidade)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

library(Seurat)
library(DropletUtils)                          # [P16] leitura do HDF5 escrito pelo Python
library(dplyr)
library(ggplot2)
library(clusterProfiler)
library(ReactomePA)
library(org.Hs.eg.db)
library(patchwork)
library(stringr)
library(cowplot)
library(future)
options(future.globals.maxSize = 10 * 1024^3)



# [P16] O espacial lia o objeto Seurat e uma tabela do FindAllMarkers já pronta
# (FindAllMarkers_results.rds). Aqui o objeto é montado das contagens SoupX e o
# FindAllMarkers roda neste script, no assay RNA (LogNormalize, 10 mil por célula:
# a mesma normalização do X do objeto final), com o ranking completo de genes, sem
# cortes (pedido do usuário): logfc.threshold = 0, min.pct = 0, return.thresh = Inf.
# Os padrões do Seurat 5 (0,1 / 0,01 / 0,01) truncariam o ranking nos genes com p < 0,01.
sce <- DropletUtils::read10xCounts(file.path(dir_in, "counts.h5"), type = "HDF5", col.names = TRUE)
m <- as(SingleCellExperiment::counts(sce), "dgCMatrix")
rownames(m) <- make.unique(SummarizedExperiment::rowData(sce)$Symbol)
obs <- read.csv(file.path(dir_in, "obs.csv"), colClasses = "character", encoding = "UTF-8")
colnames(m) <- obs$cell
rownames(obs) <- obs$cell
cluster_levels <- strsplit(readLines(file.path(dir_in, "cluster_levels.txt"), encoding = "UTF-8"), "\t")[[1]]
obs[[cluster_col]] <- factor(obs[[cluster_col]], levels = cluster_levels)

seurat_obj <- CreateSeuratObject(counts = m, meta.data = obs[, setdiff(colnames(obs), "cell"), drop = FALSE])
rm(sce, m); invisible(gc())
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE)
DefaultAssay(seurat_obj) <- "RNA"
Seurat::Idents(seurat_obj) <- seurat_obj@meta.data[[cluster_col]]
cat(sprintf("%d genes x %d células; %d clusters (%s)\n", nrow(seurat_obj), ncol(seurat_obj),
            nlevels(Idents(seurat_obj)), cluster_col))

marcadores_csv <- file.path(tab_dir, "FindAllMarkers_results.csv")
if (file.exists(marcadores_csv)) {
  df <- read.csv(marcadores_csv)
  df$cluster <- factor(df$cluster, levels = cluster_levels)
  cat("FindAllMarkers: tabela já existente reaproveitada\n")
} else {
  plan("multicore", workers = n_workers)
  t0 <- Sys.time()
  df <- FindAllMarkers(seurat_obj, assay = "RNA", test.use = "wilcox",
                       logfc.threshold = 0, min.pct = 0, only.pos = FALSE,
                       return.thresh = Inf, verbose = FALSE)
  cat(sprintf("Wilcoxon via presto: %s\n", requireNamespace("presto", quietly = TRUE)))
  plan("sequential")
  cat(sprintf("FindAllMarkers: %d linhas em %.1f min\n", nrow(df),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  utils::write.csv(df, marcadores_csv, row.names = FALSE)
}

# [P16] figuras em fig_dir e tabelas em tab_dir (no espacial, tudo na pasta corrente)
ggsave <- function(filename, ...) ggplot2::ggsave(file.path(fig_dir, filename), ...)
write.csv <- function(x, file, ...) utils::write.csv(x, file.path(tab_dir, file), ...)


# # Ajuste o caminho conforme necessário
# # seurat_obj <- readRDS("/data04/projects04/MarianaBoroni/spatial_ovary/data/Pedro_data/spatialPaper_full.RDS")
# # DefaultAssay(seurat_obj) <- "SCT"
# # Seurat::Idents(seurat_obj) <- seurat_obj@meta.data$leiden_0_8


# Chunk 3
# Garantir que as colunas estejam no formato correto
df$avg_log2FC <- as.numeric(df$avg_log2FC)
df$p_val_adj  <- as.numeric(df$p_val_adj)

# Limitar p_val_adj = 0 ao menor valor não-zero (evita Inf sem descartar os genes mais fortes)
min_p <- min(df$p_val_adj[df$p_val_adj > 0], na.rm = TRUE)

# Criar uma coluna de ranking (log2FC ponderado pela significância, reduz empates)
# [P16] Com o ranking completo, o p_val_adj do Seurat (Bonferroni sobre ~26 mil genes) é 1 em
# ~80% dos genes: o score deles vira 0, o peso se concentra em poucas dezenas de genes, conjuntos
# aleatórios também dão ES ~ ±1 e o GSEA perde o poder (teste na TregEff: MHC II p.adj 0,76 com
# p_val_adj; 0,004 com p_val). O score usa então o p bruto; a correção múltipla fica no BH do GSEA.
# O piso do p = 0 segue a mesma regra (menor p não nulo da tabela).
min_p_bruto <- min(df$p_val[df$p_val > 0], na.rm = TRUE)
df <- df %>%
  mutate(p_adj_cap  = pmax(p_val_adj, min_p),          # mantido só para registro
         p_cap      = pmax(as.numeric(p_val), min_p_bruto),
         rank_score = switch(modo_score,
                             ref          = avg_log2FC * -log10(p_cap),
                             sinal_log10p = sign(avg_log2FC) * -log10(p_cap),
                             log2fc       = avg_log2FC),
         # desempate mínimo para score exatamente 0 (evita ES NaN numa via só de zeros)
         rank_score = ifelse(rank_score == 0, avg_log2FC * 1e-9, rank_score))

# Para cada cluster, criar um vetor nomeado com os scores
clusters <- sort(unique(df$cluster))
rank_lists <- list()

for (cl in clusters) {
  cl_df <- df %>% filter(cluster == cl)

  # Remover genes com NA ou Inf
  cl_df <- cl_df[is.finite(cl_df$rank_score) & !is.na(cl_df$rank_score), ]

  # Criar vetor nomeado
  gene_list <- cl_df$rank_score
  names(gene_list) <- cl_df$gene

  # Ordenar do maior para o menor (decrescente)
  gene_list <- sort(gene_list, decreasing = TRUE)
  rank_lists[[as.character(cl)]] <- gene_list
}

# Verificar
cat("Número de clusters:", length(rank_lists), "\n")
cat("Genes por cluster:\n")
print(sapply(rank_lists, length))

# Rodar GSEA com correção
p_cutoff <- 0.05
gsea_results <- list()
# Resultado completo (sem corte de p), usado no plot combinado para ver o NES
# de uma via também nos clusters em que ela não passou no corte
gsea_full <- list()
set.seed(42)

for (cl in names(rank_lists)) {
  message(paste("Rodando GSEA para o cluster", cl))
  gene_list <- rank_lists[[cl]]

  if (length(gene_list) < 10) {
    message(paste("  Cluster", cl, "tem poucos genes. Pulando."))
    next
  }

  # 1. Converter IDs de SYMBOL para ENTREZID
  gene_df <- suppressMessages(suppressWarnings(bitr(names(gene_list),
                                   fromType = "SYMBOL",
                                   toType = "ENTREZID",
                                   OrgDb = org.Hs.eg.db,
                                   drop = TRUE)))

  # 2. Filtrar a lista original apenas para os genes mapeados com sucesso
  genes_mapeados <- gene_list[names(gene_list) %in% gene_df$SYMBOL]

  # Substituir os nomes (SYMBOL) pelos novos (ENTREZID)
  names(genes_mapeados) <- gene_df$ENTREZID[match(names(genes_mapeados), gene_df$SYMBOL)]

  # 3. Tratar duplicatas de ENTREZID mantendo o maior valor de rank
  #    (tapply devolve array 1-D; converter para vetor numérico nomeado)
  agg <- tapply(genes_mapeados, names(genes_mapeados), max)
  gene_list_entrez <- sort(setNames(as.numeric(agg), names(agg)), decreasing = TRUE)

  tryCatch({
    gsea <- gsePathway(
      geneList = gene_list_entrez,    # Usa a lista convertida
      organism = "human",
      pvalueCutoff = 1,               # sem corte aqui; o corte é aplicado abaixo
      pAdjustMethod = "BH",
      seed = TRUE,
      verbose = FALSE
    )

    # O BH é calculado sobre todas as vias testadas, então filtrar depois
    # dá exatamente o mesmo conjunto que pvalueCutoff = p_cutoff
    res_full <- if (is.null(gsea)) data.frame() else as.data.frame(gsea)
    if (nrow(res_full) > 0) gsea_full[[cl]] <- res_full
    res_sig <- res_full[which(res_full$p.adjust <= p_cutoff), , drop = FALSE]

    if (nrow(res_sig) > 0) {
      gsea_results[[cl]] <- res_sig
      message(paste("  Cluster", cl, "->", nrow(res_sig), "vias enriquecidas"))
    } else {
      message(paste("  Cluster", cl, "-> nenhuma via significativa"))
    }
  }, error = function(e) {
    message(paste("  Erro no cluster", cl, ":", e$message))
  })
}




gsea_results_df <- bind_rows(
  lapply(names(gsea_results), function(cl) {
    res <- as.data.frame(gsea_results[[cl]])
    res$cluster <- cl
    res
  })
)

gsea_results_df$cluster <- factor(gsea_results_df$cluster, levels = names(rank_lists))
head(gsea_results_df)

# Mesma tabela, mas com todas as vias testadas (significativas ou não)
gsea_full_df <- bind_rows(
  lapply(names(gsea_full), function(cl) {
    res <- gsea_full[[cl]]
    res$cluster <- cl
    res
  })
)
gsea_full_df$cluster <- factor(gsea_full_df$cluster, levels = names(rank_lists))
reactome_vazio <- nrow(gsea_results_df) == 0   # [P16]
if (reactome_vazio) {
  message("Reactome: nenhuma via com p.adjust <= ", p_cutoff, "; só o resultado completo é salvo")
  utils::write.csv(gsea_full_df, file.path(tab_dir, "gsea_reactome_completo.csv"), row.names = FALSE)
}

# Função para extrair genes do core_enrichment (formato "gene1/gene2/...")
.parse_core_genes <- function(core_string) {
  strsplit(as.character(core_string), split = "/", fixed = TRUE)
}

# Função para estimar generalidade (tamanho da via no background)
# No GSEA não temos BgRatio, mas podemos usar o tamanho do conjunto de genes da via (obtido via ID)
# Vamos buscar o tamanho da via a partir do banco de dados, se disponível.
# Uma alternativa: usar o número de genes na via (obtido da coluna 'setSize').
# Vamos usar 'setSize' como proxy.
.parse_pathway_size_gsea <- function(df, size_col = "setSize") {
  if (size_col %in% colnames(df)) {
    return(as.numeric(df[[size_col]]))
  } else {
    warning("Coluna 'setSize' não encontrada. Usando 'Count' como proxy.")
    return(as.numeric(df$Count))
  }
}

# Função de similaridade entre conjuntos (mesma do ORA)
pathway_similarity <- function(a, b, method = c("jaccard", "overlap")) {
  method <- match.arg(method)
  if (length(a) == 0 || length(b) == 0) return(0)
  inter <- length(intersect(a, b))
  if (method == "jaccard") {
    uni <- length(union(a, b))
    if (uni == 0) return(0)
    inter / uni
  } else {
    denom <- min(length(a), length(b))
    if (denom == 0) return(0)
    inter / denom
  }
}

similarity_matrix <- function(gene_sets, method = c("jaccard", "overlap")) {
  method <- match.arg(method)
  n <- length(gene_sets)
  m <- matrix(0, n, n)
  if (n < 2) return(m)
  for (i in seq_len(n - 1)) {
    for (j in seq(i + 1, n)) {
      s <- pathway_similarity(gene_sets[[i]], gene_sets[[j]], method = method)
      m[i, j] <- s
      m[j, i] <- s
    }
  }
  m
}

remove_redundant_pathways_gsea <- function(df,
                                           threshold = 0.10,
                                           gene_col  = "core_enrichment",
                                           size_col  = "setSize",
                                           p_col     = "p.adjust",
                                           method    = c("jaccard", "overlap")) {
  method <- match.arg(method)
  if (nrow(df) <= 1) {
    df$via_generalidade <- if (nrow(df) == 1) .parse_pathway_size_gsea(df, size_col) else numeric(0)
    return(df)
  }
  
  # Extrair genes
  gene_sets <- .parse_core_genes(df[[gene_col]])
  generality <- .parse_pathway_size_gsea(df, size_col)
  pvals <- suppressWarnings(as.numeric(df[[p_col]]))
  pvals[is.na(pvals)] <- 1
  
  # Ordenar por generalidade (menor = mais específica) e p.adjust
  ordem <- order(generality, pvals)
  sim_mat <- similarity_matrix(gene_sets, method = method)
  
  mantidas <- integer(0)
  for (idx in ordem) {
    if (length(mantidas) == 0) {
      mantidas <- idx
      next
    }
    if (max(sim_mat[idx, mantidas]) < threshold) {
      mantidas <- c(mantidas, idx)
    }
  }
  
  out <- df[sort(mantidas), ]
  out$via_generalidade <- generality[sort(mantidas)]
  out
}

# Aplicar o filtro por cluster
filter_redundant_GSEA <- function(results_df,
                                  threshold = 0.10,
                                  cluster_col = "cluster",
                                  gene_col = "core_enrichment",
                                  size_col = "setSize",
                                  p_col = "p.adjust",
                                  method = c("jaccard", "overlap")) {
  method <- match.arg(method)
  if (is.null(results_df) || nrow(results_df) == 0) return(results_df)
  
  clusters <- unique(results_df[[cluster_col]])
  out_list <- lapply(clusters, function(cl) {
    sub_df <- results_df[results_df[[cluster_col]] == cl, ]
    remove_redundant_pathways_gsea(
      sub_df, threshold = threshold, gene_col = gene_col,
      size_col = size_col, p_col = p_col, method = method
    )
  })
  do.call(rbind, out_list)
}

# Testar thresholds
if (!reactome_vazio) {   # [P16] o espacial não previa base sem nenhuma via significativa
thresholds <- c(0.05, 0.10, 0.15)
resultados_filtrados <- list()
resumo <- data.frame()

for (th in thresholds) {
  filtrado <- filter_redundant_GSEA(
    gsea_results_df,
    threshold = th,
    cluster_col = "cluster",
    gene_col = "core_enrichment",
    size_col = "setSize",
    p_col = "p.adjust",
    method = "jaccard"
  )
  chave <- paste0("jaccard_", th)
  resultados_filtrados[[chave]] <- filtrado
  
  n_antes <- gsea_results_df %>%
    group_by(cluster) %>%
    summarise(n_antes = n(), .groups = "drop")
  
  n_depois <- filtrado %>%
    group_by(cluster) %>%
    summarise(n_depois = n(), .groups = "drop")
  
  resumo_th <- n_antes %>%
    left_join(n_depois, by = "cluster") %>%
    mutate(
      n_depois = ifelse(is.na(n_depois), 0, n_depois),
      n_removidas = n_antes - n_depois,
      pct_removida = round(100 * n_removidas / n_antes, 1),
      threshold = th,
      method = "jaccard"
    )
  resumo <- bind_rows(resumo, resumo_th)
}



# Escolher o melhor threshold (ex.: 5%)
gsea_filtrado_5pct <- resultados_filtrados[["jaccard_0.05"]]
cat("Dimensões do GSEA filtrado (5%):", dim(gsea_filtrado_5pct), "\n")
head(gsea_filtrado_5pct)
}


# Função auxiliar para ordenar clusters
.cluster_factor <- function(x) {
  x <- as.character(x)
  lev <- unique(x)
  num <- suppressWarnings(as.numeric(lev))
  lev <- if (!any(is.na(num))) lev[order(num)] else sort(lev)
  factor(x, levels = lev)
}

select_marker_pathways_gsea <- function(df_filtrado,
                                        df_original,
                                        n_per_cluster = 5,
                                        cluster_order = NULL,
                                        cluster_col = "cluster",
                                        desc_col = "Description",
                                        p_col = "p.adjust",
                                        generality_col = "via_generalidade") {
  
  df_filtrado[[cluster_col]] <- as.character(df_filtrado[[cluster_col]])
  df_original[[cluster_col]] <- as.character(df_original[[cluster_col]])
  
  # Frequência de cada via entre os clusters (no original)
  freq_df <- df_original %>%
    dplyr::distinct(.data[[cluster_col]], .data[[desc_col]]) %>%
    dplyr::count(.data[[desc_col]], name = "n_clusters")
  
  candidatos <- df_filtrado %>%
    dplyr::left_join(freq_df, by = desc_col) %>%
    dplyr::mutate(n_clusters_ord = ifelse(is.na(n_clusters), Inf, n_clusters))
  
  if (is.null(cluster_order)) {
    cluster_order <- levels(.cluster_factor(candidatos[[cluster_col]]))
  } else {
    cluster_order <- as.character(cluster_order)
  }
  
  ja_selecionadas <- character(0)
  blocos <- list()
  tabelas <- list()
  
  for (cl in cluster_order) {
    cand_cl <- candidatos %>%
      dplyr::filter(
        .data[[cluster_col]] == cl,
        !(.data[[desc_col]] %in% ja_selecionadas)
      ) %>%
      dplyr::arrange(n_clusters_ord, .data[[generality_col]], .data[[p_col]]) %>%
      dplyr::slice_head(n = n_per_cluster)
    
    if (nrow(cand_cl) == 0) next
    
    blocos[[as.character(cl)]] <- cand_cl[[desc_col]]
    
    tabelas[[as.character(cl)]] <- cand_cl %>%
      dplyr::transmute(
        cluster_origem = as.character(cl),
        Description = .data[[desc_col]],
        n_clusters = n_clusters,
        via_generalidade = .data[[generality_col]],
        p_adjust = .data[[p_col]],
        rank_no_bloco = dplyr::row_number()
      )
    
    ja_selecionadas <- c(ja_selecionadas, cand_cl[[desc_col]])
  }
  
  list(
    descricoes = ja_selecionadas,
    blocos = blocos,
    tabela = dplyr::bind_rows(tabelas),
    freq_tbl = freq_df,
    cluster_order = cluster_order
  )
}

# Aplicar a seleção
if (!reactome_vazio) {   # [P16] o espacial não previa base sem nenhuma via significativa
source_select_gsea <- select_marker_pathways_gsea(
  df_filtrado = gsea_filtrado_5pct,
  df_original = gsea_results_df,
  n_per_cluster = 5,
  cluster_order = intersect(cluster_levels, unique(as.character(gsea_filtrado_5pct$cluster)))
  # [P16] ordem do atlas; no espacial a ordem numérica dos clusters (a padrão da função)
)

cat("Vias marcadoras selecionadas por cluster:\n")
print(source_select_gsea$blocos)
}


max_chars = 17
# Função para encurtar nomes
encurtar_nome <- function(nome, max_chars = 17) {
  ifelse(
    nchar(nome) > max_chars,
    paste0(substr(nome, 1, max_chars), "..."),
    nome
  )
}

# Extrai a legenda de um ggplot como grob (NULL se não houver).
# Lê direto do gtable, pois o retorno de cowplot::get_plot_component() muda
# entre versões do ggplot2/cowplot.
extrair_legenda <- function(p) {
  g <- ggplotGrob(p)
  idx <- which(grepl("guide-box", g$layout$name) &
                 !vapply(g$grobs, inherits, logical(1), what = "zeroGrob"))
  if (length(idx) == 0) return(NULL)
  g$grobs[[idx[1]]]
}

# Função para criar dotplot de um cluster (apenas vias marcadoras)
plot_cluster_marcadoras_gsea <- function(cluster_id,
                                         df_completo,
                                         vias_marcadoras_por_cluster,
                                         max_chars = 17,
                                         base_size = 14,
                                         remover_legenda = FALSE) {
  
  vias_do_cluster <- vias_marcadoras_por_cluster[[as.character(cluster_id)]]
  
  if (is.null(vias_do_cluster) || length(vias_do_cluster) == 0) {
    return(NULL)
  }
  
  # Filtrar dados para este cluster e estas vias
  plot_df <- df_completo %>%
    dplyr::filter(
      cluster == cluster_id,
      Description %in% vias_do_cluster
    ) %>%
    dplyr::mutate(
      # Para GSEA, usamos NES (Normalized Enrichment Score) como tamanho do ponto?
      # Ou podemos usar o tamanho do conjunto (setSize) ou Count.
      # Vamos usar setSize como tamanho do ponto e NES como eixo x.
      NES = as.numeric(NES),
      p.adjust = as.numeric(p.adjust),
      Description_short = encurtar_nome(Description, max_chars = max_chars)
    ) %>%
    dplyr::arrange(NES) %>%
    dplyr::mutate(
      Description_short = factor(Description_short, levels = unique(Description_short))
    )
  
  if (nrow(plot_df) == 0) {
    return(NULL)
  }
  
  p <- ggplot(plot_df, aes(x = NES, y = Description_short)) +
    geom_point(aes(size = setSize, color = p.adjust)) +
    scale_color_gradientn(
      colors = c("firebrick", "orange", "blue"),
      name = "p.adjust",
      trans = "reverse"
    ) +
    scale_size_continuous(range = c(3, 8), name = "Set Size") +
    guides(color = guide_colourbar(barwidth = unit(7, "cm"))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(size = base_size * 1.3, face = "bold", hjust = 0.5),
      axis.title = element_text(size = base_size * 1.1, face = "bold"),
      axis.text.y = element_text(size = base_size * 0.85, face = "bold"),
      axis.text.x = element_text(size = base_size * 0.85, face = "bold"),
      legend.title = element_text(size = base_size * 0.9, face = "bold"),
      legend.text = element_text(size = base_size * 0.75),
      legend.position = ifelse(remover_legenda, "none", "bottom"),
      legend.box = "horizontal",
      plot.margin = unit(c(0.3, 0.3, 0.3, 0.3), "cm")
    ) +
    labs(
      title = as.character(cluster_id),   # [P16] nome do tipo (no espacial, "Cluster <n>")
      x = "NES",
      y = NULL
    )
  
  return(p)
}


# Definir clusters
# clusters_interesse <- c("12", "7", "2", "5", "16", "0")
clusters_interesse = levels(seurat_obj@meta.data[[cluster_col]])   # [P16] coluna por argumento
if (!reactome_vazio) {   # [P16] o espacial não previa base sem nenhuma via significativa
vias_marcadoras <- source_select_gsea$blocos
n_colunas <- 3

# Gerar gráficos sem legenda
plots <- list()
for (cluster in clusters_interesse) {
  cat(paste("Gerando gráfico para cluster", cluster, "...\n"))
  p <- plot_cluster_marcadoras_gsea(
    cluster_id = cluster,
    df_completo = gsea_results_df,
    vias_marcadoras_por_cluster = vias_marcadoras,
    max_chars = max_chars,
    base_size = 14,
    remover_legenda = TRUE
  )
  if (!is.null(p)) {
    plots[[cluster]] <- p
    cat("  ✅ OK\n")
  } else {
    cat("  ❌ Falha (sem vias)\n")
  }
}

# Manter apenas os clusters que geraram gráfico
clusters_ok <- intersect(clusters_interesse, names(plots))
n_linhas <- ceiling(length(clusters_ok) / n_colunas)

# Criar legenda separada (a partir do primeiro cluster que deu certo)
legenda <- NULL
for (cl in clusters_ok) {
  dummy <- plot_cluster_marcadoras_gsea(
    cluster_id = cl,
    df_completo = gsea_results_df,
    vias_marcadoras_por_cluster = vias_marcadoras,
    max_chars = max_chars,
    base_size = 14,
    remover_legenda = FALSE
  )
  if (!is.null(dummy)) {
    legenda <- extrair_legenda(dummy)
    break
  }
}

# Combinar painel com patchwork (sem guides = "collect")
painel_sem_legenda <- wrap_plots(
  plots[clusters_ok],
  ncol = n_colunas
) +
  plot_annotation(
    title = "Reactome GSEA - Vias Marcadoras por Cluster (Jaccard 5%)",
    theme = theme(
      plot.title = element_text(size = 28, face = "bold", hjust = 0.5)
    )
  )

# Adicionar legenda
if (!is.null(legenda)) {
  painel_final <- cowplot::plot_grid(
    painel_sem_legenda,
    legenda,
    ncol = 1,
    rel_heights = c(n_linhas, 0.16)
  )
} else {
  painel_final <- painel_sem_legenda
}

# Mostrar
print(painel_final)

# Salvar
ggsave(
  "painel_gsea_reactome_marcadoras_todos_clusters.png",
  painel_final,
  width = 22,
  height = 9 * n_linhas,
  dpi = 300,
  limitsize = FALSE
)
}


# Plot combinado: matriz via x cluster.
#   linhas  = vias, agrupadas pelo cluster de origem (faixa à direita)
#   colunas = clusters
#   cor     = NES (azul = depletada, vermelho = enriquecida)
#   tamanho = -log10(p.adjust)
#   borda   = preta se p.adjust <= p_cutoff, cinza se não significativa
#   moldura = célula do próprio cluster de origem da via
# Como usa o resultado completo (gsea_full_df), cada via aparece em todos os
# clusters em que foi testada, e não só naquele em que foi selecionada.
plot_vias_combinado <- function(df_full,
                                blocos,
                                cluster_order = NULL,
                                clusters_destaque = NULL,
                                p_cutoff = 0.05,
                                max_chars = 60,
                                max_mlog10p = 5,
                                base_size = 11,
                                prefixo_faixa = "",   # [P16] no espacial, "C" (clusters numéricos)
                                titulo = NULL) {

  blocos <- blocos[lengths(blocos) > 0]
  if (is.null(cluster_order)) cluster_order <- levels(.cluster_factor(df_full$cluster))
  cluster_order <- as.character(cluster_order)

  vias <- data.frame(
    origem = rep(names(blocos), lengths(blocos)),
    Description = unlist(blocos, use.names = FALSE),
    stringsAsFactors = FALSE
  )

  res <- df_full[, c("cluster", "Description", "NES", "p.adjust")]
  res$cluster <- as.character(res$cluster)
  res <- res[res$cluster %in% cluster_order & res$Description %in% vias$Description, ]

  plot_df <- merge(vias, res, by = "Description")
  plot_df$NES <- as.numeric(plot_df$NES)
  plot_df$p.adjust <- as.numeric(plot_df$p.adjust)
  plot_df$mlog10p <- pmin(-log10(plot_df$p.adjust), max_mlog10p)
  plot_df$signif <- factor(plot_df$p.adjust <= p_cutoff, levels = c(TRUE, FALSE))

  # Primeira via de cada bloco no topo
  to_factors <- function(d) {
    d$cluster <- factor(d$cluster, levels = cluster_order)
    d$origem <- factor(d$origem, levels = names(blocos))
    d$Description <- factor(d$Description, levels = rev(vias$Description))
    d
  }
  plot_df <- to_factors(plot_df)

  proprio <- vias[vias$origem %in% cluster_order, ]
  proprio$cluster <- proprio$origem
  proprio <- to_factors(proprio)

  nes_lim <- max(abs(plot_df$NES), na.rm = TRUE)

  p <- ggplot(plot_df, aes(x = cluster, y = Description))

  if (length(clusters_destaque) > 0) {
    destaque <- merge(vias, data.frame(cluster = as.character(clusters_destaque)))
    destaque <- to_factors(destaque[destaque$cluster %in% cluster_order, ])
    p <- p + geom_tile(data = destaque, fill = "#FBE7A1", alpha = 0.55,
                       width = 1, height = 1)
  }

  p <- p +
    geom_tile(data = proprio, fill = NA, colour = "grey45", linewidth = 0.25,
              width = 0.92, height = 0.92) +
    geom_point(aes(size = mlog10p, fill = NES, colour = signif),
               shape = 21, stroke = 0.45) +
    scale_fill_gradient2(
      low = "#2166AC", mid = "#F2F2F2", high = "#B2182B", midpoint = 0,
      limits = c(-nes_lim, nes_lim), name = "NES"
    ) +
    scale_colour_manual(
      values = c(`TRUE` = "black", `FALSE` = "grey70"),
      labels = c(`TRUE` = paste0("p.adj <= ", p_cutoff), `FALSE` = "n.s."),
      name = NULL, drop = FALSE
    ) +
    scale_size_continuous(
      range = c(0.4, 4.5), limits = c(0, max_mlog10p),
      name = expression(-log[10](p.adj))
    ) +
    scale_x_discrete(drop = FALSE) +
    scale_y_discrete(labels = function(x) stringr::str_trunc(x, max_chars)) +
    facet_grid(origem ~ ., scales = "free_y", space = "free_y",
               labeller = labeller(origem = function(x) paste0(prefixo_faixa, x))) +
    guides(
      x.sec = "axis",
      fill = guide_colourbar(order = 1, barheight = unit(3.5, "cm")),
      size = guide_legend(order = 2),
      colour = guide_legend(order = 3, override.aes = list(size = 3.5, fill = "grey90"))
    ) +
    theme_bw(base_size = base_size) +
    theme(
      panel.grid.major = element_line(colour = "grey93", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      panel.spacing.y = unit(0.12, "lines"),
      panel.border = element_rect(colour = "grey60", linewidth = 0.3),
      strip.text.y = element_text(angle = 0, face = "bold", size = base_size * 0.8),
      strip.background = element_rect(fill = "grey92", colour = "grey60", linewidth = 0.3),
      axis.text.y = element_text(size = base_size * 0.72, colour = "grey15"),
      # [P16] nomes longos (no espacial, números): rótulos inclinados para não se sobreporem
      axis.text.x.bottom = element_text(size = base_size * 0.9, colour = "grey15", angle = 35, hjust = 1, vjust = 1),
      axis.text.x.top = element_text(size = base_size * 0.9, colour = "grey15", angle = 35, hjust = 0, vjust = 0),
      axis.ticks = element_line(colour = "grey60", linewidth = 0.25),
      axis.title = element_text(face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.title.position = "plot"
    ) +
    labs(title = titulo, x = "Cluster", y = NULL)

  attr(p, "n_vias") <- nrow(vias)
  attr(p, "n_clusters") <- length(cluster_order)
  p
}

# Tamanho de figura proporcional ao número de vias e de clusters
salvar_vias_combinado <- function(p, arquivo, dpi = 300) {
  ggsave(
    arquivo, p,
    width = 6.5 + 0.33 * attr(p, "n_clusters"),
    height = 2 + 0.17 * attr(p, "n_vias"),
    dpi = dpi, limitsize = FALSE
  )
}

# Comparação direta entre dois (ou mais) clusters: pega as top vias significativas
# de cada um e mostra o NES de todas elas em todos os clusters comparados.
# As vias são separadas em exclusivas de cada cluster e compartilhadas.
blocos_comparacao <- function(df_sig, clusters, n_per_cluster = 15) {
  clusters <- as.character(clusters)
  df_sig$cluster <- as.character(df_sig$cluster)
  df_sig <- df_sig[df_sig$cluster %in% clusters, ]

  top <- unique(unlist(lapply(clusters, function(cl) {
    d <- df_sig[df_sig$cluster == cl, ]
    d <- d[order(d$p.adjust, -abs(d$NES)), ]
    head(d$Description, n_per_cluster)
  })))

  # Em quantos dos clusters comparados cada via é significativa
  onde <- tapply(df_sig$cluster, df_sig$Description, function(x) sort(unique(x)))[top]
  rotulo <- vapply(onde, function(x) {
    if (length(x) > 1) "ambos" else x
  }, character(1))

  blocos <- split(top, factor(rotulo, levels = c(clusters, "ambos")))
  blocos[lengths(blocos) > 0]
}


# Clusters a destacar no plot combinado e a comparar diretamente
clusters_comparar <- clusters_comparar_arg   # [P16] por argumento (no espacial, c("0", "7"))

if (!reactome_vazio) {   # [P16] o espacial não previa base sem nenhuma via significativa
# 1) Todas as vias marcadoras x todos os clusters
p_combinado <- plot_vias_combinado(
  df_full = gsea_full_df,
  blocos = vias_marcadoras[intersect(clusters_interesse, names(vias_marcadoras))],
  cluster_order = clusters_interesse,
  clusters_destaque = clusters_comparar,
  p_cutoff = p_cutoff,
  titulo = "Reactome GSEA - Vias marcadoras por cluster (Jaccard 5%)"
)
print(p_combinado)
salvar_vias_combinado(p_combinado, "gsea_reactome_marcadoras_combinado.png")

# 2) Cluster 0 x cluster 7: as vias de um aparecem no outro?
#    Usa as vias significativas já sem redundância (Jaccard 5%)
p_comparacao <- plot_vias_combinado(
  df_full = gsea_full_df,
  blocos = blocos_comparacao(gsea_filtrado_5pct, clusters_comparar, n_per_cluster = 15),
  cluster_order = clusters_comparar,
  p_cutoff = p_cutoff,
  prefixo_faixa = "",
  base_size = 12,
  titulo = paste("Reactome GSEA - Cluster", paste(clusters_comparar, collapse = " x "))
)
print(p_comparacao)
ggsave(
  paste0("gsea_reactome_comparacao_", paste(clusters_comparar, collapse = "_vs_"), ".png"),
  p_comparacao,
  width = 8.5, height = 2 + 0.22 * attr(p_comparacao, "n_vias"),
  dpi = 300, limitsize = FALSE
)

# Tabela da comparação (NES e p.adjust lado a lado)
tabela_comparacao <- gsea_full_df %>%
  dplyr::filter(cluster %in% clusters_comparar,
                Description %in% unlist(blocos_comparacao(gsea_filtrado_5pct, clusters_comparar, 15))) %>%
  dplyr::select(Description, cluster, NES, p.adjust) %>%
  tidyr::pivot_wider(names_from = cluster, values_from = c(NES, p.adjust))
write.csv(
  tabela_comparacao,
  paste0("gsea_reactome_comparacao_", paste(clusters_comparar, collapse = "_vs_"), ".csv"),
  row.names = FALSE
)


# Tabela com vias marcadoras
tabela_marcadoras_gsea <- source_select_gsea$tabela %>%
  dplyr::filter(cluster_origem %in% clusters_interesse) %>%
  dplyr::arrange(match(cluster_origem, clusters_interesse), rank_no_bloco)   # [P16] nomes, não números

write.csv(tabela_marcadoras_gsea, "vias_marcadoras_gsea_6_clusters.csv", row.names = FALSE)

# Salvar o GSEA filtrado completo
write.csv(gsea_filtrado_5pct, "gsea_reactome_filtrado_5pct.csv", row.names = FALSE)

# Resultado completo (todas as vias testadas), usado pelo heatmaps_vias.Rmd
write.csv(gsea_full_df, "gsea_reactome_completo.csv", row.names = FALSE)

# Salvar o resumo do filtro
write.csv(resumo, "resumo_filtro_gsea.csv", row.names = FALSE)

cat("\nArquivos salvos:\n")
cat("- vias_marcadoras_gsea_6_clusters.csv\n")
cat("- gsea_reactome_filtrado_5pct.csv\n")
cat("- resumo_filtro_gsea.csv\n")
cat("- painel_gsea_reactome_marcadoras_todos_clusters.png\n")
cat("- gsea_reactome_marcadoras_combinado.png\n")
cat("- gsea_reactome_comparacao_", paste(clusters_comparar, collapse = "_vs_"), ".png / .csv\n", sep = "")
}

# # KEGG e Hallmark

# Reaproveita `rank_lists`, `clusters_interesse`, `clusters_comparar` e as funções
# definidas acima (filtro de redundância, seleção de vias marcadoras e plots).


# Converte um ranking nomeado por SYMBOL para ENTREZID
# (mesma lógica usada no loop do Reactome: duplicatas ficam com o maior score)
rank_para_entrez <- function(gene_list) {
  gene_df <- suppressMessages(suppressWarnings(bitr(names(gene_list),
                                   fromType = "SYMBOL",
                                   toType = "ENTREZID",
                                   OrgDb = org.Hs.eg.db,
                                   drop = TRUE)))
  genes_mapeados <- gene_list[names(gene_list) %in% gene_df$SYMBOL]
  names(genes_mapeados) <- gene_df$ENTREZID[match(names(genes_mapeados), gene_df$SYMBOL)]
  agg <- tapply(genes_mapeados, names(genes_mapeados), max)
  sort(setNames(as.numeric(agg), names(agg)), decreasing = TRUE)
}

# Roda uma função de GSEA em todos os clusters e devolve, por cluster, o
# data.frame completo (sem corte de p). fun_gsea recebe o ranking do cluster.
rodar_gsea_clusters <- function(listas, fun_gsea, min_genes = 10) {
  out <- list()
  for (cl in names(listas)) {
    message(paste("Rodando GSEA para o cluster", cl))
    gene_list <- listas[[cl]]

    if (length(gene_list) < min_genes) {
      message(paste("  Cluster", cl, "tem poucos genes. Pulando."))
      next
    }

    tryCatch({
      gsea <- fun_gsea(gene_list)
      res <- if (is.null(gsea)) data.frame() else as.data.frame(gsea)
      if (nrow(res) > 0) {
        out[[cl]] <- res
        message(paste("  Cluster", cl, "->", nrow(res), "vias testadas"))
      } else {
        message(paste("  Cluster", cl, "-> nenhuma via testada"))
      }
    }, error = function(e) {
      message(paste("  Erro no cluster", cl, ":", e$message))
    })
  }
  out
}

# Painel com um dotplot por cluster (mesmo layout do painel do Reactome)
painel_marcadoras_separado <- function(df_sig, vias_marcadoras, clusters, titulo,
                                       n_colunas = 3, max_chars = 17, base_size = 14) {
  plots <- list()
  for (cl in clusters) {
    p <- plot_cluster_marcadoras_gsea(
      cluster_id = cl,
      df_completo = df_sig,
      vias_marcadoras_por_cluster = vias_marcadoras,
      max_chars = max_chars,
      base_size = base_size,
      remover_legenda = TRUE
    )
    if (!is.null(p)) plots[[cl]] <- p
  }
  if (length(plots) == 0) return(NULL)

  n_linhas <- ceiling(length(plots) / n_colunas)

  dummy <- plot_cluster_marcadoras_gsea(
    cluster_id = names(plots)[1],
    df_completo = df_sig,
    vias_marcadoras_por_cluster = vias_marcadoras,
    max_chars = max_chars,
    base_size = base_size,
    remover_legenda = FALSE
  )
  legenda <- extrair_legenda(dummy)

  painel <- wrap_plots(plots, ncol = n_colunas) +
    plot_annotation(
      title = titulo,
      theme = theme(
        plot.title = element_text(size = 28, face = "bold", hjust = 0.5)
      )
    )

  if (!is.null(legenda)) {
    painel <- cowplot::plot_grid(
      painel,
      legenda,
      ncol = 1,
      rel_heights = c(n_linhas, 0.16)
    )
  }

  attr(painel, "n_linhas") <- n_linhas
  painel
}

# Pipeline completo a partir do resultado de rodar_gsea_clusters():
# corte de p -> filtro de redundância -> vias marcadoras -> painel separado,
# plot combinado, comparação direta entre clusters e tabelas.
#   nome    = usado nos nomes dos arquivos (ex.: "kegg")
#   rotulo  = usado nos títulos (ex.: "KEGG")
#   jaccard = threshold do filtro de redundância; NULL para não filtrar
#   selecao = "marcadoras": n_per_cluster vias por cluster, sem repetir entre
#                           clusters (mesmo critério do Reactome)
#             "todas":      para bases pequenas (ex.: Hallmark). O plot combinado
#                           mostra todas as vias significativas, cada uma no bloco
#                           do cluster em que é mais significativa, e os plots
#                           separados mostram as top n_per_cluster de cada cluster
analisar_base <- function(gsea_full,
                          nome,
                          rotulo,
                          jaccard = 0.05,
                          selecao = c("marcadoras", "todas"),
                          n_per_cluster = 5,
                          clusters = clusters_interesse,
                          comparar = clusters_comparar,
                          p_cutoff = 0.05,
                          max_chars_combinado = 60) {

  selecao <- match.arg(selecao)

  full_df <- bind_rows(
    lapply(names(gsea_full), function(cl) {
      res <- gsea_full[[cl]]
      res$cluster <- cl
      res
    })
  )
  sig_df <- full_df[which(full_df$p.adjust <= p_cutoff), , drop = FALSE]
  cat(rotulo, ":", nrow(sig_df), "vias significativas em",
      length(unique(sig_df$cluster)), "clusters\n")
  if (nrow(sig_df) == 0) {
    write.csv(full_df, paste0("gsea_", nome, "_completo.csv"), row.names = FALSE)   # [P16] salvo mesmo sem vias
    return(list(full = full_df, sig = sig_df))
  }

  if (is.null(jaccard)) {
    filtrado <- sig_df
    filtrado$via_generalidade <- .parse_pathway_size_gsea(filtrado)
    sufixo_titulo <- ""
  } else {
    filtrado <- filter_redundant_GSEA(sig_df, threshold = jaccard, method = "jaccard")
    sufixo_titulo <- paste0(" (Jaccard ", 100 * jaccard, "%)")
  }

  clusters_com_vias <- intersect(clusters, unique(as.character(filtrado$cluster)))

  if (selecao == "marcadoras") {
    sel <- select_marker_pathways_gsea(
      df_filtrado = filtrado,
      df_original = sig_df,
      n_per_cluster = n_per_cluster,
      cluster_order = clusters_com_vias
    )
    vias_marcadoras <- sel$blocos
    vias_separado <- vias_marcadoras
    tabela_vias <- sel$tabela
    nome_vias <- "Vias marcadoras"
  } else {
    ordenado <- filtrado %>%
      dplyr::mutate(cluster = as.character(cluster)) %>%
      dplyr::filter(cluster %in% clusters_com_vias) %>%
      dplyr::arrange(p.adjust, dplyr::desc(abs(NES)))

    # Combinado: cada via no cluster em que é mais significativa
    melhor <- ordenado %>% dplyr::distinct(Description, .keep_all = TRUE)
    vias_marcadoras <- split(melhor$Description, factor(melhor$cluster, levels = clusters_com_vias))
    vias_marcadoras <- vias_marcadoras[lengths(vias_marcadoras) > 0]

    # Separado: top vias de cada cluster, podendo repetir entre clusters
    vias_separado <- lapply(split(ordenado$Description, factor(ordenado$cluster, levels = clusters_com_vias)),
                            head, n = n_per_cluster)

    tabela_vias <- melhor %>%
      dplyr::transmute(cluster_origem = cluster, Description, NES, p_adjust = p.adjust) %>%
      dplyr::arrange(match(cluster_origem, clusters_com_vias), p_adjust)
    nome_vias <- "Vias significativas"
  }

  # Plots separados por cluster
  painel <- painel_marcadoras_separado(
    sig_df, vias_separado, clusters,
    titulo = paste0(rotulo, " GSEA - ", nome_vias, " por Cluster", sufixo_titulo)
  )
  if (!is.null(painel)) {
    print(painel)
    ggsave(
      paste0("painel_gsea_", nome, "_marcadoras_todos_clusters.png"),
      painel,
      width = 22,
      height = 9 * attr(painel, "n_linhas"),
      dpi = 300,
      limitsize = FALSE
    )
  }

  # Plot combinado: vias marcadoras x todos os clusters
  p_combinado <- plot_vias_combinado(
    df_full = full_df,
    blocos = vias_marcadoras,
    cluster_order = clusters,
    clusters_destaque = comparar,
    p_cutoff = p_cutoff,
    max_chars = max_chars_combinado,
    titulo = paste0(rotulo, " GSEA - ", nome_vias, " por cluster", sufixo_titulo)
  )
  print(p_combinado)
  salvar_vias_combinado(p_combinado, paste0("gsea_", nome, "_marcadoras_combinado.png"))

  # Comparação direta entre os clusters de interesse
  p_comparacao <- NULL
  tag_comp <- paste(comparar, collapse = "_vs_")
  if (any(as.character(filtrado$cluster) %in% comparar)) {
    blocos_comp <- blocos_comparacao(filtrado, comparar, n_per_cluster = 15)
    p_comparacao <- plot_vias_combinado(
      df_full = full_df,
      blocos = blocos_comp,
      cluster_order = comparar,
      p_cutoff = p_cutoff,
      max_chars = max_chars_combinado,
      prefixo_faixa = "",
      base_size = 12,
      titulo = paste(rotulo, "GSEA - Cluster", paste(comparar, collapse = " x "))
    )
    print(p_comparacao)
    ggsave(
      paste0("gsea_", nome, "_comparacao_", tag_comp, ".png"),
      p_comparacao,
      width = 8.5, height = 2 + 0.22 * attr(p_comparacao, "n_vias"),
      dpi = 300, limitsize = FALSE
    )

    tabela_comparacao <- full_df %>%
      dplyr::filter(cluster %in% comparar, Description %in% unlist(blocos_comp)) %>%
      dplyr::select(Description, cluster, NES, p.adjust) %>%
      tidyr::pivot_wider(names_from = cluster, values_from = c(NES, p.adjust))
    write.csv(tabela_comparacao,
              paste0("gsea_", nome, "_comparacao_", tag_comp, ".csv"),
              row.names = FALSE)
  } else {
    message(paste("  Nenhuma via significativa nos clusters",
                  paste(comparar, collapse = " e "), "- comparação não gerada."))
  }

  # Tabelas
  write.csv(tabela_vias, paste0("vias_marcadoras_gsea_", nome, ".csv"), row.names = FALSE)
  write.csv(filtrado, paste0("gsea_", nome, "_filtrado.csv"), row.names = FALSE)
  write.csv(full_df, paste0("gsea_", nome, "_completo.csv"), row.names = FALSE)

  list(
    full = full_df,
    sig = sig_df,
    filtrado = filtrado,
    blocos = vias_marcadoras,
    tabela = tabela_vias,
    painel = painel,
    combinado = p_combinado,
    comparacao = p_comparacao
  )
}

# ## KEGG


# gseKEGG baixa as vias da API do KEGG (precisa de internet) e usa ENTREZID
rank_lists_entrez <- lapply(rank_lists, rank_para_entrez)

set.seed(42)
gsea_kegg_full <- rodar_gsea_clusters(rank_lists_entrez, function(gene_list) {
  gseKEGG(
    geneList = gene_list,
    organism = "hsa",
    keyType = "ncbi-geneid",
    pvalueCutoff = 1,
    pAdjustMethod = "BH",
    seed = TRUE,
    verbose = FALSE
  )
})

# Algumas versões devolvem o nome com o sufixo do organismo
gsea_kegg_full <- lapply(gsea_kegg_full, function(res) {
  res$Description <- sub(" - Homo sapiens \\(human\\)$", "", res$Description)
  res
})


res_kegg <- analisar_base(gsea_kegg_full, nome = "kegg", rotulo = "KEGG", jaccard = 0.05)
print(res_kegg$blocos)

# ## Hallmark


library(msigdbr)

# msigdbr >= 10 usa 'collection'; versões anteriores usam 'category'
hallmark_df <- if ("collection" %in% names(formals(msigdbr))) {
  msigdbr(species = "Homo sapiens", collection = "H")
} else {
  msigdbr(species = "Homo sapiens", category = "H")
}
hallmark_t2g <- hallmark_df %>%
  dplyr::distinct(gs_name, gene_symbol) %>%
  as.data.frame()

# Hallmark é por SYMBOL, então usa rank_lists direto (sem conversão para ENTREZID)
set.seed(42)
gsea_hallmark_full <- rodar_gsea_clusters(rank_lists, function(gene_list) {
  GSEA(
    geneList = gene_list,
    TERM2GENE = hallmark_t2g,
    pvalueCutoff = 1,
    pAdjustMethod = "BH",
    seed = TRUE,
    verbose = FALSE
  )
})

# "HALLMARK_TNFA_SIGNALING_VIA_NFKB" -> "TNFA SIGNALING VIA NFKB"
gsea_hallmark_full <- lapply(gsea_hallmark_full, function(res) {
  res$Description <- gsub("_", " ", sub("^HALLMARK_", "", res$ID))
  res
})


# As 50 hallmarks já foram desenhadas para não serem redundantes entre si,
# então o filtro de Jaccard não é aplicado. Como são poucas, o plot combinado
# mostra todas as significativas em vez de 5 exclusivas por cluster.
res_hallmark <- analisar_base(gsea_hallmark_full, nome = "hallmark", rotulo = "Hallmark",
                              jaccard = NULL, selecao = "todas")
print(res_hallmark$blocos)


