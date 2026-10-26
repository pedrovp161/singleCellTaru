# Mapeamento no atlas de células T CD4⁺ de Yasumizu et al. (Cell Genomics 2024) com o Symphony.
# Segue reference_mapping_seuratobj() de github.com/yyoshiaki/screfmapping (R), com quatro mudanças:
# a leitura (HDF5 no formato 10x, escrito pelo Python), o SCTransform só nos genes da referência
# (os resíduos desses genes são os mesmos; poupa memória) e o vst.flavor = "v1" explícito, que era o
# padrão quando a referência foi montada (Seurat 4) e deixou de ser no Seurat 5; e a transferência
# de rótulos com a votação de knnPredict.Seurat (k = 5), vizinhos buscados com RANN.
#
# Uso: Rscript --vanilla atlas_symphony.R <dir> <ref.RData>
# Entrada em <dir>: counts.h5 (genes × células, contagens inteiras) e obs.csv (cell, batch).
# Saída em <dir>: mapeamento.csv (cell, clusterL1, clusterL1_prob, clusterL2, clusterL2_prob)
#                 e z_corr.csv (embedding corrigido da query, 50 dimensões).
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(symphony)
  library(dplyr)
  library(purrr)
  library(DropletUtils)
})
d <- args[1]
set.seed(0)

e <- new.env()
nm <- load(args[2], envir = e)
ref <- e[["ref"]]

sce <- DropletUtils::read10xCounts(file.path(d, "counts.h5"), type = "HDF5", col.names = TRUE)
m <- as(SingleCellExperiment::counts(sce), "dgCMatrix")
rownames(m) <- make.unique(SummarizedExperiment::rowData(sce)$Symbol)
obs <- read.csv(file.path(d, "obs.csv"), colClasses = "character")
colnames(m) <- obs$cell
rownames(obs) <- obs$cell
cat(sprintf("query: %d genes × %d células; %d bibliotecas\n", nrow(m), ncol(m), length(unique(obs$batch))))

genes_ref <- ref$vargenes$symbol
presentes <- intersect(genes_ref, rownames(m))
cat(sprintf("genes da referência presentes na query: %d de %d\n", length(presentes), length(genes_ref)))

q <- CreateSeuratObject(counts = m, meta.data = obs[, "batch", drop = FALSE])
q <- SCTransform(q, assay = "RNA", method = "glmGamPoi", vst.flavor = "v1",
                 residual.features = presentes, verbose = FALSE)
exp_query <- GetAssayData(q, assay = "SCT", layer = "scale.data")

# symphony::mapQuery (a do pacote): a versão dos autores (utils_seurat.R) só acrescenta o retorno
# como objeto Seurat, que escreve em slots que o Assay5 do Seurat 5 não tem, e o arquivo chama
# harmony::HarmonyMatrix, removida no harmony 2.x. O embedding corrigido é o mesmo.
# Sem UMAP: o modelo uwot da referência não vem no objeto.
res <- symphony::mapQuery(exp_query = exp_query, metadata_query = q@meta.data, ref_obj = ref,
                          vars = "batch", do_normalize = FALSE, do_umap = FALSE)
Zq <- t(res$Z)                                   # células × 50
# a votação de knnPredict.Seurat (k = 5; prob = fração de votos do vencedor), com os vizinhos
# buscados por árvore k-d (RANN) em vez da força bruta de class::knn; empate → vizinho mais próximo
nn <- RANN::nn2(data = t(ref$Z_corr), query = Zq, k = 5)$nn.idx
pred <- function(rotulo) {
  lab <- matrix(as.character(ref$meta_data[[rotulo]])[nn], nrow = nrow(nn))
  voto <- apply(lab, 1, function(v) {
    t <- table(v)
    venc <- names(t)[t == max(t)]
    if (length(venc) > 1) venc <- v[v %in% venc][1]
    c(venc, max(t) / length(v))
  })
  list(rotulo = voto[1, ], prob = as.numeric(voto[2, ]))
}
L1 <- pred("clusterL1")
L2 <- pred("clusterL2")
out <- data.frame(cell = rownames(Zq), clusterL1 = L1$rotulo, clusterL1_prob = L1$prob,
                  clusterL2 = L2$rotulo, clusterL2_prob = L2$prob)
write.csv(out, file.path(d, "mapeamento.csv"), row.names = FALSE)
write.csv(Zq, file.path(d, "z_corr.csv"))
cat(sprintf("OK: %d células mapeadas; L1: %s\n", nrow(out),
            paste(names(table(out$clusterL1)), table(out$clusterL1), sep = "=", collapse = ", ")))
