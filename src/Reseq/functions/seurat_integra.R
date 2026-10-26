# Integração com o Seurat v5 (IntegrateLayers: CCA ou RPCA) para o benchmark do P07.
# Uso: Rscript --vanilla seurat_integra.R <diretório> <cca|rpca>
# Entrada no diretório: m.mtx (genes × células, lognorm das HVG do P05), genes.csv, obs.csv (cell, library_id).
# Saída: emb.csv (células × dimensões da redução integrada).
#
# A lognorm entra pronta na camada "data" (normalizada sobre todos os genes, como nos outros métodos);
# as variáveis são as mesmas 2.000 HVG; o resto segue a vinheta de integração do Seurat v5:
# split por biblioteca → ScaleData → RunPCA(50) → IntegrateLayers(dims = 1:30).
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
})
options(future.globals.maxSize = 16 * 1024^3)
d <- args[1]
metodo <- tolower(args[2])
set.seed(123)

m <- as(Matrix::readMM(file.path(d, "m.mtx")), "CsparseMatrix")
genes <- read.csv(file.path(d, "genes.csv"), colClasses = "character")$gene
obs <- read.csv(file.path(d, "obs.csv"), colClasses = "character")
rownames(m) <- genes
colnames(m) <- obs$cell
rownames(obs) <- obs$cell

obj <- CreateSeuratObject(CreateAssay5Object(data = m), meta.data = obs[, "library_id", drop = FALSE])
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$library_id)
VariableFeatures(obj) <- genes
obj <- ScaleData(obj, features = genes, verbose = FALSE)
obj <- RunPCA(obj, features = genes, npcs = 50, verbose = FALSE)

menor <- min(table(obj$library_id))
k_weight <- min(100, menor - 1)
k_filter <- min(200, menor - 1)
dims <- seq_len(min(30, menor - 1))   # 1:30 do Seurat; menos só se uma biblioteca tiver < 31 células
fun <- switch(metodo, cca = CCAIntegration, rpca = RPCAIntegration, stop("método desconhecido: ", metodo))
t0 <- Sys.time()
# Se algum par de bibliotecas tiver menos âncoras que k.weight, o Seurat para e sugere o limite;
# tenta de novo com ele (no máximo 3 vezes) e registra o valor usado.
integrado <- NULL
for (tentativa in 1:4) {
  integrado <- tryCatch(
    IntegrateLayers(obj, method = fun, orig.reduction = "pca", new.reduction = "integrado",
                    dims = dims, k.weight = k_weight, k.filter = k_filter, verbose = FALSE),
    error = function(e) e)
  if (!inherits(integrado, "error")) break
  sug <- regmatches(conditionMessage(integrado), regexpr("less than [0-9]+", conditionMessage(integrado)))
  if (tentativa == 4 || length(sug) == 0) stop(integrado)
  k_weight <- as.integer(sub("less than ", "", sug)) - 1L
  message("refazendo com k.weight = ", k_weight)
}
obj <- integrado
emb <- Embeddings(obj, "integrado")
write.csv(emb, file.path(d, "emb.csv"))
cat(sprintf("Seurat %s %s: %d células × %d dimensões (dims 1:%d); k.weight = %d, k.filter = %d; %.1f min\n",
            as.character(packageVersion("Seurat")), toupper(metodo), nrow(emb), ncol(emb), max(dims), k_weight, k_filter,
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
