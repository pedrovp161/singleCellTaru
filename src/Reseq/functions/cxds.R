# cxds do pacote scds sobre uma matriz genes × células (MatrixMarket).
# Uso: Rscript --vanilla cxds.R <diretório com m.mtx> <ntop>
# Grava <diretório>/score.txt com um score por célula, na ordem das colunas.
args <- commandArgs(trailingOnly = TRUE)
lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(scds)
})
m <- as(Matrix::readMM(file.path(args[1], "m.mtx")), "CsparseMatrix")
rownames(m) <- paste0("g", seq_len(nrow(m)))
colnames(m) <- paste0("c", seq_len(ncol(m)))
sce <- SingleCellExperiment(assays = list(counts = m))
sce <- cxds(sce, ntop = as.integer(args[2]))
write(format(colData(sce)$cxds_score, digits = 17), file.path(args[1], "score.txt"), ncolumns = 1)
