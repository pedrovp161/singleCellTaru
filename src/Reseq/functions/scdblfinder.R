# scDblFinder numa amostra, como no capítulo de QC do sc-best-practices.
# Uso: Rscript --vanilla scdblfinder.R <diretório com m.mtx e barcodes.csv> <semente>
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(scDblFinder)
})
d <- args[1]
m <- as(Matrix::readMM(file.path(d, "m.mtx")), "CsparseMatrix")
bc <- read.csv(file.path(d, "barcodes.csv"), colClasses = "character")$barcode
rownames(m) <- paste0("g", seq_len(nrow(m)))
colnames(m) <- bc
set.seed(as.integer(args[2]))
sce <- scDblFinder(SingleCellExperiment(list(counts = m)))
write.csv(data.frame(barcode = colnames(sce), score = sce$scDblFinder.score,
                     class = sce$scDblFinder.class), file.path(d, "scdblfinder.csv"), row.names = FALSE)
cat("OK\n")
