# SoupX como no capítulo de QC do sc-best-practices.
# Uso: Rscript --vanilla soupx.R <raw_feature_bc_matrix.h5> <celulas.csv> <saída>
# celulas.csv: barcode, cluster. Grava rho.csv e, se o ajuste der certo, a matriz
# corrigida (corrigida.mtx, genes × células) com barcodes.csv e genes.csv.
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({
  library(Matrix)
  library(DropletUtils)
  library(SoupX)
})
out <- args[3]
tod <- as(counts(read10xCounts(args[1], col.names = TRUE, type = "HDF5")), "dgCMatrix")
cel <- read.csv(args[2], colClasses = c("character", "character"))
toc <- tod[, cel$barcode]
res <- tryCatch({
  sc <- SoupChannel(tod, toc, calcSoupProfile = TRUE)
  sc <- setClusters(sc, setNames(cel$cluster, cel$barcode))
  sc <- autoEstCont(sc, doPlot = FALSE)
  adj <- adjustCounts(sc, roundToInt = TRUE)
  writeMM(adj, file.path(out, "corrigida.mtx"))
  write.csv(data.frame(barcode = colnames(adj)), file.path(out, "barcodes.csv"), row.names = FALSE)
  write.csv(data.frame(gene = rownames(adj)), file.path(out, "genes.csv"), row.names = FALSE)
  data.frame(rho = sc$metaData$rho[1], status = "ok")
}, error = function(e) data.frame(rho = NA, status = gsub("[\r\n,]", " ", conditionMessage(e))))
write.csv(res, file.path(out, "rho.csv"), row.names = FALSE)
cat("OK\n")
