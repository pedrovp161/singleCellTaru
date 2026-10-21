# Gotas vazias com o DropletUtils, a partir da matriz raw do Cell Ranger.
#
# Uso: Rscript --vanilla droplets.R <raw_feature_bc_matrix.h5> <saída> <lower> <niters> <semente>
#
# Grava em <saída>:
#   ranks.csv      knee e inflexão do barcodeRanks
#   emptydrops.csv emptyDrops (lower, niters, retain = knee) para os barcodes testados
#   edcr.csv       emptyDropsCellRanger com os parâmetros padrão (imitam o Cell Ranger)
#   ambiente.csv   perfil ambiente estimado (Good-Turing) sobre os barcodes com ≤ lower UMIs
args <- commandArgs(trailingOnly = TRUE)
lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(Matrix)
  library(DropletUtils)
})
h5 <- args[1]; out <- args[2]
lower <- as.numeric(args[3]); niters <- as.integer(args[4]); seed <- as.integer(args[5])
dir.create(out, showWarnings = FALSE, recursive = TRUE)

sce <- read10xCounts(h5, col.names = TRUE, type = "HDF5")
m <- as(counts(sce), "dgCMatrix")
rm(sce); invisible(gc())

br <- barcodeRanks(m, lower = lower)
md <- metadata(br)
write.csv(data.frame(knee = md$knee, inflection = md$inflection), file.path(out, "ranks.csv"), row.names = FALSE)

set.seed(seed)
ed <- emptyDrops(m, lower = lower, niters = niters)
keep <- !is.na(ed$FDR)
write.csv(data.frame(barcode = rownames(ed)[keep], total = ed$Total[keep], logprob = ed$LogProb[keep],
                     pvalue = ed$PValue[keep], limited = ed$Limited[keep], fdr = ed$FDR[keep]),
          file.path(out, "emptydrops.csv"), row.names = FALSE)

set.seed(seed)
cr <- emptyDropsCellRanger(m, niters = niters)
keep <- !is.na(cr$FDR)
write.csv(data.frame(barcode = rownames(cr)[keep], total = cr$Total[keep], pvalue = cr$PValue[keep],
                     limited = cr$Limited[keep], fdr = cr$FDR[keep]),
          file.path(out, "edcr.csv"), row.names = FALSE)

amb <- ambientProfileEmpty(m, lower = lower, good.turing = TRUE)
write.csv(data.frame(gene = rownames(m), prop = as.numeric(amb)), file.path(out, "ambiente.csv"), row.names = FALSE)
cat("OK\n")
