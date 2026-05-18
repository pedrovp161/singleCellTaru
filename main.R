library(Seurat)

raw = file.path("data", "raw", fsep = "//")

samples <- list.dirs(raw, full.names = FALSE, recursive = FALSE)

seurat_list <- list()
for (sample in samples) {
  path <- file.path(raw, sample)
  
  counts <- Read10X(data.dir = path)
  
  obj <- CreateSeuratObject(counts = counts, project = sample)
  
  seurat_list[[sample]] <- obj
  cat(sprintf("✓ %s: %d células, %d genes\n", sample, ncol(obj), nrow(obj)))
}

cat(sprintf("\nTotal de amostras carregadas: %d\n", length(seurat_list)))

# for (name in names(seurat_list)) {
#   saveRDS(seurat_list, file = file.path("data", "raw_RDS", paste0(name, ".rds")))
# }

# preprocessamento de utils

qc <- QualityControl$new()

seurat_list_processed <- lapply(names(seurat_list), function(sample_name) {

  obj <- seurat_list[[sample_name]]

  message("Processing: ", sample_name)

  obj <- qc$preprocessing(
    obj,
    genes_and_counts_outliers = TRUE,
    mt_percentage_outliers = TRUE,
    k = 4
  )

  return(obj)
})

names(seurat_list_processed) <- names(seurat_list)

# Merge das amostras
merged <- merge(
  x = seurat_list[[1]],
  y = seurat_list[-1],
  add.cell.ids = names(seurat_list)
)

merged <- SCTransform(
  merged,
  assay = "RNA",
  verbose = FALSE,
  vst.flavor = "v2",
  vars.to.regress = c("^MT-", "^RPS", "^RPL")
)


