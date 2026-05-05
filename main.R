library(Seurat)

samples <- list.dirs("data", full.names = FALSE, recursive = FALSE)

seurat_list <- list()
for (sample in samples) {
  path <- file.path("data", sample)
  
  counts <- Read10X(data.dir = path)
  
  obj <- CreateSeuratObject(counts = counts, project = sample)
  
  seurat_list[[sample]] <- obj
  cat(sprintf("✓ %s: %d células, %d genes\n", sample, ncol(obj), nrow(obj)))
}

cat(sprintf("\nTotal de amostras carregadas: %d\n", length(seurat_list)))


