library(dplyr)
library(Seurat)
library(patchwork)
library(ggplot2)
library(readr)

# Load the PBMC dataset
pb21_ml <- Read10X(data.dir = "~/Documentos/AnalisesR/Amostras/PB21_ML")

# Initialize the Seurat object with the raw (non-normalized data).
pb21_ml <- CreateSeuratObject(counts = pb21_ml, project = "leprosy_t_cell", min.cells = 3, min.features = 200)

pb21_ml

#Filtragem posterior
pb21_ml[["percent.mt"]] <- PercentageFeatureSet(pb21_ml, pattern = "^MT-") 
VlnPlot(pb21_ml, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

#Normalizacao de Dados
pb21_ml$nUMI <- Matrix::colSums(GetAssayData(pb21_ml, slot = "counts"))
summary(pb21_ml$nUMI)

#Avalia qual valor de escala funcionara melhor para a normalizacao
# Step 1: Plot UMI distribution
vln_plot <- VlnPlot(pb21_ml, features = "nUMI", pt.size = 0.1) + ggtitle("UMI Distribution (Violin Plot)")
hist_plot <- ggplot(data = data.frame(nUMI = pb21_ml$nUMI), aes(x = nUMI)) +
  geom_histogram(biml = 50, fill = "skyblue", color = "black") +
  theme_minimal() + ggtitle("UMI Distribution (Histogram)")

vln_plot | hist_plot  # Patchwork layout

# Step 2: Compare scale factors via PCA

# Function to normalize, scale, run PCA, and extract evaluation metrics
process_and_score_obj <- function(obj, scale_factor) {
  obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = scale_factor)
  obj <- FindVariableFeatures(obj)
  obj <- ScaleData(obj)
  obj <- RunPCA(obj, npcs = 10)
  
  # Compute % variance explained by top 10 PCs
  var_exp <- obj[["pca"]]@stdev^2
  var_exp_pct <- var_exp / sum(var_exp) * 100
  top_var_pct <- sum(var_exp_pct[1:5])  # sum of top 5 PCs
  
  # Number of variable features
  num_var_genes <- length(VariableFeatures(obj))
  
  return(list(
    obj = obj,
    score = list(scale = scale_factor, top_pc_var = top_var_pct, n_hvg = num_var_genes)
  ))
}

# Apply to each scale factor
scales <- c(3000, 5000, 10000)
results <- lapply(scales, function(sf) process_and_score_obj(pb21_ml, scale_factor = sf))
names(results) <- paste0("Scale_", scales)

# Extract Seurat objects and scores
objs <- lapply(results, function(x) x$obj)
scores <- do.call(rbind, lapply(results, function(x) as.data.frame(x$score)))

# Rank by score: prioritize variance, then variable genes
scores$rank_score <- scale(scores$top_pc_var) + scale(scores$n_hvg)
scores <- scores[order(-scores$rank_score), ]

# Print ranked scale factors
print(scores)

library(cluster)   # for silhouette score

# Step 1: Normalize, scale, PCA, clustering
process_score_silhouette <- function(obj, scale_factor) {
  obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = scale_factor)
  obj <- FindVariableFeatures(obj)
  obj <- ScaleData(obj)
  obj <- RunPCA(obj, npcs = 10)
  obj <- FindNeighbors(obj, dims = 1:10)
  obj <- FindClusters(obj, resolution = 0.5)
  
  # Compute silhouette width
  dist_mat <- dist(Embeddings(obj, reduction = "pca")[, 1:10])  # Euclidean distance in PCA space
  sil <- silhouette(as.integer(Idents(obj)), dist_mat)
  sil_avg <- mean(sil[, "sil_width"])
  
  # Variance from PCA
  var_exp <- obj[["pca"]]@stdev^2
  var_pct <- var_exp / sum(var_exp) * 100
  top_pc_var <- sum(var_pct[1:5])
  
  return(list(obj = obj, score = data.frame(
    scale = scale_factor,
    top_pc_var = top_pc_var,
    silhouette_avg = sil_avg,
    n_clusters = length(unique(Idents(obj)))
  )))
}

# Run for multiple scale factors
scales <- c(3000, 5000, 10000)
results <- lapply(scales, function(sf) process_score_silhouette(pb21_ml, scale_factor = sf))

names(results) <- paste0("Scale_", scales)

# Extract scores
objs <- lapply(results, function(x) x$obj)
scores <- do.call(rbind, lapply(results, function(x) x$score))

# Rank by silhouette and top_pc_var
scores$rank_score <- scale(scores$silhouette_avg) + scale(scores$top_pc_var)
scores <- scores[order(-scores$rank_score), ]
print(scores)

FeaturePlot(objs[["Scale_5000"]], features = "FOXP3", reduction = "pca", pt.size = 0.5, ncol = 3)

# Run UMAP on Scale_3000 object
objs[["Scale_3000"]] <- RunUMAP(objs[["Scale_3000"]], dims = 1:10)

DimPlot(objs[["Scale_3000"]], reduction = "umap", label = TRUE, pt.size = 1.5) +
  ggtitle("UMAP Clustering – Scale Factor 3000")

# Run UMAP on Scale_5000 object
objs[["Scale_5000"]] <- RunUMAP(objs[["Scale_5000"]], dims = 1:10)

DimPlot(objs[["Scale_5000"]], reduction = "umap", label = TRUE, pt.size = 1.5) +
  ggtitle("UMAP Clustering – Scale Factor 5000")

# Run UMAP on Scale_10000 object
objs[["Scale_10000"]] <- RunUMAP(objs[["Scale_10000"]], dims = 1:10)

DimPlot(objs[["Scale_10000"]], reduction = "umap", label = TRUE, pt.size = 1.5) +
  ggtitle("UMAP Clustering – Scale Factor 10000")

# Define your marker genes (adjust for your data)
marker_genes <- c("CD3D", "FOXP3", "ITGB1", "IFI27", "NKG7")

# Function to score cluster-specific expression
score_marker_specificity <- function(seurat_obj, markers) {
  DefaultAssay(seurat_obj) <- "RNA"
  Idents(seurat_obj) <- "seurat_clusters"
  
  avg_exp <- AverageExpression(seurat_obj, features = markers, return.seurat = FALSE)$RNA
  
  # Normalize expression by row (gene) to get relative specificity across clusters
  rel_exp <- t(apply(avg_exp, 1, function(x) x / max(x)))
  
  # Compute entropy-like score for each gene: lower = more specific
  specificity_score <- apply(rel_exp, 1, function(x) {
    -sum(x * log2(x + 1e-9))  # small offset to avoid log(0)
  })
  
  # Average across all marker genes
  mean_specificity <- mean(specificity_score)
  
  return(mean_specificity)
}

# Apply to each Seurat object in your list `objs`
marker_scores <- sapply(objs, function(obj) {
  score_marker_specificity(obj, marker_genes)
})

# Rank: lower specificity score = more cluster-specific expression (better)
marker_scores_df <- data.frame(
  scale = names(marker_scores),
  specificity_score = marker_scores
) %>%
  arrange(specificity_score)

print(marker_scores_df)

pb21_ml <- NormalizeData(pb21_ml, normalization.method = "LogNormalize", scale.factor = 10000)

pb21_ml <- FindVariableFeatures(pb21_ml, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pb21_ml), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pb21_ml)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

pb21_ml_mvp <- FindVariableFeatures(pb21_ml, selection.method = "mvp", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pb21_ml_mvp), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pb21_ml_mvp)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

all.genes <- rownames(pb21_ml)
pb21_ml <- ScaleData(pb21_ml, features = all.genes)

pb21_ml <- RunPCA(pb21_ml, features = VariableFeatures(object = pb21_ml))

# Examine and visualize PCA results a few different ways
print(pb21_ml[["pca"]], dims = 1:5, nfeatures = 5)

DimPlot(pb21_ml, reduction = "pca") + NoLegend()

ElbowPlot(pb21_ml)

pb21_ml <- FindNeighbors(pb21_ml, dims = 1:10)

# Run clustering at multiple resolutioml
resolutioml <- c(0.2, 0.4, 0.6, 0.8, 1.0)

# Store cluster IDs in the metadata for each resolution
pb21_ml <- FindNeighbors(pb21_ml, dims = 1:10)  # Run once before clustering

for (res in resolutioml) {
  pb21_ml <- FindClusters(pb21_ml, resolution = res, verbose = FALSE)
  colname <- paste0("RNA_snn_res.", res)
  newname <- paste0("clusters_res_", gsub("\\.", "_", res))
  pb21_ml[[newname]] <- pb21_ml[[colname]]
}

sapply(paste0("clusters_res_", gsub("\\.", "_", resolutioml)), function(x) {
  length(unique(pb21_ml[[x]][,1]))
})

library(patchwork)

plots <- lapply(resolutioml, function(res) {
  DimPlot(pb21_ml, group.by = paste0("clusters_res_", gsub("\\.", "_", res)),
          label = TRUE) + ggtitle(paste("Resolution:", res))
})

wrap_plots(plots)

# Define resolutioml to test
resolutioml <- c(0.2, 0.4, 0.6, 0.8, 1.0)

# Make sure neighbors are computed
pb21_ml <- FindNeighbors(pb21_ml, dims = 1:10, verbose = FALSE)

# Function to compute silhouette score per resolution
compute_silhouette <- function(obj, cluster_col) {
  pca_data <- Embeddings(obj, reduction = "pca")[, 1:10]
  cluster_ids <- as.integer(obj[[cluster_col]][,1])
  dists <- dist(pca_data)
  sil <- silhouette(cluster_ids, dists)
  mean(sil[, 3])  # avg silhouette width
}

# Run clustering and compute scores
silhouette_summary <- data.frame()
for (res in resolutioml) {
  message("Processing resolution: ", res)
  
  # Run clustering
  obj <- FindClusters(pb21_ml, resolution = res, verbose = FALSE)
  colname <- paste0("RNA_snn_res.", res)
  cluster_col <- paste0("clusters_res_", gsub("\\.", "_", res))
  obj[[cluster_col]] <- obj[[colname]]
  
  # Compute silhouette
  sil_score <- compute_silhouette(obj, cluster_col)
  
  # Store results
  silhouette_summary <- rbind(silhouette_summary, data.frame(
    resolution = res,
    num_clusters = length(unique(obj[[cluster_col]][,1])),
    silhouette_score = sil_score
  ))
}

# Show results sorted by silhouette score
silhouette_summary <- silhouette_summary %>% arrange(desc(silhouette_score))
print(silhouette_summary)

ggplot(silhouette_summary, aes(x = factor(resolution), y = silhouette_score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(silhouette_score, 3)), vjust = -0.5) +
  theme_minimal() +
  labs(title = "Average Silhouette Score per Resolution",
       x = "Resolution", y = "Silhouette Score")

pb21_ml <- FindClusters(pb21_ml, resolution = 0.6)

pb21_ml <- RunUMAP(pb21_ml, dims = 1:10)

DimPlot(pb21_ml, reduction = "umap")

# find markers for every cluster compared to all remaining cells, report only the positive
# ones
pb21_ml.markers <- FindAllMarkers(pb21_ml)

pb21_ml.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)

cluster_tables <- pb21_ml.markers %>%
  group_by(cluster) %>%
  group_split()

# Save each cluster table to a CSV file
output_dir <- "output"
if (!dir.exists(output_dir)) dir.create(output_dir)

# Step 5: Save each cluster table to a CSV file
for (tbl in cluster_tables) {
  cluster_id <- unique(tbl$cluster)
  file_name <- paste0(output_dir, "/cluster_", cluster_id, "_markers.csv")
  write_csv(tbl, file_name)
}

VlnPlot(pb21_ml, features = c("MS4A1", "CD79A"), slot = "counts", log = TRUE)

pb21_ml.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1) %>%
  slice_head(n = 5) %>%
  ungroup() -> top10

library(viridis)
DoHeatmap(pb21_ml, features = top10$gene) +
  scale_fill_viridis(option = "D") +
  NoLegend()

DimPlot(pb21_ml, reduction = "umap", label = TRUE)

novos_nomes_clusters <- c("CD4 effector memory",           # Cluster 0
                          "CD4 Central memory",            # Cluster 1
                          "Naive T CD4",                   # Cluster 2
                          "rTreg",                         # Cluster 3
                          "Proliferation T Cells",         # Cluster 4
                          "Exhausted/Th1",            # Cluster 5
                          "Th17",                          # Cluster 6
                          "eTreg",                         # Cluster 7
                          "ISG T cells",         # Cluster 8
                          "NK cells",                      # Cluster 9
                          "B cells"                        # Cluster 10
                          
)                      
names(novos_nomes_clusters) <- levels(pb21_ml)

pb21_ml <- RenameIdents(pb21_ml, novos_nomes_clusters)

DimPlot(pb21_ml, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()