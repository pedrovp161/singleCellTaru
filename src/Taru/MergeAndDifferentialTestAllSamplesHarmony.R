library(dplyr)
library(Seurat)
library(patchwork)
library(ggplot2)
library(readr)
library(MAST)

# --- CONFIGURAÇÃO DE MEMÓRIA ---
# Aumenta o limite de exportação do pacote future para 8GB
options(future.globals.maxSize = 8000 * 1024^2)

# Load the PBMC dataset


cth92_ml <- Read10X(data.dir = "Amostras/CTH92_ML/")

cth92_ns <- Read10X(data.dir = "Amostras/CTH92_NS/")

cth94_ml <- Read10X(data.dir = "Amostras/CTH94_ML/")

cth94_ns <- Read10X(data.dir = "Amostras/CTH94_NS/")

cth97_ml <- Read10X(data.dir = "Amostras/CTH97_ML/")

cth97_ns <- Read10X(data.dir = "Amostras/CTH97_NS/")

mb34_ml <- Read10X(data.dir = "Amostras/MB34_ML/")

mb34_ns <- Read10X(data.dir = "Amostras/MB34_NS/")

pb20_ml <- Read10X(data.dir = "Amostras/PB20_ML/")

pb20_ns <- Read10X(data.dir = "Amostras/PB20_NS/")

pb21_ml <- Read10X(data.dir = "Amostras/PB21_ML/")

pb21_ns <- Read10X(data.dir = "Amostras/PB21_NS/")

# Initialize the Seurat object with the raw (non-normalized data).
cth92_ml <- CreateSeuratObject(counts = cth92_ml, project = "cth92ml", min.cells = 3, min.features = 200)

cth92_ns <- CreateSeuratObject(counts = cth92_ns, project = "cth92ns", min.cells = 3, min.features = 200)

cth94_ml <- CreateSeuratObject(counts = cth94_ml, project = "cth94ml", min.cells = 3, min.features = 200)

cth94_ns <- CreateSeuratObject(counts = cth94_ns, project = "cth94ns", min.cells = 3, min.features = 200)

cth97_ml <- CreateSeuratObject(counts = cth97_ml, project = "cth97ml", min.cells = 3, min.features = 200)

cth97_ns <- CreateSeuratObject(counts = cth97_ns, project = "cth97ns", min.cells = 3, min.features = 200)

mb34_ml <- CreateSeuratObject(counts = mb34_ml, project = "mb34ml", min.cells = 3, min.features = 200)

mb34_ns <- CreateSeuratObject(counts = mb34_ns, project = "mb34ns", min.cells = 3, min.features = 200)

pb20_ml <- CreateSeuratObject(counts = pb20_ml, project = "pb20ml", min.cells = 3, min.features = 200)

pb20_ns <- CreateSeuratObject(counts = pb20_ns, project = "pb20ns", min.cells = 3, min.features = 200)

pb21_ml <- CreateSeuratObject(counts = pb21_ml, project = "pb21ml", min.cells = 3, min.features = 200)

pb21_ns <- CreateSeuratObject(counts = pb21_ns, project = "pb21ns", min.cells = 3, min.features = 200)

# Add metadata if needed
cth92_ml$sample <- "cth92ml"
cth92_ns$sample <- "cth92ns"
cth94_ml$sample <- "cth94ml"
cth94_ns$sample <- "cth94ns"
cth97_ml$sample <- "cth97ml"
cth97_ns$sample <- "cth97ns"
mb34_ml$sample <- "mb34ml"
mb34_ns$sample <- "mb34ns"
pb20_ml$sample <- "pb20ml"
pb20_ns$sample <- "pb20ns"
pb21_ml$sample <- "pb21ml"
pb21_ns$sample <- "pb21ns"

# 3. Merge (combine all raw counts into one object)
merged <- merge(cth94_ml, y = c(cth94_ns, cth92_ml, cth92_ns, cth97_ml, cth97_ns, mb34_ml, mb34_ns, pb20_ml, pb20_ns, pb21_ml, pb21_ns), add.cell.ids = c("cth94ml", "cth94ns", "cth92_ml", "cth92_ns", "cth97_ml", "cth97_ns", "mb34ml", "mb34ns", "pb20_ml", "pb20_ns", "pb21ml", "pb21ns"))

# 1. Dividir as camadas (Prepara o objeto v5 para o SCTransform)
#merged[["RNA"]] <- split(merged[["RNA"]], f = merged$sample)

# 2. Normalização Moderna (SCTransform)
# Note: Isso substitui NormalizeData, FindVariableFeatures e ScaleData
message("Rodando SCTransform...")
merged <- SCTransform(merged, assay = "RNA", verbose = FALSE, vst.flavor = "v2")

# 3. PCA (Necessário para o Harmony)
message("Rodando PCA...")
merged <- RunPCA(merged, assay = "SCT", verbose = FALSE)

# 4. Integração com Harmony (Correção de Lote)
message("Rodando Harmony...")
library(harmony)

merged <- RunHarmony(merged, 
                     group.by.vars = "sample", 
                     reduction = "pca", 
                     assay.use = "SCT", 
                     reduction.save = "harmony")

# 5. UMAP e Clusters baseados na integração
message("Gerando UMAP e Clusters...")
merged <- RunUMAP(merged, reduction = "harmony", dims = 1:20)
merged <- FindNeighbors(merged, reduction = "harmony", dims = 1:20)
merged <- FindClusters(merged, resolution = 0.5)

# clusterTree
# analise de vias

# ###############################################################
# ### INÍCIO DA ETAPA: NOMEAÇÃO DOS CLUSTERS ###
# ###############################################################

# 1. Defina os nomes baseando-se no que você viu no top_10_markers
# IMPORTANTE: A ordem aqui deve seguir a ordem dos níveis: 0, 1, 2, 3...
new_cluster_ids <- c(
  "0" = "CD4+ SN",
  "1" = "CD4+ ISG",
  "2" = "CD4+ D2",
  "3" = "CD4+ D1",
  "4" = "CD4+ S N/CM",
  "5" = "CD4+ E",
  "6" = "CD4+ REG",
  "7" = "CD4+ D TOX",
  "8" = "N/A (QC fail)",
  "9" = "CD4+ 17",
  "10" = "CD4+ D 1 E",
  "11" = "CD4+ FH",
  "12" = "Monócitos",
  "13" = "Células B",
  "14" = "Plasmócitos"
)

# 2. Aplique a renomeação ao objeto principal
# Isso mudará o 'Ident' ativo de números para nomes
merged <- RenameIdents(merged, new_cluster_ids)

# 3. Salve em uma nova coluna de metadados para segurança
merged$cell_type <- Idents(merged)

DimPlot(merged, reduction = "harmony", split.by = "sample", label = FALSE) + ggtitle("UMAP Exploratório - 12 Amostras")

# ###############################################################
# ### FIM DA ETAPA ###
# ###############################################################

# ###############################################################
# ### INÍCIO DA ETAPA 1: CORREÇÃO FINAL ###
# ###############################################################

# PASSO 1: PREPARAR O ASSAY 'RNA'
DefaultAssay(merged) <- "RNA"
merged[["RNA"]] <- JoinLayers(merged[["RNA"]])
merged <- NormalizeData(merged, verbose = FALSE)

# PASSO 2: GARANTIR A IDENTIDADE NUMÉRICA
# Isso é crucial agora que você comentou os nomes. Garante que a tabela 
# saia com cluster 0, 1, 2, etc.
Idents(merged) <- "seurat_clusters" 

# PASSO 3: EXECUTAR O FINDALLMARKERS
message("Procurando marcadores globais para anotação...")
all_cluster_markers <- FindAllMarkers(merged, 
                                      only.pos = TRUE, 
                                      min.pct = 0.1, 
                                      logfc.threshold = 0.25, 
                                      test.use = "wilcox", 
                                      verbose = TRUE)

# PASSO 4: SALVAR PARA ANÁLISE MANUAL
all_cluster_markers$gene <- rownames(all_cluster_markers)

top_10_markers <- all_cluster_markers %>%
  group_by(cluster) %>%
  top_n(n = 10, wt = avg_log2FC)

write.csv(all_cluster_markers, file = "marcadores_globais_todos_12amostras.csv", row.names = FALSE)
write.csv(top_10_markers, file = "marcadores_globais_top10_12amostras.csv", row.names = FALSE)

message("Tabelas geradas! Interrompa o script aqui para analisar o CSV.")

# ###############################################################
# ### FIM DA ETAPA 1 ###
# ###############################################################

# Get smallest group size
min_cells <- min(table(merged$sample))

# Subsample each group
cells_to_keep <- merged@meta.data %>%
  mutate(Cell = rownames(.)) %>%
  group_by(sample) %>%
  sample_n(min(table(merged$sample))) %>%
  pull(Cell)

balanced <- subset(merged, cells = cells_to_keep)

balanced <- JoinLayers(balanced)

Idents(balanced) <- "sample"

markers <- FindMarkers(balanced, ident.1 = "mb34ml", ident.2 = "mb34ns")

# Add gene names
markers$gene <- rownames(markers)

# Categorize based on significance and log2 fold change
markers$significance <- case_when(
  markers$p_val_adj < 0.05 & markers$avg_log2FC > 0.25 ~ "Upregulated",
  markers$p_val_adj < 0.05 & markers$avg_log2FC < -0.25 ~ "Downregulated",
  TRUE ~ "Not Significant"
)

# Select top 10 upregulated and downregulated genes for labeling
top_up <- markers %>%
  filter(significance == "Upregulated") %>%
  arrange(p_val_adj) %>%
  head(10)

top_down <- markers %>%
  filter(significance == "Downregulated") %>%
  arrange(p_val_adj) %>%
  head(10)

top_genes <- bind_rows(top_up, top_down)

# Volcano plot with gene labels
ggplot(markers, aes(x = avg_log2FC, y = -log10(p_val_adj), color = significance)) +
  geom_point(alpha = 0.6, size = 1) +
  geom_text(data = top_genes, aes(label = gene), size = 3, vjust = -0.5, check_overlap = TRUE, color = "black") +
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
  theme_minimal() +
  labs(
    title = "Volcano Plot: Differentially Expressed Genes",
    x = "Log2 Fold Change",
    y = "-log10(Adjusted P-value)",
    color = "Regulation"
  ) +
  theme(legend.position = "right")

top_genes <- markers %>%
  dplyr::filter(p_val_adj < 0.05) %>%
  top_n(20, wt = abs(avg_log2FC)) %>%
  pull(gene)

balanced <- ScaleData(balanced, features = top_genes)

DoHeatmap(balanced, features = top_genes, group.by = "sample") +
  ggtitle("Top DE Genes Heatmap")

DotPlot(balanced, features = top_genes, group.by = "sample") +
  RotatedAxis() +
  ggtitle("Dot Plot of Top DE Genes")

# Example using MAST
Idents(merged) <- "sample"

DefaultAssay(merged) <- "RNA"

# Join expression layers in the "RNA" assay
merged[["RNA"]] <- JoinLayers(merged[["RNA"]])

# Compare group A vs B using MAST
markers <- FindMarkers(merged, ident.1 = "mb34ml", ident.2 = "mb34ns", test.use = "MAST")

# Add gene names and classification
markers$gene <- rownames(markers)
markers$significance <- case_when(
  markers$p_val_adj < 0.05 & markers$avg_log2FC > 0.25 ~ "Upregulated",
  markers$p_val_adj < 0.05 & markers$avg_log2FC < -0.25 ~ "Downregulated",
  TRUE ~ "Not Significant"
)

# Select top genes for labeling
top_up <- markers %>% filter(significance == "Upregulated") %>% arrange(p_val_adj) %>% head(10)
top_down <- markers %>% filter(significance == "Downregulated") %>% arrange(p_val_adj) %>% head(10)
top_genes <- bind_rows(top_up, top_down)

# Plot
ggplot(markers, aes(x = avg_log2FC, y = -log10(p_val_adj), color = significance)) +
  geom_point(alpha = 0.6, size = 1) +
  geom_text(data = top_genes, aes(label = gene), vjust = -1, size = 3, check_overlap = TRUE) +
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
  theme_minimal() +
  labs(title = "Volcano Plot: Differential Expression (MAST)",
       x = "Log2 Fold Change",
       y = "-log10(Adjusted P-value)",
       color = "Significance")

# Top 20 DE genes
top20 <- markers %>%
  filter(p_val_adj < 0.05) %>%
  arrange(-abs(avg_log2FC)) %>%
  head(20) %>%
  pull(gene)

merged <- ScaleData(merged, features = top20)

DoHeatmap(merged, features = top20, group.by = "sample") +
  ggtitle("Top 20 DE Genes by MAST")

DotPlot(merged, features = top20, group.by = "sample") +
  RotatedAxis() +
  ggtitle("DotPlot of Top DE Genes")

merged[["RNA"]] <- JoinLayers(merged[["RNA"]])

Idents(merged) <- "seurat_clusters"  # Ensure clustering identity is set

# Then repeat your DE loop
de_per_cluster <- list()
clusters <- levels(merged)

for (cl in clusters) {
  message("Analyzing cluster: ", cl)
  
  clust_subset <- subset(merged, idents = cl)
  Idents(clust_subset) <- "sample"
  
  if (length(unique(clust_subset$sample)) < 2) {
    message("  Skipping cluster ", cl, ": fewer than 2 conditions present")
    next
  }
  
  de_markers <- tryCatch({
    FindMarkers(clust_subset, ident.1 = "cth94ml", ident.2 = "cth94ns", test.use = "MAST")
  }, error = function(e) {
    message("  Error in cluster ", cl, ": ", e$message)
    return(NULL)
  })
  
  if (!is.null(de_markers) && nrow(de_markers) > 0) {
    de_markers$gene <- rownames(de_markers)
    de_markers$cluster <- cl
    de_per_cluster[[cl]] <- de_markers
    message("  --> DE genes found: ", nrow(de_markers))
  } else {
    message("  --> No DE genes found in cluster ", cl)
  }
}

# Combine DE results from each cluster into one data frame
all_cluster_de <- bind_rows(de_per_cluster, .id = "cluster")

# Make sure gene and cluster are characters
all_cluster_de$gene <- as.character(all_cluster_de$gene)
all_cluster_de$cluster <- as.character(all_cluster_de$cluster)

cl <- "6"  # Replace with any valid cluster in all_cluster_de$cluster

cluster_df <- all_cluster_de %>%
  filter(cluster == cl) %>%
  mutate(significance = case_when(
    p_val_adj < 0.05 & avg_log2FC > 0.25 ~ "Upregulated",
    p_val_adj < 0.05 & avg_log2FC < -0.25 ~ "Downregulated",
    TRUE ~ "Not Significant"
  ))

write.csv(cluster_df,"./DEG_cth94ml_vs_cth94ns.csv")

top_genes <- cluster_df %>%
  filter(significance != "Not Significant") %>%
  arrange(p_val_adj) %>%
  slice_head(n = 10)

ggplot(cluster_df, aes(x = avg_log2FC, y = -log10(p_val_adj), color = significance)) +
  geom_point(alpha = 0.6, size = 1) +
  geom_text(data = top_genes, aes(label = gene), size = 3, vjust = -0.5, check_overlap = TRUE) +
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
  theme_minimal() +
  labs(
    title = paste("Volcano Plot - Cluster", cl),
    x = "Log2 Fold Change",
    y = "-log10(Adjusted P-value)",
    color = "Significance"
  )

library(ggplot2)
library(dplyr)

# Create a list to store plots
volcano_list <- list()

for (cl in unique(all_cluster_de$cluster)) {
  
  cluster_df <- all_cluster_de %>%
    filter(cluster == cl) %>%
    mutate(significance = case_when(
      p_val_adj < 0.05 & avg_log2FC > 0.25 ~ "Upregulated",
      p_val_adj < 0.05 & avg_log2FC < -0.25 ~ "Downregulated",
      TRUE ~ "Not Significant"
    ))
  
  top_genes <- cluster_df %>%
    filter(significance != "Not Significant") %>%
    arrange(p_val_adj) %>%
    slice_head(n = 10)
  
  p <- ggplot(cluster_df, aes(x = avg_log2FC, y = -log10(p_val_adj), color = significance)) +
    geom_point(alpha = 0.6, size = 1) +
    geom_text(data = top_genes, aes(label = gene), size = 3, vjust = -0.5, check_overlap = TRUE) +
    scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
    theme_minimal() +
    labs(
      title = paste("Volcano Plot - Cluster", cl),
      x = "Log2 Fold Change",
      y = "-log10(Adjusted P-value)",
      color = "Significance"
    )
  
  # Store the plot
  volcano_list[[cl]] <- p
}

for (cl in names(volcano_list)) {
  cat("### Volcano Plot - Cluster", cl, "\n")
  print(volcano_list[[cl]])
  cat("\n\n")
}

# Annotate DE status
plot_df <- all_cluster_de %>%
  mutate(significance = case_when(
    p_val_adj < 0.05 & avg_log2FC > 0.25 ~ "Upregulated",
    p_val_adj < 0.05 & avg_log2FC < -0.25 ~ "Downregulated",
    TRUE ~ "Not Significant"
  ))

# Faceted volcano plot
ggplot(plot_df, aes(x = avg_log2FC, y = -log10(p_val_adj), color = significance)) +
  geom_point(alpha = 0.5, size = 0.8) +
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
  facet_wrap(~ cluster, scales = "free") +
  theme_minimal() +
  labs(
    title = "Faceted Volcano Plots by Cluster",
    x = "Log2 Fold Change",
    y = "-log10(Adjusted P-value)",
    color = "Significance"
  )

top_genes <- all_cluster_de %>%
  filter(p_val_adj < 0.05) %>%
  group_by(cluster) %>%
  slice_max(order_by = abs(avg_log2FC), n = 5) %>%
  pull(gene) %>%
  unique()

DotPlot(merged, features = top_genes, group.by = "seurat_clusters") +
  RotatedAxis() +
  labs(title = "Top DE Genes per Cluster")

library(DT)

datatable(all_cluster_de, options = list(pageLength = 10), filter = "top")
