import scanpy as sc
import os

adata = sc.read(r'C:\Users\IOC\OneDrive - FIOCRUZ\Área de Trabalho\analysis\data\adata_harmony.h5ad')
adata = adata.raw.to_adata()


from decontx import decontx
# Rodar decontX
result = decontx(
    adata=adata,
    batch_key='library_id',
    seed=42,
    verbose=True,
    cluster_key="leiden_0_5"
)

adata.obs["decontX_contamination"] = result["contamination"]
adata.layers["decontXcounts"] = sp.csr_matrix(result["decontXcounts"])
adata.X = adata.layers["decontXcounts"].copy()

print(f"Células antes: {adata.n_obs}")
adata = adata[adata.obs["decontX_contamination"] < 0.3].copy()
print(f"Células depois: {adata.n_obs}")

sc.pl.umap(
    adata,
    color="decontX_contamination",
    cmap="viridis",
    vmax=0.5  # opcional: corta outliers
)

adata.write_h5ad(r"C:\Users\IOC\OneDrive - FIOCRUZ\Área de Trabalho\analysis\data\adata_harmony_clean.h5ad")