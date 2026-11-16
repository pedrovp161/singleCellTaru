"""Objeto final único: o h5ad do P05 com a integração (P07), clusters (P08),
anotação (P09), programas de NMF (P10) e os resultados dos benchmarks (P07, P12).

O h5ad do P05 não é alterado; o objeto montado vai para
``dataReseq/analises_previas/P13_objeto_final/reseq_final.h5ad`` (versionado por ``io.save_data``).

Uso: python raw/analises_previas/P13_objeto_final.py
"""
import sys
sys.path.insert(0, "/mnt/SATA/singleCellTaru/src/Reseq")

import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd

from functions import config, io, pipeline as pl

D, T = config.DATA_OUT, config.TABLES

a = io.load_data("P05_integracao", "reseq_raw_integrado", "h5ad")
key = a.obs["cell_key"].astype(str).to_numpy()


def junta(df, cols, prefixo=""):
    df = df.set_index("cell_key")
    assert df.index.is_unique and set(df.index) == set(key), "chaves não fecham"
    for c in cols:
        a.obs[prefixo + c] = df.loc[key, c].to_numpy()


# ── integração escolhida (P07: ComBat escalado) ─────────────────────────────
integ = io.load_data("P07_benchmark_integracao", "integracao_escolhida", "h5ad")
assert (integ.obs["cell_key"].astype(str).to_numpy() == key).all()
a.obsm["X_umap_harmony"] = a.obsm.pop("X_umap")          # UMAP do P05 (Harmony 15 dims)
a.obsm["X_emb_combat"] = integ.obsm["X_emb"]
a.obsm["X_umap"] = integ.obsm["X_umap"]                  # UMAP padrão = ComBat
a.obsp["connectivities"] = integ.obsp["connectivities"]
a.obsp["distances"] = integ.obsp["distances"]
a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                      "params": {"method": "umap", "metric": "euclidean", "n_neighbors": 15,
                                 "use_rep": "X_emb_combat", "random_state": 0}}
a.uns["integracao"] = dict(integ.uns["integracao"])
a.obs = a.obs.rename(columns={"leiden": "leiden_harmony_P05"})

# ── clusters (P08), anotação (P09), programas de NMF (P10) ──────────────────
cl = io.load_data("P08_clusterizacao", "clusters")
junta(cl, [c for c in cl.columns if c not in ("cell_key", "library_id")])
an = io.load_data("analises_previas/P09_anotacao", "anotacao")
junta(an, [c for c in an.columns if c not in ("cell_key", "library_id")])
pr = io.load_data("analises_previas/P10_nmf_programas", "programas")
H14 = [f"H14_P{k}" for k in range(14)]
Hpj = [c for c in pr.columns if c.startswith("Hproj_")]
junta(pr, [c for c in pr.columns if c not in ("cell_key", "library_id", *H14, *Hpj)])
prk = pr.set_index("cell_key").loc[key]
a.obsm["X_nmf14"] = prk[H14].to_numpy(np.float32)
a.obsm["X_nmf_projecao"] = prk[Hpj].to_numpy(np.float32)
for c in a.obs.columns:
    if a.obs[c].dtype == object:
        a.obs[c] = a.obs[c].astype("category")

# ── metadados dos processamentos ────────────────────────────────────────────
hvg = pl.select_hvg(a)
a.uns["hvg_spans_loess"] = {k: float(v) for k, v in hvg.attrs["spans"].items()}
a.uns["nmf"] = {
    "programas": pd.read_csv(T / "analises_previas/P10_nmf_programas" / "programas.csv"),
    "genes_por_programa": pd.read_csv(T / "analises_previas/P10_nmf_programas" / "genes_por_programa.csv"),
    "colunas_X_nmf14": H14, "colunas_X_nmf_projecao": Hpj,
}
bench = io.load_data("P07_benchmark_integracao", "benchmark")
a.uns["benchmark_integracao"] = {
    "P07_azimuth": bench.astype({c: str for c in bench.columns if bench[c].dtype == object}),
    "P12_posicoes": pd.read_csv(T / "analises_previas/P12_benchmark_integracao_rotulos" / "posicoes.csv"),
    "P12_escolhas": pd.read_csv(T / "analises_previas/P12_benchmark_integracao_rotulos" / "escolhas.csv"),
}
a.uns["soupx_rho"] = a.uns["soupx_rho"]
a.uns["guia"] = {
    "X": "lognorm (10 mil por célula, log1p) das contagens SoupX",
    "layers": "counts = SoupX (inteiras); counts_raw = Cell Ranger sem correção",
    "obsm": "X_pca e X_emb_harmony/X_umap_harmony = P05 (Harmony 15 dims); "
            "X_emb_combat, X_umap e grafo = integração escolhida no P07 (ComBat escalado); "
            "X_nmf14 = atividade dos 14 programas de NMF (P10); X_nmf_projecao = projeção nos programas publicados",
    "obs": "QC e doublets (P01–P04); leiden_harmony_P05; leiden_* e cluster_tecnico (P08, sobre o ComBat); "
           "anotacao, grupo, atlas_* e azimuth_* (P09); nmf14_*, nmf9_*, proj_* (P10)",
    "filtro_CD4": "obs['grupo'] fora de {'contaminante', 'baixa qualidade'}",
    "fonte": "raw/analises_previas/P13_objeto_final.py",
}
for k in ("benchmark_integracao", "nmf"):
    for kk, v in a.uns[k].items():
        if isinstance(v, pd.DataFrame):
            v.columns = v.columns.astype(str)
            a.uns[k][kk] = v.astype(str)

print(a)
p = io.save_data(a, "analises_previas/P13_objeto_final", "reseq_final",
                 description="Objeto final: P05 + integração ComBat (P07), clusters (P08), anotação (P09), "
                             "NMF (P10) e benchmarks (P07, P12)")
print(p)
