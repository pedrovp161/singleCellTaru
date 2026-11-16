"""Subconjuntos das CD4 limpas (P09): reprocessamento, varredura de resolução e nomes por referência.

Usado pelos notebooks de subset (P10 Treg, P11 naive e memória). A receita é a mesma do P09 (ComBat
escalado escolhido no P07), refeita só nas células do subconjunto:

    HVG seurat_v3 por biblioteca → lognorm → scale(10) → ComBat(library_id) → PCA → 15 vizinhos → UMAP

Bibliotecas com poucas células do subconjunto (``min_lote``) ficam fora da escolha das HVG — o loess
da média–variância não se ajusta com poucas células —, mas entram no ComBat e em todo o resto.
"""

from __future__ import annotations

import itertools

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from . import config, io
from . import pipeline as pl

COLS_PREV = ["anotacao", "grupo", "atlas_L1", "atlas_L2"]
COLS_NMF = ["nmf14_rotulo", "proj_rotulo"]


def carregar_cd4() -> ad.AnnData:
    """Objeto principal restrito às CD4 limpas do P09, com a anotação, o atlas e a NMF da 1ª rodada."""
    a = ad.read_h5ad(io.data_path("P05_integracao", "reseq_raw_integrado", "h5ad"))
    keys = a.obs["cell_key"].astype(str).to_numpy()
    filtro = io.load_data("P09_cd4_limpo", "filtro_celulas").set_index("cell_key").reindex(keys)
    a = a[(filtro["motivo"] == "mantida").to_numpy()].copy()
    keys = a.obs["cell_key"].astype(str).to_numpy()
    prev = io.load_data("analises_previas/P09_anotacao", "anotacao").set_index("cell_key").reindex(keys)
    nmf = io.load_data("analises_previas/P10_nmf_programas", "programas").set_index("cell_key").reindex(keys)
    for c in COLS_PREV:
        a.obs[c] = prev[c].astype(object).to_numpy()
    for c in COLS_NMF:
        a.obs[c] = nmf[c].astype(object).to_numpy()
    for c in [c for c in nmf.columns if c.startswith(("H14_", "Hproj_"))]:
        a.obs[c] = nmf[c].to_numpy(float)
    a.var["gene_ids"] = a.var_names.astype(str)
    a.var_names = pd.Index(ad.utils.make_index_unique(pd.Index(a.var["gene_symbol"].astype(str))))
    return a


def reprocessar(sub: ad.AnnData, n_hvg: int = config.N_HVG, min_lote: int = 20, n_pcs: int = 30,
                seed: int = 0) -> tuple[ad.AnnData, dict]:
    """Receita do P09 no subconjunto. Devolve o AnnData com X = lognorm (todos os genes com ≥ 3 células),
    ``layers['counts']``, ``obsm['X_emb']``, grafo e UMAP; e um dicionário com as HVG e as notas."""
    n_cel = np.asarray((sp.csr_matrix(sub.layers["counts"]) > 0).sum(axis=0)).ravel()
    b = sub[:, n_cel >= config.GLOBAL_MIN_CELLS].copy()
    b.obs["library_id"] = pd.Categorical(b.obs["library_id"].astype(str))
    tam = b.obs["library_id"].value_counts()
    grandes = tam.index[tam >= min_lote].astype(str)
    bh = b[b.obs["library_id"].astype(str).isin(grandes)].copy()
    bh.obs["library_id"] = pd.Categorical(bh.obs["library_id"].astype(str))
    hvg = pl.select_hvg(bh, n_top=n_hvg)
    HVG = hvg.index[hvg["highly_variable"]]
    b.X = pl.lognorm(b.layers["counts"])
    pre = ad.AnnData(X=b[:, HVG].X.toarray(), obs=b.obs[["library_id"]].copy(), var=pd.DataFrame(index=HVG))
    sc.pp.scale(pre, max_value=10)
    sc.pp.combat(pre, key="library_id")
    sc.tl.pca(pre, n_comps=n_pcs, random_state=config.RANDOM_STATE)
    b.obsm["X_emb"] = np.asarray(pre.obsm["X_pca"], np.float32)
    sc.pp.neighbors(b, n_neighbors=15, use_rep="X_emb", random_state=seed)
    sc.tl.umap(b, random_state=seed)
    info = {"hvg": list(map(str, HVG)), "bibliotecas_fora_das_hvg": {str(k): int(v) for k, v in tam[tam < min_lote].items()},
            "variancia_pca": float(pre.uns["pca"]["variance_ratio"].sum()), "n_pcs": n_pcs}
    return b, info


def varrer(b: ad.AnnData, res, seeds=(0, 1, 2, 3, 4), referencias: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leiden em cada resolução e semente. Devolve (rótulos, tabela de estabilidade).

    Estabilidade = ARI médio e mínimo entre os pares de sementes; Jaccard ≥ 0,75 por cluster.
    ``referencias`` = {nome: rótulos por célula} acrescenta o ARI da semente 0 com cada referência.
    """
    sw = pl.leiden_sweep(b, res, seeds=seeds)
    linhas = []
    for r in res:
        cols = [f"leiden_r{r:g}_s{s}" for s in seeds]
        aris = [pl.clustering_agreement(sw[x], sw[y])["ARI"] for x, y in itertools.combinations(cols, 2)]
        j = pl.cluster_stability(sw[cols[0]], [sw[c] for c in cols[1:]])
        tam = sw[cols[0]].value_counts()
        lin = {"resolução": r, "clusters": int(sw[cols[0]].nunique()),
               "clusters por semente": " / ".join(str(sw[c].nunique()) for c in cols),
               "ARI entre sementes (média)": float(np.mean(aris)), "ARI entre sementes (mínimo)": float(np.min(aris)),
               "células em clusters estáveis": float(tam.reindex(j.index[j >= 0.75]).sum() / len(sw)),
               "menor cluster": int(tam.min())}
        for nome, ref in (referencias or {}).items():
            ok = pd.Series(ref).notna().to_numpy()
            lin[f"ARI com {nome}"] = pl.clustering_agreement(sw.loc[ok, cols[0]], pd.Series(ref)[ok].astype(str))["ARI"]
        linhas.append(lin)
    return sw, pd.DataFrame(linhas).set_index("resolução")


def composicao(clusters, ref) -> pd.DataFrame:
    """Fração de cada rótulo de referência dentro de cada cluster (linhas somam 1; sem rótulo fica fora)."""
    s = pd.Series(np.asarray(ref, dtype=object))
    ok = s.notna().to_numpy()
    ct = pd.crosstab(np.asarray(clusters).astype(str)[ok], s[ok].astype(str))
    ct = ct.loc[sorted(ct.index, key=lambda x: int(x) if str(x).isdigit() else str(x))]
    return ct.div(ct.sum(axis=1), axis=0)


def escores(b: ad.AnnData, paineis: dict[str, list[str]], grupos, seed: int = 0) -> pd.DataFrame:
    """Média por grupo do ``sc.tl.score_genes`` de cada painel (lognorm em ``b.X``)."""
    out = {}
    for nome, genes in paineis.items():
        gs = [g for g in genes if g in b.var_names]
        sc.tl.score_genes(b, gs, score_name="_tmp", random_state=seed)
        out[nome] = pd.Series(b.obs["_tmp"].to_numpy()).groupby(np.asarray(grupos).astype(str)).mean()
    b.obs.drop(columns="_tmp", inplace=True)
    return pd.DataFrame(out)
