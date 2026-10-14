"""
Réplica do processamento original (`src/scripts/processing.ipynb`) e as duas
variantes que mudam uma coisa por vez.

==========  ==================  ===========================  ================
variante    universo de genes   matriz que entra na PCA      papel
==========  ==================  ===========================  ================
V1 fiel     inner join          contagens cruas dos HVGs     igual ao original
V2          inner join          lognorm escalonada dos HVGs  isola a PCA
V3          união + min_cells   lognorm escalonada dos HVGs  isola o universo
==========  ==================  ===========================  ================

Tudo o mais é idêntico nas três: HVG ``seurat_v3`` com 2000 genes e
``batch_key="library_id"``, PCA com 50 componentes, Harmony por biblioteca,
vizinhança sobre as 15 primeiras dimensões, UMAP e Leiden 0,5.

O que o original fazia e por quê a V1 o repete: em ``OptimalClustering.process``
a chamada ``normalize_pearson_residuals(adata, layer="counts")`` grava os
resíduos na layer e deixa ``adata.X`` intacto; o ``sc.tl.pca`` seguinte roda
sobre ``X``, que são as contagens cruas. A V1 reproduz exatamente esse efeito
(sem recalcular os resíduos, que nunca foram usados).

Único desvio em relação ao original, nas três variantes: os símbolos dos genes
vêm da tabela de features do Cell Ranger, não de uma consulta ao ``mygene``
(rede). Não altera HVGs, PCA nem clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from . import config, io

VARIANTS = ["V1", "V2", "V3"]


# ─────────────────────────────────────────────────────────────────────────────
# Montagem do objeto concatenado
# ─────────────────────────────────────────────────────────────────────────────

def symbol_map() -> pd.Series:
    """Ensembl ID → símbolo único, igual ao ``var_names_make_unique`` da leitura."""
    import anndata as ad
    ref = io.reference_features()
    sym = ad.utils.make_index_unique(pd.Index(ref["gene_name"].astype(str)))
    return pd.Series(np.asarray(sym), index=ref["gene_id"].to_numpy())


def load_postqc(stage: str = "A01_preprocessamento/new") -> dict:
    """Objetos por biblioteca gravados por A01 (formato de `data/new`)."""
    import anndata as ad
    return {lib: ad.read_h5ad(io.data_path(stage, lib, "h5ad")) for lib in config.LIBRARIES}


def load_old_postqc() -> dict:
    """Objetos por biblioteca da corrida antiga (`data/new`), somente leitura.

    Só ``X``, ``obs``, ``var`` e as estatísticas de pré-processamento entram na
    concatenação. PCA, UMAP e grafos por biblioteca ficam de fora: têm
    dimensões diferentes entre bibliotecas e nunca foram usados adiante.
    """
    import anndata as ad
    out = {}
    for lib in config.LIBRARIES:
        a = ad.read_h5ad(config.OLD_POSTDOUBLET / f"{lib}.h5ad")
        uns = {k: v for k, v in a.uns.items() if k.startswith("preprocessing_stats_")}
        out[lib] = ad.AnnData(X=a.X, obs=a.obs, var=a.var[[]], uns=uns)
    return out


def add_design(obs: pd.DataFrame) -> pd.DataFrame:
    """Metadados do desenho: as colunas originais, com as regras originais, e as corrigidas."""
    lib = obs["library_id"].astype(str)
    # processing.ipynb, célula 037 — preservado como estava, inclusive HNP44 → "MB"
    obs["controles"] = ["controle" if l.startswith("CTH") else "estudo" for l in lib]
    obs["classificacao"] = ["PB" if l.startswith("PB") else "controle" if l.startswith("CTH") else "MB"
                            for l in lib]
    obs["estimulo"] = ["nao estimulado" if l.endswith("NS") else "estimulado" for l in lib]
    # corrigidas
    obs["patient"] = pd.Categorical([config.patient_of(l) for l in lib], config.PATIENTS)
    obs["classification"] = pd.Categorical([config.classification_of(l) for l in lib],
                                           config.CLASSIFICATION_ORDER)
    obs["stimulation"] = pd.Categorical([config.stimulation_of(l) for l in lib],
                                        config.STIMULATION_ORDER)
    return obs


def concat_libraries(adatas: dict, join: str = "inner", order=None):
    """Concatena como a célula 007 do original; ``join="outer"`` dá a união.

    A ordem padrão é a alfabética dos nomes de arquivo, como no original
    (``sorted(os.listdir(...))``): ela decide qual célula recebe o sufixo quando
    o mesmo barcode aparece em duas bibliotecas.
    """
    import anndata as an
    order = list(order or sorted(adatas))
    bdata = an.concat([adatas[l] for l in order], join=join, fill_value=0,
                      uns_merge="first", label="library_id", keys=order)
    bdata.obs["barcode"] = bdata.obs_names.to_numpy()
    bdata.obs_names_make_unique()
    bdata.var_names_make_unique()
    if not sp.issparse(bdata.X):
        bdata.X = sp.csr_matrix(bdata.X)
    bdata.X = bdata.X.tocsr().astype(np.float32)
    bdata.obs["library_id"] = pd.Categorical(bdata.obs["library_id"].astype(str), order)
    bdata.obs["cell_key"] = (bdata.obs["library_id"].astype(str) + "|" + bdata.obs["barcode"]).to_numpy()
    add_design(bdata.obs)

    bdata.layers["counts"] = bdata.X.copy()
    sym = symbol_map()
    bdata.var["gene_ids"] = bdata.var_names.to_numpy()
    bdata.var["gene_symbol"] = sym.reindex(bdata.var_names).to_numpy()
    bdata.var_names = bdata.var["gene_symbol"].to_numpy()
    bdata.var.index.name = None
    assert bdata.var_names.is_unique

    bdata.var["mt"] = bdata.var_names.str.startswith("MT-")
    bdata.var["rps"] = bdata.var_names.str.startswith("RPS")
    bdata.var["rpl"] = bdata.var_names.str.startswith("RPL")
    sc.pp.calculate_qc_metrics(bdata, qc_vars=["mt", "rps", "rpl"], inplace=True)
    for k in list(bdata.uns.keys()):
        if not k.startswith("preprocessing_stats_"):
            del bdata.uns[k]
    return bdata


def build_universe(adatas: dict, universe: str = "inner", min_cells: int = config.GLOBAL_MIN_CELLS):
    """``inner``: interseção (original). ``union``: união + um filtro global."""
    if universe == "inner":
        return concat_libraries(adatas, join="inner")
    if universe == "union":
        bdata = concat_libraries(adatas, join="outer")
        keep = np.asarray((bdata.layers["counts"] > 0).sum(axis=0)).ravel() >= min_cells
        return bdata[:, keep].copy()
    raise ValueError(universe)


# ─────────────────────────────────────────────────────────────────────────────
# HVG
# ─────────────────────────────────────────────────────────────────────────────

def select_hvg(bdata, n_top: int = config.N_HVG, batch_key: str = "library_id",
               span: float = 0.3, fallback_spans=(0.4, 0.5, 0.75, 1.0),
               exclude=None) -> pd.DataFrame:
    """HVG como no original: ``seurat_v3`` sobre contagens, por lote. Devolve ``var``.

    É a mesma conta de ``sc.pp.highly_variable_genes(flavor="seurat_v3",
    batch_key=...)``, linha por linha, com uma diferença: quando o loess da
    tendência média–variância de um lote é numericamente singular com
    ``span=0.3``, o ajuste desse lote é refeito com um span maior, em vez de
    abortar a seleção inteira.

    Isso só acontece no universo completo de genes: em bibliotecas pequenas,
    milhares de genes têm exatamente 1 UMI, logo a mesma média, e a janela local
    do loess fica degenerada (o erro ``reciprocal condition number`` do
    scikit-misc). Na interseção nenhum lote falha e o resultado é idêntico ao
    do scanpy — ``hvg_matches_scanpy`` confere isso.

    Os spans usados por lote ficam em ``df.attrs["spans"]``.

    ``exclude`` — genes que não podem ser escolhidos (continuam no objeto e nas
    estatísticas); os ``n_top`` saem dos demais, pela mesma ordenação.
    """
    from scanpy.preprocessing._highly_variable_genes import _sum_and_sum_squares_clipped
    from scanpy.preprocessing._utils import _get_mean_var
    from skmisc.loess import loess

    data = sp.csr_matrix(bdata.layers["counts"])
    df = pd.DataFrame(index=bdata.var_names)
    df["means"], df["variances"] = _get_mean_var(data)
    batch_info = bdata.obs[batch_key].astype(str).to_numpy()

    norm_gene_vars, spans = [], {}
    for b in np.unique(batch_info):
        data_batch = data[batch_info == b]
        mean, var = _get_mean_var(data_batch)
        not_const = var > 0
        estimat_var = np.zeros(data.shape[1], dtype=np.float64)
        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])
        for sp_try in (span, *fallback_spans):
            try:
                model = loess(x, y, span=sp_try, degree=2)
                model.fit()
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"loess não convergiu no lote {b} com nenhum span")
        spans[b] = sp_try
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        n_obs = data_batch.shape[0]
        clip_val = reg_std * np.sqrt(n_obs) + mean
        sq_sum, c_sum = _sum_and_sum_squares_clipped(
            data_batch.indices, data_batch.data, n_cols=data_batch.shape[1],
            clip_val=clip_val, nnz=data_batch.nnz)
        norm_gene_var = (1 / ((n_obs - 1) * np.square(reg_std))) * (
            (n_obs * np.square(mean)) + sq_sum - 2 * c_sum * mean)
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    ranked = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1).astype(np.float32)
    n_batches_hv = np.sum((ranked < n_top).astype(int), axis=0)
    ranked[ranked >= n_top] = np.nan
    median_ranked = np.ma.median(np.ma.masked_invalid(ranked), axis=0).filled(np.nan)

    df["highly_variable_nbatches"] = n_batches_hv
    df["highly_variable_rank"] = median_ranked
    df["variances_norm"] = np.mean(norm_gene_vars, axis=0)
    df.attrs["spans"] = spans
    return pick_hvg(df, n_top, exclude)


def pick_hvg(df: pd.DataFrame, n_top: int = config.N_HVG, exclude=None) -> pd.DataFrame:
    """Escolhe os ``n_top`` genes pela ordenação do ``seurat_v3`` — mediana do
    rank entre lotes e, no empate, número de lotes —, sem refazer as contas.
    ``exclude`` proíbe um conjunto de genes.
    """
    order = df[["highly_variable_rank", "highly_variable_nbatches"]].sort_values(
        ["highly_variable_rank", "highly_variable_nbatches"], ascending=[True, False],
        na_position="last").index
    if exclude:
        order = order[~order.isin(list(exclude))]
    out = df.copy()
    out["highly_variable"] = False
    out.loc[order[:int(n_top)], "highly_variable"] = True
    out.attrs = dict(df.attrs)
    return out


def hvg_matches_scanpy(bdata, hvg: pd.DataFrame, batch_key: str = "library_id") -> bool:
    """A seleção acima é idêntica à do scanpy? (só faz sentido onde o scanpy não falha)"""
    ref = sc.pp.highly_variable_genes(bdata, n_top_genes=int(hvg["highly_variable"].sum()),
                                      layer="counts", flavor="seurat_v3", batch_key=batch_key,
                                      inplace=False)
    ref.index = bdata.var_names
    same_set = bool((ref["highly_variable"].to_numpy() == hvg["highly_variable"].to_numpy()).all())
    same_rank = bool(np.allclose(ref["highly_variable_rank"].to_numpy(dtype=float),
                                 hvg["highly_variable_rank"].to_numpy(dtype=float), equal_nan=True))
    same_var = bool(np.allclose(ref["variances_norm"].to_numpy(dtype=float),
                                hvg["variances_norm"].to_numpy(dtype=float)))
    return same_set and same_rank and same_var


GENE_CLASSES = {
    "TCR (V)": r"^TR[ABGD]V\d",
    "imunoglobulina": r"^IG[HKL][VDJC]|^JCHAIN$",
    "mitocondrial": r"^MT-",
    "ribossomal": r"^RP[SL]\d|^RPLP\d|^RPSA$",
    "hemoglobina": r"^HB[ABDEGMQZ]\d?$",
    "HLA": r"^HLA-",
}


def classify_genes(names, extra: dict[str, set] | None = None) -> pd.Series:
    """Classe de cada gene para a composição dos HVGs. O que não casa é 'outros'."""
    names = pd.Index(names)
    out = pd.Series("outros", index=names)
    for cls, pat in GENE_CLASSES.items():
        out[names.str.match(pat)] = cls
    for cls, members in (extra or {}).items():
        out[names.isin(members)] = cls
    return out


# ─────────────────────────────────────────────────────────────────────────────
# As três variantes
# ─────────────────────────────────────────────────────────────────────────────

def _harmony(X_pca: np.ndarray, obs: pd.DataFrame, key: str = "library_id") -> tuple[np.ndarray, str]:
    """``hm.run_harmony`` exatamente como na célula 057 do original (``key="library_id"``).

    Devolve também a última linha do log do harmonypy sobre convergência.
    """
    import logging

    import harmonypy as hm

    msgs: list[str] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            msgs.append(record.getMessage())

    logger = logging.getLogger("harmonypy")
    quiet = [(h, h.level) for h in logger.handlers]
    for h, _ in quiet:                      # o log por iteração não vai para o notebook
        h.setLevel(logging.WARNING)
    handler = _Keep()
    logger.addHandler(handler)
    try:
        ho = hm.run_harmony(data_mat=np.asarray(X_pca), meta_data=obs, vars_use=key)
    finally:
        logger.removeHandler(handler)
        for h, level in quiet:
            h.setLevel(level)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != X_pca.shape[0]:        # harmonypy < 2 devolvia (d, N)
        Z = Z.T
    conv = next((m for m in reversed(msgs) if m.startswith("Converged")), None)
    if conv is None:
        last = next((m for m in reversed(msgs) if m.startswith("Iteration")), "sem registro de iterações")
        conv = f"sem convergência ({last})"
    return np.ascontiguousarray(Z), conv


def lognorm(counts) -> sp.csr_matrix:
    """normalize_total(1e4) + log1p sobre todos os genes do universo."""
    C = sp.csr_matrix(counts, dtype=np.float64)
    tot = np.asarray(C.sum(axis=1)).ravel()
    C = sp.diags(1e4 / tot) @ C
    C.data = np.log1p(C.data)
    return C.tocsr().astype(np.float32)


def run_variant(bdata, variant: str, hvg: pd.DataFrame | None = None, verbose: bool = True,
                harmony_key: str = "library_id"):
    """Do objeto concatenado ao Leiden. Devolve o objeto restrito aos HVGs.

    ``bdata`` tem de vir do universo certo: interseção para V1/V2, união para V3.
    ``harmony_key`` é a variável corrigida pelo Harmony; o original usa a
    biblioteca, e só a análise de sensibilidade de A04 muda isso.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    hvg = select_hvg(bdata) if hvg is None else hvg
    mask = hvg["highly_variable"].to_numpy()
    adata = bdata[:, mask].copy()
    for c in ("highly_variable", "highly_variable_rank", "highly_variable_nbatches",
              "means", "variances", "variances_norm"):
        adata.var[c] = hvg.loc[adata.var_names, c].to_numpy()

    if variant == "V1":
        adata.X = adata.layers["counts"].copy()          # PCA sobre contagens cruas, como no original
        pca_input = "contagens cruas"
    else:
        adata.X = lognorm(bdata.layers["counts"])[:, np.flatnonzero(mask)]
        sc.pp.scale(adata, max_value=10)
        pca_input = "lognorm escalonada"

    sc.tl.pca(adata, n_comps=config.N_PCS, random_state=config.RANDOM_STATE)
    adata.obsm["X_emb_harmony"], conv = _harmony(adata.obsm["X_pca"], adata.obs, harmony_key)
    sc.pp.neighbors(adata, use_rep="X_emb_harmony", n_pcs=config.N_DIMS_NEIGHBORS)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=config.LEIDEN_RESOLUTION)
    adata.uns["reseq_variant"] = {"variant": variant, "pca_input": pca_input,
                                  "n_genes_universe": int(bdata.n_vars), "harmony": conv,
                                  "harmony_key": harmony_key}
    if verbose:
        print(f"  {variant}: {adata.n_obs:,} células × {bdata.n_vars:,} genes no universo; "
              f"PCA sobre {pca_input}; variância em 50 PCs "
              f"{adata.uns['pca']['variance_ratio'].sum():.3f}; "
              f"{adata.obs['leiden'].nunique()} clusters; Harmony: {conv}")
    return adata


def neighbors_before_harmony(adata) -> sp.csr_matrix:
    """Grafo de vizinhança sobre a PCA não corrigida (para medir a mistura antes do Harmony)."""
    tmp = sc.AnnData(obs=adata.obs[[]].copy())
    tmp.obsm["X_pca"] = adata.obsm["X_pca"]
    sc.pp.neighbors(tmp, use_rep="X_pca", n_pcs=config.N_DIMS_NEIGHBORS)
    return tmp.obsp["connectivities"]


def leiden_sweep(adata, resolutions, seeds=(0,), prefix: str = "leiden",
                 obsp: str | None = None) -> pd.DataFrame:
    """Leiden em várias resoluções e sementes sobre um grafo já calculado.

    ``obsp`` escolhe o grafo (por exemplo ``connectivities_V1``); sem ele, o
    grafo padrão do objeto. Os parâmetros do Leiden são os do original.
    """
    out = {}
    for r in resolutions:
        for s in seeds:
            key = f"{prefix}_r{r:g}_s{s}"
            sc.tl.leiden(adata, resolution=float(r), random_state=int(s), key_added=key, obsp=obsp)
            out[key] = adata.obs[key].astype(str).to_numpy()
            del adata.obs[key]
            adata.uns.pop(key, None)
    return pd.DataFrame(out, index=adata.obs_names)


# ─────────────────────────────────────────────────────────────────────────────
# Um único objeto com as três variantes
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDINGS = ("X_pca", "X_emb_harmony", "X_umap")


def _plain(val):
    """Escalar nativo, para o h5ad e para o scanpy lerem os parâmetros de volta."""
    if isinstance(val, np.generic):
        return val.item()
    return val if isinstance(val, (str, int, float, bool)) else str(val)


def assemble(union, results: dict, default: str = "V3"):
    """Junta as variantes num objeto só, sobre o universo completo de genes.

    ``X`` passa a ser a lognorm de todos os genes e ``layers["counts"]`` guarda
    as contagens. Cada variante entra com sufixo — ``leiden_V1``,
    ``X_umap_V1``, ``connectivities_V1``, ``highly_variable_V1`` —, e as chaves
    sem sufixo (``leiden``, ``X_umap``, ``neighbors``, ``highly_variable``)
    apontam para ``default``. Modifica ``union`` no lugar.
    """
    out = union
    out.X = lognorm(out.layers["counts"])
    for v, a in results.items():
        assert (a.obs_names == out.obs_names).all(), f"{v}: células fora de ordem"
        assert a.var_names.isin(out.var_names).all(), f"{v}: HVG fora do universo completo"
        cats = sorted(a.obs["leiden"].astype(str).unique(), key=int)
        out.obs[f"leiden_{v}"] = pd.Categorical(a.obs["leiden"].astype(str).to_numpy(), cats)
        for k in EMBEDDINGS:
            out.obsm[f"{k}_{v}"] = np.asarray(a.obsm[k], dtype=np.float32)
        out.obsp[f"connectivities_{v}"] = sp.csr_matrix(a.obsp["connectivities"])
        out.obsp[f"distances_{v}"] = sp.csr_matrix(a.obsp["distances"])
        params = {k: _plain(val) for k, val in a.uns["neighbors"]["params"].items()}
        out.uns[f"neighbors_{v}"] = {"connectivities_key": f"connectivities_{v}",
                                     "distances_key": f"distances_{v}", "params": params}
        out.var[f"highly_variable_{v}"] = out.var_names.isin(a.var_names)
        out.var[f"highly_variable_rank_{v}"] = (
            a.var["highly_variable_rank"].reindex(out.var_names).to_numpy(dtype=float))
        out.uns[f"pca_{v}"] = {"variance_ratio": np.asarray(a.uns["pca"]["variance_ratio"]),
                               "variance": np.asarray(a.uns["pca"]["variance"])}
        out.uns[f"variant_{v}"] = dict(a.uns["reseq_variant"])
    out.obs["leiden"] = out.obs[f"leiden_{default}"]
    for k in EMBEDDINGS:
        out.obsm[k] = out.obsm[f"{k}_{default}"]
    out.obsp["connectivities"] = out.obsp[f"connectivities_{default}"]
    out.obsp["distances"] = out.obsp[f"distances_{default}"]
    out.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                            "params": dict(out.uns[f"neighbors_{default}"]["params"])}
    out.var["highly_variable"] = out.var[f"highly_variable_{default}"].to_numpy()
    out.uns["reseq"] = {"variantes": list(results), "variante_padrao": default,
                        "n_hvg": int(config.N_HVG), "n_pcs": int(config.N_PCS),
                        "n_dims_vizinhos": int(config.N_DIMS_NEIGHBORS),
                        "leiden_resolucao": float(config.LEIDEN_RESOLUTION),
                        "min_cells_global": int(config.GLOBAL_MIN_CELLS)}
    return out


def counts_match_cellranger(adata) -> pd.DataFrame:
    """Confere ``layers["counts"]`` contra as matrizes filtradas do Cell Ranger.

    Entrada por entrada, para todos os genes do objeto e todas as células de
    cada biblioteca: fecha a cadeia leitura → QC → concatenação → filtro global.
    """
    gid = adata.var["gene_ids"].to_numpy()
    lib_of = adata.obs["library_id"].astype(str).to_numpy()
    barcode = adata.obs["barcode"].astype(str).to_numpy()
    rows = []
    for lib in config.LIBRARIES:
        m = io.read_filtered(lib)
        cols = pd.Index(m["gene_ids"]).get_indexer(gid)
        sel = lib_of == lib
        idx = pd.Index(m["barcodes"]).get_indexer(barcode[sel])
        assert (cols >= 0).all() and (idx >= 0).all(), lib
        ref = m["X"][idx][:, cols].astype(np.float32)
        mine = sp.csr_matrix(adata.layers["counts"][sel])
        rows.append({"library_id": lib, "celulas": int(sel.sum()),
                     "umis_cellranger": int(ref.sum()), "umis_objeto": int(mine.sum()),
                     "entradas_diferentes": int((ref != mine).nnz)})
    return pd.DataFrame(rows).set_index("library_id")


def _crosstab(a, b) -> pd.DataFrame:
    ct = pd.crosstab(pd.Series(np.asarray(a), name="a").astype(str),
                     pd.Series(np.asarray(b), name="b").astype(str))
    key = (lambda v: int(v)) if all(str(v).isdigit() for v in list(ct.index) + list(ct.columns)) else str
    return ct.loc[sorted(ct.index, key=key), sorted(ct.columns, key=key)]


def matched_contingency(a, b) -> pd.DataFrame:
    """Tabela cruzada de dois agrupamentos, com as colunas na ordem do melhor
    pareamento com as linhas (para a diagonal ficar visível).

    Colunas sem par — quando há mais colunas que linhas — entram logo depois da
    linha em que mais aparecem.
    """
    from scipy.optimize import linear_sum_assignment
    ct = _crosstab(a, b)
    rows, cols = linear_sum_assignment(-ct.to_numpy())
    where = {int(c): float(r) for r, c in zip(rows, cols)}
    for c in range(ct.shape[1]):
        where.setdefault(c, float(np.argmax(ct.iloc[:, c].to_numpy())) + 0.5)
    return ct.iloc[:, sorted(range(ct.shape[1]), key=lambda c: (where[c], c))]


def matched_fraction(a, b) -> float:
    """Fração das células que fica no cluster pareado (pareamento um-para-um ótimo)."""
    from scipy.optimize import linear_sum_assignment
    ct = _crosstab(a, b).to_numpy()
    rows, cols = linear_sum_assignment(-ct)
    return float(ct[rows, cols].sum() / ct.sum())


def clustering_agreement(a, b) -> dict:
    """ARI e NMI entre dois agrupamentos das mesmas células."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    return {"ARI": float(adjusted_rand_score(a, b)),
            "NMI": float(normalized_mutual_info_score(a, b))}


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de avaliação do clustering
# ─────────────────────────────────────────────────────────────────────────────

def knn_library_mixing(connectivities, library, k_report: int = 15) -> pd.DataFrame:
    """Mistura de bibliotecas na vizinhança de cada célula.

    Para cada célula, o inverso do índice de Simpson das bibliotecas entre os
    vizinhos do grafo (1 = todos da mesma biblioteca). O valor esperado sob
    mistura perfeita é o inverso de Simpson da composição global.
    """
    lib = pd.Categorical(library)
    codes = lib.codes
    n_lib = len(lib.categories)
    A = sp.csr_matrix(connectivities)
    A = (A > 0).astype(np.float32)
    onehot = sp.csr_matrix((np.ones(len(codes), np.float32), (np.arange(len(codes)), codes)),
                           shape=(len(codes), n_lib))
    counts = np.asarray((A @ onehot).todense())
    deg = counts.sum(axis=1, keepdims=True)
    p = counts / np.maximum(deg, 1)
    ilisi = 1.0 / np.maximum((p ** 2).sum(axis=1), 1e-12)
    same = counts[np.arange(len(codes)), codes] / np.maximum(deg.ravel(), 1)
    return pd.DataFrame({"ilisi": ilisi, "frac_vizinhos_mesma_biblioteca": same,
                         "n_vizinhos": deg.ravel()})


def expected_ilisi(library) -> float:
    p = pd.Series(library).value_counts(normalize=True).to_numpy()
    return float(1.0 / (p ** 2).sum())


def depth_correlation(emb: np.ndarray, total_counts: np.ndarray, n: int = 5) -> np.ndarray:
    """|Spearman| entre as primeiras dimensões do embedding e o nº de UMIs."""
    from scipy.stats import spearmanr
    return np.array([abs(spearmanr(emb[:, k], total_counts).statistic) for k in range(n)])


def cluster_composition(obs: pd.DataFrame, cluster_key: str, by: str = "library_id") -> pd.DataFrame:
    """Células por cluster × grupo."""
    return pd.crosstab(obs[cluster_key].astype(str), obs[by].astype(str))


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação do agrupamento (A05)
# ─────────────────────────────────────────────────────────────────────────────

def share_within_group(obs: pd.DataFrame, cluster_key: str, by: str, order=None) -> pd.DataFrame:
    """Linhas = clusters, colunas = grupos; valor = fração das células **do grupo**
    que cai no cluster (cada coluna soma 1). Normalizar pela coluna tira o
    tamanho do grupo da conta: uma biblioteca de 10 mil células não domina a
    leitura só por ser grande."""
    ct = pd.crosstab(obs[cluster_key].astype(str), obs[by].astype(str))
    ct = ct.loc[sorted(ct.index, key=int)]
    if order is not None:
        ct = ct.reindex(columns=[o for o in order if o in ct.columns])
    return ct / ct.sum(axis=0)


def effective_number(share: pd.DataFrame) -> pd.Series:
    """Número efetivo de grupos em cada cluster (inverso de Simpson).

    Calculado sobre as frações de ``share_within_group``: vale o número de
    grupos quando todos contribuem com a mesma proporção das suas células, e 1
    quando o cluster é de um grupo só.
    """
    p = share.div(share.sum(axis=1), axis=0)
    return 1.0 / (p ** 2).sum(axis=1)


def eta_squared(values, groups) -> float:
    """Fração da variância de ``values`` explicada pelos grupos (SQ entre / SQ total)."""
    v = np.asarray(values, dtype=float)
    sst = float(((v - v.mean()) ** 2).sum())
    if sst == 0 or len(v) < 3:
        return np.nan
    agg = pd.Series(v).groupby(np.asarray(groups).astype(str)).agg(["mean", "size"])
    return float((agg["size"] * (agg["mean"] - v.mean()) ** 2).sum() / sst)


def eta_squared_within(values, groups, strata, min_cells: int = 200) -> pd.Series:
    """η² calculado dentro de cada estrato (biblioteca) com pelo menos ``min_cells``."""
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups).astype(str),
                       "s": np.asarray(strata).astype(str)})
    out = {s: eta_squared(d["v"], d["g"]) for s, d in df.groupby("s") if len(d) >= min_cells}
    return pd.Series(out, dtype=float)


def within_patient_separation(connectivities, library, patient) -> pd.DataFrame:
    """Separação entre as duas bibliotecas de cada paciente na vizinhança.

    Para cada célula olha só os vizinhos do mesmo paciente e mede que fração é
    da mesma biblioteca (isto é, do mesmo estímulo). Sob mistura perfeita dentro
    do paciente o esperado é ``n_biblioteca / n_paciente``; o índice
    ``(observado − esperado) / (1 − esperado)`` vai de 0 (misturadas) a 1
    (separadas). Devolve uma linha por biblioteca.
    """
    lib = pd.Categorical(np.asarray(library).astype(str))
    pat = pd.Series(np.asarray(patient).astype(str))
    n = len(lib)
    A = (sp.csr_matrix(connectivities) > 0).astype(np.float32)
    onehot = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), lib.codes)), shape=(n, len(lib.categories)))
    counts = np.asarray((A @ onehot).todense())
    lib_patient = pd.Series(pat.to_numpy(), index=np.asarray(lib)).groupby(level=0).first()
    same_lib = counts[np.arange(n), lib.codes]
    same_pat = np.zeros(n)
    for j, l in enumerate(lib.categories):
        same_pat += counts[:, j] * (pat.to_numpy() == lib_patient[l])
    size_lib = pd.Series(np.asarray(lib)).value_counts()
    size_pat = pat.value_counts()
    rows = []
    for l in lib.categories:
        m = np.asarray(lib) == l
        ok = m & (same_pat > 0)
        obs_frac = float((same_lib[ok] / same_pat[ok]).mean()) if ok.any() else np.nan
        exp = float(size_lib[l] / size_pat[lib_patient[l]])
        rows.append({"library_id": l, "celulas": int(m.sum()),
                     "vizinhos_mesma_biblioteca_obs": obs_frac, "esperado_sob_mistura": exp,
                     "separacao": (obs_frac - exp) / (1 - exp) if exp < 1 else np.nan,
                     "celulas_com_vizinho_do_paciente": float(ok.sum() / m.sum())})
    return pd.DataFrame(rows).set_index("library_id")


def knn_overlap(dist_a, dist_b) -> np.ndarray:
    """Fração dos vizinhos de cada célula que é a mesma nos dois grafos kNN.

    Usa os grafos de ``distances`` (k − 1 vizinhos por célula). Ao contrário do
    ARI, não depende do Leiden nem da semente: compara a estrutura local dos
    dois espaços diretamente. Vale 1 quando os espaços dão os mesmos vizinhos.
    """
    A = sp.csr_matrix(dist_a) > 0
    B = sp.csr_matrix(dist_b) > 0
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    k = np.asarray(A.sum(axis=1)).ravel()
    return inter / np.maximum(k, 1)


def cluster_stability(ref, others) -> pd.Series:
    """Estabilidade de cada cluster de ``ref`` frente a outras partições.

    Para cada outra partição, o maior índice de Jaccard entre o cluster e
    algum cluster dela; devolve a média sobre as partições. Perto de 1: o
    cluster reaparece quase igual; abaixo de 0,5: é um recorte arbitrário.
    """
    ref = pd.Series(np.asarray(ref).astype(str))
    vals = []
    for o in others:
        ct = pd.crosstab(ref.to_numpy(), np.asarray(o).astype(str))
        jac = ct / (ct.sum(axis=1).to_numpy()[:, None] + ct.sum(axis=0).to_numpy()[None, :] - ct)
        vals.append(jac.max(axis=1))
    out = pd.concat(vals, axis=1).mean(axis=1)
    key = (lambda v: int(v)) if all(str(v).isdigit() for v in out.index) else str
    return out.loc[sorted(out.index, key=key)]
