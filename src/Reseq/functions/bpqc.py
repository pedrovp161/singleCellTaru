"""
Controle de qualidade como no capítulo *quality control* do sc-best-practices.

==========================  =======================  ==========================
métrica                     transformação            corte
==========================  =======================  ==========================
nº de UMIs                  log1p                    5 MADs, dois lados
nº de genes                 log1p                    5 MADs, dois lados
% nos 20 genes mais altos   —                        5 MADs, dois lados
fração mitocondrial         logit((mt + 1)/(n + 2))  3 MADs acima **e** % MT > 8
==========================  =======================  ==========================

MAD sem fator de escala (``scipy.stats.median_abs_deviation`` com o padrão do
capítulo), por amostra. Ribossomais e hemoglobina entram só como diagnóstico.

Duas diferenças em relação ao spatools usado no notebook original, ambas do
capítulo: os limites de contagem e genes valem nos **dois** lados, e o MT é
cortado em logit com a regra conjunta. Além disso, todos os limiares são
calculados sobre a **mesma** população e ficam registrados — o plot desenha o
limiar que de fato foi aplicado.

Uma métrica a mais, fora do capítulo: a fração de UMIs intrônicos (de
``molecule_info.h5``). Ela separa células de material nuclear de células
mortas, que o emptyDrops chama porque difere do perfil ambiente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from . import config

#: Métrica → (coluna, lado, k).
RULES = {
    "umis": ("log1p_total_counts", "both", config.BP_K),
    "genes": ("log1p_n_genes_by_counts", "both", config.BP_K),
    "top20": ("pct_counts_in_top_20_genes", "both", config.BP_K),
    "mt": ("logit_mt", "upper", config.BP_K_MT),
}
LABELS = {"umis": "nº de UMIs", "genes": "nº de genes", "top20": "% nos 20 genes mais altos",
          "mt": "fração mitocondrial"}


def qc_metrics(adata) -> pd.DataFrame:
    """Métricas do capítulo. ``adata`` com contagens em ``X`` e símbolos em ``var['gene_symbol']``."""
    sym = pd.Index(adata.var["gene_symbol"].astype(str))
    adata.var["mt"] = sym.str.startswith("MT-")
    adata.var["ribo"] = sym.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = sym.str.contains(r"^HB[^(P)]", regex=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], percent_top=[20], log1p=True,
                               inplace=True)
    obs = adata.obs
    frac = (obs["total_counts_mt"] + 1) / (obs["total_counts"] + 2)
    obs["logit_mt"] = np.log(frac / (1 - frac))
    return obs


def mad_limits(x, k: float, side: str) -> tuple[float, float]:
    from scipy.stats import median_abs_deviation
    x = np.asarray(x, dtype=float)
    med, mad = float(np.median(x)), float(median_abs_deviation(x))
    lo = med - k * mad if side in ("both", "lower") else -np.inf
    hi = med + k * mad if side in ("both", "upper") else np.inf
    return lo, hi


def thresholds(obs: pd.DataFrame, ref: np.ndarray | None = None) -> dict:
    """Limiares de cada métrica, calculados sobre ``obs[ref]`` (ou todas as linhas)."""
    sub = obs if ref is None else obs[np.asarray(ref, dtype=bool)]
    return {m: mad_limits(sub[col], k, side) for m, (col, side, k) in RULES.items()}


def flags(obs: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Uma coluna booleana por regra (True = reprovado) e ``passa`` no fim."""
    out = pd.DataFrame(index=obs.index)
    for m, (col, side, k) in RULES.items():
        lo, hi = thr[m]
        v = obs[col].to_numpy(dtype=float)
        out[f"fora_{m}"] = (v < lo) | (v > hi)
    out["fora_mt"] = out["fora_mt"] & (obs["pct_counts_mt"].to_numpy() > config.BP_MT_PCT)
    out["passa"] = ~out.any(axis=1)
    return out


def thresholds_table(thr_by_lib: dict) -> pd.DataFrame:
    """Limiares por biblioteca, na escala original (UMIs, genes, %, % MT)."""
    rows = []
    for lib, thr in thr_by_lib.items():
        r = {"library_id": lib}
        for m, (lo, hi) in thr.items():
            if m in ("umis", "genes"):
                lo, hi = np.expm1(lo), np.expm1(hi)
            elif m == "mt":
                lo, hi = (np.nan, 100 / (1 + np.exp(-hi)))
            r[f"{m}_min"], r[f"{m}_max"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows).set_index("library_id")


def intronic_flag(frac: pd.Series, library: pd.Series, ref: np.ndarray, k: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """Fração intrônica acima de mediana + k·MAD das células de referência da biblioteca."""
    from scipy.stats import median_abs_deviation
    lim = {}
    for lib in pd.unique(library):
        m = (library == lib).to_numpy() & np.asarray(ref, dtype=bool)
        x = frac[m].dropna().to_numpy()
        lim[lib] = float(np.median(x) + k * median_abs_deviation(x)) if len(x) else np.nan
    lim = pd.Series(lim)
    return frac > library.map(lim), lim


def lineage_scores(X, gene_names, panels: dict) -> pd.DataFrame:
    """Fração dos UMIs de cada célula nos genes de cada painel (T, B, mieloide…)."""
    X = sp.csr_matrix(X)
    tot = np.asarray(X.sum(axis=1)).ravel()
    names = pd.Index(gene_names)
    out = {}
    for k, genes in panels.items():
        idx = np.flatnonzero(names.isin(genes))
        out[k] = np.asarray(X[:, idx].sum(axis=1)).ravel() / np.maximum(tot, 1)
    return pd.DataFrame(out)
