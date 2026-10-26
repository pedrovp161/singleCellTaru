"""
Métricas para comparar clusterizações (P11: Leiden × NMF).

Cada partição atribui todas as células. As métricas se dividem em cinco grupos:

* **concordância** com rótulos independentes de qualquer clusterização nossa:
  o atlas de Yasumizu et al. (2024; L1 e L2, nas células CD4⁺ pelo Azimuth) e o
  Azimuth do Cell Ranger (7 grupos) — ARI e NMI;
* **robustez**: ARI entre sementes e entre a partição completa e a refeita em
  subamostras de 80% das células;
* **técnica**: quanto a partição se organiza por profundidade (η² dos UMIs dentro
  da biblioteca) e por biblioteca/paciente;
* **marcadores**: quão específicos são os melhores genes de cada grupo (AUROC);
* **anotação**: concordância dos rótulos do pipeline com o tipo L1 do atlas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

from . import pipeline as pl


def ari_nmi(part, ref, mask=None) -> tuple[float, float]:
    part, ref = np.asarray(part).astype(str), np.asarray(ref).astype(object)
    ok = pd.notna(ref) if mask is None else (np.asarray(mask) & pd.notna(ref))
    if ok.sum() == 0:
        return np.nan, np.nan
    return (float(adjusted_rand_score(ref[ok].astype(str), part[ok])),
            float(normalized_mutual_info_score(ref[ok].astype(str), part[ok], average_method="arithmetic")))


def mean_pairwise_ari(parts: list) -> float:
    vals = [adjusted_rand_score(np.asarray(parts[i]).astype(str), np.asarray(parts[j]).astype(str))
            for i in range(len(parts)) for j in range(i + 1, len(parts))]
    return float(np.mean(vals)) if vals else np.nan


def depth_eta2(part, umis, library) -> float:
    """η² do log10 dos UMIs explicado pela partição, dentro de cada biblioteca (mediana)."""
    return float(pl.eta_squared_within(np.log10(np.asarray(umis, float)), np.asarray(part).astype(str),
                                       np.asarray(library).astype(str)).median())


def patient_mixing(part, patient) -> float:
    """Número efetivo de pacientes por grupo, ponderado pelo tamanho do grupo."""
    ct = pd.crosstab(np.asarray(part).astype(str), np.asarray(patient).astype(str))
    share = ct / ct.sum(axis=0)                     # como pipeline.share_within_group, sem exigir rótulo inteiro
    ne = pl.effective_number(share)
    return float(np.average(ne.to_numpy(), weights=ct.sum(axis=1).reindex(ne.index).to_numpy()))


def single_library_fraction(part, library, thr: float = 0.5) -> float:
    """Fração das células que estão em grupos com mais de ``thr`` das células numa biblioteca só."""
    ct = pd.crosstab(np.asarray(part).astype(str), np.asarray(library).astype(str))
    dom = (ct.max(axis=1) / ct.sum(axis=1)) > thr
    return float(ct.sum(axis=1)[dom].sum() / ct.values.sum())


def marker_auroc(X, part, n_top: int = 10, max_cells: int = 20000, seed: int = 0) -> tuple[float, pd.Series]:
    """Para cada grupo, a AUROC média dos ``n_top`` genes mais específicos (grupo × resto).

    AUROC pela soma de postos (Mann–Whitney), numa subamostra de até
    ``max_cells`` células. Devolve a média entre grupos e o valor por grupo.
    """
    rng = np.random.default_rng(seed)
    part = np.asarray(part).astype(str)
    idx = np.arange(len(part)) if len(part) <= max_cells else np.sort(rng.choice(len(part), max_cells, replace=False))
    M = X[idx]
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    R = stats.rankdata(M, axis=0)
    g = part[idx]
    n = len(g)
    out = {}
    for c in pd.unique(g):
        m = g == c
        n1 = int(m.sum())
        if n1 < 5 or n1 == n:
            continue
        U = R[m].sum(0) - n1 * (n1 + 1) / 2
        auc = U / (n1 * (n - n1))
        out[c] = float(np.sort(auc)[-n_top:].mean())
    s = pd.Series(out)
    return float(s.mean()), s


def annotation_agreement(labels, atlas_l1, mask, to_l1: dict) -> dict:
    """Concordância dos rótulos (traduzidos para L1) com o L1 do atlas, nas células comparáveis."""
    lab = pd.Series(np.asarray(labels).astype(str)).map(to_l1)
    ref = pd.Series(np.asarray(atlas_l1, dtype=object))
    ok = np.asarray(mask) & lab.notna().to_numpy() & ref.notna().to_numpy()
    if ok.sum() == 0:
        return {"acuracia": np.nan, "kappa": np.nan, "cobertura": 0.0}
    return {"acuracia": float((lab[ok].to_numpy() == ref[ok].to_numpy()).mean()),
            "kappa": float(cohen_kappa_score(lab[ok], ref[ok])),
            "cobertura": float(ok.sum() / np.asarray(mask).sum())}


def minmax(df: pd.DataFrame) -> pd.DataFrame:
    lo, hi = df.min(), df.max()
    return (df - lo) / (hi - lo).replace(0, np.nan)
