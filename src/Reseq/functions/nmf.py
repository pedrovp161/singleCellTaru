"""
Programas gênicos por NMF, como em Yasumizu et al. (2024), Cell Genomics 4:100473.

Os autores argumentam que delimitar clusters num contínuo como o das células T
CD4⁺ é artificial, e descrevem cada célula como uma combinação de programas
gênicos obtidos por fatoração não negativa (NMF). A receita (tutorial
``PBMC.ipynb`` do repositório NMFprojection e Métodos do artigo):

1. lognorm (10⁴ + log1p), genes variáveis;
2. ``regress_out`` de ``total_counts``, ``pct_counts_mt`` e dos escores de ciclo
   celular (S, G2M); ``scale(max_value=10)``;
3. os valores negativos viram zero (a NMF exige matriz não negativa);
4. ``sklearn.decomposition.NMF(n_components=k, init="random")``;
5. k escolhido por dois critérios: depois do cotovelo da variância explicada e
   antes de a correlação máxima entre componentes subir.

Aqui a entrada do passo 2 é a matriz corrigida pelo ComBat (a integração
escolhida no P07), nas 2.000 HVG do P05. Para comparar com o Leiden, cada célula
recebe o programa dominante (`dominant`) — o "cluster" do método.

Também a projeção nos 12 programas publicados (`project`), como o NMFproj:
W fixa, só H é estimada, sobre a lognorm sem escala.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config, io

STAGE = "P10_nmf"
CACHE = Path(os.environ.get("RESEQ_NMF_CACHE", config.HEAVY_OUT / STAGE))
COVARIATES = ["total_counts", "pct_counts_mt", "S_score", "G2M_score"]
KS = list(range(2, 21))
SEEDS = (0, 1, 2)
MAX_ITER = 400
TOL_CORR = 0.10        # "antes de a correlação máxima subir": até +0,10 sobre o valor no cotovelo


def cell_cycle(full) -> pd.DataFrame:
    """Escores S e G2M (listas de Tirosh et al. 2016, as do scib) sobre a lognorm de todos os genes."""
    import scanpy as sc
    import scib

    res = Path(scib.__file__).parent / "resources"
    s = [g.strip() for g in open(res / "s_genes_tirosh_hm.txt") if g.strip()]
    g2m = [g.strip() for g in open(res / "g2m_genes_tirosh_hm.txt") if g.strip()]
    a = full.copy()
    sc.tl.score_genes_cell_cycle(a, s_genes=[g for g in s if g in a.var_names],
                                 g2m_genes=[g for g in g2m if g in a.var_names])
    return a.obs[["S_score", "G2M_score"]].copy()


def regress_out(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Resíduos de mínimos quadrados de cada gene sobre as covariáveis (com intercepto).

    É a conta do ``sc.pp.regress_out`` para covariáveis contínuas (GLM
    gaussiano por gene), feita de uma vez com álgebra matricial.
    """
    Z1 = np.column_stack([np.ones(len(Z)), (Z - Z.mean(0)) / Z.std(0)])
    beta, *_ = np.linalg.lstsq(Z1, X, rcond=None)
    return X - Z1 @ beta


def prepare_input(X_corr: np.ndarray, covars: pd.DataFrame) -> np.ndarray:
    """Passos 2–3 da receita: regressão das covariáveis, escala (corte em 10) e negativos → 0."""
    R = regress_out(np.asarray(X_corr, dtype=np.float64), covars[COVARIATES].to_numpy(np.float64))
    R = (R - R.mean(0)) / np.where(R.std(0) > 0, R.std(0), 1.0)
    np.clip(R, None, 10, out=R)
    R[R < 0] = 0.0
    return R.astype(np.float32)


def max_corr(W: np.ndarray, method: str = "spearman") -> float:
    """Maior correlação entre dois componentes (linhas de W, pesos dos genes)."""
    if W.shape[0] < 2:
        return 0.0
    M = stats.rankdata(W, axis=1) if method == "spearman" else W
    C = np.corrcoef(M)
    np.fill_diagonal(C, -np.inf)
    return float(C.max())


def fit(X: np.ndarray, k: int, seed: int, max_iter: int = MAX_ITER):
    """Uma NMF; devolve (H células × k, W k × genes, variância explicada, iterações)."""
    from sklearn.decomposition import NMF

    m = NMF(n_components=k, init="random", random_state=seed, max_iter=max_iter)
    H = m.fit_transform(X)
    W = m.components_
    evar = 1.0 - m.reconstruction_err_ ** 2 / float((X.astype(np.float64) ** 2).sum())
    return H.astype(np.float32), W.astype(np.float32), float(evar), int(m.n_iter_)


def sweep_one(X_path: str, k: int, seed: int, out_dir: str) -> dict:
    """Um ponto da varredura (roda num processo à parte)."""
    X = np.load(X_path)
    t0 = time.perf_counter()
    H, W, evar, it = fit(X, k, seed)
    row = {"k": k, "seed": seed, "evar": evar, "maxcorr_spearman": max_corr(W, "spearman"),
           "maxcorr_pearson": max_corr(W, "pearson"), "n_iter": it, "tempo_s": time.perf_counter() - t0}
    np.savez_compressed(Path(out_dir) / f"nmf_k{k}_s{seed}.npz", H=H, W=W)
    (Path(out_dir) / f"nmf_k{k}_s{seed}.json").write_text(json.dumps(row))
    return row


def collect_sweep(out_dir: Path = CACHE / "varredura") -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("nmf_k*_s*.json"))]
    return pd.DataFrame(rows).sort_values(["k", "seed"]).reset_index(drop=True)


def seed_stability(sweep_dir: Path = CACHE / "varredura") -> pd.DataFrame:
    """Reprodutibilidade dos programas entre as sementes, por k.

    Para cada par de sementes: os programas são pareados pela correlação dos
    pesos (W, húngaro) e o programa dominante de cada célula é comparado (ARI).
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score

    rows = []
    for k in KS:
        fits = [np.load(Path(sweep_dir) / f"nmf_k{k}_s{s}.npz") for s in SEEDS]
        Ws, doms = [], []
        for z in fits:
            H, W = normalize(z["H"], z["W"])
            Ws.append(W)
            doms.append(dominant(H)[0])
        cors, aris = [], []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                C = np.corrcoef(Ws[i], Ws[j])[:k, k:]
                r, c = linear_sum_assignment(-C)
                cors.append(float(C[r, c].mean()))
                aris.append(float(adjusted_rand_score(doms[i], doms[j])))
        rows.append({"k": k, "corr_programas": float(np.mean(cors)), "corr_programas_min": float(np.min(cors)),
                     "ARI_dominante": float(np.mean(aris)), "ARI_dominante_min": float(np.min(aris))})
    return pd.DataFrame(rows).set_index("k")


def choose_rank(sweep: pd.DataFrame, stab: pd.DataFrame | None = None, tol: float = TOL_CORR,
                corr: str = "maxcorr_spearman", min_repro: float = 0.99) -> dict:
    """Os dois critérios do artigo, em forma de regra, e o desempate.

    * cotovelo: o k de maior distância acima da reta entre o primeiro e o último
      ponto da variância explicada (média das sementes), em escala 0–1 (kneedle);
    * candidatos: os k acima do cotovelo até a correlação máxima entre
      componentes subir mais que ``tol`` em relação ao cotovelo;
    * desempate (quando a correlação não sobe e todos os candidatos servem): o
      maior candidato cujos programas se reproduzem nas três sementes
      (correlação pareada de W e ARI do programa dominante ≥ ``min_repro``);
      sem nenhum, o cotovelo.
    """
    m = sweep.groupby("k")[["evar", corr]].mean()
    ks = m.index.to_numpy()
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (m["evar"] - m["evar"].min()) / (m["evar"].max() - m["evar"].min())
    k_elbow = int(ks[np.argmax(y.to_numpy() - x)])
    base = float(m.loc[k_elbow, corr])
    cand = []
    for k in ks[ks > k_elbow]:
        if float(m.loc[k, corr]) > base + tol:
            break
        cand.append(int(k))
    k_rise = cand[-1] + 1 if cand and cand[-1] < ks.max() else None
    repro = []
    if stab is not None:
        repro = [k for k in cand if stab.loc[k, "corr_programas_min"] >= min_repro
                 and stab.loc[k, "ARI_dominante_min"] >= min_repro]
    k_star = max(repro) if repro else k_elbow
    return {"k_cotovelo": k_elbow, "candidatos": cand, "subida_da_correlacao_em": k_rise,
            "reprodutiveis": repro, "k_escolhido": int(k_star), "corr_no_cotovelo": base, "tolerancia": tol,
            "criterio_corr": corr, "min_reprodutibilidade": min_repro}


def normalize(H: np.ndarray, W: np.ndarray):
    """Cada programa com ‖W_k‖₂ = 1, e H na mesma escala (X ≈ H W não muda)."""
    s = np.linalg.norm(W, axis=1)
    s = np.where(s > 0, s, 1.0)
    return H * s[None, :], W / s[:, None]


def dominant(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Programa de maior contribuição em cada célula e a fração que ele tem do total."""
    tot = H.sum(1)
    return H.argmax(1), np.where(tot > 0, H.max(1) / np.where(tot > 0, tot, 1), 0.0)


def top_genes(W: np.ndarray, genes, n: int = 30) -> pd.DataFrame:
    genes = np.asarray(genes)
    return pd.DataFrame({f"P{k}": genes[np.argsort(-W[k])[:n]] for k in range(W.shape[0])})


def project_fixed(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """H com W fixa (o que o NMFproj faz): ``non_negative_factorization(update_H=False)``.

    ``X`` células × genes; ``W`` programas × genes. Devolve células × programas.
    """
    from sklearn.decomposition import non_negative_factorization

    Hc, _, _ = non_negative_factorization(np.asarray(X, dtype=np.float64), n_components=W.shape[0],
                                          init="custom", update_H=False, H=np.asarray(W, dtype=np.float64),
                                          random_state=0, max_iter=MAX_ITER)
    return Hc.astype(np.float32)


def explained_variance(X: np.ndarray, H: np.ndarray, W: np.ndarray) -> pd.Series:
    """``calc_EV`` do NMFproj: por programa e de todos juntos."""
    X = np.asarray(X, dtype=np.float64)
    tot = float((X ** 2).sum())
    ev = {}
    for k in range(W.shape[0]):
        R = X - np.outer(H[:, k], W[k])
        ev[k] = 1.0 - float((R ** 2).sum()) / tot
    R = X - H.astype(np.float64) @ W.astype(np.float64)
    ev["ALL"] = 1.0 - float((R ** 2).sum()) / tot
    return pd.Series(ev)


def hvg_overlap(lognorm_adata, W_genes, n_top: int = 500) -> dict:
    """POH do NMFproj: fração das 500 HVG da query (sem TCR/IG) que estão em W. Abaixo de 0,1 = ajuste ruim."""
    import scanpy as sc

    a = lognorm_adata
    bl = ("TRAV", "TRAJ", "TRBV", "TRBD", "TRBJ", "TRGV", "TRGJ", "TRDV", "TRDD", "TRDJ",
          "IGKV", "IGKJ", "IGLV", "IGLJ", "IGHV", "IGHD", "IGHJ")
    a = a[:, ~a.var_names.str.startswith(bl)].copy()
    sc.pp.highly_variable_genes(a, min_mean=0.0125, max_mean=3, min_disp=0.1, n_bins=300)
    v = a.var[a.var["highly_variable"]]
    top = v.index[v["dispersions_norm"].rank(ascending=False) <= n_top]
    sel = pd.Index(top).isin(pd.Index(W_genes))
    return {"poh": float(sel.mean()), "hvg_query": int(len(top)), "hvg_em_W": int(sel.sum())}


def save_json(obj, path: Path) -> None:
    io._assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))
