"""
Gotas vazias a partir da matriz raw (P01, P02).

Duas fontes, além da chamada do próprio Cell Ranger:

* **DropletUtils** (R): ``barcodeRanks`` (knee e inflexão), ``emptyDrops`` e
  ``emptyDropsCellRanger`` — os métodos citados pelo capítulo *raw data
  processing* do sc-best-practices. Rodam por ``Rscript --vanilla`` com a
  biblioteca R isolada, lendo o ``raw_feature_bc_matrix.h5`` direto de
  ``data/allOuts`` (somente leitura).
* **CellBender** ``remove-background`` (Python, GPU), num ambiente separado.
  O treino roda fora dos notebooks (``raw/run_cellbender.py``); aqui só se
  leem as saídas.

As saídas brutas das duas ferramentas ficam em ``config.HEAVY_OUT`` (NVMe).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import config, io

_FUNCS = Path(__file__).parent


def run_r(script: str, args: list, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Roda um script de ``functions/`` no R do ambiente conda (``config.RESEQ_RSCRIPT``).

    Sem esse ambiente, cai no R do sistema com a biblioteca isolada. As duas
    instalações nunca se misturam: com o R do conda, ``R_LIBS_USER`` é apagado.
    """
    env = dict(os.environ)
    if config.RESEQ_RSCRIPT.exists():
        rscript = str(config.RESEQ_RSCRIPT)
        env.pop("R_LIBS_USER", None)
        env["R_LIBS_USER"] = ""
    else:
        rscript = "Rscript"
        env["R_LIBS_USER"] = str(config.RESEQ_R_LIB)
    r = subprocess.run([rscript, "--vanilla", str(_FUNCS / script), *map(str, args)],
                       env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{script} falhou:\n{r.stderr[-2000:]}")
    return r


# ─────────────────────────────────────────────────────────────────────────────
# DropletUtils
# ─────────────────────────────────────────────────────────────────────────────

def emptydrops_dir(library_id: str) -> Path:
    return config.HEAVY_OUT / "emptydrops" / library_id


def run_emptydrops(library_id: str, force: bool = False) -> Path:
    """barcodeRanks + emptyDrops + emptyDropsCellRanger de uma biblioteca.

    Idempotente: se as quatro saídas já existem, não roda de novo.
    """
    out = emptydrops_dir(library_id)
    files = ["ranks.csv", "emptydrops.csv", "edcr.csv", "ambiente.csv"]
    if not force and all((out / f).exists() for f in files):
        return out
    io._assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    h5 = config.outs_dir(library_id) / "raw_feature_bc_matrix.h5"
    run_r("droplets.R", [h5, out, config.EMPTYDROPS_LOWER, config.EMPTYDROPS_NITERS, config.SEED])
    for f in files:
        io.register(stage="P01_gotas_vazias/externo", name=f"{library_id}_{f}", kind="csv",
                    path=out / f, description=f"DropletUtils ({f}) — {library_id}")
    return out


def read_emptydrops(library_id: str) -> dict:
    out = emptydrops_dir(library_id)
    ranks = pd.read_csv(out / "ranks.csv").iloc[0]
    return {
        "knee": float(ranks["knee"]), "inflection": float(ranks["inflection"]),
        "emptydrops": pd.read_csv(out / "emptydrops.csv"),
        "edcr": pd.read_csv(out / "edcr.csv"),
        "ambiente": pd.read_csv(out / "ambiente.csv"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CellBender
# ─────────────────────────────────────────────────────────────────────────────

def cellbender_dir(library_id: str) -> Path:
    return config.HEAVY_OUT / "cellbender" / library_id


def cellbender_done(library_id: str) -> bool:
    return (cellbender_dir(library_id) / "output_cell_barcodes.csv").exists()


def read_cellbender(library_id: str) -> dict:
    """Probabilidade de célula por barcode analisado, perfil ambiente e métricas."""
    d = cellbender_dir(library_id)
    with h5py.File(d / "output.h5", "r") as f:
        barcodes = np.array([b.decode() for b in f["matrix/barcodes"][()]])
        idx = f["droplet_latents/barcode_indices_for_latents"][()]
        prob = f["droplet_latents/cell_probability"][()]
        genes = np.array([g.decode() for g in f["matrix/features/id"][()]])
        ambient = f["global_latents/ambient_expression"][()]
    cells = pd.read_csv(d / "output_cell_barcodes.csv", header=None)[0].to_numpy()
    metrics = pd.read_csv(d / "output_metrics.csv", header=None, index_col=0)[1] \
        if (d / "output_metrics.csv").exists() else pd.Series(dtype=object)
    return {
        "prob": pd.Series(prob, index=barcodes[idx], name="cellbender_prob"),
        "cells": cells,
        "ambient": pd.Series(ambient, index=genes, name="ambient"),
        "metrics": metrics,
    }


def read_cellbender_counts(library_id: str, barcodes) -> dict:
    """Contagens corrigidas pelo CellBender para um conjunto de barcodes.

    Lê ``output.h5`` (formato CSC do Cell Ranger: genes × barcodes) só nas
    colunas pedidas. Barcodes fora do conjunto analisado pelo CellBender
    voltam como ausentes.
    """
    import scipy.sparse as sp
    d = cellbender_dir(library_id)
    with h5py.File(d / "output.h5", "r") as f:
        g = f["matrix"]
        shape = tuple(int(x) for x in g["shape"][()])
        M = sp.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shape)
        bc = pd.Index([b.decode() for b in g["barcodes"][()]])
        genes = np.array([x.decode() for x in g["features/id"][()]])
    pos = bc.get_indexer(pd.Index(barcodes))
    ok = pos >= 0
    X = M[:, pos[ok]].T.tocsr()
    return {"X": X, "barcodes": np.asarray(barcodes)[ok], "missing": np.asarray(barcodes)[~ok],
            "gene_ids": genes}


# ─────────────────────────────────────────────────────────────────────────────
# SoupX e scDblFinder (P04)
# ─────────────────────────────────────────────────────────────────────────────

def run_soupx(library_id: str, cells: pd.DataFrame, out: Path) -> dict:
    """SoupX como no capítulo de QC: raw + células + clusters → ρ e contagens corrigidas.

    ``cells``: colunas ``barcode`` e ``cluster``. Devolve ρ e a matriz corrigida
    (células × genes, na ordem de ``cells``), ou ρ = NaN se o ajuste falhar.
    """
    import scipy.io
    io._assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    cells[["barcode", "cluster"]].to_csv(out / "celulas.csv", index=False)
    h5 = config.outs_dir(library_id) / "raw_feature_bc_matrix.h5"
    run_r("soupx.R", [h5, out / "celulas.csv", out])
    info = pd.read_csv(out / "rho.csv").iloc[0]
    res = {"rho": float(info["rho"]), "status": str(info["status"])}
    if (out / "corrigida.mtx").exists():
        M = scipy.io.mmread(out / "corrigida.mtx").T.tocsr()
        res["X"] = M
        res["barcodes"] = pd.read_csv(out / "barcodes.csv")["barcode"].to_numpy()
        res["genes"] = pd.read_csv(out / "genes.csv")["gene"].to_numpy()
    return res


def run_scdblfinder(X, barcodes, out: Path) -> pd.DataFrame:
    """scDblFinder numa amostra (``set.seed(123)``, parâmetros padrão)."""
    import scipy.io
    import scipy.sparse as sp
    io._assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(out / "m.mtx", sp.csr_matrix(X).T.astype(np.int64))
    pd.Series(barcodes, name="barcode").to_csv(out / "barcodes.csv", index=False)
    run_r("scdblfinder.R", [out, config.SEED])
    res = pd.read_csv(out / "scdblfinder.csv")
    (out / "m.mtx").unlink()          # entrada temporária, recriável
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Os barcodes extras parecem célula? (teste com controle de profundidade)
# ─────────────────────────────────────────────────────────────────────────────

def _thin(M, targets, rng):
    """Subamostra binomial de cada linha de ``M`` até ``targets`` UMIs."""
    import scipy.sparse as sp
    M = sp.csr_matrix(M, dtype=np.int64)
    rows = []
    for i, t in enumerate(targets):
        row = M[i]
        n = row.data.sum()
        d = np.random.Generator.binomial(rng, row.data, min(1.0, t / max(n, 1)))
        rows.append(sp.csr_matrix((d, row.indices, [0, len(d)]), shape=(1, M.shape[1])))
    return sp.vstack(rows).tocsr()


def cell_likeness(X, totals, is_cell, test, *, amb_max: int = 100, amb_min: int = 20,
                  n_cells: int = 500, reps: int = 2, n_amb: int = 600, seed: int = 0) -> dict:
    """Quanto cada barcode de ``test`` se parece com as células da biblioteca, e não com o ambiente.

    Medida: log-razão de verossimilhança por UMI entre o perfil somado das
    células (``is_cell``) e o das gotas vazias (1–``amb_max`` UMIs). Não
    depende da profundidade, mas o ruído sim; por isso os dois controles, com
    a mesma distribuição de UMIs dos barcodes testados:

    * **positivo** — células reais subamostradas até essa profundidade;
    * **negativo** — gotas vazias somadas até essa profundidade.

    Limiar = percentil 99 do controle negativo (1% de falsos positivos).
    ``n_semelhantes`` estima quantos dos testados têm perfil de célula,
    corrigindo pela sensibilidade e pelos falsos positivos:
    ``(acima − n·fp) / (sens − fp)``.
    """
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(X)
    cell_idx = np.flatnonzero(is_cell)
    amb_idx = np.flatnonzero((totals >= 1) & (totals <= amb_max))
    pc = np.asarray(X[cell_idx].sum(axis=0)).ravel() + 0.5
    pa = np.asarray(X[amb_idx].sum(axis=0)).ravel() + 0.5
    lr = np.log(pc / pc.sum()) - np.log(pa / pa.sum())

    def llr(M):
        M = sp.csr_matrix(M)
        return np.asarray(M @ lr).ravel() / np.maximum(np.asarray(M.sum(axis=1)).ravel(), 1)

    test = np.asarray(test)
    if len(test) == 0:
        return {"llr": np.array([]), "limiar": np.nan, "sensibilidade": np.nan, "fp": np.nan,
                "acima": 0, "n_semelhantes": 0.0, "llr_celulas_reduzidas": np.array([]), "llr_vazias": np.array([]),
                "lr": lr}
    alvo = totals[test]
    sub = rng.choice(cell_idx, min(n_cells, len(cell_idx)), replace=False)
    pos = sp.vstack([_thin(X[sub], rng.choice(alvo, len(sub)), rng) for _ in range(reps)])
    big = amb_idx[totals[amb_idx] >= amb_min]
    neg = []
    for _ in range(n_amb):
        t = rng.choice(alvo)
        pick = []
        s = 0
        while s < t:
            j = rng.choice(big)
            pick.append(j)
            s += totals[j]
        neg.append(_thin(X[pick].sum(axis=0), [t], rng))
    neg = sp.vstack(neg)
    l_pos, l_neg, l_test = llr(pos), llr(neg), llr(X[test])
    lim = float(np.quantile(l_neg, 0.99))
    sens = float((l_pos > lim).mean())
    fp = float((l_neg > lim).mean())
    acima = int((l_test > lim).sum())
    est = (acima - len(test) * fp) / max(sens - fp, 1e-9)
    return {"llr": l_test, "limiar": lim, "sensibilidade": sens, "fp": fp, "acima": acima,
            "n_semelhantes": float(np.clip(est, 0, len(test))),
            "llr_celulas_reduzidas": l_pos, "llr_vazias": l_neg, "lr": lr}
