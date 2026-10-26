"""NMF de programas gênicos (Yasumizu et al. 2024) sobre a matriz corrigida pelo ComBat.

Etapas (cache em ``HEAVY_OUT/P10_nmf/``; o que já está pronto não roda de novo):

* ``entradas``     — ComBat escalonado do P07 → regressão de UMIs, %MT e ciclo
                     celular → escala → negativos = 0 (``X_nmf.npy``);
* ``varredura``    — NMF para k = 2…20 e sementes 0, 1, 2, em paralelo;
* ``escolha``      — k pelos dois critérios do artigo (`nmf.choose_rank`);
* ``estabilidade`` — no k escolhido: 3 ajustes em subamostras de 80%, com as
                     demais células projetadas na W da subamostra;
* ``projecao``     — projeção nos 12 programas publicados (NMFproj), sobre a
                     lognorm sem escala, com variância explicada e POH.

Uso (de ``src/Reseq``):

    python raw/run_nmf.py tudo --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from functions import atlas, benchmark as bm, io, nmf  # noqa: E402

warnings.filterwarnings("ignore")
ENT = nmf.CACHE / "entradas"
VAR = nmf.CACHE / "varredura"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    nmf.CACHE.mkdir(parents=True, exist_ok=True)
    with open(nmf.CACHE / "log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def entradas() -> str:
    import anndata as ad

    inp = bm.load_inputs(bm.CACHE / "entradas", full=True)
    sig = inp["assinatura"]
    meta = ENT / "assinatura.json"
    if meta.exists() and json.loads(meta.read_text())["assinatura"] == sig:
        _log("entradas: no cache")
        return sig
    pre = inp["pre"]
    X_corr = bm.read_output("combat_esc")["X"]
    main = ad.read_h5ad(io.data_path("P05_integracao", "reseq_raw_integrado", "h5ad"), backed="r")
    assert (main.obs["cell_key"].astype(str).to_numpy() == pre.obs["cell_key"].to_numpy()).all()
    cov = nmf.cell_cycle(inp["full"])
    cov["total_counts"] = pre.obs["total_counts"].to_numpy()
    cov["pct_counts_mt"] = main.obs["pct_counts_mt"].to_numpy()
    cov.index = pre.obs["cell_key"].to_numpy()
    X = nmf.prepare_input(X_corr, cov)
    io._assert_writable(ENT / "X_nmf.npy")
    ENT.mkdir(parents=True, exist_ok=True)
    np.save(ENT / "X_nmf.npy", X)
    cov.to_parquet(ENT / "covariaveis.parquet")
    pd.Series(pre.var["gene_symbol"].astype(str).to_numpy(), name="gene").to_frame().to_parquet(ENT / "genes.parquet")
    meta.write_text(json.dumps({"assinatura": sig, "celulas": int(X.shape[0]), "genes": int(X.shape[1]),
                                "fracao_zeros": float((X == 0).mean())}))
    io.register(stage=nmf.STAGE, name="entrada_nmf", kind="cache", path=ENT / "X_nmf.npy", signature=sig,
                description="Entrada da NMF: ComBat escalonado, regressão de UMIs/%MT/ciclo, escala, negativos = 0")
    _log(f"entradas: {X.shape[0]} células × {X.shape[1]} genes; {100 * (X == 0).mean():.0f}% zeros")
    return sig


def _job(args):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=3):
        return nmf.sweep_one(*args)


def varredura(workers: int) -> None:
    VAR.mkdir(parents=True, exist_ok=True)
    jobs = [(str(ENT / "X_nmf.npy"), k, s, str(VAR)) for k in nmf.KS for s in nmf.SEEDS
            if not (VAR / f"nmf_k{k}_s{s}.json").exists()]
    _log(f"varredura: {len(jobs)} ajustes a rodar ({len(nmf.KS) * len(nmf.SEEDS)} no total)")
    jobs.sort(key=lambda j: -j[1])                       # os maiores k primeiro
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            _log(f"  k={r['k']:>2} s={r['seed']}: Evar {r['evar']:.4f}, corr máx {r['maxcorr_spearman']:.3f}, "
                 f"{r['n_iter']} iterações, {r['tempo_s']:.0f} s")


def escolha() -> dict:
    sw = nmf.collect_sweep(VAR)
    stab = nmf.seed_stability(VAR)
    stab.to_csv(nmf.CACHE / "estabilidade_sementes.csv")
    esc = nmf.choose_rank(sw, stab)
    esc_p = nmf.choose_rank(sw, stab, corr="maxcorr_pearson")
    esc["k_escolhido_pearson"] = esc_p["k_escolhido"]
    nmf.save_json(esc, nmf.CACHE / "escolha.json")
    _log(f"escolha: cotovelo k={esc['k_cotovelo']}; candidatos {esc['candidatos']}; reprodutíveis "
         f"{esc['reprodutiveis']}; k escolhido={esc['k_escolhido']} (com Pearson: {esc['k_escolhido_pearson']})")
    return esc


def estabilidade(k: int, reps: int = 3, frac: float = 0.8) -> None:
    from threadpoolctl import threadpool_limits

    X = np.load(ENT / "X_nmf.npy")
    out = nmf.CACHE / "estabilidade"
    out.mkdir(parents=True, exist_ok=True)
    for r in range(reps):
        f = out / f"sub{r}_k{k}.npz"
        if f.exists():
            continue
        rng = np.random.default_rng(100 + r)
        idx = np.sort(rng.choice(X.shape[0], int(frac * X.shape[0]), replace=False))
        t0 = time.perf_counter()
        with threadpool_limits(limits=8):
            _, W, evar, it = nmf.fit(X[idx], k, seed=100 + r)
            H = nmf.project_fixed(X, W)
        np.savez_compressed(f, H=H, W=W, idx=idx)
        _log(f"estabilidade: subamostra {r} (k={k}): Evar {evar:.4f}, {it} iterações, {time.perf_counter() - t0:.0f} s")


def projecao() -> None:
    from threadpoolctl import threadpool_limits

    f = nmf.CACHE / "projecao.npz"
    if f.exists():
        _log("projecao: no cache")
        return
    inp = bm.load_inputs(bm.CACHE / "entradas", full=True)
    full = inp["full"]
    W = atlas.published_W()
    genes = W.index.intersection(full.var_names)
    Xq = full[:, genes].X
    Xq = np.asarray(Xq.toarray() if hasattr(Xq, "toarray") else Xq, dtype=np.float64)
    Wt = W.loc[genes].to_numpy().T                       # programas × genes
    t0 = time.perf_counter()
    with threadpool_limits(limits=8):
        H = nmf.project_fixed(Xq, Wt)
    ev = nmf.explained_variance(Xq, H, Wt)
    ev.index = [W.columns[i] if isinstance(i, (int, np.integer)) else i for i in ev.index]
    poh = nmf.hvg_overlap(full, W.index)
    np.savez_compressed(f, H=H, W=Wt, genes=np.asarray(genes), programas=np.asarray(W.columns))
    nmf.save_json({"genes_W": int(W.shape[0]), "genes_presentes": int(len(genes)), **poh,
                   "variancia_explicada": ev.to_dict(), "tempo_s": time.perf_counter() - t0},
                  nmf.CACHE / "projecao_qc.json")
    _log(f"projecao: {len(genes)} de {W.shape[0]} genes de W; Evar {ev['ALL']:.3f}; POH {poh['poh']:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("etapa", choices=["entradas", "varredura", "escolha", "estabilidade", "projecao", "tudo"])
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    io._assert_writable(nmf.CACHE / "log.txt")
    if a.etapa in ("entradas", "tudo"):
        entradas()
    if a.etapa in ("varredura", "tudo"):
        varredura(a.workers)
    if a.etapa in ("escolha", "tudo"):
        esc = escolha()
    if a.etapa in ("estabilidade", "tudo"):
        k = json.loads((nmf.CACHE / "escolha.json").read_text())["k_escolhido"]
        estabilidade(k)
    if a.etapa in ("projecao", "tudo"):
        projecao()
    _log("fim")


if __name__ == "__main__":
    main()
