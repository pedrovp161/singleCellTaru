"""Roda o benchmark de integração do P07 (fora do notebook, em segundo plano).

Cada ajuste e cada avaliação roda num processo próprio, para que o tempo e a
memória de um método não se misturem com os de outro (como no artigo). Tudo é
idempotente: o que já está no cache com a mesma assinatura das entradas não
roda de novo.

Uso (de ``src/Reseq``):

    python raw/run_benchmark.py tudo                 # entradas → ajustes → métricas
    python raw/run_benchmark.py integrar harmony scvi
    python raw/run_benchmark.py metricas harmony_esc
    python raw/run_benchmark.py tudo --forcar        # refaz mesmo com cache
    RESEQ_BENCH_CACHE=<HEAVY_OUT>/P07_benchmark_integracao/_teste \
        python raw/run_benchmark.py tudo --teste 3000   # subamostra para testar

O log vai para ``<cache>/log.txt``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions import benchmark as bm, config, io  # noqa: E402


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    bm.CACHE.mkdir(parents=True, exist_ok=True)
    with open(bm.CACHE / "log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _entradas(teste: int | None) -> str:
    import anndata as ad
    import numpy as np

    caminho = io.data_path("P05_integracao", "reseq_raw_integrado", "h5ad")
    adata = ad.read_h5ad(caminho)
    if teste:                                  # estratificada: até teste/14 células por biblioteca
        rng = np.random.default_rng(0)
        lib = adata.obs["library_id"].astype(str).to_numpy()
        por = teste // len(config.LIBRARIES)
        idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(lib == l), min(por, int((lib == l).sum())),
                                                 replace=False) for l in config.LIBRARIES]))
        adata = adata[idx].copy()
    inp = bm.prepare(adata, bm.CACHE / "entradas")
    _log(f"entradas: {inp['pre'].n_obs} células × {inp['pre'].n_vars} HVG; assinatura {inp['assinatura']}")
    return inp["assinatura"]


def _rotulo(nome: str | None):
    """Coluna ``nome`` de ``rotulos_benchmark`` (P09), indexada por ``cell_key``; None = Azimuth."""
    if nome is None:
        return None
    import pandas as pd

    rb = pd.read_parquet(io.data_path("P09_anotacao", "rotulos_benchmark", "parquet")).set_index("cell_key")
    if nome not in rb.columns:
        sys.exit(f"rótulo desconhecido: {nome} (há: {', '.join(rb.columns)})")
    return rb[nome]


def _filho(args: list[str]) -> int:
    """Roda este script num processo novo e devolve o código de saída."""
    env = {**__import__("os").environ, "PYTHONWARNINGS": "ignore"}
    with open(bm.CACHE / "log.txt", "a", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, __file__, *args], stdout=fh, stderr=subprocess.STDOUT, env=env)
    return r.returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("etapa", choices=["entradas", "integrar", "metricas", "tudo", "_ajuste", "_metricas"])
    p.add_argument("alvos", nargs="*", help="ajustes (integrar) ou execuções (metricas); vazio = todos")
    p.add_argument("--forcar", action="store_true")
    p.add_argument("--teste", type=int, default=None, help="subamostra de N células (só com RESEQ_BENCH_CACHE)")
    p.add_argument("--rotulo", default=None, help="métricas com outro rótulo no lugar do Azimuth: uma coluna de "
                   "dataReseq/P09_anotacao/rotulos_benchmark (anotacao, atlas_L1, atlas_L2)")
    a = p.parse_intermixed_args()
    if a.teste and "RESEQ_BENCH_CACHE" not in __import__("os").environ:
        sys.exit("--teste exige RESEQ_BENCH_CACHE apontando para um cache separado")
    io._assert_writable(bm.CACHE / "log.txt")
    warnings.filterwarnings("ignore")

    if a.etapa == "_ajuste":                       # processo filho: um ajuste
        inp = bm.load_inputs(bm.CACHE / "entradas", full=False)
        for k, meta in bm.integrate(a.alvos[0], inp).items():
            print(f"  {k}: {meta['tempo_s']:.0f} s; memória de pico +{meta['memoria_pico_gb']:.1f} GB; "
                  f"GPU {meta['gpu_pico_gb']:.1f} GB; {meta['info']}", flush=True)
        return
    if a.etapa == "_metricas":                     # processo filho: uma avaliação
        inp = bm.load_inputs(bm.CACHE / "entradas")
        labels = _rotulo(a.rotulo)
        out = bm.evaluate(a.alvos[0], inp, labels=labels, tag=a.rotulo)
        print("  " + "; ".join(f"{k} {v:.3f}" for k, v in out["metricas"].items()), flush=True)
        return

    if a.rotulo and a.etapa != "metricas":
        sys.exit("--rotulo só vale para a etapa metricas")
    if a.etapa in ("entradas", "tudo"):
        sig = _entradas(a.teste)
    else:
        sig = bm.load_inputs(bm.CACHE / "entradas", full=False)["assinatura"]

    if a.etapa in ("integrar", "tudo"):
        fits = a.alvos if (a.alvos and a.etapa == "integrar") else bm.FITS
        for f in fits:
            keys = [r.key for r in bm.RUNS if r.fit == f]
            if not a.forcar and all(bm.is_integrated(k, sig) for k in keys):
                _log(f"ajuste {f}: no cache")
                continue
            _log(f"ajuste {f}: iniciando")
            rc = _filho(["_ajuste", f])
            _log(f"ajuste {f}: {'ok' if rc == 0 else f'FALHOU (código {rc})'}")

    if a.etapa in ("metricas", "tudo"):
        keys = a.alvos if (a.alvos and a.etapa == "metricas") else [r.key for r in bm.RUNS]
        for k in keys:
            if not bm.is_integrated(k, sig):
                _log(f"métricas {k}: sem ajuste, pulando")
                continue
            rot = f" [{a.rotulo}]" if a.rotulo else ""
            if not a.forcar and bm.is_evaluated(k, sig, a.rotulo):
                _log(f"métricas {k}{rot}: no cache")
                continue
            _log(f"métricas {k}{rot}: iniciando")
            rc = _filho(["_metricas", k] + (["--rotulo", a.rotulo] if a.rotulo else []))
            _log(f"métricas {k}{rot}: {'ok' if rc == 0 else f'FALHOU (código {rc})'}")
    _log("fim")


if __name__ == "__main__":
    main()
