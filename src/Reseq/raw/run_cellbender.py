"""
Treina o CellBender ``remove-background`` nas 14 bibliotecas, em sequência, na GPU.

Uso (no .venv do projeto; o CellBender roda no seu próprio ambiente):

    python src/Reseq/raw/run_cellbender.py [biblioteca ...] [--epochs N] [--saida DIR]

Para cada biblioteca, lê ``raw_feature_bc_matrix.h5`` de ``data/allOuts``
(somente leitura) e grava em ``config.HEAVY_OUT/cellbender/<biblioteca>/``.
É idempotente: biblioteca com ``output_cell_barcodes.csv`` é pulada. O número
esperado de células é a chamada do Cell Ranger.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions import config, droplets, io  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("libs", nargs="*", default=config.LIBRARIES)
    ap.add_argument("--epochs", type=int, default=config.CELLBENDER_EPOCHS)
    ap.add_argument("--saida", type=Path, default=None, help="pasta de saída (teste); padrão: HEAVY_OUT")
    ap.add_argument("--total-droplets", type=int, default=None,
                    help="--total-droplets-included do CellBender; reduzir em bibliotecas com poucas células "
                         "(com o padrão, minilotes com menos de 4 células abortam o treino)")
    a = ap.parse_args()

    for lib in a.libs:
        out = (a.saida / lib) if a.saida else droplets.cellbender_dir(lib)
        if (out / "output_cell_barcodes.csv").exists():
            print(f"[{lib}] já feito, pulado", flush=True)
            continue
        if a.saida is None:
            io._assert_writable(out)
        out.mkdir(parents=True, exist_ok=True)
        h5 = config.outs_dir(lib) / "raw_feature_bc_matrix.h5"
        n_cr = len(io.read_filtered(lib)["barcodes"])
        cmd = [str(config.CELLBENDER_BIN), "remove-background", "--cuda",
               "--input", str(h5), "--output", str(out / "output.h5"),
               "--expected-cells", str(n_cr), "--fpr", str(config.CELLBENDER_FPR),
               "--epochs", str(a.epochs)]
        if a.total_droplets:
            cmd += ["--total-droplets-included", str(a.total_droplets)]
        t0 = time.time()
        print(f"[{lib}] início: {n_cr} células esperadas", flush=True)
        with open(out / "cellbender.log", "w") as log:
            r = subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{lib}] fim em {(time.time() - t0) / 60:.1f} min, código {r.returncode}", flush=True)
        if r.returncode == 0 and a.saida is None:
            io.register(stage="P02_cellbender/externo", name=lib, kind="cellbender", path=out / "output.h5",
                        description=f"CellBender remove-background {lib}",
                        extra={"comando": " ".join(cmd)})


if __name__ == "__main__":
    main()
