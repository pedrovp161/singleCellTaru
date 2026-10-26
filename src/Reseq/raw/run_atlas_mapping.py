"""Mapeia as células CD4⁺ do objeto principal no atlas de Yasumizu et al. (2024).

Fica de fora quem o Azimuth do Cell Ranger chama de outra linhagem (CD8 e T não
convencional, B, NK/ILC, mieloide): mapear outra linhagem numa referência de
CD4⁺ dá rótulos confiantes e sem sentido (screfmapping, "query input contract").
As células sem Azimuth entram.

Saída: ``HEAVY_OUT/atlas_yasumizu/mapeamento/mapeamento.parquet`` (chave
``cell_key``), lida pelo P09. Uso (de ``src/Reseq``):

    python raw/run_atlas_mapping.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402

from functions import atlas, benchmark as bm, io  # noqa: E402

warnings.filterwarnings("ignore")
NAO_CD4 = ["CD8 e T não convencional", "B e plasmócito", "NK e ILC", "mieloide"]


def main() -> None:
    t0 = time.time()
    adata = ad.read_h5ad(io.data_path("P05_integracao", "reseq_raw_integrado", "h5ad"))
    lab = bm.azimuth_labels(adata.obs)
    cells = ~lab[bm.LABEL].isin(NAO_CD4).to_numpy()
    print(f"{int(cells.sum())} de {adata.n_obs} células para o mapeamento "
          f"({int((~cells).sum())} de outra linhagem pelo Azimuth ficam de fora)", flush=True)
    res = atlas.run_mapping(adata, cells)
    out = atlas.DIR / "mapeamento" / "mapeamento.parquet"
    res.to_parquet(out)
    io.register(stage="P09_anotacao", name="mapeamento_atlas", kind="cache", path=out,
                description="Mapeamento Symphony no atlas CD4⁺ de Yasumizu et al. (2024): L1, L2 e fração de votos")
    print(res["atlas_L1"].value_counts().to_string())
    print(f"pronto em {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
