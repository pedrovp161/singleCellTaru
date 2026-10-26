"""
Atlas de células T CD4⁺ de Yasumizu et al. (2024), *Single-cell transcriptome
landscape of circulating CD4⁺ T cell populations in autoimmune diseases*,
Cell Genomics 4:100473 (doi:10.1016/j.xgen.2023.100473).

Duas coisas vêm do atlas, ambas baixadas para ``HEAVY_OUT/atlas_yasumizu/``:

* a referência do Symphony (``ref_Reference_Mapping_20220525.RData``, Figshare
  25052648, CC BY 4.0): 103.153 células, 5 tipos L1 e 18 tipos L2. O
  mapeamento (`run_mapping`) dá a cada célula nossa o tipo do atlas, sem usar
  a nossa integração nem a nossa clusterização — é a régua independente da
  anotação e do benchmark;
* a matriz W dos 12 programas de NMF (``NMF.W.CD4T.csv.gz``, repositório
  NMFprojection, CC BY-NC 4.0, uso acadêmico), usada em `nmf.project`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config, io

DIR = config.HEAVY_OUT / "atlas_yasumizu"
REF_RDATA = DIR / "ref_Reference_Mapping_20220525.RData"
UTILS_R = DIR / "utils_seurat.R"
NMF_W = DIR / "NMF.W.CD4T.csv.gz"

#: Os 5 tipos L1 e os 18 tipos L2 do atlas, com o nome por extenso.
L1 = ["Tnaive", "Tcm", "Tem", "Temra", "Treg"]
L2 = ["Tnaive", "TnaiveAct", "TnaiveMX1", "TnaiveSOX4", "TcmTh0", "TcmTh0Act", "TcmTh17", "TcmTh2", "TcmTfh",
      "TcmPHLDA3", "TemTh1", "TemTh117", "TemTh1pre", "TemTph", "TemraTh1", "TregNaive", "TregAct", "TregEff"]
#: Programas de NMF publicados (NMFprojection/README).
NMF_PROGRAMS = ["NMF0 Cytotoxic-F", "NMF1 Treg-F", "NMF2 Th17-F", "NMF3 Naive-F", "NMF4 Act-F",
                "NMF5 TregEff/Th2-F", "NMF6 Tfh-F", "NMF7 IFN-F", "NMF8 Cent. Mem.-F", "NMF9 Thymic Emi.-F",
                "NMF10 Tissue-F", "NMF11 Th1-F"]


def write_10x_h5(counts, barcodes, gene_ids, gene_names, path: Path) -> None:
    """Matriz células × genes em HDF5 no formato do Cell Ranger 3 (genes × células, CSC)."""
    X = sp.csr_matrix(counts)
    if not np.allclose(X.data, np.round(X.data)):
        raise ValueError("as contagens têm de ser inteiras")
    io._assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=X.data.astype(np.int32), compression="gzip")
        g.create_dataset("indices", data=X.indices.astype(np.int64), compression="gzip")
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array([X.shape[1], X.shape[0]], dtype=np.int32))
        g.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))
        ft = g.create_group("features")
        ft.create_dataset("id", data=np.asarray(gene_ids, dtype="S"))
        ft.create_dataset("name", data=np.asarray(gene_names, dtype="S"))
        ft.create_dataset("feature_type", data=np.asarray(["Gene Expression"] * X.shape[1], dtype="S"))
        ft.create_dataset("genome", data=np.asarray(["GRCh38"] * X.shape[1], dtype="S"))
        ft.create_dataset("_all_tag_keys", data=np.asarray(["genome"], dtype="S"))


def run_mapping(adata, cells: np.ndarray, out: Path = DIR / "mapeamento", timeout: int = 4 * 3600) -> pd.DataFrame:
    """Mapeia as células ``cells`` (máscara booleana) no atlas; devolve L1/L2 e a fração de votos.

    O lote do Symphony é a biblioteca. As contagens são as do SoupX
    (``layers["counts"]``), com todos os genes, porque o SCTransform usa o total
    da célula.
    """
    from . import droplets

    sub = adata[cells]
    names = [f"c{i}" for i in range(sub.n_obs)]
    write_10x_h5(sub.layers["counts"], names, sub.var_names.astype(str),
                 sub.var["gene_symbol"].astype(str), out / "counts.h5")
    pd.DataFrame({"cell": names, "batch": sub.obs["library_id"].astype(str).to_numpy()}).to_csv(
        out / "obs.csv", index=False)
    r = droplets.run_r("atlas_symphony.R", [out, REF_RDATA], timeout=timeout)
    res = pd.read_csv(out / "mapeamento.csv").set_index("cell").reindex(names)
    res.index = sub.obs["cell_key"].astype(str).to_numpy()
    res.index.name = "cell_key"
    (out / "log.txt").write_text(r.stdout + "\n" + r.stderr[-4000:])
    (out / "counts.h5").unlink(missing_ok=True)
    return res.rename(columns={"clusterL1": "atlas_L1", "clusterL1_prob": "atlas_L1_prob",
                               "clusterL2": "atlas_L2", "clusterL2_prob": "atlas_L2_prob"})


def published_W() -> pd.DataFrame:
    """Os 12 programas de NMF do atlas: genes × programas (pesos não negativos)."""
    W = pd.read_csv(NMF_W, index_col=0)
    if W.shape[1] == len(NMF_PROGRAMS):
        W.columns = NMF_PROGRAMS
    return W
