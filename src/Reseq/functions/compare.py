"""
Comparação célula a célula entre a corrida antiga e o ressequenciamento.

As duas corridas são as mesmas bibliotecas físicas, então o mesmo barcode é a
mesma gota. Isso permite parear células e medir o que a profundidade mudou em
cada uma, em vez de comparar distribuições.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config, io


def load_pair(library_id: str) -> dict:
    """Matrizes antiga e nova de uma biblioteca, com as células pareadas alinhadas.

    Devolve ``Xo`` e ``Xn`` (células pareadas × genes, mesma ordem de linhas e
    colunas), ``barcodes`` (pareados), ``only_old``/``only_new`` (barcodes
    chamados como célula em só uma corrida), as matrizes completas ``Xo_all`` /
    ``Xn_all`` com seus barcodes, e ``gene_names`` / ``gene_ids``.
    """
    old = io.read_old_matrix(library_id)
    new = io.read_filtered(library_id)
    if not np.array_equal(old["gene_ids"], new["gene_ids"]):
        raise ValueError(f"{library_id}: as duas corridas não usam a mesma lista de genes")

    pos_new = {b: i for i, b in enumerate(new["barcodes"])}
    set_old = set(old["barcodes"])
    paired = [b for b in old["barcodes"] if b in pos_new]
    io_ = np.array([i for i, b in enumerate(old["barcodes"]) if b in pos_new], dtype=np.int64)
    in_ = np.array([pos_new[b] for b in paired], dtype=np.int64)

    return {
        "library_id": library_id,
        "Xo": old["X"][io_], "Xn": new["X"][in_],
        "barcodes": np.array(paired),
        "only_old": np.array([b for b in old["barcodes"] if b not in pos_new]),
        "only_new": np.array([b for b in new["barcodes"] if b not in set_old]),
        "Xo_all": old["X"], "Xn_all": new["X"],
        "barcodes_old": old["barcodes"], "barcodes_new": new["barcodes"],
        "gene_names": new["gene_names"], "gene_ids": new["gene_ids"],
    }


def per_cell(X) -> tuple[np.ndarray, np.ndarray]:
    """UMIs e genes detectados por célula."""
    X = X.tocsr()
    return np.rint(np.asarray(X.sum(axis=1)).ravel()).astype(np.int64), np.diff(X.indptr).astype(np.int64)


def per_gene(X) -> tuple[np.ndarray, np.ndarray]:
    """Células positivas e UMIs por gene."""
    Xc = X.tocsc()
    return (np.diff(Xc.indptr).astype(np.int64),
            np.rint(np.asarray(Xc.sum(axis=0)).ravel()).astype(np.int64))


def column(X, j: int) -> np.ndarray:
    return np.rint(np.asarray(X[:, j].todense()).ravel()).astype(np.int64)


def entry_consistency(Xo, Xn) -> dict:
    """Quantas entradas (célula, gene) diminuíram da corrida antiga para a nova.

    Se a corrida nova contém os reads da antiga (ou é uma reamostragem muito
    mais profunda da mesma biblioteca), quase nenhuma contagem deveria cair.
    """
    D = (Xn.astype(np.int64) - Xo.astype(np.int64)).tocoo()
    neg = D.data < 0
    pos = D.data > 0
    return {
        "entradas_nao_nulas_antiga": int(Xo.nnz),
        "entradas_que_cairam": int(neg.sum()),
        "frac_entradas_que_cairam": float(neg.sum() / max(Xo.nnz, 1)),
        "umis_perdidos": int(-D.data[neg].sum()),
        "umis_ganhos": int(D.data[pos].sum()),
        "frac_umis_perdidos": float(-D.data[neg].sum() / max(Xo.sum(), 1)),
    }


def barcode_overlap(old_barcodes: dict[str, np.ndarray],
                    new_barcodes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Fração das células antigas (colunas) presentes entre as novas (linhas)."""
    libs = list(new_barcodes)
    sets_old = {l: set(old_barcodes[l]) for l in libs}
    out = pd.DataFrame(index=libs, columns=libs, dtype=float)
    for ln in libs:
        sn = set(new_barcodes[ln])
        for lo in libs:
            out.loc[ln, lo] = len(sn & sets_old[lo]) / len(sets_old[lo])
    out.index.name = "ressequenciamento"
    out.columns.name = "corrida_antiga"
    return out


def sex_signal(X, gene_names) -> dict:
    """Fração de células com XIST e com algum gene do Y — sexo do doador."""
    idx_x = io.gene_index(gene_names, config.SEX_GENES["feminino"])
    idx_y = io.gene_index(gene_names, config.SEX_GENES["masculino"])
    Xc = X.tocsc()
    xist = (Xc[:, list(idx_x.values())].sum(axis=1) > 0)
    y = (Xc[:, list(idx_y.values())].sum(axis=1) > 0)
    return {"frac_XIST": float(np.asarray(xist).mean()),
            "frac_genes_Y": float(np.asarray(y).mean())}
