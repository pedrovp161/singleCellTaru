"""
Análise molécula a molécula a partir de ``molecule_info.h5``.

Cada linha do arquivo é uma molécula (barcode, UMI, gene) com o número de
reads que a sustentam. Isso permite três coisas que a matriz de contagens não
permite:

* **Rarefação exata.** Mantendo cada read com probabilidade ``f``, uma molécula
  com ``c`` reads continua visível com probabilidade ``1 − (1 − f)^c``. O valor
  esperado de qualquer contagem em qualquer profundidade menor sai em forma
  fechada, sem sortear nada.
* **Suporte de reads.** Numa biblioteca lida dezenas de vezes por molécula, uma
  molécula com 1 read é atípica; dá para medir quanto da matriz depende delas.
* **Testes de artefato.** Erro de UMI (mesma célula e gene, UMI vizinho de uma
  molécula bem sustentada), erro de barcode (mesmo gene e UMI noutro barcode
  com muito mais reads) e troca de índice entre bibliotecas.

``analyze_library`` faz uma passada por biblioteca e devolve tabelas pequenas;
nada do tamanho do arquivo fica na memória depois.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import brentq

from . import config, io

#: Frações de reads em que as curvas são avaliadas.
F_GRID = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 1.0])

#: Faixas de reads por molécula usadas nas tabelas de suporte.
SUPPORT_BINS = [1, 2, 3, 6, 11, 31, 101, np.inf]
SUPPORT_LABELS = ["1", "2", "3–5", "6–10", "11–30", "31–100", "> 100"]

LOW_READS = 2      # molécula de baixo suporte: até 2 reads
HIGH_READS = 10    # molécula bem sustentada: 10 reads ou mais

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_N_NEIGHBORS_D2 = 1 + 12 * 3 + 66 * 9          # UMIs de 12 bases a distância ≤ 2
_UMI_SPACE = 4 ** 12


# ─────────────────────────────────────────────────────────────────────────────
# Leitura
# ─────────────────────────────────────────────────────────────────────────────

def load_molecules(library_id: str) -> SimpleNamespace:
    mi = io.read_molecule_info(
        library_id, fields=("count", "barcode_idx", "feature_idx", "umi", "umi_type"))
    n_bc = len(mi["barcodes"])
    cell_idx = mi["cell_barcode_idx"]
    cell_slot = np.full(n_bc, -1, dtype=np.int32)
    cell_slot[cell_idx] = np.arange(len(cell_idx), dtype=np.int32)
    lib0 = mi["libraries"].get("0", {})
    return SimpleNamespace(
        library_id=library_id,
        count=mi["count"].astype(np.int32),
        bidx=mi["barcode_idx"].astype(np.int32),
        fidx=mi["feature_idx"].astype(np.int32),
        umi=mi["umi"].astype(np.int32),
        exonic=mi["umi_type"].astype(np.uint8),
        barcodes=mi["barcodes"], gene_names=mi["gene_names"],
        cell_idx=cell_idx, cell_slot=cell_slot, n_cells=len(cell_idx),
        n_genes=len(mi["gene_names"]),
        raw_reads=int(lib0.get("raw_read_pairs", 0)),
        feature_reads=int(lib0.get("feature_read_pairs", 0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rarefação
# ─────────────────────────────────────────────────────────────────────────────

def survive(c, f: float):
    """P(molécula com `c` reads é vista) mantendo cada read com probabilidade `f`."""
    if f >= 1.0:
        return np.ones(np.shape(c), dtype=np.float64)
    return -np.expm1(np.asarray(c, dtype=np.float64) * np.log1p(-f))


def expected_from_hist(hist: np.ndarray, f: float) -> float:
    """Σ_k hist[k]·(1 − (1 − f)^k), com `hist[k]` = nº de itens com k reads."""
    k = np.flatnonzero(hist)
    return float((hist[k] * survive(k, f)).sum())


def good_toulmin_doubling(hist: np.ndarray) -> float:
    """Itens novos esperados ao DOBRAR a profundidade (Good–Toulmin, t = 1).

    ``Δ = n₁ − n₂ + n₃ − …``, onde ``n_k`` é o número de itens vistos com
    exatamente k reads. É o estimador clássico de espécies não vistas; em
    t = 1 ainda é não viesado, mas a variância cresce, e itens espúrios de 1
    read entram como se fossem reais. Serve como teto do ganho.
    """
    k = np.arange(len(hist))
    sign = np.where(k % 2 == 1, 1.0, -1.0)
    sign[0] = 0.0
    return float((sign * hist).sum())


def solve_fraction(hist: np.ndarray, target: float) -> float:
    """Fração de reads `f` em que o total esperado de itens iguala `target`."""
    total = float(hist[1:].sum())
    if target >= total:
        return 1.0
    return float(brentq(lambda f: expected_from_hist(hist, f) - target, 1e-6, 1.0, xtol=1e-7))


def _popcount24_pairs(x: np.ndarray) -> np.ndarray:
    """Distância de Hamming em BASES entre UMIs codificados em 2 bits (x = a XOR b)."""
    d = (x | (x >> 1)) & 0x555555
    return (_POP8[d & 0xFF].astype(np.int16) + _POP8[(d >> 8) & 0xFF] + _POP8[(d >> 16) & 0xFF])


def _bin_support(c: np.ndarray) -> np.ndarray:
    """Contagem de moléculas por faixa de reads (SUPPORT_LABELS)."""
    edges = np.array(SUPPORT_BINS[:-1])
    idx = np.searchsorted(edges, c, side="right") - 1
    return np.bincount(idx, minlength=len(SUPPORT_LABELS))[:len(SUPPORT_LABELS)]


# ─────────────────────────────────────────────────────────────────────────────
# Uma biblioteca
# ─────────────────────────────────────────────────────────────────────────────

def analyze_library(library_id: str, *, paired: pd.DataFrame, other_cells: dict[str, set],
                    panel_genes: list[str], n_neighbor_sample: int = 200_000,
                    seed: int = config.RANDOM_STATE, verbose: bool = True) -> dict:
    """Passada única por uma biblioteca.

    ``paired`` — células pareadas desta biblioteca (colunas ``barcode``,
    ``umis_antiga``, ``genes_antiga``), vindas de C02.
    ``other_cells`` — barcodes de células das outras bibliotecas (teste de troca
    de índice).
    """
    rng = np.random.default_rng(seed)
    M = load_molecules(library_id)
    lib = library_id
    n_mol = len(M.count)
    slot_all = M.cell_slot[M.bidx]
    in_cell = slot_all >= 0
    out: dict[str, pd.DataFrame] = {}

    # ── 1. suporte de reads ──────────────────────────────────────────────────
    sup = pd.DataFrame({
        "library_id": lib, "faixa_reads": SUPPORT_LABELS,
        "moleculas_celulas": _bin_support(M.count[in_cell]),
        "moleculas_fundo": _bin_support(M.count[~in_cell]),
    })
    out["suporte"] = sup

    c_cell = M.count[in_cell]
    slot = slot_all[in_cell]
    gene = M.fidx[in_cell]
    hist_cell = np.bincount(c_cell)
    hist_all = np.bincount(M.count)
    vals = np.flatnonzero(hist_cell)
    out["hist_reads"] = pd.DataFrame({"library_id": lib, "reads": vals,
                                      "moleculas": hist_cell[vals]})

    resumo = {
        "library_id": lib, "moleculas": n_mol, "reads_feature": int(M.count.sum(dtype=np.int64)),
        "reads_brutos": M.raw_reads, "celulas": M.n_cells,
        "moleculas_em_celulas": int(in_cell.sum()),
        "reads_em_celulas": int(c_cell.sum(dtype=np.int64)),
        "reads_por_molecula_mediana": float(np.median(c_cell)),
        "reads_por_molecula_media": float(c_cell.mean()),
        "frac_1_read": float((c_cell == 1).mean()),
        "frac_ate_2_reads": float((c_cell <= LOW_READS).mean()),
        "frac_exonica": float(M.exonic[in_cell].mean()),
        "saturacao_moleculas": 1.0 - n_mol / float(M.count.sum(dtype=np.int64)),
    }

    # ── 2. matrizes por (célula, gene): moléculas e reads ────────────────────
    shape = (M.n_cells, M.n_genes)
    R = sp.coo_matrix((c_cell.astype(np.int64), (slot, gene)), shape=shape).tocsr()
    U = sp.coo_matrix((np.ones(len(c_cell), dtype=np.int32), (slot, gene)), shape=shape).tocsr()
    R.sort_indices()
    U.sort_indices()
    assert np.array_equal(R.indices, U.indices) and np.array_equal(R.indptr, U.indptr)
    row_of_entry = np.repeat(np.arange(M.n_cells, dtype=np.int32), np.diff(U.indptr))
    umis_cell = np.asarray(U.sum(axis=1)).ravel()
    genes_cell = np.diff(U.indptr)
    resumo["umis_medianos"] = float(np.median(umis_cell))
    resumo["genes_medianos"] = float(np.median(genes_cell))

    # ── 3. rarefação ─────────────────────────────────────────────────────────
    rows = []
    for f in F_GRID:
        w = survive(c_cell, f)
        u_f = np.bincount(slot, weights=w, minlength=M.n_cells)
        g_f = np.bincount(row_of_entry, weights=survive(R.data, f), minlength=M.n_cells)
        exp_all = expected_from_hist(hist_all, f)
        rows.append({
            "library_id": lib, "f": f, "reads_por_celula": f * M.raw_reads / M.n_cells,
            "umis_em_celulas": float(u_f.sum()), "umis_medianos": float(np.median(u_f)),
            "genes_medianos": float(np.median(g_f)),
            "saturacao": 1.0 - exp_all / (f * float(M.count.sum(dtype=np.int64))),
        })
        del w
    out["rarefacao"] = pd.DataFrame(rows)

    resumo["gt_moleculas_ao_dobrar"] = good_toulmin_doubling(hist_cell)
    resumo["gt_entradas_ao_dobrar"] = good_toulmin_doubling(np.bincount(R.data))
    resumo["entradas_celula_gene"] = int(U.nnz)

    # ── 4. limiar mínimo de reads por UMI ────────────────────────────────────
    rows = []
    for k in (1, 2, 3):
        m = c_cell >= k
        Uk = sp.coo_matrix((np.ones(int(m.sum()), dtype=np.int32), (slot[m], gene[m])),
                           shape=shape).tocsr()
        uk = np.asarray(Uk.sum(axis=1)).ravel()
        rows.append({"library_id": lib, "min_reads": k, "umis_em_celulas": int(m.sum()),
                     "umis_medianos": float(np.median(uk)),
                     "genes_medianos": float(np.median(np.diff(Uk.indptr))),
                     "entradas": int(Uk.nnz)})
        del Uk
    out["limiar_reads"] = pd.DataFrame(rows)

    # ── 5. profundidade antiga por rarefação, e teste "mesma biblioteca" ─────
    bc_cells = pd.Index(M.barcodes[M.cell_idx])
    pr = paired.set_index("barcode")
    pslot = bc_cells.get_indexer(pr.index)
    assert (pslot >= 0).all(), "célula pareada ausente do molecule_info"
    is_paired_cell = np.zeros(M.n_cells, dtype=bool)
    is_paired_cell[pslot] = True
    mol_paired = is_paired_cell[slot]
    hist_p = np.bincount(c_cell[mol_paired])
    f_old = solve_fraction(hist_p, float(pr["umis_antiga"].sum()))
    resumo["f_antiga"] = f_old
    resumo["frac_umis_ja_na_antiga"] = expected_from_hist(hist_cell, f_old) / len(c_cell)
    resumo["reads_por_celula_antiga_est"] = f_old * M.raw_reads / M.n_cells

    u_pred = np.bincount(slot, weights=survive(c_cell, f_old), minlength=M.n_cells)[pslot]
    g_pred = np.bincount(row_of_entry, weights=survive(R.data, f_old), minlength=M.n_cells)[pslot]
    uo = pr["umis_antiga"].to_numpy(dtype=float)
    go = pr["genes_antiga"].to_numpy(dtype=float)
    resumo["corr_log_umis_previsto_antigo"] = float(np.corrcoef(np.log(u_pred), np.log(uo))[0, 1])
    resumo["razao_mediana_umis_antigo_previsto"] = float(np.median(uo / u_pred))
    resumo["razao_mediana_genes_antigo_previsto"] = float(np.median(go / g_pred))
    resumo["genes_medianos_previsto_antiga"] = float(np.median(g_pred))
    resumo["genes_medianos_obs_antiga"] = float(np.median(go))

    # entradas com UMA molécula na nova: estavam na antiga? por nº de reads da molécula
    old = io.read_old_matrix(lib)
    old_pos = pd.Index(old["barcodes"]).get_indexer(pr.index)
    assert (old_pos >= 0).all()
    Xo = old["X"][old_pos].tocoo()
    key_old = pslot[Xo.row].astype(np.int64) * M.n_genes + Xo.col
    del old, Xo
    single = (U.data == 1) & is_paired_cell[row_of_entry]
    key_new = row_of_entry[single].astype(np.int64) * M.n_genes + U.indices[single]
    reads_single = R.data[single]
    absent = ~np.isin(key_new, key_old)
    edges = np.array(SUPPORT_BINS[:-1])
    b = np.searchsorted(edges, reads_single, side="right") - 1
    pred_absent = (1.0 - f_old) ** reads_single.astype(np.float64)
    out["ausencia_na_antiga"] = pd.DataFrame({
        "library_id": lib, "faixa_reads": SUPPORT_LABELS,
        "entradas": np.bincount(b, minlength=len(SUPPORT_LABELS)),
        "ausentes_na_antiga": np.bincount(b, weights=absent, minlength=len(SUPPORT_LABELS)),
        "ausentes_previstas": np.bincount(b, weights=pred_absent, minlength=len(SUPPORT_LABELS)),
    })
    del key_old, key_new, absent, pred_absent

    # ── 6. painel de genes: reads por (célula, gene) ─────────────────────────
    gi = io.gene_index(M.gene_names, panel_genes)
    rows = []
    Ucsc, Rcsc = U.tocsc(), R.tocsc()
    # reads máximos por entrada: a partir das moléculas do painel
    sel = np.isin(gene, np.array(list(gi.values())))
    mx = pd.DataFrame({"slot": slot[sel], "gene": gene[sel], "c": c_cell[sel]}) \
        .groupby(["slot", "gene"])["c"].max()
    for g, j in gi.items():
        a, e = Ucsc.indptr[j], Ucsc.indptr[j + 1]
        if e == a:
            continue
        cells_j = Ucsc.indices[a:e]
        rows.append(pd.DataFrame({
            "library_id": lib, "barcode": bc_cells[cells_j], "gene": g,
            "umis": Ucsc.data[a:e], "reads": Rcsc.data[a:e],
            "reads_max": mx.reindex(pd.MultiIndex.from_arrays(
                [cells_j, np.full(len(cells_j), j)])).to_numpy(),
        }))
    out["painel"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["library_id", "barcode", "gene", "umis", "reads", "reads_max"])
    del Ucsc, Rcsc

    # ── 7. por gene: moléculas, baixo suporte, exônicas ──────────────────────
    low = c_cell <= LOW_READS
    out["genes"] = pd.DataFrame({
        "library_id": lib, "gene_idx": np.arange(M.n_genes),
        "moleculas": np.bincount(gene, minlength=M.n_genes),
        "baixo_suporte": np.bincount(gene, weights=low, minlength=M.n_genes).astype(np.int64),
        "exonicas": np.bincount(gene, weights=M.exonic[in_cell], minlength=M.n_genes).astype(np.int64),
    }).query("moleculas > 0")

    # ── 8. artefatos ─────────────────────────────────────────────────────────
    # (a) erro de barcode: mesma (gene, UMI) noutro barcode com ≥ 10× mais reads
    key = (M.fidx.astype(np.int64) << 24) | M.umi.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks, cs = key[order], M.count[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    sizes = np.diff(np.r_[start, len(ks)])
    gmax = np.repeat(np.maximum.reduceat(cs, start), sizes)
    is_copy = np.zeros(n_mol, dtype=bool)
    is_copy[order] = (np.repeat(sizes, sizes) > 1) & (cs.astype(np.int64) * 10 <= gmax)
    del key, order, ks, cs, start, sizes, gmax
    low_all = M.count <= LOW_READS
    resumo["copia_barcode_celulas"] = float(is_copy[in_cell].mean())
    resumo["copia_barcode_celulas_baixo_suporte"] = float(is_copy[in_cell & low_all].mean())
    resumo["copia_barcode_fundo"] = float(is_copy[~in_cell].mean())
    resumo["copia_barcode_fundo_baixo_suporte"] = float(is_copy[~in_cell & low_all].mean())
    del is_copy, low_all

    # (b) erro de UMI: molécula de baixo suporte com UMI vizinho (≤ 2 bases) de
    #     uma molécula bem sustentada na mesma célula e no mesmo gene
    umi_cell = M.umi[in_cell]
    gkey = slot.astype(np.int64) * M.n_genes + gene
    order = np.argsort(gkey, kind="stable")
    gk, cc, uu = gkey[order], c_cell[order], umi_cell[order]
    start = np.r_[0, np.flatnonzero(np.diff(gk)) + 1]
    end = np.r_[start[1:], len(gk)]
    grp = np.repeat(np.arange(len(start), dtype=np.int32), end - start)
    n_high = np.bincount(grp, weights=(cc >= HIGH_READS), minlength=len(start)).astype(np.int32)
    low_pos = np.flatnonzero(cc <= LOW_READS)
    has_partner = n_high[grp[low_pos]] > 0
    resumo["baixo_suporte_com_parceira"] = float(has_partner.mean()) if len(low_pos) else np.nan
    cand = low_pos[has_partner]
    if len(cand) > n_neighbor_sample:
        cand = rng.choice(cand, n_neighbor_sample, replace=False)
    dmin = np.empty(len(cand), dtype=np.int16)
    chance = np.empty(len(cand), dtype=np.float64)
    for i, p in enumerate(cand):
        s, e = start[grp[p]], end[grp[p]]
        hi = uu[s:e][cc[s:e] >= HIGH_READS]
        dmin[i] = _popcount24_pairs(hi ^ uu[p]).min()
        chance[i] = 1.0 - (1.0 - _N_NEIGHBORS_D2 / _UMI_SPACE) ** len(hi)
    if len(cand):
        resumo["vizinho_umi_d1"] = float((dmin == 1).mean())
        resumo["vizinho_umi_d2"] = float((dmin == 2).mean())
        resumo["vizinho_umi_ate_2"] = float((dmin <= 2).mean())
        resumo["vizinho_umi_ate_2_acaso"] = float(chance.mean())
    resumo["n_amostra_vizinhos"] = int(len(cand))
    del gkey, order, gk, cc, uu, grp

    # (c) troca de índice: barcodes de fundo que são célula noutra biblioteca
    mol_bc = np.bincount(M.bidx, minlength=len(M.barcodes))
    reads_bc = np.bincount(M.bidx, weights=M.count, minlength=len(M.barcodes))
    is_cell_bc = M.cell_slot >= 0
    bc_index = pd.Index(M.barcodes)
    rows = []
    union_other = np.zeros(len(M.barcodes), dtype=bool)
    for src, cells in other_cells.items():
        pos = bc_index.get_indexer(list(cells))
        pos = pos[pos >= 0]
        pos = pos[~is_cell_bc[pos]]
        union_other[pos] = True
        rows.append({"library_id": lib, "origem": src, "barcodes": int(len(pos)),
                     "moleculas_por_barcode": float(mol_bc[pos].mean()) if len(pos) else np.nan,
                     "reads_por_molecula": float(reads_bc[pos].sum() / max(mol_bc[pos].sum(), 1))})
    rest = ~is_cell_bc & ~union_other
    rows.append({"library_id": lib, "origem": "demais barcodes de fundo", "barcodes": int(rest.sum()),
                 "moleculas_por_barcode": float(mol_bc[rest].mean()),
                 "reads_por_molecula": float(reads_bc[rest].sum() / max(mol_bc[rest].sum(), 1))})
    out["troca_de_indice"] = pd.DataFrame(rows)

    # ── 9. resumo por barcode (células e fundo com ≥ 50 moléculas) ───────────
    exo_bc = np.bincount(M.bidx, weights=M.exonic, minlength=len(M.barcodes))
    keep = is_cell_bc | (mol_bc >= 50)
    out["barcodes"] = pd.DataFrame({
        "library_id": lib, "barcode": M.barcodes[keep], "celula": is_cell_bc[keep],
        "moleculas": mol_bc[keep].astype(np.int64), "reads": reads_bc[keep].astype(np.int64),
        "frac_intronica": 1.0 - exo_bc[keep] / mol_bc[keep],
    })

    out["resumo"] = pd.DataFrame([resumo])
    if verbose:
        print(f"  {lib}: {n_mol / 1e6:.1f} M moléculas, {resumo['reads_por_molecula_mediana']:.0f} "
              f"reads/molécula (mediana), f_antiga = {f_old:.3f}")
    return out
