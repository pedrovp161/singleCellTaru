"""
Réplica do pré-processamento original (`src/scripts/preprocessing.ipynb`).

A regra aqui é chamar **as mesmas funções** que o notebook original chamou, com
os mesmos argumentos, e só acrescentar o que ele não registrava: de qual filtro
saiu cada célula, e onde ficaram os limiares.

Cadeia original, célula por célula do notebook:

==========  ================================================================
célula      operação
==========  ================================================================
002         ``sc.read_10x_mtx(path, var_names="gene_symbols")`` + nomes únicos
010         ``st.read(...)`` → ``var_names_make_unique`` +
            ``sc.pp.filter_genes(min_cells=1)``
013         ``st.pp.Preprocessing.run("MAD_combined", adatas_dict, k=5)``
016         ``obs["mt_outlier"] = pct_counts_mt > 8`` (criado, nunca aplicado)
024         índice de ``var`` passa a Ensembl ID; símbolos em ``gene_symbols``
031         Scrublet por amostra, ``expected_doublet_rate=0.06``, threshold
            automático; células preditas como doublet são removidas
==========  ================================================================

Não replicado, de propósito: ``cache=True`` (escreve em ``src/scripts/cache``);
o PCA/UMAP por amostra sobre contagens cruas, que só servia às figuras; e a
célula 036, que sobrescreveu ``data/filtered_h5ad``.
"""

from __future__ import annotations

import contextlib
import io as _io

import numpy as np
import pandas as pd
import scanpy as sc

from . import config


# ─────────────────────────────────────────────────────────────────────────────
# Leitura
# ─────────────────────────────────────────────────────────────────────────────

def read_cellranger_filtered(library_id: str):
    """Matriz filtrada do Cell Ranger, lida como na célula 002 do original."""
    adata = sc.read_10x_mtx(config.outs_dir(library_id) / "filtered_feature_bc_matrix",
                            var_names="gene_symbols", cache=False)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def read_old_raw(library_id: str):
    """O mesmo objeto, da corrida antiga, lido da fonte (`data/raw/<lib>/`) como
    na célula 002 do original. Serve para validar a réplica contra os números
    gravados no notebook."""
    adata = sc.read_10x_mtx(config.OLD_RAW_MTX / library_id, var_names="gene_symbols", cache=False)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def post_read(adata):
    """Equivalente a ``st.read`` para um h5ad: nomes únicos + genes com ≥ 1 célula."""
    from spatools.reading.read_ import _post_process_adata
    return _post_process_adata(adata, True)


# ─────────────────────────────────────────────────────────────────────────────
# Filtro MAD (spatools)
# ─────────────────────────────────────────────────────────────────────────────

def _mad_limits(x: np.ndarray, k: float) -> tuple[float, float, float]:
    """(mediana, MAD, k·MAD) — a mesma conta de ``spatools.processing.utils.is_outlier``.

    Atenção: é a MAD **bruta**, sem o fator 1,4826. ``k = 5`` MADs brutas
    equivalem a ~3,4 desvios-padrão numa normal.
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, mad, k * mad


def mad_flags(adata, k: float = config.MAD_K) -> tuple[pd.DataFrame, dict]:
    """Refaz, célula a célula, as máscaras do pipeline ``MAD_combined``.

    ``adata`` é o objeto depois de ``post_read``. Devolve a tabela de flags de
    todas as células de entrada e os limiares efetivos. Serve para saber *por
    que* cada célula saiu; a filtragem em si é feita pelo spatools.
    """
    a = adata.copy()
    a.var_names_make_unique()
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True)
    obs = a.obs

    lc, lg = obs["log1p_total_counts"].to_numpy(), obs["log1p_n_genes_by_counts"].to_numpy()
    mc, madc, tc = _mad_limits(lc, k)
    mg, madg, tg = _mad_limits(lg, k)
    out_c = (lc - mc) < -tc if madc > 0 else np.zeros(len(lc), bool)
    out_g = (lg - mg) < -tg if madg > 0 else np.zeros(len(lg), bool)
    keep1 = ~(out_c | out_g)

    mt = obs["pct_counts_mt"].to_numpy()
    mm, madm, tm = _mad_limits(mt[keep1], k)
    out_mt = np.zeros(len(mt), bool)
    if madm > 0:
        out_mt[keep1] = (mt[keep1] - mm) > tm

    flags = pd.DataFrame({
        "total_counts": obs["total_counts"].to_numpy(),
        "n_genes_by_counts": obs["n_genes_by_counts"].to_numpy(),
        "pct_counts_mt": mt,
        "fora_contagem": out_c, "fora_genes": out_g, "fora_mt": out_mt,
        "passou_mad": keep1 & ~out_mt,
        "mt_acima_8pct": mt > config.MT_DIAGNOSTIC_PCT,
    }, index=adata.obs_names)
    limits = {
        "umis_mediana": float(np.expm1(mc)), "umis_limiar_inferior": float(np.expm1(mc - tc)),
        "genes_mediana": float(np.expm1(mg)), "genes_limiar_inferior": float(np.expm1(mg - tg)),
        "mt_mediana": mm, "mt_limiar_superior": mm + tm,
        "mad_log_umis": madc, "mad_log_genes": madg, "mad_mt": madm,
    }
    return flags, limits


def run_original_qc(adatas: dict, k: float = config.MAD_K) -> dict:
    """Células 010–024 do original sobre um dicionário {library_id: AnnData}.

    Devolve ``{"adatas", "flags", "limits", "steps"}``: os objetos filtrados
    (índice de ``var`` em Ensembl, como saíam do notebook), as flags por
    célula, os limiares e a tabela de etapas.
    """
    import spatools as st

    lidos = {lib: post_read(a) for lib, a in adatas.items()}
    flags, limits = {}, {}
    for lib, a in lidos.items():
        flags[lib], limits[lib] = mad_flags(a, k)

    filtrados = st.pp.Preprocessing.run("MAD_combined", adatas_dict=lidos, k=k)

    steps = []
    for lib, a in filtrados.items():
        # a filtragem do spatools tem de coincidir com as máscaras refeitas
        esperado = flags[lib].index[flags[lib]["passou_mad"]]
        if not a.obs_names.equals(esperado):
            raise AssertionError(f"{lib}: máscaras refeitas ≠ saída do spatools")
        stats = a.uns[f"preprocessing_stats_{lib}"]
        a.obs["mt_outlier"] = a.obs["pct_counts_mt"] > config.MT_DIAGNOSTIC_PCT
        stats["n_after_additional_filter"] = int((~a.obs["mt_outlier"]).sum())
        a.var_names_make_unique()
        a.obs_names_make_unique()
        if not a.var["gene_ids"].iloc[0].startswith("ENS"):
            raise ValueError(f"{lib}: gene_ids não são Ensembl")
        a.var["gene_symbols"] = a.var.index
        a.var.index = a.var["gene_ids"]
        filtrados[lib] = a.copy()
        steps.append({
            "library_id": lib, "cellranger": stats["initial_n_spots"],
            "apos_contagem_genes": stats["n_after_combined_filter"],
            "apos_mt": stats["n_after_mt_filter"],
            "mt_acima_8pct_nao_aplicado": int(a.obs["mt_outlier"].sum()),
            "genes_iniciais": stats["initial_n_genes"], "genes_finais": stats["final_n_genes"],
        })
    return {"adatas": filtrados, "flags": flags, "limits": pd.DataFrame(limits).T,
            "steps": pd.DataFrame(steps).set_index("library_id")}


# ─────────────────────────────────────────────────────────────────────────────
# Scrublet, como no original
# ─────────────────────────────────────────────────────────────────────────────

def run_original_scrublet(adata, expected_doublet_rate: float = config.SCRUBLET_EXPECTED_RATE) -> dict:
    """Célula 031: ``scr.Scrublet(adata.X, expected_doublet_rate=0.06)`` e
    ``scrub_doublets()`` com tudo no padrão, threshold automático.

    Além do que o original guardava (score e chamada), devolve o threshold e o
    resumo dos doublets simulados, que é o que permite julgar se o automático
    fez sentido. Se o Scrublet não achar um mínimo no histograma simulado ele
    não chama ninguém; o original quebraria nesse caso, aqui fica registrado.
    """
    import scrublet as scr

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), np.errstate(all="ignore"):
        scrub = scr.Scrublet(adata.X, expected_doublet_rate=expected_doublet_rate)
        scores, predicted = scrub.scrub_doublets()
    falhou = predicted is None
    if falhou:
        predicted = np.zeros(adata.n_obs, dtype=bool)
    sim = np.asarray(scrub.doublet_scores_sim_)
    thr = getattr(scrub, "threshold_", np.nan)
    return {
        "scores": np.asarray(scores), "predicted": np.asarray(predicted, dtype=bool),
        "threshold": float(thr) if thr is not None else np.nan,
        "threshold_falhou": falhou,
        "sim_mediana": float(np.median(sim)), "sim_p90": float(np.quantile(sim, 0.9)),
        "obs_mediana": float(np.median(scores)),
        "sim_acima_do_threshold": float((sim > thr).mean()) if np.isfinite(thr) else np.nan,
        "sim_scores": sim, "log": buf.getvalue(),
    }


def expected_multiplet_rate(n_cells: int) -> float:
    """Taxa de multipletos do 10x 3' v3: ~0,8% por 1.000 células recuperadas."""
    return config.MULTIPLET_RATE_PER_1000 * n_cells / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# RNA ambiente
# ─────────────────────────────────────────────────────────────────────────────

def ambient_profile(library_id: str, max_umis: int = 100) -> dict:
    """Perfil do RNA ambiente de uma biblioteca, a partir das gotas vazias.

    Soma as contagens dos barcodes **não** chamados como célula que têm de 1 a
    ``max_umis`` UMIs (a mesma faixa que o SoupX usa por padrão). Devolve o
    perfil (UMIs por gene), quantos barcodes entraram, e a distribuição de UMIs
    dos barcodes de fundo por faixa.
    """
    from . import io
    raw = io.read_raw(library_id)
    cells = set(io.read_filtered(library_id)["barcodes"])
    X = raw["X"]
    tot = np.asarray(X.sum(axis=1)).ravel()
    is_cell = np.fromiter((b in cells for b in raw["barcodes"]), bool, count=len(tot))
    amb = (~is_cell) & (tot >= 1) & (tot <= max_umis)
    profile = np.asarray(X[amb].sum(axis=0)).ravel().astype(np.int64)
    bg = ~is_cell
    return {
        "profile": profile, "gene_names": raw["gene_names"], "gene_ids": raw["gene_ids"],
        "n_barcodes": int(amb.sum()), "umis": int(profile.sum()),
        "umis_em_celulas": int(tot[is_cell].sum()), "umis_fora_de_celulas": int(tot[bg].sum()),
        "fundo_100_499": int((bg & (tot >= 100) & (tot < 500)).sum()),
        "fundo_500_mais": int((bg & (tot >= 500)).sum()),
        "fundo_1_99": int((bg & (tot >= 1) & (tot < 100)).sum()),
    }


def contamination_from_markers(cell_marker: np.ndarray, cell_total: np.ndarray,
                               ambient_marker_frac: float, trim: float = 0.98) -> dict:
    """Fração de contaminação ρ pelo método das células não expressoras.

    Para um conjunto de genes que as células em questão não expressam, toda
    contagem observada veio do ambiente. Então
    ``ρ = (Σ marcador nas células / Σ UMIs nas células) / (fração do marcador no ambiente)``.
    As células acima do percentil ``trim`` da fração do marcador são excluídas,
    para que uma célula que de fato expressa (ou um doublet) não infle a conta.
    """
    frac = cell_marker / np.maximum(cell_total, 1)
    keep = frac <= np.quantile(frac, trim)
    obs = cell_marker[keep].sum() / max(cell_total[keep].sum(), 1)
    rho = obs / ambient_marker_frac if ambient_marker_frac > 0 else np.nan
    return {"rho": float(rho), "umis_marcador_celulas": int(cell_marker[keep].sum()),
            "frac_marcador_celulas": float(obs), "n_celulas": int(keep.sum())}


def pseudobulk_correlation(a: np.ndarray, b: np.ndarray, min_total: int = 20) -> float:
    """Correlação de Pearson entre dois perfis em log1p(CPM), nos genes com contagem."""
    keep = (a + b) >= min_total
    la = np.log1p(a[keep] / max(a.sum(), 1) * 1e6)
    lb = np.log1p(b[keep] / max(b.sum(), 1) * 1e6)
    return float(np.corrcoef(la, lb)[0, 1])
