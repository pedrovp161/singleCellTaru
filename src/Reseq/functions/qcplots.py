"""
Plots de controle de qualidade.

Versões corrigidas e ampliadas dos plots do spatools que o
``src/scripts/preprocessing.ipynb`` usava, mais os do capítulo de QC do
sc-best-practices.

==================================  ========================================================
aqui                                 origem e o que mudou
==================================  ========================================================
``outlier_quality``                  ``st.pl.outlier_quality``: recebe a tabela de células
                                     (não lê arquivos), ordena por paciente com NS/ML lado a
                                     lado, cor só para o estímulo, limiares dos dois lados
                                     tirados do que foi **aplicado**, % removido por amostra
``qc_funnel``                        ``preprocessing_quality_metrics``: total por etapa com a
                                     perda em %, e as amostras em pequenos múltiplos em vez
                                     de 14 cores
``barcode_rank``                     knee plot (capítulo *raw data processing*), com knee,
                                     inflexão, piso do Cell Ranger e os barcodes recuperados
``qc_scatter``                       UMIs × genes colorido por % MT (capítulo de QC)
``threshold_histograms``             histograma por métrica com os limiares (capítulo de QC)
==================================  ========================================================
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, fmt as F, plotting
from .plotting import INK, STIMULATION_COLORS

LIBS = config.LIBRARIES


def _stim(lib: str) -> str:
    return config.stimulation_of(lib)


# ─────────────────────────────────────────────────────────────────────────────
# Distribuição por amostra com limiares
# ─────────────────────────────────────────────────────────────────────────────

def outlier_quality(ax, cells: pd.DataFrame, column: str, limits: pd.DataFrame | None = None, *,
                    removed: pd.Series | None = None, ylabel: str = "", log: bool = False,
                    order=None, width: float = 0.72):
    """Violino + caixa por biblioteca, com os limiares aplicados.

    ``cells``: uma linha por célula, com ``library_id`` e ``column``.
    ``limits``: índice = biblioteca, colunas ``min`` e ``max`` (na escala de
    ``column``; ausentes ou infinitos não são desenhados).
    ``removed``: fração removida por biblioteca, escrita acima de cada uma.
    """
    order = list(order or [l for l in LIBS if l in set(cells["library_id"])])
    pos = plotting.library_positions(order)
    for lib in order:
        v = cells.loc[cells["library_id"] == lib, column].dropna().to_numpy(dtype=float)
        if log:
            v = np.log10(np.maximum(v, 1e-9))
        if len(v) < 2:
            continue
        x = pos[lib]
        parts = ax.violinplot(v, positions=[x], widths=width, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(STIMULATION_COLORS[_stim(lib)])
            b.set_edgecolor("none")
            b.set_alpha(0.85)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([x, x], [q1, q3], color=INK["primary"], lw=3, solid_capstyle="butt", zorder=3)
        ax.plot(x, med, "o", ms=4.5, mfc=INK["surface"], mec=INK["primary"], mew=1, zorder=4)
        if limits is not None and lib in limits.index:
            for side in ("min", "max"):
                t = limits.loc[lib, side] if side in limits.columns else np.nan
                if np.isfinite(t):
                    t = np.log10(t) if log else t
                    ax.plot([x - width / 2, x + width / 2], [t, t], color=INK["primary"], lw=1.6, zorder=5)
    if removed is not None:
        top = ax.get_ylim()[1]
        for lib in order:
            if lib in removed.index and np.isfinite(removed[lib]):
                ax.annotate(F.pct(removed[lib], 1), (pos[lib], top), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=INK["secondary"], annotation_clip=False)
    ax.set_xticks([pos[l] for l in order])
    ax.set_xticklabels([l.split("_")[1] for l in order], fontsize=8)
    for pat in dict.fromkeys(config.patient_of(l) for l in order):
        xs = [pos[l] for l in order if config.patient_of(l) == pat]
        ax.annotate(pat, (np.mean(xs), 0), xycoords=("data", "axes fraction"), xytext=(0, -22),
                    textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK["secondary"])
    ax.set_ylabel(ylabel)
    if log:
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: F.n(10 ** v)))
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    return ax


def outlier_legend(fig_or_ax, *, loc="upper right", **kw):
    handles = plotting.stim_handles() + [
        mpl.lines.Line2D([], [], color=INK["primary"], lw=1.6, label="limiar aplicado"),
        mpl.lines.Line2D([], [], color=INK["primary"], lw=3, label="quartis"),
    ]
    return fig_or_ax.legend(handles=handles, loc=loc, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Funil de células por etapa
# ─────────────────────────────────────────────────────────────────────────────

def qc_funnel(steps: pd.DataFrame, *, title: str = "", ncols: int = 7):
    """``steps``: linhas = bibliotecas, colunas = etapas em ordem; valores = células.

    Em cima, o total por etapa com a perda de uma etapa para a outra. Embaixo,
    um painel por biblioteca com a fração que sobra em cada etapa (100% = a
    primeira coluna), escrita no último ponto.
    """
    steps = steps.reindex([l for l in LIBS if l in steps.index])
    etapas = list(steps.columns)
    tot = steps.sum()
    nrows = int(np.ceil(len(steps) / ncols))
    fig = plt.figure(figsize=(2.2 * ncols, 3.4 + 1.9 * nrows))
    gs = fig.add_gridspec(1 + nrows, ncols, height_ratios=[2.2] + [1] * nrows, hspace=0.75, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    xs = np.arange(len(etapas))
    ax.plot(xs, tot.to_numpy(), "-o", color=plotting.ACCENT, lw=2, ms=8, mec=INK["surface"], mew=1.5, zorder=3)
    for i, v in enumerate(tot):
        ax.annotate(F.n(v), (i, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9,
                    color=INK["primary"], fontweight="bold")
        if i:
            d = v / tot.iloc[i - 1] - 1
            ax.annotate(F.pct(d, 1, signed=True), ((i - 0.5), (v + tot.iloc[i - 1]) / 2), ha="center",
                        va="center", fontsize=8.5, color=INK["secondary"],
                        bbox=dict(boxstyle="round,pad=0.2", fc=INK["surface"], ec="none"))
    ax.set_xticks(xs)
    ax.set_xticklabels(etapas)
    ax.set_xlim(-0.4, len(etapas) - 0.6)
    ax.set_ylim(0, tot.max() * 1.18)
    plotting.thousands(ax, "y")
    ax.set_ylabel("células (soma das bibliotecas)")
    ax.grid(axis="x", visible=False)
    if title:
        ax.set_title(title)
    frac = steps.div(steps.iloc[:, 0], axis=0)
    for k, lib in enumerate(steps.index):
        a = fig.add_subplot(gs[1 + k // ncols, k % ncols])
        a.plot(xs, frac.loc[lib].to_numpy(), "-o", color=STIMULATION_COLORS[_stim(lib)], lw=1.6, ms=4)
        a.set_ylim(0, max(1.05, frac.loc[lib].max() * 1.08))
        a.set_xticks([])
        a.set_yticks([0, 0.5, 1])
        plotting.percent(a, "y")
        a.tick_params(labelsize=7)
        a.grid(axis="x", visible=False)
        last = steps.loc[lib].iloc[-1]
        a.set_title(f"{lib}\n{F.n(steps.loc[lib].iloc[0])} → {F.n(last)}", fontsize=8.5, fontweight="normal")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Knee plot
# ─────────────────────────────────────────────────────────────────────────────

CALL_COLORS = {"Cell Ranger": plotting.CATEGORICAL[0], "recuperado": plotting.CATEGORICAL[1]}


def barcode_rank(ax, totals: np.ndarray, called_cr: np.ndarray, recovered: np.ndarray | None = None, *,
                 knee: float | None = None, inflection: float | None = None,
                 floor: float | None = config.CELLRANGER_UMI_FLOOR, title: str = ""):
    """UMIs por barcode × posto, log-log. Pontos subamostrados no posto (log)."""
    o = np.argsort(-totals, kind="stable")
    t = totals[o]
    ok = t > 0
    t, o = t[ok], o[ok]
    rank = np.arange(1, len(t) + 1)
    keep = np.unique(np.concatenate([np.arange(min(len(t), 3000)),
                                     np.unique(np.geomspace(1, len(t), 4000).astype(int) - 1)]))
    cls = np.full(len(t), "outros", dtype=object)
    cls[called_cr[o]] = "Cell Ranger"
    if recovered is not None:
        cls[recovered[o]] = "recuperado"
    ax.plot(rank[keep], t[keep], color=plotting.CONTEXT, lw=1.6, zorder=1)
    for c, col in CALL_COLORS.items():
        m = cls == c
        if m.any():
            ax.scatter(rank[m], t[m], s=4, color=col, lw=0, zorder=3, rasterized=True, label=c)
    for val, lab, ls in [(knee, "knee", "-"), (inflection, "inflexão", (0, (4, 3))),
                         (floor, "piso do Cell Ranger", (0, (1, 2)))]:
        if val is not None and np.isfinite(val):
            ax.axhline(val, color=INK["axis"], lw=1, ls=ls, zorder=0)
            ax.annotate(f"{lab} {F.n(val)}", (1, val), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                        textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK["secondary"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("posto do barcode")
    ax.set_ylabel("UMIs")
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="normal")
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Dispersão e histogramas do capítulo de QC
# ─────────────────────────────────────────────────────────────────────────────

def qc_scatter(ax, total, genes, pct_mt, *, vmax: float = 15, s: float = 2, title: str = ""):
    """UMIs × genes (log-log), cor = % MT (rampa de um matiz, cortada em ``vmax``)."""
    total, genes, pct_mt = (np.asarray(v, dtype=float) for v in (total, genes, pct_mt))
    o = np.argsort(pct_mt)
    sc_ = ax.scatter(total[o], genes[o], c=np.minimum(pct_mt[o], vmax), s=s, lw=0, cmap=plotting.SEQUENTIAL,
                     vmin=0, vmax=vmax, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("UMIs")
    ax.set_ylabel("genes")
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="normal")
    return sc_


def threshold_histograms(axes, obs: pd.DataFrame, thr: dict, rules: dict, labels: dict, bins: int = 80):
    """Um histograma por métrica com os limiares, como no capítulo de QC."""
    for ax, (m, (col, side, k)) in zip(axes, rules.items()):
        v = obs[col].to_numpy(dtype=float)
        ax.hist(v, bins=bins, color=plotting.ACCENT, edgecolor=INK["surface"], linewidth=0.4)
        lo, hi = thr[m]
        for t in (lo, hi):
            if np.isfinite(t):
                ax.axvline(t, color=INK["primary"], lw=1.4)
        ax.set_title(f"{labels[m]} ({col})", fontsize=9, fontweight="normal")
        ax.set_yticks([])
        ax.grid(False)
    return axes
