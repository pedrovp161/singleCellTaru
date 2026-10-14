"""
Estilo e primitivas de figura.

A paleta e os rcParams são os da auditoria anterior
(`src/scripts/claude/functions/plotting.py`), reutilizados sem alteração. As
combinações usadas aqui foram conferidas com o validador do guia de dataviz
(modo claro, superfície #fcfcfb):

==========================  ============  ==========================
conjunto                    pares         pior ΔE (CVD / visão normal)
==========================  ============  ==========================
8 slots em ordem            adjacentes    9,1 / 19,6
estímulo (azul, laranja)    todos         24,7 / 33,6
variantes V1–V3 (slots 1–3) todos         9,2 / 24,0
classificação (1/2/3/7)     todos         9,2 / 16,3
==========================  ============  ==========================

Aqua, amarelo e magenta ficam abaixo de 3:1 contra a superfície, então toda
figura categórica leva legenda ou rótulo direto, e toda figura grava a tabela
correspondente em CSV.

Escolhas de forma que seguem do guia:

* 14 bibliotecas nunca viram 14 cores. A biblioteca vai na posição (eixo ou
  painel); a cor fica para o estímulo (2 níveis).
* antes → depois por item é haltere, um matiz em dois tons.
* UMAP com mais de 3 grupos é faceteado: um grupo em destaque por painel.
* magnitude em grade é mapa de calor de um matiz; valor com sinal é divergente
  azul ↔ vermelho com meio neutro.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, fmt as F, io, legacy

_lp = legacy.module("plotting")

CATEGORICAL = _lp.CATEGORICAL
INK = _lp.INK
SEQUENTIAL = _lp.SEQUENTIAL
DIVERGING = _lp.DIVERGING
SEVERITY_COLORS = _lp.SEVERITY_COLORS
CLASSIFICATION_COLORS = _lp.CLASSIFICATION_COLORS
STIMULATION_COLORS = _lp.STIMULATION_COLORS   # nao_estimulado azul, estimulado laranja
annotate_n = _lp.annotate_n

#: Antes → depois: um matiz, dois tons (passos 200 e 600 da rampa azul).
OLD_NEW_COLORS = {"antiga": "#9ec5f4", "nova": "#184f95"}

#: Variantes do processamento, slots 1–3.
VARIANT_COLORS = {"V1": CATEGORICAL[0], "V2": CATEGORICAL[1], "V3": CATEGORICAL[2]}
VARIANT_LABEL = {"V1": "V1 fiel", "V2": "V2 PCA corrigida", "V3": "V3 + universo completo"}

#: Classes ordenadas: um matiz, passos claros → escuros (validados com --ordinal).
ORDINAL3 = ["#86b6ef", "#2a78d6", "#0d366b"]
ORDINAL4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

#: Contexto / de-ênfase.
CONTEXT = "#d5d4cc"
ACCENT = CATEGORICAL[0]

N_DESIGN = f"{config.N_LIBRARIES} bibliotecas · {config.N_PATIENTS} pacientes"

#: Rótulo curto do grupo clínico sob o nome do paciente.
CLASS_SHORT = {"controle": "ctrl", "PB": "PB", "MB": "MB", "NP": "NP"}


def apply_style() -> None:
    """rcParams do projeto. Chame uma vez no topo de cada notebook."""
    _lp.apply_style()
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "grid.linestyle": "-",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "legend.handlelength": 1.2,
        "legend.borderaxespad": 0.2,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Gravação
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fig, notebook: str, name: str, *, description: str,
            data: pd.DataFrame | None = None, data_index: bool = True) -> Path:
    """PNG + PDF em ``resultsReseq/figures/<notebook>/`` e, se ``data`` for
    dado, a tabela da figura em ``resultsReseq/tables/<notebook>/fig_<name>.csv``.
    """
    outdir = config.FIGURES / notebook
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    pdf = outdir / f"{name}.pdf"
    io._assert_writable(png)
    fig.savefig(png, format="png")
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    io.register(stage=notebook, name=f"fig_{name}", kind="figure", path=png,
                description=description, extra={"pdf": str(pdf)})
    if data is not None:
        io.save_table(data, notebook, f"fig_{name}", description=f"Dados da figura: {description}",
                      index=data_index)
    return png


# ─────────────────────────────────────────────────────────────────────────────
# Eixo de bibliotecas
# ─────────────────────────────────────────────────────────────────────────────

def library_positions(libraries=None, gap: float = 0.6) -> dict[str, float]:
    """Posição de cada biblioteca num eixo, com espaço entre pacientes."""
    libraries = list(libraries or config.LIBRARIES)
    pos, x, prev = {}, 0.0, None
    for lib in libraries:
        pat = config.patient_of(lib)
        if prev is not None and pat != prev:
            x += gap
        pos[lib] = x
        x += 1.0
        prev = pat
    return pos


def stim_color(library_id: str) -> str:
    return STIMULATION_COLORS[config.stimulation_of(library_id)]


def stim_handles():
    return [mpl.lines.Line2D([], [], marker="o", ls="", ms=7, mfc=STIMULATION_COLORS[s],
                             mec=INK["surface"], mew=1.2, label=config.STIMULATION_LABEL[s])
            for s in config.STIMULATION_ORDER]


def stim_legend(target, *, loc="upper right", **kw):
    """Legenda do estímulo num eixo ou na figura (``target`` pode ser ``fig``)."""
    return target.legend(handles=stim_handles(), loc=loc, **kw)


def paired_dots(ax, values: pd.Series, *, ylabel: str = "", log: bool = False,
                fmt=None, label_points: bool = False):
    """Um valor por biblioteca, pareado dentro do paciente.

    x = paciente; dois pontos por paciente (NS azul, ML laranja) ligados por um
    traço. É a forma natural deste desenho: 7 pacientes × 2 condições.
    """
    xs = {p: i for i, p in enumerate(config.PATIENTS)}
    off = {"nao_estimulado": -0.16, "estimulado": 0.16}
    for p in config.PATIENTS:
        pts = []
        for s in config.STIMULATION_ORDER:
            suffix = "NS" if s == "nao_estimulado" else "ML"
            lib = f"{p}_{suffix}"
            if lib in values.index and pd.notna(values[lib]):
                pts.append((xs[p] + off[s], values[lib], s))
        if len(pts) == 2:
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                    color=INK["axis"], lw=1.2, zorder=1)
        for x, y, s in pts:
            ax.plot(x, y, "o", ms=8, mfc=STIMULATION_COLORS[s], mec=INK["surface"],
                    mew=1.5, zorder=3)
            if label_points and fmt is not None:
                ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7.5, color=INK["secondary"])
    ax.set_xticks(range(len(config.PATIENTS)))
    ax.set_xticklabels([f"{p}\n{CLASS_SHORT[config.PATIENT_CLASSIFICATION[p]]}"
                        for p in config.PATIENTS])
    ax.set_xlim(-0.6, len(config.PATIENTS) - 0.4)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.tick_params(axis="x", length=0)
    return ax


def dumbbell(ax, before: pd.Series, after: pd.Series, *, xlabel: str = "", log: bool = False,
             labels=None, before_label="corrida antiga", after_label="ressequenciamento",
             value_fmt=None, legend: bool = True, legend_loc: str = "lower right"):
    """Antes → depois por item (linhas = itens), um matiz em dois tons."""
    items = list(labels if labels is not None else before.index)
    y = np.arange(len(items))[::-1]
    for yi, it in zip(y, items):
        b, a = before[it], after[it]
        ax.plot([b, a], [yi, yi], color=INK["axis"], lw=1.6, zorder=1, solid_capstyle="round")
        ax.plot(b, yi, "o", ms=8, mfc=OLD_NEW_COLORS["antiga"], mec=INK["surface"], mew=1.5, zorder=3)
        ax.plot(a, yi, "o", ms=8, mfc=OLD_NEW_COLORS["nova"], mec=INK["surface"], mew=1.5, zorder=4)
        if value_fmt is not None:
            ax.annotate(value_fmt(b, a), (max(a, b), yi), textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=8, color=INK["secondary"])
    ax.set_yticks(y)
    ax.set_yticklabels(items)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    if legend:
        ax.legend(handles=old_new_handles(before_label, after_label), loc=legend_loc)
    return ax


def old_new_handles(before_label="corrida antiga", after_label="ressequenciamento"):
    return [
        mpl.lines.Line2D([], [], marker="o", ls="", ms=7, mfc=OLD_NEW_COLORS["antiga"],
                         mec=INK["surface"], mew=1.2, label=before_label),
        mpl.lines.Line2D([], [], marker="o", ls="", ms=7, mfc=OLD_NEW_COLORS["nova"],
                         mec=INK["surface"], mew=1.2, label=after_label),
    ]


def stacked_barh(ax, frac: pd.DataFrame, colors, *, height: float = 0.58,
                 label_min: float = 0.07, fmt=None):
    """Barras horizontais empilhadas a 100% (linhas = itens, colunas = classes).

    Segmentos separados por uma faixa da cor da superfície, não por borda.
    Rotula um segmento só quando ele tem largura para o texto.
    """
    items = list(frac.index)
    y = np.arange(len(items))[::-1]
    left = np.zeros(len(items))
    fmt = fmt or (lambda v: F.pct(v, 0))
    for col, color in zip(frac.columns, colors):
        vals = frac[col].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=height, color=color, edgecolor=INK["surface"],
                linewidth=1.6, label=str(col), zorder=3)
        rgb = mpl.colors.to_rgb(color)
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        for yi, v, l in zip(y, vals, left):
            if v >= label_min:
                ax.text(l + v / 2, yi, fmt(v), ha="center", va="center", fontsize=8,
                        color=INK["primary"] if lum > 0.5 else "white", zorder=4)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(items)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xlim(0, 1)
    percent(ax, "x")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Mapa de calor
# ─────────────────────────────────────────────────────────────────────────────

def heatmap(ax, df: pd.DataFrame, *, cmap=None, vmin=None, vmax=None, fmt: str | None = None,
            cbar_label: str = "", annotate_above: float | None = None, center: float | None = None,
            fontsize: float = 7.5, cbar: bool = True, xrotation: float = 45):
    """Grade de magnitudes. Sequencial azul por padrão; divergente se `center`.

    Rotula só as células acima de `annotate_above` (ou todas, se `fmt` for dado
    e `annotate_above` for None) — nunca um número em cada célula por reflexo.
    """
    values = df.to_numpy(dtype=float)
    if center is not None:
        cmap = cmap or DIVERGING
        span = np.nanmax(np.abs(values - center)) if vmax is None else max(abs(vmax - center), abs((vmin or 0) - center))
        norm = mpl.colors.TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)
        im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    else:
        cmap = cmap or SEQUENTIAL
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1]))
    if xrotation:
        ax.set_xticklabels(df.columns, rotation=xrotation, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(df.columns)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # separação por faixa de superfície, não por borda desenhada
    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color=INK["surface"], linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    if fmt is not None:
        lo, hi = im.norm.vmin, im.norm.vmax
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = values[i, j]
                if np.isnan(v):
                    continue
                if annotate_above is not None and v < annotate_above:
                    continue
                frac = (v - lo) / (hi - lo) if hi > lo else 0.0
                dark = frac > 0.55 if center is None else abs(frac - 0.5) > 0.3
                ax.text(j, i, F._swap(format(v, fmt)), ha="center", va="center", fontsize=fontsize,
                        color="white" if dark else INK["primary"])
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(cbar_label)
        _style_colorbar(cb)
    return im


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def _umap_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def umap_highlight_grid(coords: np.ndarray, groups, *, order=None, ncols: int = 5,
                        panel: float = 2.2, s: float = 1.2, title_fmt=None, suptitle: str = "",
                        max_points: int = 60000, seed: int = 0):
    """Um painel por grupo: o grupo em azul sobre todas as células em cinza.

    Substitui o UMAP de muitas cores — acima de 3 grupos num espalhamento,
    quaisquer dois podem ficar lado a lado e a paleta não sustenta.
    """
    groups = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    order = [str(o) for o in (order if order is not None else sorted(set(groups)))]
    n = len(order)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows + 0.5),
                             squeeze=False)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(coords.shape[0])
    if coords.shape[0] > max_points:
        idx_all = rng.choice(coords.shape[0], max_points, replace=False)
    for k, g in enumerate(order):
        ax = axes[k // ncols][k % ncols]
        m = groups == g
        ax.scatter(coords[idx_all, 0], coords[idx_all, 1], s=s, c=CONTEXT, lw=0, rasterized=True)
        ax.scatter(coords[m, 0], coords[m, 1], s=s * 1.6, c=ACCENT, lw=0, rasterized=True)
        ax.set_title(title_fmt(g, int(m.sum())) if title_fmt else f"{g}  (n={F.n(m.sum())})",
                     fontsize=9, fontweight="normal")
        _umap_axes(ax)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def umap_continuous(ax, coords: np.ndarray, values, *, label: str = "", s: float = 1.2,
                    vmin=None, vmax=None, cmap=None, order_by_value: bool = True):
    """UMAP colorido por uma magnitude (rampa de um matiz), com barra de escala."""
    values = np.asarray(values, dtype=float)
    idx = np.argsort(values) if order_by_value else np.arange(len(values))
    sc_ = ax.scatter(coords[idx, 0], coords[idx, 1], c=values[idx], s=s, lw=0,
                     cmap=cmap or SEQUENTIAL, vmin=vmin, vmax=vmax, rasterized=True)
    _umap_axes(ax)
    cb = ax.figure.colorbar(sc_, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(label)
    _style_colorbar(cb)
    return sc_


def umap_labeled(ax, coords: np.ndarray, groups, *, s: float = 1.0, title: str = ""):
    """Visão geral em cinza com o rótulo de cada grupo no seu centroide."""
    groups = pd.Series(np.asarray(groups)).astype(str)
    ax.scatter(coords[:, 0], coords[:, 1], s=s, c=CONTEXT, lw=0, rasterized=True)
    for g in sorted(groups.unique(), key=lambda v: (len(v), v)):
        m = (groups == g).to_numpy()
        cx, cy = np.median(coords[m, 0]), np.median(coords[m, 1])
        ax.text(cx, cy, g, ha="center", va="center", fontsize=9, fontweight="bold",
                color=INK["primary"],
                bbox=dict(boxstyle="round,pad=0.18", fc=INK["surface"], ec="none", alpha=0.85))
    _umap_axes(ax)
    if title:
        ax.set_title(title)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _style_colorbar(cb) -> None:
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.formatter = mpl.ticker.FuncFormatter(lambda v, _: F._swap(f"{v:g}"))
    cb.update_ticks()


def decimals(ax, axis: str = "y", nd: int = 1):
    """Rótulos do eixo com vírgula decimal."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: F.f(v, nd))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def thousands(ax, axis: str = "y"):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: F.n(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def percent(ax, axis: str = "y", decimals: int = 0):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: F.pct(v, decimals))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def note(fig, text: str, *, y: float = -0.01):
    """Nota de rodapé com n e ressalvas, em tinta secundária."""
    fig.text(0.01, y, text, ha="left", va="top", fontsize=8, color=INK["muted"], wrap=True)
