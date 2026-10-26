"""
Figuras do benchmark de integração (P07), no formato das do scIB.

* `results_table` — a tabela-resumo das Figs. 2a e 3b do artigo: uma linha
  por execução, ordenada pelo score geral; barras para os scores agregados;
  um círculo por métrica (tamanho e cor = score escalonado), azul para a
  remoção de lote e rosa para a conservação biológica, como no artigo.
* `batch_bio_scatter` — Fig. 3a / Extended Data Fig. 4: remoção de lote ×
  conservação biológica; cor = tipo de saída, forma = escala, tamanho = geral.
* `umap_grid` — Figs. 2b–c: não integrado + melhores métodos; linhas = rótulo,
  estímulo e mistura local de bibliotecas.
* `runtime_bars` — Extended Data Fig. 7: tempo e memória por ajuste.
* `stimulus_scatter` — métrica do estudo: estímulo preservado × score geral.

Regras de `plotting.py`: no máximo 3 matizes num espalhamento, legenda sempre
que houver ≥ 2 séries, texto em tinta (nunca na cor da série).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import benchmark as bm, fmt as F, plotting

INK = plotting.INK
CAT = plotting.CATEGORICAL

#: Rampas de um matiz, claro → escuro. A azul é a do projeto; a rosa tem os
#: mesmos degraus de claridade, a partir do slot magenta da paleta.
BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PINK_STEPS = ["#fadbe7", "#f4b5cc", "#ec8eb2", "#de6595", "#bd4677", "#90305a", "#62203f"]
BATCH_CMAP = LinearSegmentedColormap.from_list("lote", BLUE_STEPS)
BIO_CMAP = LinearSegmentedColormap.from_list("bio", PINK_STEPS)
OVERALL_COLOR = INK["secondary"]

#: Tipo de saída → cor (slots 1–3, validados para todos os pares); escala → forma.
OUTPUT_COLORS = {"embedding": CAT[0], "genes": CAT[1], "grafo": CAT[2]}
OUTPUT_LABEL = {"embedding": "embedding", "genes": "genes", "grafo": "grafo"}
OUTPUT_LEGEND = {"embedding": "embedding", "genes": "genes (matriz corrigida)", "grafo": "grafo"}
SCALE_MARKERS = {"escalado": "o", "sem escala": "s", "—": "^"}

BAND = "#f0efec"
GPU_FITS = {"scvi", "scanvi", "scgen"}


def _short(key: str) -> str:
    r = bm.RUN[key]
    if r.dims:
        return f"{r.method.split(' ·')[0]} {r.dims}d (P05)"
    extra = {"escalado": " · esc.", "sem escala": "", "—": ""}[r.scaling]
    out = {"embedding": "", "genes": " (genes)", "grafo": ""}[r.output] if r.method == "Scanorama" else ""
    return f"{r.label}{out}{extra}"


# ─────────────────────────────────────────────────────────────────────────────
# Tabela-resumo (Figs. 2a e 3b)
# ─────────────────────────────────────────────────────────────────────────────

def results_table(summary: pd.DataFrame, scaled: pd.DataFrame, *, chosen: str | None = None,
                  title: str = "", note: str = ""):
    """Tabela do scIB: execuções ordenadas pelo geral; barras e círculos.

    ``summary``: colunas Overall score / Batch correction / Bio conservation;
    ``scaled``: métricas escalonadas (0–1), NaN = não se aplica.
    """
    order = summary.sort_values("Overall score", ascending=False).index.tolist()
    n = len(order)
    agg = [("Overall score", "Geral", OVERALL_COLOR), ("Batch correction", "Remoção\nde lote", BLUE_STEPS[4]),
           ("Bio conservation", "Conservação\nbiológica", PINK_STEPS[4])]
    x_rank, x_name, x_out, x_scale = 0.25, 0.55, 3.55, 4.75
    x_bar0, bar_w, bar_gap = 5.75, 1.25, 0.35
    x_met0 = x_bar0 + 3 * (bar_w + bar_gap) + 0.35
    step = 0.72
    xb = [x_met0 + i * step for i in range(len(bm.BATCH_METRICS))]
    xo = [xb[-1] + 0.45 + (i + 1) * step for i in range(len(bm.BIO_METRICS))]
    x_end = xo[-1] + 0.5
    fig_w = x_end * 0.74
    fig_h = 0.36 * n + 4.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, x_end)
    ax.set_ylim(n - 0.4, -5.8)
    ax.axis("off")

    smin, smax = 8, 165
    for i, k in enumerate(order):
        r = bm.RUN[k]
        if k == chosen:
            ax.add_patch(Rectangle((0.05, i - 0.48), x_end - 0.1, 0.96, color=BAND, zorder=0, lw=0))
        elif i % 2 == 1:
            ax.add_patch(Rectangle((0.05, i - 0.48), x_end - 0.1, 0.96, color=INK["surface"], zorder=0, lw=0))
        muted = r.method.startswith("Não integrado")
        tcolor = INK["muted"] if muted else INK["primary"]
        weight = "bold" if k == chosen else "normal"
        ax.text(x_rank, i, str(i + 1), ha="center", va="center", fontsize=9, color=INK["secondary"])
        name = r.label + ("  ◂ escolhido" if k == chosen else "")
        ax.text(x_name, i, name, ha="left", va="center", fontsize=9.5, color=tcolor, fontweight=weight,
                style="italic" if muted else "normal")
        ax.text(x_out, i, OUTPUT_LABEL[r.output], ha="left", va="center", fontsize=8.5, color=INK["secondary"])
        ax.text(x_scale, i, r.scaling, ha="left", va="center", fontsize=8.5, color=INK["secondary"])
        for j, (col, _, color) in enumerate(agg):
            x0 = x_bar0 + j * (bar_w + bar_gap)
            v = summary.loc[k, col]
            ax.add_patch(Rectangle((x0, i - 0.3), bar_w, 0.6, color=INK["grid"], lw=0, zorder=1, alpha=0.45))
            if np.isfinite(v):
                ax.add_patch(Rectangle((x0, i - 0.3), bar_w * max(v, 0), 0.6, color=color, lw=0, zorder=2))
                inside = v > 0.62
                ax.text(x0 + (bar_w * v - 0.06 if inside else bar_w * max(v, 0) + 0.06), i, F.f(v, 2),
                        ha="right" if inside else "left", va="center", fontsize=8,
                        color="white" if inside else INK["primary"], zorder=3)
        for xs, mets, cmap in ((xb, bm.BATCH_METRICS, BATCH_CMAP), (xo, bm.BIO_METRICS, BIO_CMAP)):
            for x, mname in zip(xs, mets):
                v = scaled.loc[k, mname] if mname in scaled.columns else np.nan
                if not np.isfinite(v):
                    ax.text(x, i, "–", ha="center", va="center", fontsize=9, color=INK["axis"])
                    continue
                ax.scatter([x], [i], s=smin + (smax - smin) * v, color=cmap(0.12 + 0.88 * v), lw=0, zorder=3)

    # cabeçalhos
    hy = -0.85
    for txt, x in (("#", x_rank), ("Método", x_name), ("Saída", x_out), ("Escala", x_scale)):
        ax.text(x, hy, txt, ha="center" if txt == "#" else "left", va="bottom", fontsize=9.5,
                color=INK["primary"], fontweight="bold")
    for j, (_, lab, _) in enumerate(agg):
        x0 = x_bar0 + j * (bar_w + bar_gap)
        ax.text(x0 + bar_w / 2, hy, lab, ha="center", va="bottom", fontsize=9, color=INK["primary"], fontweight="bold")
    for xs, mets in ((xb, bm.BATCH_METRICS), (xo, bm.BIO_METRICS)):
        for x, mname in zip(xs, mets):
            ax.text(x, hy, mname, ha="left", va="bottom", rotation=45, rotation_mode="anchor", fontsize=8.5,
                    color=INK["primary"])
    gy = -5.3
    for xs, lab, color in ((xb, "Remoção de lote", BLUE_STEPS[4]), (xo, "Conservação biológica", PINK_STEPS[4])):
        ax.plot([xs[0] - 0.3, xs[-1] + 0.3], [gy + 0.35, gy + 0.35], color=color, lw=3, solid_capstyle="round")
        ax.text((xs[0] + xs[-1]) / 2, gy, lab, ha="center", va="bottom", fontsize=10, color=INK["primary"],
                fontweight="bold")
    ax.plot([x_bar0, x_bar0 + 3 * bar_w + 2 * bar_gap], [gy + 0.35, gy + 0.35], color=OVERALL_COLOR, lw=3,
            solid_capstyle="round")
    ax.text(x_bar0 + 1.5 * bar_w + bar_gap, gy, "Scores agregados", ha="center", va="bottom", fontsize=10,
            color=INK["primary"], fontweight="bold")
    ax.plot([0.05, x_end - 0.05], [-0.55, -0.55], color=INK["axis"], lw=0.8)

    # legenda dos círculos
    ly = n + 0.25
    ax.text(x_met0 - 0.3, ly, "score escalonado:", ha="right", va="center", fontsize=8.5, color=INK["secondary"])
    for j, v in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        x = x_met0 + j * 0.55
        ax.scatter([x], [ly], s=smin + (smax - smin) * v, color=BATCH_CMAP(0.12 + 0.88 * v), lw=0)
        ax.scatter([x + 2.9], [ly], s=smin + (smax - smin) * v, color=BIO_CMAP(0.12 + 0.88 * v), lw=0)
        ax.text(x, ly + 0.55, F.f(v, 2), ha="center", va="center", fontsize=7.5, color=INK["muted"])
        ax.text(x + 2.9, ly + 0.55, F.f(v, 2), ha="center", va="center", fontsize=7.5, color=INK["muted"])
    ax.text(xo[2], ly, "–  não se aplica à saída (como no artigo)", ha="left", va="center", fontsize=8.5,
            color=INK["secondary"])
    ax.set_ylim(n + 1.0, -5.8)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    if note:
        plotting.note(fig, note, y=0.005)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Remoção de lote × conservação biológica (Fig. 3a / ED Fig. 4)
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_runs(ax, x: pd.Series, y: pd.Series, size: pd.Series | None, *, chosen=None, labels=True):
    from adjustText import adjust_text

    texts = []
    for k in x.index:
        r = bm.RUN[k]
        s = 70 if size is None else 30 + 260 * max(size[k], 0)
        face = "none" if r.uses_labels else OUTPUT_COLORS[r.output]
        ax.scatter([x[k]], [y[k]], s=s, marker=SCALE_MARKERS[r.scaling], facecolor=face,
                   edgecolor=OUTPUT_COLORS[r.output] if r.uses_labels else INK["surface"],
                   lw=1.8 if r.uses_labels else 1.2, zorder=3)
        if k == chosen:
            ax.scatter([x[k]], [y[k]], s=s * 2.6 + 60, marker="o", facecolor="none", edgecolor=INK["primary"],
                       lw=1.3, zorder=2)
        if labels:
            texts.append(ax.text(x[k], y[k], _short(k), fontsize=8, color=INK["primary"], zorder=4))
    if texts:
        adjust_text(texts, x=x.to_numpy(float), y=y.to_numpy(float), ax=ax, expand=(1.3, 1.8),
                    force_text=(0.3, 0.6), force_static=(0.4, 0.8), min_arrow_len=8,
                    arrowprops=dict(arrowstyle="-", color=INK["axis"], lw=0.6))


def _scatter_legend(fig, ax, *, size_note: bool = True, chosen: bool = True, extra=()):
    h = [Line2D([], [], marker="o", ls="", ms=8, mfc=c, mec=INK["surface"], label=OUTPUT_LEGEND[o])
         for o, c in OUTPUT_COLORS.items()]
    h += [Line2D([], [], marker=m, ls="", ms=8, mfc=INK["muted"], mec=INK["surface"], label=f"{s}")
          for s, m in SCALE_MARKERS.items() if s != "—"]
    h += [Line2D([], [], marker="^", ls="", ms=8, mfc=INK["muted"], mec=INK["surface"], label="escala não se aplica"),
          Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=INK["muted"], mew=1.6, label="usa rótulo (não elegível)")]
    if chosen:
        h.append(Line2D([], [], marker="o", ls="", ms=12, mfc="none", mec=INK["primary"], mew=1.2, label="escolhido"))
    h += list(extra)
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5,
              title="saída · escala" + (" · tamanho = geral" if size_note else ""), title_fontsize=8.5,
              alignment="left")


def batch_bio_scatter(summary: pd.DataFrame, *, chosen: str | None = None, reference: str = bm.REFERENCE,
                      title: str = "", note: str = ""):
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    x, y = summary["Batch correction"], summary["Bio conservation"]
    ax.axvline(x[reference], color=plotting.CATEGORICAL[7], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(y[reference], color=plotting.CATEGORICAL[7], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axvline(x.median(), color=INK["axis"], lw=1, zorder=1)
    ax.axhline(y.median(), color=INK["axis"], lw=1, zorder=1)
    lo = min(x.min(), y.min()) - 0.05
    ax.set_xlim(max(-0.02, lo), 1.0)
    ax.set_ylim(max(-0.02, lo), 1.0)
    _scatter_runs(ax, x, y, summary["Overall score"], chosen=chosen)
    ax.set_xlabel("remoção de lote (média das métricas escalonadas)")
    ax.set_ylabel("conservação biológica (média das métricas escalonadas)")
    plotting.decimals(ax, "x", 1)
    plotting.decimals(ax, "y", 1)
    ax.text(x[reference], ax.get_ylim()[1], " não integrado (escalado)", ha="left", va="top", fontsize=8,
            color=INK["secondary"])
    ax.text(ax.get_xlim()[1], y.median(), "mediana ", ha="right", va="bottom", fontsize=8, color=INK["muted"])
    _scatter_legend(fig, ax)
    if title:
        ax.set_title(title, loc="left")
    if note:
        plotting.note(fig, note, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def stimulus_scatter(summary: pd.DataFrame, retention: pd.Series, *, chosen: str | None = None,
                     current: str = bm.CURRENT, floor: float | None = None, title: str = "", note: str = ""):
    """Métrica do estudo: fração da separação NS × ML preservada × score geral."""
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    x, y = summary["Overall score"], retention.reindex(summary.index)
    ax.axhline(1.0, color=plotting.CATEGORICAL[7], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(y[current], color=INK["secondary"], lw=1, zorder=1)
    refs = [Line2D([], [], color=plotting.CATEGORICAL[7], lw=1, ls=(0, (4, 3)), label="não integrado (escalado) = 100%"),
            Line2D([], [], color=INK["secondary"], lw=1, label="Harmony atual (P05)")]
    if floor is not None:
        ax.axhspan(min(y.min(), floor) - 0.5, floor, color=BAND, zorder=0, lw=0)
        refs.append(mpl.patches.Patch(color=BAND, label="faixa vetada"))
    dx = x.max() - x.min()
    ax.set_xlim(x.min() - 0.06 * dx, x.max() + 0.08 * dx)
    y_lo = min(y.min(), floor if floor is not None else y.min())
    ax.set_ylim(y_lo - 0.08, max(y.max(), 1.0) + 0.08)
    _scatter_runs(ax, x, y, None, chosen=chosen)
    ax.set_xlabel("score geral (0,4 × lote + 0,6 × bio)")
    ax.set_ylabel("separação NS × ML preservada\n(fração do não integrado)")
    plotting.decimals(ax, "x", 2)
    plotting.percent(ax, "y")
    _scatter_legend(fig, ax, size_note=False, extra=refs)
    if title:
        ax.set_title(title, loc="left")
    if note:
        plotting.note(fig, note, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UMAPs (Figs. 2b–c)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_HIGHLIGHT = {"CD4 naive": CAT[0], "CD4 memória": CAT[1], "Treg": CAT[2]}


def umap_grid(umaps: dict, keys: list, obs: pd.DataFrame, ilisi: dict, *, titles: dict | None = None,
              ilisi_max: float | None = None, panel: float = 2.6, s: float = 0.6, seed: int = 0, note: str = ""):
    """Colunas = execuções; linhas = rótulo Azimuth, estímulo, iLISI local de bibliotecas."""
    rng = np.random.default_rng(seed)
    n = len(obs)
    perm = rng.permutation(n)
    lab = obs[bm.LABEL].astype(object).to_numpy()
    stim = obs["stimulation"].astype(str).to_numpy()
    rows = ["rótulo (Azimuth)", "estímulo", "mistura local de bibliotecas (iLISI)"]
    fig, axes = plt.subplots(3, len(keys), figsize=(panel * len(keys) + 1.6, panel * 3 + 0.9), squeeze=False)
    vmax = ilisi_max or max(np.nanpercentile(v, 99) for v in ilisi.values())
    for j, k in enumerate(keys):
        um = umaps[k]
        # rótulo: três grupos principais em cor; outros rótulos em cinza escuro; sem rótulo em cinza claro
        ax = axes[0][j]
        c = np.full(n, plotting.CONTEXT, dtype=object)
        other = pd.notna(lab)
        c[other] = INK["muted"]
        for g, col in LABEL_HIGHLIGHT.items():
            c[lab == g] = col
        ax.scatter(um[perm, 0], um[perm, 1], c=c[perm], s=s, lw=0, rasterized=True)
        ax.set_title((titles or {}).get(k, _short(k)), fontsize=9.5, fontweight="bold")
        ax = axes[1][j]
        c = np.array([plotting.STIMULATION_COLORS.get(v, INK["muted"]) for v in stim], dtype=object)
        ax.scatter(um[perm, 0], um[perm, 1], c=c[perm], s=s, lw=0, rasterized=True)
        ax = axes[2][j]
        v = ilisi[k]
        idx = np.argsort(v)
        sc_ = ax.scatter(um[idx, 0], um[idx, 1], c=v[idx], s=s, lw=0, cmap=plotting.SEQUENTIAL, vmin=1, vmax=vmax,
                         rasterized=True)
        for i in range(3):
            plotting._umap_axes(axes[i][j])
    for i, r in enumerate(rows):
        axes[i][0].set_ylabel(r, fontsize=9, color=INK["secondary"])
    h1 = [Line2D([], [], marker="o", ls="", ms=6, mfc=col, mec="none", label=g) for g, col in LABEL_HIGHLIGHT.items()]
    h1 += [Line2D([], [], marker="o", ls="", ms=6, mfc=INK["muted"], mec="none", label="outros rótulos"),
           Line2D([], [], marker="o", ls="", ms=6, mfc=plotting.CONTEXT, mec="none", label="sem rótulo")]
    axes[0][-1].legend(handles=h1, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    h2 = [Line2D([], [], marker="o", ls="", ms=6, mfc=col, mec="none",
                 label={"nao_estimulado": "não estimulado (NS)", "estimulado": "estimulado (ML)"}[g])
          for g, col in plotting.STIMULATION_COLORS.items()]
    axes[1][-1].legend(handles=h2, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    cb = fig.colorbar(sc_, ax=axes[2][-1], fraction=0.05, pad=0.04)
    cb.set_label("bibliotecas efetivas\nentre os vizinhos", fontsize=8)
    plotting._style_colorbar(cb)
    cb.ax.tick_params(labelsize=8)
    if note:
        plotting.note(fig, note, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tempo e memória (ED Fig. 7)
# ─────────────────────────────────────────────────────────────────────────────

def runtime_bars(meta: pd.DataFrame, *, note: str = ""):
    """Uma barra por ajuste (fora o não integrado): tempo (escala log) e memória de pico adicional."""
    m = meta.drop_duplicates("ajuste").set_index("ajuste")
    m = m.drop(index=[f for f in ("pca", "pca_esc") if f in m.index])   # o não integrado não tem ajuste
    names = {}
    for r in bm.RUNS:
        names.setdefault(r.fit, (r.method if r.method != "Não integrado" else "Não integrado (PCA)")
                         + {"escalado": " · esc.", "sem escala": "", "—": ""}[r.scaling])
    m = m.sort_values("tempo_s")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 0.34 * len(m) + 1.8), sharey=True)
    y = np.arange(len(m))
    a1.barh(y, m["tempo_s"] / 60, color=CAT[0], height=0.62)
    a1.set_xscale("log")
    a1.set_xlabel("tempo do ajuste (min, escala log)")
    a1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: F._swap(f"{v:g}")))
    for yi, (k, r) in zip(y, m.iterrows()):
        tag = []
        if k in GPU_FITS:
            tag.append("GPU")
        if bm.RUN[[x.key for x in bm.RUNS if x.fit == k][0]].language == "R":
            tag.append("R")
        t = r["tempo_s"]
        lab_t = f"{F.n(t)} s" if t < 60 else f"{F.f(t / 60, 1)} min"
        a1.text(t / 60 * 1.08, yi, lab_t + (f"  ({', '.join(tag)})" if tag else ""),
                va="center", fontsize=8, color=INK["primary"])
    a2.barh(y, m["memoria_pico_gb"].clip(lower=0), color=CAT[0], height=0.62)
    for yi, v in zip(y, m["memoria_pico_gb"].clip(lower=0)):
        a2.text(v + 0.05, yi, F.f(v, 1), va="center", fontsize=8, color=INK["primary"])
    a2.set_xlabel("memória de pico além da base (GB, processo + R)")
    a1.set_yticks(y)
    a1.set_yticklabels([names[k] for k in m.index])
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    a1.set_xlim(right=a1.get_xlim()[1] * 4)
    a2.set_xlim(right=max(1.0, m["memoria_pico_gb"].max()) * 1.25)
    if note:
        plotting.note(fig, note, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tabela-resumo genérica (P11): os mesmos círculos e barras, grupos à escolha
# ─────────────────────────────────────────────────────────────────────────────

GROUP_STEPS = [BLUE_STEPS, PINK_STEPS,
               ["#d1f2e6", "#a3e5cd", "#6fd2ae", "#1baf7a", "#138a5f", "#0d6646", "#08432e"],
               ["#fde9c2", "#fbd385", "#f6bb45", "#eda100", "#b97e00", "#875c00", "#563a00"],
               ["#dcd8f1", "#b9b1e3", "#9286d2", "#4a3aa7", "#3b2e86", "#2c2264", "#1d1642"]]


def group_table(summary: pd.DataFrame, scaled: pd.DataFrame, *, groups: list, bars: list, labels: dict,
                info: list | None = None, chosen: str | None = None, sort_by: str | None = None,
                title: str = "", note: str = ""):
    """Tabela no formato do scIB com grupos de métricas arbitrários.

    ``groups``: [(rótulo do grupo, [métricas])] — cada grupo ganha uma rampa de um
    matiz; ``bars``: [(coluna de ``summary``, cabeçalho)] — a primeira é o geral
    (cinza), as demais ganham a cor do grupo na mesma ordem; ``labels``: linha →
    nome exibido; ``info``: [(cabeçalho, {linha: texto})].
    """
    sort_by = sort_by or bars[0][0]
    order = summary.sort_values(sort_by, ascending=False).index.tolist()
    n = len(order)
    info = info or []
    x = 0.25
    x_rank = x
    x_name = 0.55
    x_info = [x_name + 3.4 + i * 1.5 for i in range(len(info))]
    x_bar0 = (x_info[-1] + 1.4) if info else x_name + 3.4
    bar_w, bar_gap = 1.15, 0.3
    x_met0 = x_bar0 + len(bars) * (bar_w + bar_gap) + 0.35
    step = 0.72
    xs_groups, cur = [], x_met0
    for _, mets in groups:
        xs_groups.append([cur + i * step for i in range(len(mets))])
        cur = xs_groups[-1][-1] + 0.45 + step
    x_end = xs_groups[-1][-1] + 0.6
    fig, ax = plt.subplots(figsize=(x_end * 0.74, 0.42 * n + 4.6))
    ax.axis("off")
    smin, smax = 8, 165
    cmaps = [LinearSegmentedColormap.from_list(f"g{i}", GROUP_STEPS[i % len(GROUP_STEPS)]) for i in range(len(groups))]
    bar_colors = [OVERALL_COLOR] + [GROUP_STEPS[i % len(GROUP_STEPS)][4] for i in range(len(bars) - 1)]
    for i, k in enumerate(order):
        if k == chosen:
            ax.add_patch(Rectangle((0.05, i - 0.48), x_end - 0.1, 0.96, color=BAND, zorder=0, lw=0))
        ax.text(x_rank, i, str(i + 1), ha="center", va="center", fontsize=9, color=INK["secondary"])
        ax.text(x_name, i, labels.get(k, k), ha="left", va="center", fontsize=9.5, color=INK["primary"],
                fontweight="bold" if k == chosen else "normal")
        for xi, (_, d) in zip(x_info, info):
            ax.text(xi, i, str(d.get(k, "")), ha="left", va="center", fontsize=8.5, color=INK["secondary"])
        for j, ((col, _), color) in enumerate(zip(bars, bar_colors)):
            x0 = x_bar0 + j * (bar_w + bar_gap)
            v = summary.loc[k, col]
            ax.add_patch(Rectangle((x0, i - 0.3), bar_w, 0.6, color=INK["grid"], lw=0, zorder=1, alpha=0.45))
            if np.isfinite(v):
                ax.add_patch(Rectangle((x0, i - 0.3), bar_w * max(v, 0), 0.6, color=color, lw=0, zorder=2))
                inside = v > 0.62
                ax.text(x0 + (bar_w * v - 0.06 if inside else bar_w * max(v, 0) + 0.06), i, F.f(v, 2),
                        ha="right" if inside else "left", va="center", fontsize=8,
                        color="white" if inside else INK["primary"], zorder=3)
            else:
                ax.text(x0 + 0.1, i, "–", ha="left", va="center", fontsize=9, color=INK["axis"])
        for xs, (_, mets), cmap in zip(xs_groups, groups, cmaps):
            for xx, mname in zip(xs, mets):
                v = scaled.loc[k, mname] if mname in scaled.columns else np.nan
                if not np.isfinite(v):
                    ax.text(xx, i, "–", ha="center", va="center", fontsize=9, color=INK["axis"])
                    continue
                ax.scatter([xx], [i], s=smin + (smax - smin) * v, color=cmap(0.12 + 0.88 * v), lw=0, zorder=3)
    hy = -0.85
    for txt, xx, ha in [("#", x_rank, "center"), ("Clusterização", x_name, "left")] + \
            [(h, xi, "left") for xi, (h, _) in zip(x_info, info)]:
        ax.text(xx, hy, txt, ha=ha, va="bottom", fontsize=9.5, color=INK["primary"], fontweight="bold")
    for j, (_, head) in enumerate(bars):
        x0 = x_bar0 + j * (bar_w + bar_gap)
        ax.text(x0 + bar_w / 2, hy, head, ha="center", va="bottom", fontsize=8.5, color=INK["primary"],
                fontweight="bold", rotation=45 if j else 0, rotation_mode="anchor")
    for xs, (_, mets) in zip(xs_groups, groups):
        for xx, mname in zip(xs, mets):
            ax.text(xx, hy, mname, ha="left", va="bottom", rotation=45, rotation_mode="anchor", fontsize=8.5,
                    color=INK["primary"])
    gy = -4.6
    for xs, (lab, _), steps in zip(xs_groups, groups, GROUP_STEPS):
        ax.plot([xs[0] - 0.3, xs[-1] + 0.3], [gy + 0.35, gy + 0.35], color=steps[4], lw=3, solid_capstyle="round")
        ax.text((xs[0] + xs[-1]) / 2, gy, lab, ha="center", va="bottom", fontsize=9.5, color=INK["primary"],
                fontweight="bold")
    ax.plot([0.05, x_end - 0.05], [-0.55, -0.55], color=INK["axis"], lw=0.8)
    ly = n + 0.25
    ax.text(x_met0 - 0.3, ly, "métrica escalonada:", ha="right", va="center", fontsize=8.5, color=INK["secondary"])
    for j, v in enumerate([0.0, 0.5, 1.0]):
        xx = x_met0 + j * 0.6
        ax.scatter([xx], [ly], s=smin + (smax - smin) * v, color=cmaps[0](0.12 + 0.88 * v), lw=0)
        ax.text(xx, ly + 0.55, F.f(v, 1), ha="center", va="center", fontsize=7.5, color=INK["muted"])
    ax.text(x_met0 + 2.2, ly, "–  não se aplica", ha="left", va="center", fontsize=8.5, color=INK["secondary"])
    ax.set_xlim(0, x_end)
    ax.set_ylim(n + 1.0, -5.2)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    if note:
        plotting.note(fig, note, y=0.005)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig
