# [P15] Cópia de /mnt/SATA/spatialPaper/src/forest_composicao/forest.py (projeto espacial), sem mudança de método.
# Edições: "spots" → "células" e "nicho" → "tipo celular" nos textos; o resto é o arquivo original.
"""Forest plot da composicao de nichos -- UM grafico, UM teste.

**Um unico forest plot.** O contraste e um so, `FB - UB`, estimado uma vez por
nicho. Uma versao anterior deste modulo desenhava duas figuras, uma "orientada"
para cada grupo; era o mesmo teste espelhado, e duas figuras com os mesmos
p-valores sugerem dois modelos onde ha um. Agora ha um grafico com todos os
nichos e o sinal do efeito diz qual grupo tem maior abundancia.

A direcao e codificada tres vezes, de proposito redundante:

  * **posicao** -- a esquerda do zero = maior em UB, a direita = maior em FB;
  * **tinta de fundo** -- cada metade do painel recebe um lavado da cor do seu
    grupo, para que o lado seja legivel sem ler o eixo;
  * **cor da marca** -- o ponto e o intervalo saem na cor do grupo favorecido
    (cinza quando o nicho nao alcanca significancia).

**Um ponto por amostra, ao lado do intervalo.** Com n=28 um intervalo estreito e
um largo tem a mesma aparencia de barra, e o leitor nao consegue ver que um
nicho inteiro esta sendo carregado por duas amostras. Os dois paineis tem
escalas de x diferentes (CLR e % de spots) e ficam separados de proposito: sao
dois graficos alinhados pelo eixo y, nunca dois eixos no mesmo grafico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURF, INK, INK2, GRID, BANDA, NS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#f4f3ef", "#c9c8c1"

# par validado para daltonismo (ΔE 24,7 protanopia / 33,6 visao normal)
CORES_GRUPO = {"FB": "#2a78d6", "UB": "#eb6834"}
ROTULOS = {"FB": "Respondedor (FB)", "UB": "Não respondedor (UB)"}

FORMAS = {"FDR < 0.05": "*", "p < 0.05": "D", "Não significativo": "o"}
TAMANHOS = {"FDR < 0.05": 15.0, "p < 0.05": 7.0, "Não significativo": 6.5}
TINTA = 0.075          # opacidade do lavado direcional


def _fmt_p(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.3f}" if v >= 1e-3 else f"{v:.1e}"


def forest(res: pd.DataFrame, comp: dict,
           cores_grupo: dict | None = None, rotulos: dict | None = None,
           titulo: str | None = None, subtitulo: str | None = None,
           figsize=(15.4, None), rotulos_linha: dict | None = None):
    """Forest unico do contraste `g1 - g2` ja estimado em `res`.

    Nao recalcula nem reorienta nada: usa `estimate`, `conf_low`, `conf_high`,
    `p_value`, `FDR` e `sig_shape` como vieram de `composicao.run_contrast`.
    """
    cores_grupo = cores_grupo or CORES_GRUPO
    rotulos = rotulos or ROTULOS
    res = res.copy()

    g1, g2 = res["g1"].iloc[0], res["g2"].iloc[0]   # contraste = g1 - g2
    ordem = res.sort_values("estimate")["cell_population"].tolist()
    res = res.set_index("cell_population").loc[ordem].reset_index()
    n = len(ordem)
    y = np.arange(n)

    prop = 100.0 * comp["proportion"]
    grupos = comp["meta"].loc[prop.index, comp["group_col"]].astype(str)

    CAB, ROD = 1.55, 1.15
    alt = figsize[1] or max(3.8, 0.44 * n + CAB + ROD)
    fig = plt.figure(figsize=(figsize[0], alt), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[3.15, 2.25, 0.9], wspace=0.07,
                          left=0.215, right=0.985,
                          top=1 - CAB / alt, bottom=ROD / alt)
    axF, axP, axT = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax in (axF, axP, axT):
        ax.set_facecolor(SURF)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        ax.set_ylim(-0.6, n - 0.4)
    for i in range(n):
        if i % 2 == 0:
            for ax in (axF, axP, axT):
                ax.axhspan(i - 0.5, i + 0.5, color=BANDA, zorder=0, lw=0)

    # ------------------------------------------------------------------ forest
    for i, r in res.iterrows():
        sig = r["sig_shape"] != "Não significativo"
        cor = cores_grupo.get(r["direction"], INK2) if sig else NS
        axF.plot([r["conf_low"], r["conf_high"]], [i, i], lw=2.0, color=cor,
                 solid_capstyle="round", zorder=3, alpha=0.95)
        axF.plot([r["estimate"]], [i], marker=FORMAS[r["sig_shape"]],
                 ms=TAMANHOS[r["sig_shape"]], color=cor, zorder=4,
                 mec=SURF, mew=1.4)

    etiquetas = [rotulos_linha.get(c, c) if rotulos_linha else c for c in ordem]
    axF.set_yticks(y)
    axF.set_yticklabels(etiquetas, fontsize=10, color=INK)
    axF.grid(axis="x", color=GRID, lw=0.8, zorder=1)
    axF.set_axisbelow(True)

    # lavado direcional: cobre cada metade do painel, por cima das bandas de
    # linha (zorder 0) e por baixo dos dados (zorder >= 3)
    x0, x1 = axF.get_xlim()
    margem = 0.04 * (x1 - x0)
    x0, x1 = x0 - margem, x1 + margem
    axF.set_xlim(x0, x1)
    axF.axvspan(x0, 0, color=cores_grupo.get(g2, NS), alpha=TINTA, lw=0, zorder=0.5)
    axF.axvspan(0, x1, color=cores_grupo.get(g1, NS), alpha=TINTA, lw=0, zorder=0.5)
    axF.axvline(0, ls="--", lw=1.2, color="#6f6d68", zorder=2)

    # rotulos de direcao dentro do painel, na cor de cada grupo
    axF.text(0.015, 1.012, f"◀ maior em {rotulos.get(g2, g2)}",
             transform=axF.transAxes, ha="left", va="bottom", fontsize=10,
             fontweight="bold", color=cores_grupo.get(g2, INK2))
    axF.text(0.985, 1.012, f"maior em {rotulos.get(g1, g1)} ▶",
             transform=axF.transAxes, ha="right", va="bottom", fontsize=10,
             fontweight="bold", color=cores_grupo.get(g1, INK2))
    axF.set_xlabel(f"efeito (CLR) do contraste {g1} − {g2}, ajustado, com IC 95 %",
                   fontsize=9.5, color=INK2, labelpad=7)

    # ---------------------------------------------------------------- amostras
    rng = np.random.default_rng(0)
    for i, ct in enumerate(ordem):
        for gr in (g1, g2):
            amostras = [s for s in grupos[grupos == gr].index if s in prop.index]
            v = prop.loc[amostras, ct].to_numpy(dtype=float)
            v = np.clip(v[np.isfinite(v)], 1e-2, None)
            if not len(v):
                continue
            jit = (rng.random(len(v)) - 0.5) * 0.32
            axP.scatter(v, i + jit, s=26, color=cores_grupo.get(gr, INK2),
                        alpha=0.75, linewidths=0.8, edgecolors=SURF, zorder=3)
            axP.plot([np.median(v)], [i], marker="|", ms=17, mew=2.4,
                     color=cores_grupo.get(gr, INK2), zorder=4)
    axP.set_xscale("log")
    axP.set_xlabel("% das células da amostra (escala log)", fontsize=9.5,
                   color=INK2, labelpad=7)
    axP.grid(axis="x", color=GRID, lw=0.8, zorder=1)
    axP.set_axisbelow(True)
    # curto de propósito: o painel é estreito e o texto encostava na coluna do p
    axP.text(0.0, 1.012, "dados brutos · 1 ponto = 1 amostra · | = mediana",
             transform=axP.transAxes, ha="left", va="bottom", fontsize=10, color=INK2)
    plt.setp(axP.get_yticklabels(), visible=False)

    # ------------------------------------------------------------------ p, FDR
    axT.set_xlim(0, 1); axT.set_xticks([])
    plt.setp(axT.get_yticklabels(), visible=False)
    axT.text(0.02, 1.012, "p", transform=axT.transAxes, ha="left", va="bottom",
             fontsize=10, color=INK2, fontweight="bold")
    axT.text(0.56, 1.012, "FDR", transform=axT.transAxes, ha="left", va="bottom",
             fontsize=10, color=INK2, fontweight="bold")
    for i, r in res.iterrows():
        forte = r["sig_shape"] != "Não significativo"
        axT.text(0.02, i, _fmt_p(r["p_value"]), fontsize=8.6, va="center",
                 color=INK if forte else INK2,
                 fontweight="bold" if forte else "normal")
        axT.text(0.56, i, _fmt_p(r["FDR"]), fontsize=8.6, va="center",
                 color=INK if r["FDR"] < 0.05 else INK2,
                 fontweight="bold" if r["FDR"] < 0.05 else "normal")

    # ------------------------------------------------------------------ legenda
    n_por_grupo = grupos.value_counts()
    handles = [
        Patch(facecolor=cores_grupo.get(g1, INK2),
              label=f"maior em {rotulos.get(g1, g1)}  (n={n_por_grupo.get(g1, 0)})"),
        Patch(facecolor=cores_grupo.get(g2, INK2),
              label=f"maior em {rotulos.get(g2, g2)}  (n={n_por_grupo.get(g2, 0)})"),
        Patch(facecolor=NS, label="não significativo"),
        Line2D([], [], marker="", ls="", label="      "),
    ]
    handles += [Line2D([], [], marker=FORMAS[k], ls="", ms=TAMANHOS[k] * 0.8,
                       color=INK2, label=k) for k in FORMAS]
    leg = fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                     fontsize=9.5, bbox_to_anchor=(0.57, 0.06 / alt),
                     handletextpad=0.6, columnspacing=1.3)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.text(0.215, 1 - 0.30 / alt,
             titulo or f"Composição de nichos · {rotulos.get(g1, g1)} vs "
                       f"{rotulos.get(g2, g2)}",
             fontsize=14.5, fontweight="bold", color=INK, ha="left", va="top")
    if subtitulo:
        fig.text(0.215, 1 - 0.68 / alt, subtitulo, fontsize=10, color=INK2,
                 ha="left", va="top")
    fig.text(0.215, 1 - 1.02 / alt,
             f"Um único teste por tipo celular ({g1} vs {g2}); o sinal do efeito diz "
             f"qual grupo tem maior abundância.",
             fontsize=9.5, color=INK2, ha="left", va="top", style="italic")
    return fig, (axF, axP, axT)
