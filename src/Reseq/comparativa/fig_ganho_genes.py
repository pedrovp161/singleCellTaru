"""Barras do ganho de genes por célula (antigo → ressequenciamento), por biblioteca.

Lê ``resultsReseq/tables/C02_antigo_vs_novo/ganho_por_biblioteca.csv`` (C02) e
grava ``resultsReseq/figures/C02_antigo_vs_novo/ganho_genes_barras.png``.

Uso: python comparativa/fig_ganho_genes.py
"""
import sys
sys.path.insert(0, "/mnt/SATA/singleCellTaru/src/Reseq")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions import config, fmt as F, plotting

plotting.apply_style()
NB = "C02_antigo_vs_novo"
INK = plotting.INK

g = pd.read_csv(config.TABLES / NB / "ganho_por_biblioteca.csv")
g["ganho_genes"] = g["razao_genes"] - 1
g["ganho_umis"] = g["razao_umis"] - 1
g = g.sort_values("ganho_genes")
med = float(g["ganho_genes"].median())

fig, ax = plt.subplots(figsize=(7.2, 5.4))
y = np.arange(len(g))
cores = [plotting.stim_color(lib) for lib in g["library_id"]]
ax.barh(y, g["ganho_genes"] * 100, height=0.62, color=cores, edgecolor=plotting.mpl.rcParams["axes.facecolor"],
        linewidth=1)
for yi, (_, r) in zip(y, g.iterrows()):
    ax.text(r["ganho_genes"] * 100 + 0.4, yi, f"+{F.f(r['ganho_genes'] * 100, 1)}%  "
            f"({F.n(round(r['genes_antiga']))} → {F.n(round(r['genes_nova']))})",
            va="center", ha="left", fontsize=8, color=INK["secondary"],
            bbox={"facecolor": plotting.mpl.rcParams["axes.facecolor"], "edgecolor": "none", "pad": 0.5})
ax.axvline(med * 100, color=INK["muted"], lw=1, ls="--", zorder=0)
ax.text(med * 100 + 0.3, len(g) - 0.4, f"mediana +{F.f(med * 100, 1)}%", fontsize=8, color=INK["muted"],
        va="bottom", ha="left")
ax.set_yticks(y, g["library_id"])
ax.set_xlim(0, 40)
ax.set_xlabel("ganho na mediana de genes por célula (%)")
ax.grid(axis="y", visible=False)
plotting.stim_legend(ax, loc="lower right")
ax.set_title(f"Genes por célula: ressequenciamento × sequenciamento antigo\n"
             f"{F.n(int(g['pareadas'].sum()))} células pareadas · {plotting.N_DESIGN}",
             loc="left", fontsize=11)
plotting.note(fig, "Mediana de genes por célula nas mesmas células (barcodes presentes nas duas corridas); "
                   "entre parênteses, antigo → novo. Fonte: C02.", y=-0.02)
fig.tight_layout()
out = plotting.savefig(fig, NB, "ganho_genes_barras",
                       description="Ganho na mediana de genes por célula por biblioteca (células pareadas)",
                       data=g[["library_id", "pareadas", "genes_antiga", "genes_nova", "ganho_genes",
                               "umis_antiga", "umis_nova", "ganho_umis", "saturacao_antiga_est",
                               "saturacao_nova"]],
                       data_index=False)
print(out)
