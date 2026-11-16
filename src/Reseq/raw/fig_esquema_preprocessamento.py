"""Esquema do pré-processamento a partir da matriz raw (P01–P05), com as contagens de cada etapa.

Lê as tabelas já gravadas por P01, P02, P03 e P06 e grava
``resultsReseq/figures/P06_balanco_final/esquema_preprocessamento.png``.

Uso: python raw/fig_esquema_preprocessamento.py
"""
import sys
sys.path.insert(0, "/mnt/SATA/singleCellTaru/src/Reseq")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from functions import config, fmt as F, plotting

plotting.apply_style()
NB = "P06_balanco_final"
INK = plotting.INK
T = config.TABLES

funil = pd.read_csv(T / NB / "funil_novo.csv", index_col=0)
cham = pd.read_csv(T / "P02_cellbender" / "chamadas_por_biblioteca.csv", index_col=0)
pad = pd.read_csv(T / "P03_qc_best_practices" / "padrao_por_biblioteca.csv", index_col=0)
amb = pd.read_csv(T / "P04_ambiente_e_doublets" / "rna_ambiente.csv", index_col=0)
genes = pd.read_csv(T / "P05_integracao" / "filtro_de_genes.csv", index_col=0)

col_bc, col_cr = "barcodes com UMI", "Cell Ranger"
col_qc, col_fim = "QC (200 UMIs; extras só com perfil de célula)", "sem doublets"
n_bc, n_cr, n_qc, n_fim = (int(funil[c].sum()) for c in (col_bc, col_cr, col_qc, col_fim))
qc_cr, qc_extra = int(pad["padrão: do Cell Ranger"].sum()), int(pad["padrão: baixa profundidade"].sum())
n_ed, n_cb = int(cham["emptyDrops"].sum()), int(cham["CellBender"].sum())
g20 = genes.loc["min_cells = 20"]


def faixa(c):
    s = funil[c]
    return f"{F.n(s.min())} ({s.idxmin()}) a {F.n(s.max())} ({s.idxmax()}) por biblioteca"


LN, GAP = 0.0122, 0.036    # passo de linha e espaço entre caixas (fração da altura da figura)


def caixa(ax, x, y, w, titulo, linhas, *, destaque=None, tracejado=False, cor=INK["primary"]):
    """Desenha a caixa com o topo em ``y`` e devolve a base (altura pelo conteúdo)."""
    h = 0.034 + (0.032 if destaque else 0) + LN * len(linhas) + 0.008
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.01",
                                fc=plotting.mpl.rcParams["axes.facecolor"], ec=cor, lw=1.2,
                                ls=(0, (4, 3)) if tracejado else "-"))
    ax.text(x + 0.015, y - 0.012, titulo, fontsize=10, fontweight="bold", color=INK["primary"], va="top")
    yy = y - 0.034
    if destaque:
        ax.text(x + 0.015, yy, destaque, fontsize=13, fontweight="bold", color=plotting.ACCENT, va="top")
        yy -= 0.032
    for ln in linhas:
        ax.text(x + 0.015, yy, ln, fontsize=8.3, color=INK["secondary"], va="top")
        yy -= LN
    return y - h


def seta(ax, x, y0, texto="", *, tracejada=False):
    """Seta vertical de ``y0`` até ``y0 - GAP``; devolve o topo da próxima caixa."""
    y1 = y0 - GAP
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color=INK["muted"], ls=(0, (4, 3)) if tracejada else "-"))
    if texto:
        ax.text(x + 0.012, (y0 + y1) / 2, texto, fontsize=8.3, color=INK["muted"], va="center", ha="left",
                style="italic")
    return y1


fig = plt.figure(figsize=(11, 15))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

X, W = 0.04, 0.56          # coluna principal
XR, WR = 0.65, 0.32        # coluna lateral
XA = X + W / 2
ax.text(X, 0.985, "Pré-processamento do ressequenciamento a partir da matriz raw (P01–P05)",
        fontsize=14, fontweight="bold", color=INK["primary"], va="top")
ax.text(X, 0.962, f"{plotting.N_DESIGN} · cada biblioteca = 1 paciente × 1 condição (NS ou ML) · "
                  "Cell Ranger 10.1.0", fontsize=9.5, color=INK["secondary"], va="top")

topo = 0.935
y = caixa(ax, X, topo, W, "1 · Matriz raw do Cell Ranger (raw_feature_bc_matrix)",
          [f"todos os barcodes com ≥ 1 UMI: {faixa(col_bc).replace(' por biblioteca', '')}",
           "quase todos são gotas vazias com RNA ambiente"],
          destaque=f"{F.n(n_bc)} barcodes")
y_raw = (topo + y) / 2
y = seta(ax, XA, y, "chamada de células")
y = caixa(ax, X, y, W, "2 · Células chamadas pelo Cell Ranger (matriz filtered)",
          [faixa(col_cr), "base do dado final: o emptyDrops e o CellBender não tiram nenhuma delas"],
          destaque=f"{F.n(n_cr)} células")
y = seta(ax, XA, y, "controle de qualidade")
y_qc = y
y = caixa(ax, X, y, W, "3 · QC do sc-best-practices com valor único de 200 UMIs (P03)",
          ["mínimo de 200 UMIs em todas as bibliotecas, no lugar de 5 MADs por",
           "  biblioteca (que iam de ~150 a ~3.800 UMIs e descartavam células boas)",
           "UMIs e genes: até 5 MADs acima da mediana da biblioteca (log)",
           "% de UMIs nos 20 genes mais expressos: 5 MADs, nos dois lados",
           f"fração mitocondrial: 3 MADs acima (logit) e > {F.f(config.BP_MT_PCT, 0)}%",
           "sem material nuclear (fração intrônica alta)",
           f"entram {F.n(qc_cr)} do Cell Ranger (saem {F.n(n_cr - qc_cr)})",
           f"  + {F.n(qc_extra)} fora do Cell Ranger, só com perfil de célula",
           "    → marcadas em obs['baixa_profundidade']"],
          destaque=f"{F.n(n_qc)} células")
y = seta(ax, XA, y, "RNA ambiente e doublets")
y = caixa(ax, X, y, W, "4 · SoupX + doublets (P04)",
          [f"SoupX corrige as contagens: ρ de {F.pct(amb['SoupX ρ'].min(), 1)} a {F.pct(amb['SoupX ρ'].max(), 1)}",
           "scDblFinder com limiar calibrado pela taxa esperada do 10x",
           "  + doublets T + B validados (A03)",
           f"saem {F.n(n_qc - n_fim)} doublets · {faixa(col_fim)}"],
          destaque=f"{F.n(n_fim)} células")
y = seta(ax, XA, y, "genes e normalização")
y = caixa(ax, X, y, W, "5 · Objeto principal (P05)",
          [f"gene em ≥ 20 células no conjunto: {F.n(g20['genes'])} genes, "
           f"{int(g20['citocinas mantidas (de 16)'])} de 16 citocinas",
           "lognorm (10 mil por célula, log1p) · 2.000 genes variáveis",
           "  (seurat_v3 por biblioteca) · escala · PCA",
           "dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad"],
          destaque=f"{F.n(n_fim)} células × {F.n(g20['genes'])} genes")
y = seta(ax, XA, y, tracejada=True)
y = caixa(ax, X, y, W, "Depois do pré-processamento (P07–P12)",
          ["integração ComBat (escolhida no benchmark scIB) · Leiden 1,0 (P08)",
           "anotação (P09) · programas de NMF (P10)"],
          tracejado=True, cor=INK["muted"])

# coluna lateral: avaliação em paralelo
yb = caixa(ax, XR, topo, WR, "Avaliação em paralelo (P01, P02)",
           ["rodam sobre a matriz raw inteira;",
            "NÃO filtram: só conferem o Cell Ranger",
            "",
            f"emptyDrops (DropletUtils): {F.n(n_ed)}",
            "  · imitando o Cell Ranger: 37.783 de 37.898",
            "  · os ~38 mil extras: mediana de 153 UMIs,",
            "    metade com perfil de núcleo",
            f"CellBender 0.4.0 (GPU): {F.n(n_cb)}",
            "  · perde ≤ 11 células do CR por biblioteca",
            "  · aceita 18.584 núcleos em PB21_NS",
            "",
            "Teste de perfil de célula (P01): o barcode",
            "  parece célula ou gota vazia na mesma",
            "  profundidade? Usado no passo 3."],
           tracejado=True, cor=INK["muted"])
ax.add_patch(FancyArrowPatch((X + W, y_raw), (XR, y_raw), arrowstyle="-|>", mutation_scale=12, lw=1.2,
                             color=INK["muted"], ls=(0, (4, 3))))

# coluna lateral: contagens por biblioteca
linhas = [f"{'':<10}{'CR':>7}{'QC':>7}{'final':>7}"]
for lib in funil.index:
    linhas.append(f"{lib:<10}{F.n(funil.loc[lib, col_cr]):>7}{F.n(funil.loc[lib, col_qc]):>7}"
                  f"{F.n(funil.loc[lib, col_fim]):>7}")
linhas.append(f"{'total':<10}{F.n(n_cr):>7}{F.n(n_qc):>7}{F.n(n_fim):>7}")
yt = min(y_qc, yb - GAP)
ax.text(XR, yt, "Células por biblioteca", fontsize=10, fontweight="bold", color=INK["primary"], va="top")
ax.text(XR, yt - 0.024, "\n".join(linhas), fontsize=8.3, family="monospace", color=INK["secondary"], va="top",
        linespacing=1.45)

fig.text(X, 0.012, "Fontes: tabelas de P01–P06 em resultsReseq/tables/. CR = Cell Ranger. "
                   "Caixas tracejadas = etapas que não alteram as células do dado final. Na tabela, QC inclui as "
                   "células de baixa profundidade, por isso pode passar do CR.",
         fontsize=8, color=INK["muted"])
assert y > 0.03, f"esquema não coube: base em {y:.3f}"

tab = pd.DataFrame({"etapa": ["matriz raw (barcodes ≥ 1 UMI)", "Cell Ranger", "emptyDrops (avaliação)",
                              "CellBender (avaliação)", "QC 200 UMIs", "QC: do Cell Ranger",
                              "QC: baixa profundidade", "sem doublets (final)"],
                    "n": [n_bc, n_cr, n_ed, n_cb, n_qc, qc_cr, qc_extra, n_fim]})
out = plotting.savefig(fig, NB, "esquema_preprocessamento",
                       description="Esquema do pré-processamento a partir da matriz raw, com contagens por etapa",
                       data=tab, data_index=False)
print(out)
