"""
Anotação por marcadores canônicos (P09) e o que dela depende (P10–P12).

O painel (`PANEL`) junta, para cada tipo celular, os marcadores consagrados na
literatura (`REFS`). A anotação é feita por cluster: perfil de detecção dos
marcadores, genes diferenciais e o atlas de Yasumizu et al. (2024) como
conferência. A decisão de cada cluster fica escrita em `LEIDEN_P08`, com a
evidência que a sustenta; o notebook confere os números.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

#: Referências do painel (citadas por chave).
REFS = {
    "Sallusto1999": "Sallusto F et al. Two subsets of memory T lymphocytes with distinct homing potentials and "
                    "effector functions. Nature 401:708–712 (1999). doi:10.1038/44385",
    "Mahnke2013": "Mahnke YD et al. The who's who of T-cell differentiation: human memory T-cell subsets. "
                  "Eur J Immunol 43:2797–2809 (2013). doi:10.1002/eji.201343751",
    "Fontenot2003": "Fontenot JD et al. Foxp3 programs the development and function of CD4+CD25+ regulatory T "
                    "cells. Nat Immunol 4:330–336 (2003). doi:10.1038/ni904",
    "Thornton2010": "Thornton AM et al. Expression of Helios, an Ikaros transcription factor family member, "
                    "differentiates thymic-derived from peripherally induced Foxp3+ T regulatory cells. "
                    "J Immunol 184:3433–3441 (2010). doi:10.4049/jimmunol.0904028",
    "Crotty2011": "Crotty S. Follicular helper CD4 T cells (TFH). Annu Rev Immunol 29:621–663 (2011). "
                  "doi:10.1146/annurev-immunol-031210-101400",
    "AcostaRodriguez2007": "Acosta-Rodriguez EV et al. Surface phenotype and antigenic specificity of human "
                           "interleukin 17-producing T helper memory cells. Nat Immunol 8:639–646 (2007). "
                           "doi:10.1038/ni1467",
    "Cosmi2008": "Cosmi L et al. Human interleukin 17-producing cells originate from a CD161+CD4+ T cell "
                 "precursor. J Exp Med 205:1903–1916 (2008). doi:10.1084/jem.20080397",
    "Szabo2019": "Szabo PA et al. Single-cell transcriptomics of human T cells reveals tissue and activation "
                 "signatures in health and disease. Nat Commun 10:4706 (2019). doi:10.1038/s41467-019-12464-3",
    "Patil2018": "Patil VS et al. Precursors of human CD4+ cytotoxic T lymphocytes identified by single-cell "
                 "transcriptome analysis. Sci Immunol 3:eaan8664 (2018). doi:10.1126/sciimmunol.aan8664",
    "Hao2021": "Hao Y et al. Integrated analysis of multimodal single-cell data. Cell 184:3573–3587 (2021). "
               "doi:10.1016/j.cell.2021.04.048",
    "Yasumizu2024": "Yasumizu Y et al. Single-cell transcriptome landscape of circulating CD4+ T cell "
                    "populations in autoimmune diseases. Cell Genomics 4:100473 (2024). "
                    "doi:10.1016/j.xgen.2023.100473",
    "Papavasileiou2025": "Papavasileiou et al. Single-cell omics-based characterization of human basophils "
                         "reveals two transcriptionally distinct populations. bioRxiv (2025), Allergy (2026). "
                         "doi:10.1101/2025.02.20.639059",
}

#: Painel de marcadores: tipo → (genes, referências). A ordem é a do gráfico de pontos.
PANEL = {
    "célula T": (["CD3E", "CD3D", "CD2", "TRAC"], ["Hao2021"]),
    "CD4": (["CD4", "CD40LG"], ["Hao2021"]),
    "CD8": (["CD8A", "CD8B"], ["Hao2021"]),
    "naive": (["CCR7", "SELL", "LEF1", "TCF7"], ["Sallusto1999", "Mahnke2013"]),
    "memória": (["FAS", "S100A4", "ANXA1", "AQP3"], ["Mahnke2013", "Yasumizu2024"]),
    "efetora (Tem)": (["GZMK", "CCL5", "KLRB1"], ["Sallusto1999", "Szabo2019"]),
    "citotóxica (Temra)": (["GZMB", "PRF1", "GNLY", "NKG7", "GZMH", "CX3CR1"], ["Patil2018", "Mahnke2013"]),
    "Treg": (["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"], ["Fontenot2003", "Thornton2010"]),
    "Th17": (["RORC", "CCR6", "IL23R"], ["AcostaRodriguez2007", "Cosmi2008"]),
    "Th2": (["GATA3", "CCR4"], ["Yasumizu2024"]),
    "Th1": (["TBX21", "CXCR3", "IFNG"], ["Yasumizu2024"]),
    "Tfh": (["CXCR5", "PDCD1", "ICOS", "BCL6"], ["Crotty2011"]),
    "ativação (TCR)": (["CD69", "TNFRSF4", "TNFRSF9", "MIR155HG", "HLA-DRA"], ["Szabo2019"]),
    "proliferação": (["MKI67", "TOP2A"], ["Szabo2019"]),
    "B / plasmócito": (["MS4A1", "CD79A", "CD19", "JCHAIN", "MZB1"], ["Hao2021"]),
    "monócito": (["CD14", "LYZ", "FCGR3A", "CD68"], ["Hao2021"]),
    "NK": (["KLRF1", "NCAM1", "KLRD1"], ["Hao2021"]),
    "progenitor CD34⁺": (["CD34", "KIT", "GATA2", "SOX4"], ["Hao2021"]),
    "basófilo": (["HDC", "CLC", "CPA3", "IL3RA", "FCER1A"], ["Papavasileiou2025"]),
}

#: Anotação dos clusters do P08 (Leiden 1,0 sobre o ComBat), decidida com o perfil de
#: marcadores, os genes diferenciais e o atlas. ``marcadores``: os que sustentam a decisão
#: (o notebook imprime a % de células que os detectam). ``grupo`` é a classe ampla.
LEIDEN_P08 = {
    "0": ("CD4 naive", "CD4 naive", ["CCR7", "SELL", "LEF1", "TCF7"], "naive clássica (TSHZ2, FHIT, LEF1 nos diferenciais)"),
    "2": ("CD4 naive", "CD4 naive", ["CCR7", "SELL", "LEF1", "TCF7"], "naive; difere do 0 por genes ribossomais"),
    "12": ("CD4 naive", "CD4 naive", ["CCR7", "SELL", "LEF1", "TCF7"], "naive; difere do 0 por dois lncRNAs"),
    "5": ("CD4 transição naive/Tcm", "CD4 memória", ["CCR7", "LEF1", "S100A4", "ANXA1"],
          "CCR7⁺ com marcadores de memória em parte; o atlas divide 44% naive / 40% Tcm"),
    "1": ("CD4 Tcm", "CD4 memória", ["CCR7", "ANXA1", "S100A4", "GATA3", "CCR4"],
          "memória central CCR7⁺; inclina a Th2 (GATA3, CCR4); atlas: TcmTh0"),
    "3": ("CD4 Tcm/Tfh", "CD4 memória", ["CCR7", "KLRB1", "CXCR5", "TIGIT"],
          "memória central com CXCR5 e TIGIT; atlas: TcmTh0 e TcmTfh"),
    "6": ("CD4 Th17", "CD4 memória", ["CCR6", "KLRB1", "RORC", "S100A4"],
          "CCR6⁺ KLRB1⁺ (CD161) RORC⁺; atlas: TcmTh17"),
    "16": ("CD4 Tem GZMK⁺", "CD4 memória", ["GZMK", "TIGIT", "CTLA4", "CCR7"],
           "GZMK⁺ TIGIT⁺; atlas: TemTph (49%); poucas células"),
    "4": ("CD4 ativada", "CD4 ativada", ["TNFRSF4", "IL2RA", "MIR155HG", "CD69"],
          "ativação pelo TCR (OX40, CD25, MIR155HG); mistura de naive e memória no atlas"),
    "13": ("CD4 ativada (Th17)", "CD4 ativada", ["CD69", "TNFRSF4", "IL2RA", "CCR6"],
           "ativação forte com CCR6/KLRB1; atlas: TcmTh17"),
    "8": ("Treg", "Treg", ["FOXP3", "IL2RA", "CTLA4", "IKZF2"], "FOXP3⁺ Helios⁺; atlas: TregEff, TregAct, TregNaive"),
    "9": ("CD4 citotóxica (Temra)", "CD4 citotóxica", ["GZMB", "PRF1", "GNLY", "NKG7"],
          "programa citotóxico; atlas: TemraTh1; uma minoria tem marcadores de NK (KLRF1, FCGR3A)"),
    "7": ("baixa qualidade", "baixa qualidade", ["CD3E", "CCR7"], "baixo conteúdo de RNA (mediana de 1.353 UMIs)"),
    "10": ("baixa qualidade", "baixa qualidade", ["CD3E", "CCR7"],
           "baixo conteúdo (2.568 UMIs), genes ribossomais, 74% de CTH97"),
    "11": ("baixa qualidade", "baixa qualidade", ["CD3E", "CCR7"], "422 UMIs; 60% de baixa profundidade"),
    "14": ("B e plasmócito", "contaminante", ["MS4A1", "CD79A", "JCHAIN", "MZB1"], "linhagem B; CD3 em <25%"),
    "15": ("monócito", "contaminante", ["CD14", "LYZ", "FCGR3A", "HLA-DRA"], "linhagem mieloide"),
    "17": ("progenitor CD34⁺", "contaminante", ["CD34", "KIT", "GATA2", "SOX4"], "progenitor hematopoético"),
    "18": ("basófilo", "contaminante", ["HDC", "CLC", "GATA2", "IL3RA"], "basófilo maduro (KIT⁻ CD34⁻)"),
}

#: Programas da NMF de novo (k = 14, semente 0) → (rótulo, grupo, genes de maior peso que o sustentam).
#: Decididos pelos genes de maior peso de W e conferidos contra o atlas (P10).
NMF14_DENOVO = {
    0: ("B e plasmócito", "contaminante", "BANK1, MS4A1, PAX5, CD79A"),
    1: ("monócito", "contaminante", "FCER1G, MS4A7, TYROBP, CD68, CST3"),
    2: ("programa nuclear (MALAT1/NEAT1)", "baixa qualidade", "TALAM1, MALAT1, NEAT1, ARHGAP15, ZBTB20"),
    3: ("Treg", "Treg", "IKZF2, FOXP3, RTKN2, TIGIT, IL2RA"),
    4: ("CD4 ativada", "CD4 ativada", "MIR155HG, LTA, CCL20, TNFRSF4, IL2RA, TNF"),
    5: ("CD4 transição naive/Tcm", "CD4 memória", "CD7, MIF, S100A11, S100A6, VIM"),
    6: ("CD4 Tcm", "CD4 memória", "NR3C1, TGFBR3, PHACTR2, SAMSN1"),
    7: ("CD4 Th17", "CD4 memória", "CCR6, KLRB1, SLC4A10, MYBL1"),
    8: ("CD4 citotóxica (Temra)", "CD4 citotóxica", "NKG7, GNLY, GZMH, GZMB, PRF1"),
    9: ("CD4 naive IFN⁺", "CD4 naive", "IFIT3, RSAD2, MX1, ISG15, OAS3"),
    10: ("CD4 Tcm/Tfh", "CD4 memória", "MAF, TOX, PRKCB, ITGA4, IFNG-AS1"),
    11: ("Treg efetora (HLA-DR⁺)", "Treg", "HLA-DRB1, HLA-DQB1, ENTPD1, FOXP3, LGALS3"),
    12: ("progenitor/basófilo (GATA2⁺)", "contaminante", "ATP8B4, ZNF521, SOX4, GATA2, MEIS1"),
    13: ("CD4 naive", "CD4 naive", "LEF1-AS1, IGF1R, FHIT, TXK, CERS6"),
}

#: Programas publicados (projeção) → rótulo no nosso vocabulário.
NMF_PUBLICADOS = {
    "NMF0 Cytotoxic-F": "CD4 citotóxica (Temra)", "NMF1 Treg-F": "Treg", "NMF2 Th17-F": "CD4 Th17",
    "NMF3 Naive-F": "CD4 naive", "NMF4 Act-F": "programa Act-F", "NMF5 TregEff/Th2-F": "Treg efetora/Th2",
    "NMF6 Tfh-F": "CD4 Tcm/Tfh", "NMF7 IFN-F": "CD4 IFN⁺", "NMF8 Cent. Mem.-F": "CD4 Tcm",
    "NMF9 Thymic Emi.-F": "CD4 emigrante tímica", "NMF10 Tissue-F": "programa Tissue-F", "NMF11 Th1-F": "CD4 Th1",
}

#: Ordem de exibição dos rótulos finos.
LABEL_ORDER = ["CD4 naive", "CD4 transição naive/Tcm", "CD4 Tcm", "CD4 Tcm/Tfh", "CD4 Th17", "CD4 Tem GZMK⁺",
               "CD4 ativada", "CD4 ativada (Th17)", "Treg", "CD4 citotóxica (Temra)", "B e plasmócito",
               "monócito", "progenitor CD34⁺", "basófilo", "baixa qualidade"]
GROUP_ORDER = ["CD4 naive", "CD4 memória", "CD4 ativada", "Treg", "CD4 citotóxica", "contaminante",
               "baixa qualidade"]

#: Rótulo fino → tipo L1 do atlas (para medir a concordância). Sem equivalente: None.
TO_ATLAS_L1 = {"CD4 naive": "Tnaive", "CD4 transição naive/Tcm": None, "CD4 Tcm": "Tcm", "CD4 Tcm/Tfh": "Tcm",
               "CD4 Th17": "Tcm", "CD4 Tem GZMK⁺": "Tem", "CD4 ativada": None, "CD4 ativada (Th17)": None,
               "Treg": "Treg", "CD4 citotóxica (Temra)": "Temra",
               # rótulos que só a NMF produz
               "CD4 naive IFN⁺": "Tnaive", "Treg efetora (HLA-DR⁺)": "Treg", "CD4 emigrante tímica": "Tnaive",
               "CD4 Th1": "Tem", "Treg efetora/Th2": None, "CD4 IFN⁺": None, "programa Act-F": None,
               "programa Tissue-F": None}


def panel_genes() -> list[str]:
    return [g for genes, _ in PANEL.values() for g in genes]


def symbol_view(adata):
    """Cópia leve com símbolos como nomes de genes (únicos), para indexar por marcador."""
    a = adata.copy() if adata.is_view else adata
    sym = pd.Index(a.var["gene_symbol"].astype(str)).str.replace("_", "-")
    out = a.copy()
    out.var_names = sym
    out.var_names_make_unique()
    return out


def dot_stats(adata, groups, genes, layer_counts: str = "counts"):
    """% de células que detectam (≥ 1 UMI) e média da lognorm, por grupo × gene."""
    genes = [g for g in genes if g in adata.var_names]
    idx = adata.var_names.get_indexer(genes)
    X = adata.X[:, idx]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    C = adata.layers[layer_counts][:, idx]
    C = (C.toarray() if sp.issparse(C) else np.asarray(C)) > 0
    g = pd.Series(np.asarray(groups).astype(str))
    frac = pd.DataFrame(C, columns=genes).groupby(g.to_numpy()).mean()
    mean = pd.DataFrame(X, columns=genes).groupby(g.to_numpy()).mean()
    return frac, mean


def module_scores(adata, seed: int = 0) -> pd.DataFrame:
    """Escore de cada tipo do painel por célula (``sc.tl.score_genes`` sobre a lognorm)."""
    import scanpy as sc

    out = {}
    for name, (genes, _) in PANEL.items():
        gs = [g for g in genes if g in adata.var_names]
        if len(gs) < 2:
            continue
        sc.tl.score_genes(adata, gs, score_name="_tmp", random_state=seed)
        out[name] = adata.obs["_tmp"].to_numpy()
    adata.obs.drop(columns="_tmp", inplace=True)
    return pd.DataFrame(out, index=adata.obs_names)


def dotplot(ax, frac: pd.DataFrame, mean: pd.DataFrame, *, rows, cols, groups_of_cols=None, smax: float = 120,
            cmap=None):
    """Gráfico de pontos: tamanho = % que detecta; cor = média escalonada por gene (0–1)."""
    import matplotlib.pyplot as plt

    from . import plotting

    cmap = cmap or plotting.SEQUENTIAL
    f = frac.reindex(index=rows, columns=cols)
    m = mean.reindex(index=rows, columns=cols)
    rng = (m.max() - m.min()).replace(0, 1)
    ms = (m - m.min()) / rng
    yy, xx = np.meshgrid(np.arange(len(rows)), np.arange(len(cols)), indexing="ij")
    sc_ = ax.scatter(xx.ravel(), yy.ravel(), s=(f.to_numpy().ravel() * smax), c=ms.to_numpy().ravel(),
                     cmap=cmap, vmin=0, vmax=1, lw=0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7.5, fontstyle="italic")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlim(-0.7, len(cols) - 0.3)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if groups_of_cols:
        x0 = 0
        for name, n in groups_of_cols:
            ax.text(x0 + (n - 1) / 2, -0.9, name, ha="center", va="bottom", fontsize=7, rotation=90,
                    color=plotting.INK["secondary"])
            if x0 > 0:
                ax.axvline(x0 - 0.5, color=plotting.INK["grid"], lw=0.8, zorder=0)
            x0 += n
    return sc_
