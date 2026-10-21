# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # P07 — Integração com Harmony e clusterização do objeto principal
#
# Objeto principal: `dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad`,
# com 35.247 células — padrão de 200 UMIs, contagens corrigidas pelo SoupX,
# sem doublets (P01–P05). Nele, P05 já fez o processamento V3:
#
# 1. 2.000 genes variáveis `seurat_v3` por biblioteca;
# 2. PCA (50 componentes) sobre a lognorm escalonada;
# 3. **Harmony por biblioteca** (`vars_use="library_id"`, como no original);
# 4. vizinhança nas 15 primeiras dimensões do Harmony, UMAP, Leiden 0,5.
#
# Este notebook avalia a integração e fecha a clusterização, **sem anotar**:
#
# * a integração misturou as bibliotecas e os pacientes? Apagou a diferença
#   entre estimulado e não estimulado do mesmo paciente?
# * que resolução do Leiden os dados sustentam (estabilidade entre sementes)?
# * que clusters são técnicos (baixa profundidade, não-T)?
# * composição por paciente, profundidade, citocinas e estímulo por cluster.
#
# Os rótulos de cluster vão para um arquivo à parte
# (`dataReseq/P07_integracao_clusterizacao/clusters.parquet`, chave
# `cell_key`), sem criar outra versão do objeto principal.
#
# Limite de inferência: 7 pacientes, 14 bibliotecas; MB e NP com um paciente
# cada. Só o contraste pareado estimulado × não estimulado (n = 7) é testado, e
# de forma exploratória.

# %%
import sys
sys.path.insert(0, "/mnt/SATA/singleCellTaru/src/Reseq")

import itertools
import time
import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

from functions import config, fmt as F, io, pipeline as pl, plotting
from functions.issues import IssueLog, show

plotting.apply_style()
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 250)
sc.settings.verbosity = 0

NB = "P07_integracao_clusterizacao"
log = IssueLog(NB)
LIBS = config.LIBRARIES
PATS = config.PATIENTS
INK = plotting.INK


def _bool(s):
    return pd.Series(s).astype(str).str.lower().isin(["true", "1"]).to_numpy()


# %%
caminho = io.data_path("P05_integracao", "reseq_raw_integrado", "h5ad")
adata = ad.read_h5ad(caminho)
obs = adata.obs
N = adata.n_obs
lib_arr = obs["library_id"].astype(str).to_numpy()
pat_arr = obs["patient"].astype(str).to_numpy()
n_lib = pd.Series(lib_arr).value_counts().reindex(LIBS)
print(f"{caminho.name}: {F.n(N)} células × {F.n(adata.n_vars)} genes · {plotting.N_DESIGN}")
print(f"de {F.n(n_lib.min())} ({n_lib.idxmin()}) a {F.n(n_lib.max())} ({n_lib.idxmax()}) células por biblioteca")

# %% [markdown]
# ## 1. A integração com Harmony
#
# Para cada célula, a fração dos vizinhos no grafo que vem da mesma biblioteca,
# no grafo construído sobre a PCA (antes do Harmony) e no grafo usado no
# agrupamento (depois). Sob mistura perfeita, essa fração é a parcela da
# biblioteca no conjunto. Também o iLISI (número efetivo de bibliotecas ou
# pacientes entre os vizinhos) e a separação entre as duas bibliotecas de cada
# paciente (0 = misturadas, 1 = separadas).

# %%
tmp = sc.AnnData(obs=obs[[]].copy())
tmp.obsm["X_pca"] = adata.obsm["X_pca"]
sc.pp.neighbors(tmp, use_rep="X_pca", n_pcs=config.N_DIMS_NEIGHBORS)
grafos = {"antes do Harmony (PCA)": tmp.obsp["connectivities"], "depois do Harmony": adata.obsp["connectivities"]}
mix, linhas = {}, []
for nome, G in grafos.items():
    k = pl.knn_library_mixing(G, lib_arr)
    k["library_id"] = lib_arr
    mix[nome] = k.groupby("library_id")["frac_vizinhos_mesma_biblioteca"].mean().reindex(LIBS)
    kp = pl.knn_library_mixing(G, pat_arr)
    sep = pl.within_patient_separation(G, lib_arr, pat_arr)
    linhas.append({"grafo": nome,
                   "vizinhos da mesma biblioteca (mediana das bibliotecas)": float(mix[nome].median()),
                   "iLISI de bibliotecas (mediana)": float(k["ilisi"].median()),
                   "iLISI de pacientes (mediana)": float(kp["ilisi"].median()),
                   "separação NS × ML no paciente (mediana)": float(sep["separacao"].median())})
mistura = pd.DataFrame(linhas).set_index("grafo")
mistura["iLISI de bibliotecas, mistura perfeita"] = pl.expected_ilisi(lib_arr)
mistura["iLISI de pacientes, mistura perfeita"] = pl.expected_ilisi(pat_arr)
esperado = n_lib / N
mix_tab = pd.DataFrame(mix)
mix_tab["esperado sob mistura perfeita"] = esperado
io.save_table(mistura, NB, "mistura_resumo", description="Mistura de bibliotecas e pacientes antes e depois do Harmony")
io.save_table(mix_tab, NB, "mistura_por_biblioteca", description="Fração de vizinhos da mesma biblioteca, antes e depois do Harmony")
umis = obs["total_counts"].to_numpy()
rho = pd.DataFrame({"PCA": pl.depth_correlation(adata.obsm["X_pca"], umis),
                    "Harmony": pl.depth_correlation(adata.obsm["X_emb_harmony"], umis)},
                   index=[f"dim {i}" for i in range(1, 6)]).T
io.save_table(rho, NB, "profundidade_por_dimensao", description="|ρ de Spearman| entre as 5 primeiras dimensões e o nº de UMIs")
display(mistura.round(3))
rho.round(2)

# %%
fig, ax = plt.subplots(figsize=(8.5, 5.6))
y = np.arange(len(LIBS))[::-1]
antes, depois = mix["antes do Harmony (PCA)"], mix["depois do Harmony"]
for yi, l in zip(y, LIBS):
    ax.plot([antes[l], depois[l]], [yi, yi], color=INK["axis"], lw=1.6, zorder=1, solid_capstyle="round")
    ax.plot(esperado[l], yi, marker="|", ms=13, mew=2, color=INK["primary"], zorder=2, ls="")
    ax.plot(antes[l], yi, "o", ms=8, mfc=plotting.OLD_NEW_COLORS["antiga"], mec=INK["surface"], mew=1.5, zorder=3)
    ax.plot(depois[l], yi, "o", ms=8, mfc=plotting.OLD_NEW_COLORS["nova"], mec=INK["surface"], mew=1.5, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([f"{l}  ({F.n(n_lib[l])})" for l in LIBS])
ax.set_xlim(-0.03, 1.03)
plotting.percent(ax, "x")
ax.set_xlabel("vizinhos da mesma biblioteca")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
fig.legend(handles=plotting.old_new_handles("antes do Harmony (PCA)", "depois do Harmony") + [
    mpl.lines.Line2D([], [], marker="|", ls="", ms=11, mew=2, color=INK["primary"], label="esperado sob mistura perfeita")],
    loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
plotting.note(fig, f"{F.n(N)} células · {plotting.N_DESIGN}. Média, por biblioteca, da fração de vizinhos no grafo que pertence "
                   "à mesma biblioteca. Entre parênteses, células da biblioteca.", y=0.0)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
plotting.savefig(fig, NB, "mistura_de_bibliotecas", description="Vizinhos da mesma biblioteca antes e depois do Harmony",
                 data=mix_tab.rename_axis("library_id").reset_index(), data_index=False)
plt.show()

# %%
mb = mistura.loc["depois do Harmony"]
ma = mistura.loc["antes do Harmony (PCA)"]
resid = (mix["depois do Harmony"] / esperado).sort_values(ascending=False)
print(f"vizinhos da mesma biblioteca: {F.pct(ma.iloc[0], 0)} antes → {F.pct(mb.iloc[0], 0)} depois (mediana das bibliotecas); "
      f"iLISI de pacientes {F.f(ma['iLISI de pacientes (mediana)'], 2)} → {F.f(mb['iLISI de pacientes (mediana)'], 2)} "
      f"(mistura perfeita {F.f(mb['iLISI de pacientes, mistura perfeita'], 2)})")
print(f"separação NS × ML no paciente: {F.f(ma['separação NS × ML no paciente (mediana)'], 2)} → "
      f"{F.f(mb['separação NS × ML no paciente (mediana)'], 2)}")
print("bibliotecas que continuam mais separadas (vizinhos próprios ÷ esperado): "
      + "; ".join(f"{l} {F.x(v, 1)}" for l, v in resid.head(4).items()))

# %% [markdown]
# ## 2. Que resolução os dados sustentam
#
# O Leiden é estocástico e a resolução é uma escolha. Para cada resolução, o
# agrupamento roda com três sementes sobre o mesmo grafo; mede-se o ARI entre
# as sementes e, para cada cluster da semente 0, o Jaccard com o melhor par nas
# outras sementes (≥ 0,75 = cluster que reaparece quase igual).
#
# O ARI favorece resoluções grossas (poucos clusters grandes se reproduzem com
# facilidade), então duas escalas são escolhidas:
#
# * **grossa** — a de maior ARI médio entre sementes;
# * **fina (principal)** — a de maior ARI médio entre as que dão pelo menos 10
#   clusters. É a usada adiante: separa os clusters técnicos e dá a
#   granularidade de que a anotação vai precisar.

# %%
RES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
SEEDS = (0, 1, 2)
t0 = time.time()
# cache da varredura (NVMe): só refaz se o grafo, as resoluções ou as sementes mudarem
G_ = adata.obsp["connectivities"]
chave = f"{G_.nnz}_{float(G_.sum()):.6f}_{'-'.join(map(str, RES))}_{'-'.join(map(str, SEEDS))}"
cache = config.HEAVY_OUT / NB / "leiden_sweep.parquet"
if cache.exists() and pd.read_parquet(cache, columns=["chave"])["chave"].iloc[0] == chave:
    sw = pd.read_parquet(cache).drop(columns="chave")
    sw.index = obs.index
    print("varredura do Leiden lida do cache")
else:
    sw = pl.leiden_sweep(adata, RES, seeds=SEEDS)
    io._assert_writable(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    sw.assign(chave=chave).to_parquet(cache)
linhas, jac = [], {}
for r in RES:
    cols = [f"leiden_r{r:g}_s{s}" for s in SEEDS]
    aris = [pl.clustering_agreement(sw[a], sw[b])["ARI"] for a, b in itertools.combinations(cols, 2)]
    j = pl.cluster_stability(sw[cols[0]], [sw[c] for c in cols[1:]])
    jac[r] = j
    tam = sw[cols[0]].value_counts()
    linhas.append({"resolução": r, "clusters": int(sw[cols[0]].nunique()),
                   "clusters por semente": " / ".join(str(sw[c].nunique()) for c in cols),
                   "ARI entre sementes (média)": float(np.mean(aris)), "ARI entre sementes (mínimo)": float(np.min(aris)),
                   "clusters estáveis (Jaccard ≥ 0,75)": int((j >= 0.75).sum()),
                   "células em clusters estáveis": float(tam.reindex(j.index[j >= 0.75]).sum() / N),
                   "menor cluster": int(tam.min())})
estab = pd.DataFrame(linhas).set_index("resolução")
RES_GROSSA = float(estab["ARI entre sementes (média)"].idxmax())
finas = estab[estab["clusters"] >= 10]
RES_FINAL = float(finas["ARI entre sementes (média)"].idxmax())
io.save_table(estab, NB, "estabilidade_por_resolucao", description="Leiden por resolução: número de clusters e estabilidade entre sementes")
print(f"{len(RES) * len(SEEDS)} execuções do Leiden em {time.time() - t0:.0f} s; escala grossa: {F.f(RES_GROSSA, 1)} "
      f"({estab.loc[RES_GROSSA, 'clusters']} clusters); escala fina (principal): {F.f(RES_FINAL, 1)} "
      f"({estab.loc[RES_FINAL, 'clusters']} clusters); resolução do original (0,5): ARI {F.f(estab.loc[0.5, 'ARI entre sementes (média)'], 2)}")
estab.round(3)

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
a1.plot(RES, estab["clusters"], "-o", color=plotting.CATEGORICAL[0], lw=2, ms=7, mec=INK["surface"], mew=1.2)
a1.set_xlabel("resolução do Leiden")
a1.set_ylabel("clusters (semente 0)")
a1.set_title("Número de clusters")
a2.plot(RES, estab["ARI entre sementes (média)"], "-o", color=plotting.CATEGORICAL[0], lw=2, ms=7, mec=INK["surface"],
        mew=1.2, label="ARI entre sementes (média)")
a2.plot(RES, estab["células em clusters estáveis"], "-s", color=plotting.CATEGORICAL[1], lw=2, ms=7, mec=INK["surface"],
        mew=1.2, label="células em clusters estáveis (Jaccard ≥ 0,75)")
a2.set_ylim(0, 1.02)
plotting.decimals(a2, "y", 1)
a2.set_xlabel("resolução do Leiden")
a2.set_title("Estabilidade entre sementes")
a2.legend(loc="lower left")
for ax in (a1, a2):
    for r_, ls_ in ((RES_GROSSA, (0, (1, 2))), (RES_FINAL, (0, (4, 3)))):
        ax.axvline(r_, color=INK["axis"], lw=1, ls=ls_, zorder=0)
    ax.set_xticks(RES)
    plotting.decimals(ax, "x", 1)
for r_, nome in ((RES_GROSSA, "grossa"), (RES_FINAL, "fina (principal)")):
    a1.annotate(nome, (r_, a1.get_ylim()[1]), xytext=(4, -4), textcoords="offset points", va="top",
                fontsize=8.5, color=INK["secondary"])
plotting.note(fig, f"{F.n(N)} células · {plotting.N_DESIGN}. Três sementes por resolução, sobre o grafo do Harmony.", y=0.0)
fig.tight_layout()
plotting.savefig(fig, NB, "resolucao", description="Número de clusters e estabilidade por resolução do Leiden",
                 data=estab.reset_index(), data_index=False)
plt.show()

# %% [markdown]
# ## 3. Os clusters na resolução escolhida

# %%
cl = pd.Series(sw[f"leiden_r{RES_FINAL:g}_s0"].to_numpy(), index=obs.index)
CL = sorted(cl.unique(), key=int)
j_final = jac[RES_FINAL].reindex(CL)
share_pat = pl.share_within_group(obs.assign(cl=cl.to_numpy()), "cl", "patient", PATS)
share_lib = pl.share_within_group(obs.assign(cl=cl.to_numpy()), "cl", "library_id", LIBS)
ne_pat = pl.effective_number(share_pat).reindex(CL)
ct_pat = pd.crosstab(cl, obs["patient"].astype(str)).reindex(index=CL, columns=PATS)
bp = _bool(obs["baixa_profundidade"])
so = _bool(obs["so_no_padrao"])
g = obs.groupby(cl.to_numpy())
tab = pd.DataFrame({
    "células": cl.value_counts().reindex(CL),
    "estabilidade (Jaccard)": j_final,
    "nº efetivo de pacientes": ne_pat,
    "paciente dominante": ct_pat.idxmax(axis=1),
    "fração do paciente dominante": ct_pat.max(axis=1) / ct_pat.sum(axis=1),
    "UMIs medianos": g["total_counts"].median().reindex(CL),
    "genes medianos": g["n_genes_by_counts"].median().reindex(CL),
    "% MT mediana": g["pct_counts_mt"].median().reindex(CL),
    "baixa profundidade": pd.Series(bp, index=obs.index).groupby(cl).mean().reindex(CL),
    "do Cell Ranger que o MAD descartaria": pd.Series(so, index=obs.index).groupby(cl).mean().reindex(CL),
    "marcadores T (mediana)": g["frac_T"].median().reindex(CL),
    "marcadores B (mediana)": g["frac_B"].median().reindex(CL),
    "marcadores mieloides (mediana)": g["frac_mieloide"].median().reindex(CL),
})
motivo = []
for c in CL:
    r = tab.loc[c]
    m = []
    if r["baixa profundidade"] > 0.3:
        m.append(f"{F.pct(r['baixa profundidade'], 0)} de baixa profundidade")
    if r["UMIs medianos"] < 0.25 * np.median(umis):
        m.append(f"baixo conteúdo de RNA (mediana de {F.n(r['UMIs medianos'])} UMIs)")
    if r["marcadores T (mediana)"] < 0.001 and (r["marcadores B (mediana)"] > 0.001 or r["marcadores mieloides (mediana)"] > 0.001):
        m.append("não-T (marcadores de B ou mieloides)")
    if r["nº efetivo de pacientes"] < 2 and r["fração do paciente dominante"] > 0.5:
        m.append(f"de um paciente só ({r['paciente dominante']})")
    motivo.append("; ".join(m))
tab["marca"] = motivo
tab["técnico"] = tab["marca"].str.contains("baixa profundidade|não-T|baixo conteúdo")
io.save_table(tab, NB, "clusters", description="Clusters na resolução escolhida: estabilidade, composição, QC e linhagem")
print(f"resolução {F.f(RES_FINAL, 1)}: {len(CL)} clusters; técnicos: "
      + (", ".join(f"{c} ({tab.loc[c, 'marca']})" for c in CL if tab.loc[c, "técnico"]) or "nenhum"))
tab.round(3)

# %%
um = adata.obsm["X_umap"]
fig = plotting.umap_highlight_grid(um, cl.to_numpy(), order=CL, ncols=6, panel=2.0, s=1.3,
                                   title_fmt=lambda gname, n: f"cluster {gname}  (n={F.n(n)})"
                                   + ("  · técnico" if tab.loc[gname, "técnico"] else ""))
plotting.note(fig, f"Resolução {F.f(RES_FINAL, 1)} · {F.n(N)} células · {plotting.N_DESIGN}. Os números só identificam os "
                   "clusters; não há anotação.", y=0.0)
plotting.savefig(fig, NB, "umap_por_cluster", description="UMAP com cada cluster em destaque",
                 data=tab[["células", "estabilidade (Jaccard)", "marca"]].rename_axis("cluster").reset_index(), data_index=False)
plt.show()

# %%
fig = plotting.umap_highlight_grid(um, lib_arr, order=LIBS, ncols=7, panel=2.0, s=1.3)
plotting.note(fig, f"{F.n(N)} células · {plotting.N_DESIGN}. Em cada painel, azul = células da biblioteca; cinza = demais.", y=0.0)
plotting.savefig(fig, NB, "umap_por_biblioteca", description="UMAP com cada biblioteca em destaque",
                 data=n_lib.rename("células").rename_axis("library_id").reset_index(), data_index=False)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(CL) + 2.2))
m = share_lib.reindex(CL) * 100
m.index = [f"{c}  ·  n={F.n(tab.loc[c, 'células'])}  ·  {F.f(tab.loc[c, 'nº efetivo de pacientes'], 1)} pac."
           + ("  · técnico" if tab.loc[c, "técnico"] else "") for c in CL]
m.columns = [f"{l.split('_')[0]}\n{l.split('_')[1]}" for l in LIBS]
plotting.heatmap(ax, m, vmin=0, vmax=60, fmt=".0f", annotate_above=10, cbar_label="% das células da biblioteca", xrotation=0)
ax.set_ylabel("cluster · células · nº efetivo de pacientes")
ax.set_title(f"Fração de cada biblioteca em cada cluster (resolução {F.f(RES_FINAL, 1)})")
plotting.note(fig, f"{F.n(N)} células · {plotting.N_DESIGN}. Cada coluna soma 100%; escala cortada em 60%; só valores a partir "
                   "de 10% são escritos.", y=0.0)
fig.tight_layout()
plotting.savefig(fig, NB, "composicao", description="Fração das células de cada biblioteca em cada cluster",
                 data=share_lib.reindex(CL).rename_axis("cluster").reset_index(), data_index=False)
plt.show()

# %% [markdown]
# ## 4. Profundidade dentro de cada biblioteca

# %%
eta = pl.eta_squared_within(np.log10(umis), cl.to_numpy(), lib_arr)
nt = ~cl.map(tab["técnico"]).to_numpy().astype(bool)
eta_nt = pl.eta_squared_within(np.log10(umis[nt]), cl.to_numpy()[nt], lib_arr[nt])
grossa = sw[f"leiden_r{RES_GROSSA:g}_s0"].to_numpy()
eta_g = pl.eta_squared_within(np.log10(umis), grossa, lib_arr)
print(f"variância de log10(UMIs) explicada pelo cluster, dentro de cada biblioteca (mediana): escala fina {F.f(eta.median(), 2)}; "
      f"escala fina sem os clusters técnicos {F.f(eta_nt.median(), 2)}; escala grossa {F.f(eta_g.median(), 2)} "
      f"(V3 de A05: 0,22; processamento original, V1: 0,57)")
io.save_table(pd.DataFrame({"escala fina": eta, "escala fina sem técnicos": eta_nt, "escala grossa": eta_g}).rename_axis("library_id"),
              NB, "profundidade_eta2", description="Fração da variância de log10(UMIs) explicada pelo cluster, por biblioteca")

# %% [markdown]
# ## 5. Citocinas por cluster
#
# Contagens corrigidas pelo SoupX (`layers["counts"]`). Positividade com
# ≥ 1 UMI e, mais conservadora, ≥ 2 UMIs.

# %%
CYT = [gname for gname in config.CYTOKINES if gname in adata.var_names]
C = adata[:, CYT].layers["counts"]
C = pd.DataFrame(C.toarray() if sp.issparse(C) else C, index=obs.index, columns=CYT)
pos1 = (C >= 1).groupby(cl).mean().reindex(CL) * 100
pos2 = (C >= 2).groupby(cl).mean().reindex(CL) * 100
io.save_table(pos1, NB, "citocinas_por_cluster_1umi", description="% de células positivas (≥ 1 UMI) por cluster")
io.save_table(pos2, NB, "citocinas_por_cluster_2umis", description="% de células positivas (≥ 2 UMIs) por cluster")
ordem_g = (C >= 1).sum().sort_values(ascending=False).index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(CL) + 2.4), sharey=True)
for ax, (nome, m) in zip(axes, [("≥ 1 UMI", pos1), ("≥ 2 UMIs", pos2)]):
    mm = m[ordem_g].copy()
    mm.index = [f"{c}  (n={F.n(tab.loc[c, 'células'])})" for c in CL]
    plotting.heatmap(ax, mm, vmin=0, vmax=20, fmt=".1f", annotate_above=1.0, cbar_label="% das células do cluster")
    ax.set_title(f"Células positivas: {nome}")
axes[0].set_ylabel(f"cluster (resolução {F.f(RES_FINAL, 1)})")
plotting.note(fig, f"{F.n(N)} células · {plotting.N_DESIGN}. Contagens SoupX. Escala cortada em 20%; só valores a partir de 1% são "
                   "escritos.", y=0.0)
fig.tight_layout()
plotting.savefig(fig, NB, "citocinas_por_cluster", description="% de células positivas para cada citocina, por cluster",
                 data=pos1.rename_axis("cluster").reset_index(), data_index=False)
plt.show()

# %% [markdown]
# ## 6. Estimulado × não estimulado na composição dos clusters
#
# Para cada cluster, a fração das células de cada biblioteca que cai nele,
# pareada dentro do paciente (n = 7). Descritivo: com 7 pares, o menor p exato
# possível é 0,016, e nenhum cluster resiste à correção para o número de
# clusters. A integração por biblioteca também aproxima as duas bibliotecas de
# cada paciente (seção 1).

# %%
sh = share_lib.reindex(CL)
linhas = []
for c in CL:
    ns = pd.Series({p: sh.loc[c, f"{p}_NS"] for p in PATS})
    ml = pd.Series({p: sh.loc[c, f"{p}_ML"] for p in PATS})
    d = ml - ns
    pval = (float(stats.wilcoxon(ml, ns, zero_method="wilcox", alternative="two-sided", method="exact").pvalue)
            if (d != 0).all() else np.nan)
    linhas.append({"cluster": c, "mediana_NS": float(ns.median()), "mediana_ML": float(ml.median()),
                   "pacientes_ML_maior": int((d > 0).sum()), "p_wilcoxon_exato": pval})
estim = pd.DataFrame(linhas).set_index("cluster")
asc = estim["p_wilcoxon_exato"].dropna().sort_values()
estim["p_BH"] = (asc * len(asc) / np.arange(1, len(asc) + 1))[::-1].cummin()[::-1].clip(upper=1).reindex(estim.index)
io.save_table(estim, NB, "estimulo_por_cluster", description="Fração de cada biblioteca por cluster, estimulado × não estimulado (n = 7)")
estim.round(4)

# %% [markdown]
# ## 7. Rótulos

# %%
rot = pd.DataFrame({"cell_key": obs["cell_key"].to_numpy(), "library_id": lib_arr,
                    "leiden_final": cl.to_numpy(), "leiden_grossa": sw[f"leiden_r{RES_GROSSA:g}_s0"].to_numpy(),
                    "cluster_tecnico": cl.map(tab["técnico"]).to_numpy()})
for r in RES:
    rot[f"leiden_r{r:g}"] = sw[f"leiden_r{r:g}_s0"].to_numpy()
rot.attrs["resolucao_final"] = RES_FINAL
io.save_data(rot, NB, "clusters",
             description=f"Rótulos do Leiden por resolução (semente 0), final (resolução {RES_FINAL:g}) e grossa "
                         f"(resolução {RES_GROSSA:g}) do objeto principal; chave cell_key")
gros = pd.crosstab(rot["leiden_final"], rot["leiden_grossa"]).reindex(CL)
io.save_table(gros, NB, "fina_x_grossa", description="Clusters da escala fina dentro dos da escala grossa")
print(f"{F.n(len(rot))} células; final: resolução {F.f(RES_FINAL, 1)}, {len(CL)} clusters, "
      f"{F.n(rot['cluster_tecnico'].sum())} células em clusters técnicos")

# %% [markdown]
# ## Problemas encontrados

# %%
log.add("INFO", "integração com Harmony (por biblioteca)", "[VERIFICAÇÃO] o Harmony mistura bibliotecas e pacientes",
        evidence=f"vizinhos da mesma biblioteca {F.pct(ma.iloc[0], 0)} → {F.pct(mb.iloc[0], 0)} (mediana); iLISI de pacientes "
                 f"{F.f(ma['iLISI de pacientes (mediana)'], 2)} → {F.f(mb['iLISI de pacientes (mediana)'], 2)} "
                 f"(máximo {F.f(mb['iLISI de pacientes, mistura perfeita'], 2)}); as mais separadas depois: "
                 + ", ".join(f"{l} {F.x(v, 1)}" for l, v in resid.head(4).items()),
        impact="as bibliotecas que continuam separadas são as de perfil atípico (CTH97_ML) ou poucas células",
        recommendation="nenhuma; manter a correção por biblioteca, como no original")
log.add("MAJOR", "Harmony por biblioteca × estímulo",
        "a correção por biblioteca também aproxima as duas bibliotecas de cada paciente (estímulo e canal confundidos)",
        evidence=f"separação NS × ML no paciente {F.f(ma['separação NS × ML no paciente (mediana)'], 2)} → "
                 f"{F.f(mb['separação NS × ML no paciente (mediana)'], 2)}",
        impact="clusters próprios do estímulo ficam atenuados no espaço integrado",
        recommendation="testar o estímulo por pseudobulk pareado nas contagens, não pela composição dos clusters")
log.add("MINOR", "resolução do Leiden", "a resolução do original (0,5) está entre as menos estáveis",
        evidence=f"ARI entre sementes: 0,5 → {F.f(estab.loc[0.5, 'ARI entre sementes (média)'], 2)}; escala grossa "
                 f"{F.f(RES_GROSSA, 1)} → {F.f(estab.loc[RES_GROSSA, 'ARI entre sementes (média)'], 2)} "
                 f"({estab.loc[RES_GROSSA, 'clusters']} clusters); escala fina {F.f(RES_FINAL, 1)} → "
                 f"{F.f(estab.loc[RES_FINAL, 'ARI entre sementes (média)'], 2)} ({estab.loc[RES_FINAL, 'clusters']} clusters)",
        impact="a partição da massa central de células T é um recorte de um contínuo; só parte dos clusters é estável",
        recommendation=f"usar a escala fina ({RES_FINAL:g}) para anotar e a grossa ({RES_GROSSA:g}) como agrupamento robusto")
tec = tab[tab["técnico"]]
log.add("MINOR", f"clusters técnicos (resolução {RES_FINAL:g})", "parte dos clusters é técnica ou não-T",
        evidence="; ".join(f"cluster {c}: {F.n(r['células'])} células ({r['marca']})" for c, r in tec.iterrows()),
        impact=f"a profundidade explica {F.f(eta.median(), 2)} da variância de UMIs dentro da biblioteca com eles e "
               f"{F.f(eta_nt.median(), 2)} sem eles",
        recommendation="excluir ou marcar esses clusters antes da anotação (coluna cluster_tecnico)")
issues = log.save()
print(log.summary().to_string())
show(issues)

# %% [markdown]
# ## Conclusão
#
# **Integração.** O Harmony por biblioteca mistura bem: os vizinhos da mesma
# biblioteca caem de 41% para 11% (mediana) e o número efetivo de pacientes
# na vizinhança vai de 1,4 para 2,9. Como já visto em A04, a correção também
# aproxima as duas bibliotecas de cada paciente (0,54 → 0,22), porque
# estímulo e canal estão confundidos.
#
# **Clusterização.** A resolução 0,5 do original é das menos estáveis. Duas
# escalas ficam no arquivo de rótulos: a **fina (0,8, 14 clusters)**, a
# principal, e a **grossa (0,2, 6 clusters)**, a mais estável. Na escala fina,
# os clusters técnicos (não-T e de baixo conteúdo de RNA, onde caem as células
# de baixa profundidade) estão marcados em `cluster_tecnico`; fora deles, a
# profundidade pesa pouco na partição. Um cluster com mais células nas
# bibliotecas estimuladas em todos os 7 pacientes concentra parte de CTH97_ML —
# hipótese para a anotação, não resultado.
#
# Rótulos: `dataReseq/P07_integracao_clusterizacao/clusters__v3.parquet`
# (`leiden_final`, `leiden_grossa`, `leiden_r*`, `cluster_tecnico`), unidos ao
# objeto principal por `cell_key`.

# %%
feitos = io.manifest()
feitos = feitos[feitos["stage"] == NB][["name", "kind"]].drop_duplicates(keep="last")
print(f"{len(feitos)} artefatos gravados por este notebook:")
print(feitos.to_string(index=False))
