# [P15] Cópia de /mnt/SATA/spatialPaper/src/forest_composicao/composicao.py (projeto espacial), sem mudança de método.
# Edições: "spots" → "células" nos textos; o resto é o arquivo original.
"""Composicao de nichos entre grupos de resposta -- CLR + modelo linear + BH.

Metodo: proporcao por AMOSTRA -> CLR -> `lm(CLR ~ resposta + covariaveis)` ->
contraste por medias marginais -> Benjamini-Hochberg entre os nichos.

Para um modelo aditivo o contraste de medias marginais do `emmeans` (R) e
exatamente o coeficiente do grupo -- os termos das covariaveis se cancelam na
diferenca -- entao o OLS do `statsmodels` reproduz `pairs(emmeans(fit, specs =
group_col))` sem aproximacao.

Tres pontos que o desenho deste dado impoe:

**1. A amostra e a `library_id`, nao o spot.** Sao 51.263 spots em 28 amostras
(uma por paciente). Testar no spot trata 51 mil observacoes como independentes
quando o n real e 28: os intervalos ficariam ~40x estreitos demais e quase tudo
sairia "significativo". A agregacao para amostra e o que torna o teste valido.

**2. O pseudocount do CLR vai na escala de CONTAGEM**, `(k+0,5)/(N+0,5K)`.
Somar 1e-6 a proporcao -- o atalho usual -- cria um degrau: `log(1e-6) = -13,8`
contra `log(0,05) = -3,0` de um nicho normal, e cada amostra sem aquele nicho
vira um outlier que domina o `lm`. O ajuste por contagem depende da
profundidade da amostra e nao cria esse degrau.

**3. Filtro de prevalencia, nao apenas variancia > 0.** Um nicho so e testado se
aparece a `min_frac` dos spots em pelo menos `min_pac` amostras de CADA grupo;
caso contrario o "efeito" e a amostra de uma ou duas amostras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  1. Composicao por amostra
# --------------------------------------------------------------------------- #
def build_composition(adata, cell_col: str, sample_col: str = "library_id",
                      group_col: str = "response",
                      covars: list[str] | None = None,
                      min_cells: int = 30) -> dict:
    """Contagem e proporcao de cada nicho por amostra, com a metadata da amostra."""
    obs = adata.obs if hasattr(adata, "obs") else adata
    covars = list(covars or [])
    cols = [sample_col, group_col] + covars
    # `astype(str)` so nas colunas categoricas: uma covariavel numerica (ex.: o
    # %MT da amostra) precisa chegar ao modelo como numero, nao como rotulo.
    meta = obs[cols].copy()
    for c in meta.columns:
        if not pd.api.types.is_numeric_dtype(meta[c]):
            meta[c] = meta[c].astype(str)
    meta = meta.drop_duplicates(subset=[sample_col])

    n_por_amostra = obs.groupby(sample_col, observed=True)[cell_col].size()
    fora = n_por_amostra[n_por_amostra < min_cells].index.tolist()
    if fora:
        print(f"amostras com menos de {min_cells} células, removidas: {fora}")
    manter = n_por_amostra[n_por_amostra >= min_cells].index.astype(str)

    counts = pd.crosstab(obs[sample_col].astype(str),
                         obs[cell_col].astype(str)).loc[manter].sort_index()
    prop = counts.div(counts.sum(axis=1), axis=0)
    meta = meta.set_index(sample_col).loc[counts.index]

    return {"counts": counts, "proportion": prop, "meta": meta,
            "cell_cols": list(counts.columns), "sample_col": sample_col,
            "group_col": group_col, "covars": covars}


# --------------------------------------------------------------------------- #
#  2. Filtro de prevalencia
# --------------------------------------------------------------------------- #
def filtro_prevalencia(comp: dict, min_pac: int = 3, min_frac: float = 0.01) -> dict:
    """Nichos presentes a >= `min_frac` em >= `min_pac` amostras de CADA grupo."""
    prop, meta = comp["proportion"], comp["meta"]
    g = meta[comp["group_col"]]
    linhas = {}
    for ct in comp["cell_cols"]:
        por_grupo = {gr: int((prop.loc[g[g == gr].index, ct] >= min_frac).sum())
                     for gr in g.unique()}
        linhas[ct] = {**{f"n_{k}": v for k, v in por_grupo.items()},
                      "min_grupos": min(por_grupo.values()),
                      "passa": min(por_grupo.values()) >= min_pac}
    tab = pd.DataFrame(linhas).T
    tab["passa"] = tab["passa"].astype(bool)
    return {"tabela": tab, "testaveis": tab.index[tab["passa"]].tolist(),
            "excluidos": tab.index[~tab["passa"]].tolist()}


# --------------------------------------------------------------------------- #
#  3. CLR
# --------------------------------------------------------------------------- #
def clr_transform(comp: dict, cell_cols=None, pseudo="contagem") -> pd.DataFrame:
    """Log-razao centrada. `pseudo='contagem'` usa (k+0,5)/(N+0,5K)."""
    cell_cols = cell_cols or comp["cell_cols"]
    k = comp["counts"][cell_cols].to_numpy(dtype=float)
    if pseudo == "contagem":
        p = (k + 0.5) / (k.sum(axis=1, keepdims=True) + 0.5 * k.shape[1])
    else:
        p = comp["proportion"][cell_cols].to_numpy(dtype=float) + float(pseudo)
    gm = np.exp(np.log(p).mean(axis=1, keepdims=True))
    return pd.DataFrame(np.log(p / gm), index=comp["counts"].index, columns=cell_cols)


# --------------------------------------------------------------------------- #
#  4. Modelo e contraste
# --------------------------------------------------------------------------- #
def _matriz_covariaveis(meta: pd.DataFrame, covars: list[str], idx) -> pd.DataFrame:
    """Dummies das covariaveis, descartando as que nao variam (1 nivel so).

    Uma covariavel com um unico nivel gera zero coluna util; uma com um nivel de
    UMA observacao gera um dummy que ajusta aquele ponto perfeitamente, tirando-o
    do contraste em silencio. As duas situacoes sao reportadas.
    """
    partes, notas = [], []
    for c in covars:
        col = meta[c]
        # Numerica entra como UMA coluna padronizada. Sem esta ramificacao o
        # `%MT` -- que tem um valor distinto por amostra -- viraria 27 dummies e
        # saturaria o modelo (28 amostras).
        if pd.api.types.is_numeric_dtype(col):
            v = pd.to_numeric(col, errors="coerce").astype(float)
            dp = v.std()
            if not np.isfinite(dp) or dp == 0:
                notas.append(f"'{c}' é constante — descartada")
                continue
            z = ((v - v.mean()) / dp).to_frame(f"{c}_z")
            z.index = idx
            partes.append(z)
            continue

        v = col.astype(str)
        vc = v.value_counts()
        if len(vc) < 2:
            notas.append(f"'{c}' tem 1 nível só ({vc.index[0]}) — descartada")
            continue
        singleton = vc[vc == 1]
        if len(singleton):
            notas.append(
                f"'{c}' tem nível(is) com 1 amostra: {dict(singleton)} — o dummy "
                f"ajusta esse ponto perfeitamente e ele sai do contraste")
        D = pd.get_dummies(v, drop_first=True, prefix=c, dtype=float)
        D.index = idx
        partes.append(D)
    X = pd.concat(partes, axis=1) if partes else pd.DataFrame(index=idx)
    return X, notas


def run_contrast(comp: dict, cell_cols=None, ordem_grupos=None,
                 covars: list[str] | None = None, pseudo="contagem",
                 verbose: bool = True) -> pd.DataFrame:
    """`lm(CLR ~ grupo [+ covariaveis])` por nicho, com BH entre nichos.

    `ordem_grupos = [g1, g2]` fixa o contraste como `g1 - g2`: efeito positivo
    significa enriquecido em `g1`.
    """
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    cell_cols = cell_cols or comp["cell_cols"]
    covars = comp["covars"] if covars is None else list(covars)
    clr = clr_transform(comp, cell_cols, pseudo=pseudo)
    meta = comp["meta"]
    g = meta[comp["group_col"]].astype(str)
    niveis = ordem_grupos or sorted(g.unique())
    if len(niveis) != 2:
        raise ValueError(f"esperados 2 grupos, achei {niveis}")
    g1, g2 = niveis

    X = pd.get_dummies(pd.Categorical(g, categories=[g2, g1]),
                       drop_first=True, prefix="grp", dtype=float)
    X.index = clr.index
    C, notas = _matriz_covariaveis(meta, covars, clr.index)
    if verbose:
        for n in notas:
            print("  ⚠ " + n)
    X = sm.add_constant(pd.concat([X, C], axis=1), has_constant="add")
    jcol = list(X.columns).index(f"grp_{g1}")

    Xv = X.to_numpy(dtype=float)
    posto = np.linalg.matrix_rank(Xv)
    if posto < Xv.shape[1]:
        raise ValueError(
            f"matriz de desenho deficiente (posto {posto} < {Xv.shape[1]} colunas): "
            f"covariáveis colineares. Colunas: {list(X.columns)}")

    linhas = []
    for ct in cell_cols:
        fit = sm.OLS(clr[ct].to_numpy(dtype=float), Xv).fit()
        lo, hi = fit.conf_int()[jcol]
        linhas.append({
            "cell_population": ct, "comparison": f"{g1} - {g2}", "g1": g1, "g2": g2,
            "estimate": fit.params[jcol], "std_error": fit.bse[jcol],
            "conf_low": lo, "conf_high": hi, "p_value": fit.pvalues[jcol],
            "df_resid": fit.df_resid,
            "prop_media_g1": float(comp["proportion"].loc[g[g == g1].index, ct].mean()),
            "prop_media_g2": float(comp["proportion"].loc[g[g == g2].index, ct].mean()),
        })
    res = pd.DataFrame(linhas)
    res["FDR"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["direction"] = np.where(res["estimate"] > 0, res["g1"], res["g2"])
    res["sig_shape"] = np.select(
        [res["FDR"] < 0.05, res["p_value"] < 0.05],
        ["FDR < 0.05", "p < 0.05"], default="Não significativo")
    res.attrs["modelo"] = "CLR ~ " + " + ".join([comp["group_col"]] + list(covars))
    res.attrs["covars"] = covars
    res.attrs["n_amostras"] = int(len(clr))
    return res.sort_values("p_value").reset_index(drop=True)
