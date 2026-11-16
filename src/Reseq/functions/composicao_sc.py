"""[P17] Adaptações do `forest_composicao` (projeto espacial) ao desenho do single-cell.

No espacial, cada amostra é um paciente (28 amostras, 28 pacientes): um único
`lm(CLR ~ grupo + covariáveis)` com todas as amostras. Aqui cada paciente tem
duas bibliotecas, NS e ML, o que é um desenho *split-plot*:

* **dentro do paciente** (estímulo): `lm(CLR ~ estímulo + paciente [+ %MT])` nas 14
  bibliotecas, que é o `composicao.run_contrast` sem alteração, com o paciente
  como covariável;
* **entre pacientes** (grupo clínico): as duas bibliotecas de um paciente não são
  independentes. Usar as 14 como se fossem 14 pacientes dobra o n de mentira.
  O teste entre pacientes usa uma observação por paciente: a **média do CLR das
  suas bibliotecas** (NS e ML). Com o desenho balanceado (todo paciente tem as
  duas), isso é exatamente o teste do estrato entre pacientes de um modelo misto
  com paciente aleatório. O CLR de cada biblioteca é o mesmo de
  `composicao.clr_transform` (pseudocontagem na escala de contagem).

O resto (filtro de prevalência, BH, `direction`, `sig_shape`, colunas do
resultado) é o do `composicao.py`, para que o `forest_composicao.forest`
desenhe os dois casos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import composicao as CP


def comp_paciente(comp: dict, patient_col: str, group_col: str, grupos, estimulo: str | None = None) -> dict:
    """Composição por paciente (média das proporções das suas bibliotecas), no formato de `composicao`.

    Covariáveis numéricas viram a média do paciente; categóricas, o valor (constante no paciente).
    `estimulo` restringe às bibliotecas de um estímulo (análise por condição).
    """
    m = comp["meta"]
    sel = m[group_col].astype(str).isin(grupos)
    if estimulo is not None:
        sel &= m["estimulo"].astype(str) == estimulo
    libs = m.index[sel]
    pac = m.loc[libs, patient_col].astype(str)
    prop = comp["proportion"].loc[libs].groupby(pac).mean()
    partes = {}
    for c in m.columns:
        v = m.loc[libs, c]
        partes[c] = v.groupby(pac).mean() if pd.api.types.is_numeric_dtype(v) else v.astype(str).groupby(pac).first()
    meta = pd.DataFrame(partes).loc[prop.index]
    meta.index.name = patient_col
    return {"proportion": prop, "meta": meta, "cell_cols": comp["cell_cols"], "group_col": group_col,
            "sample_col": patient_col, "covars": [], "libs": libs, "counts": None}


def run_contrast_paciente(comp: dict, comp_p: dict, cell_cols, ordem_grupos, covars=(),
                          pseudo="contagem") -> pd.DataFrame:
    """CLR por biblioteca → média por paciente → `OLS(CLR ~ grupo [+ covariáveis])` → BH entre os tipos.

    Mesmas colunas de `composicao.run_contrast`; `ordem_grupos = [g1, g2]` fixa o contraste g1 − g2.
    """
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    cell_cols = list(cell_cols)
    libs = comp_p["libs"]
    clr = CP.clr_transform(comp, cell_cols, pseudo=pseudo).loc[libs]
    pac = comp["meta"].loc[libs, comp_p["sample_col"]].astype(str)
    clr_p = clr.groupby(pac).mean()
    meta_p = comp_p["meta"].loc[clr_p.index]
    g = meta_p[comp_p["group_col"]].astype(str)
    g1, g2 = ordem_grupos

    X = pd.DataFrame({f"grp_{g1}": (g == g1).astype(float)}, index=clr_p.index)
    C, notas = CP._matriz_covariaveis(meta_p, list(covars), clr_p.index)
    for n in notas:
        print("  ⚠ " + n)
    X = sm.add_constant(pd.concat([X, C], axis=1), has_constant="add")
    jcol = list(X.columns).index(f"grp_{g1}")
    Xv = X.to_numpy(dtype=float)
    if np.linalg.matrix_rank(Xv) < Xv.shape[1]:
        raise ValueError(f"matriz de desenho deficiente: {list(X.columns)}")

    prop = comp_p["proportion"]
    linhas = []
    for ct in cell_cols:
        fit = sm.OLS(clr_p[ct].to_numpy(dtype=float), Xv).fit()
        lo, hi = fit.conf_int()[jcol]
        linhas.append({
            "cell_population": ct, "comparison": f"{g1} - {g2}", "g1": g1, "g2": g2,
            "estimate": fit.params[jcol], "std_error": fit.bse[jcol],
            "conf_low": lo, "conf_high": hi, "p_value": fit.pvalues[jcol],
            "df_resid": fit.df_resid,
            "prop_media_g1": float(prop.loc[g[g == g1].index, ct].mean()),
            "prop_media_g2": float(prop.loc[g[g == g2].index, ct].mean()),
        })
    res = pd.DataFrame(linhas)
    res["FDR"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["direction"] = np.where(res["estimate"] > 0, res["g1"], res["g2"])
    res["sig_shape"] = np.select([res["FDR"] < 0.05, res["p_value"] < 0.05],
                                 ["FDR < 0.05", "p < 0.05"], default="Não significativo")
    res.attrs["modelo"] = "média do CLR por paciente ~ " + " + ".join([comp_p["group_col"]] + list(covars))
    res.attrs["covars"] = list(covars)
    res.attrs["n_amostras"] = int(len(clr_p))
    return res.sort_values("p_value").reset_index(drop=True)


def filtro_prevalencia(comp: dict, min_pac: int = 3, min_frac: float = 0.01) -> dict:
    """`composicao.filtro_prevalencia` com `min_pac` limitado ao tamanho do menor grupo.

    Com 2 pacientes PB, "≥ 3 amostras de cada grupo" excluiria tudo; vale então "todas as do grupo".
    """
    n_menor = int(comp["meta"][comp["group_col"]].astype(str).value_counts().min())
    out = CP.filtro_prevalencia(comp, min_pac=min(min_pac, n_menor), min_frac=min_frac)
    out["min_pac"] = min(min_pac, n_menor)
    return out
