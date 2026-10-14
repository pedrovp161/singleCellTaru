"""
Registro de problemas encontrados, no mesmo formato da auditoria anterior
(`severity`, `location`, `problem`, `impact`, `recommendation`) mais a
evidência numérica que sustenta cada linha.

Gravidade:

* ``CRITICAL`` — compromete o objetivo do ressequenciamento ou uma conclusão a jusante;
* ``MAJOR``    — limita de forma material o que pode ser analisado;
* ``MINOR``    — precisa de cuidado, mas não muda conclusões;
* ``INFO``     — registro, verificação que deu certo ou não-problema.
"""

from __future__ import annotations

import pandas as pd

from . import io

SEVERITY_ORDER = ["CRITICAL", "MAJOR", "MINOR", "INFO"]
COLUMNS = ["severity", "location", "problem", "evidence", "impact", "recommendation"]


class IssueLog:
    def __init__(self, notebook: str) -> None:
        self.notebook = notebook
        self._rows: list[dict] = []

    def add(self, severity: str, location: str, problem: str, *, evidence: str,
            impact: str, recommendation: str) -> None:
        if severity not in SEVERITY_ORDER:
            raise ValueError(f"gravidade inválida: {severity}")
        self._rows.append(dict(severity=severity, location=location, problem=problem,
                               evidence=evidence, impact=impact,
                               recommendation=recommendation))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=COLUMNS)
        df["severity"] = pd.Categorical(df["severity"], SEVERITY_ORDER, ordered=True)
        return df.sort_values("severity", kind="stable").reset_index(drop=True)

    def summary(self) -> pd.Series:
        return self.to_frame()["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)

    def save(self) -> pd.DataFrame:
        df = self.to_frame()
        io.save_table(df.assign(notebook=self.notebook), self.notebook, "issues",
                      description="Problemas encontrados neste notebook", index=False)
        return df


def show(df: pd.DataFrame):
    """Exibe a tabela de problemas com as células de texto por inteiro."""
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        try:
            from IPython.display import display
            display(df.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"})
                    .hide(axis="index"))
        except Exception:
            print(df.to_string(index=False))
