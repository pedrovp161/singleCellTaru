"""
Formatação de números em português (ponto de milhar, vírgula decimal).

Usada em textos impressos, rótulos de figura e tabelas exibidas, para que
"37.898 células" e "97,3%" nunca apareçam como "37,898" e "97.3%".
Os CSV gravados continuam com ponto decimal, que é o que os leitores de dados
esperam.
"""

from __future__ import annotations

import math


def _swap(s: str) -> str:
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def n(x) -> str:
    """Inteiro com ponto de milhar: 37898 → '37.898'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return _swap(f"{float(x):,.0f}")


def f(x, nd: int = 1) -> str:
    """Decimal com vírgula: 1.234,5."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return _swap(f"{float(x):,.{nd}f}")


def pct(x, nd: int = 1, signed: bool = False) -> str:
    """Fração → percentual: 0.973 → '97,3%'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    sign = "+" if signed else ""
    return _swap(f"{float(x) * 100:{sign},.{nd}f}") + "%"


def x(v, nd: int = 2) -> str:
    """Razão: 1.09 → '1,09×'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return _swap(f"{float(v):,.{nd}f}") + "×"


def compact(v) -> str:
    """Número grande compacto: 1.76e9 → '1,76 bi'; 443895 → '444 mil'."""
    v = float(v)
    if abs(v) >= 1e9:
        return f(v / 1e9, 2) + " bi"
    if abs(v) >= 1e6:
        return f(v / 1e6, 1) + " mi"
    if abs(v) >= 1e3:
        return f(v / 1e3, 0) + " mil"
    return f(v, 0)


def styler(df, formats: dict):
    """`df.style.format` com os formatadores acima. Ex.: {"celulas": n, "sat": pct}."""
    return df.style.format(formats, na_rep="–")
