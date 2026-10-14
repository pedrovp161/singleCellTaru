"""
Testes da hipótese de excesso de doublets.

O que esta tecnologia permite e o que não permite. Cada biblioteca tem um único
doador e não há BAM: não existe demultiplexação genética nem reconstrução de
TCR, que seriam as provas diretas. Sobram quatro linhas de evidência indireta,
cada uma com um alcance diferente:

* **Aritmética do carregamento** — quantas células por gota seriam precisas.
* **Régua de RNA** — um doublet de duas células do mesmo tipo tem ~2× os UMIs.
  Mistura de duas gaussianas em log-UMI com deslocamento fixo de log 2: estima
  um *teto* para doublets homotípicos (células grandes entram na conta).
* **"Barnyard" interno** — os poucos contaminantes (células B) funcionam como a
  segunda espécie de um experimento de mistura: a fração deles que também
  carrega uma célula T dá a taxa de coencapsulamento.
* **Scrublet** — só enxerga doublets heterotípicos.

As funções de taxa esperada e de threshold calibrado são as da auditoria
anterior (`src/scripts/claude/functions/doublets.py`), reutilizadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import legacy

_ld = legacy.module("doublets")
expected_doublet_rate_10x = _ld.expected_doublet_rate_10x
threshold_by_expected_rate = _ld.threshold_by_expected_rate

LOG10_2 = float(np.log10(2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Régua de RNA
# ─────────────────────────────────────────────────────────────────────────────

def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def fit_doublet_mixture(x: np.ndarray, shift: float = LOG10_2, n_iter: int = 300,
                        tol: float = 1e-8) -> dict:
    """Mistura de dois componentes em log10(UMIs): singletos N(μ, σ²) e
    doublets N(μ + log10 2, σ²), mesma variância.

    Devolve ``pi`` (fração no componente deslocado), ``mu``, ``sd`` e o ganho
    de log-verossimilhança frente a uma única gaussiana. ``pi`` é um **teto**
    para doublets homotípicos: qualquer assimetria para a direita — células
    maiores, ativadas — é absorvida por ele.
    """
    x = np.asarray(x, dtype=float)
    mu = float(np.median(x))
    sd = float(max(1.4826 * np.median(np.abs(x - mu)), 1e-3))
    pi = 0.05
    last = -np.inf
    for _ in range(n_iter):
        p1 = (1 - pi) * _norm_pdf(x, mu, sd)
        p2 = pi * _norm_pdf(x, mu + shift, sd)
        tot = p1 + p2 + 1e-300
        r = p2 / tot
        ll = float(np.log(tot).sum())
        pi = float(np.clip(r.mean(), 1e-6, 0.999))
        mu = float(((1 - r) * x + r * (x - shift)).mean())
        sd = float(np.sqrt(((1 - r) * (x - mu) ** 2 + r * (x - mu - shift) ** 2).mean()))
        sd = max(sd, 1e-3)
        if abs(ll - last) < tol * max(1.0, abs(ll)):
            break
        last = ll
    ll1 = float(np.log(_norm_pdf(x, x.mean(), max(x.std(), 1e-3)) + 1e-300).sum())
    return {"pi": pi, "mu": mu, "sd": sd, "loglik": ll, "delta_loglik": ll - ll1, "n": int(len(x))}


def bootstrap_pi(x: np.ndarray, n_boot: int = 200, seed: int = 42) -> tuple[float, float]:
    """Intervalo percentil de 95% para ``pi``."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    vals = [fit_doublet_mixture(rng.choice(x, len(x), replace=True))["pi"] for _ in range(n_boot)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ─────────────────────────────────────────────────────────────────────────────
# Barnyard interno
# ─────────────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Intervalo de Wilson de 95% para uma proporção."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_from_contaminants(r: float, p_t: float, p_c: float) -> float:
    """Taxa de doublets ``d`` (fração dos barcodes com duas células) a partir de
    ``r``, a fração dos barcodes com um contaminante que também têm uma célula T.

    Sob coencapsulamento ao acaso, com frações ``p_t`` de T e ``p_c`` do
    contaminante entre as células:

        P(contaminante sozinho) = (1 − d)·p_c
        P(T + contaminante)     = d·2·p_t·p_c
        P(dois contaminantes)   = d·p_c²

    logo ``r = 2·d·p_t / (1 − d + 2·d·p_t + d·p_c)`` e
    ``d = r / (2·p_t − r·(2·p_t + p_c − 1))``.
    """
    if not np.isfinite(r):
        return np.nan
    den = 2 * p_t - r * (2 * p_t + p_c - 1)
    return float(r / den) if den > 0 else np.nan


def r_expected(d: float, p_t: float, p_c: float) -> float:
    """O inverso: ``r`` esperado para uma taxa de doublets ``d``."""
    return float(2 * d * p_t / (1 - d + 2 * d * p_t + d * p_c))


# ─────────────────────────────────────────────────────────────────────────────
# TRBC1 / TRBC2
# ─────────────────────────────────────────────────────────────────────────────

def mantel_haenszel_or(tables: list[np.ndarray]) -> tuple[float, float, float]:
    """Razão de chances comum de Mantel–Haenszel e IC 95% (Robins–Breslow–Greenland).

    Cada tabela é 2×2: linhas = exposto / não exposto, colunas = evento / não evento.
    """
    num = den = 0.0
    P = Q = R = S = PR = PSQR = QS = 0.0
    for t in tables:
        a, b, c, d = (float(v) for v in np.asarray(t).ravel())
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
        P_i, Q_i, R_i, S_i = (a + d) / n, (b + c) / n, a * d / n, b * c / n
        PR += P_i * R_i
        PSQR += P_i * S_i + Q_i * R_i
        QS += Q_i * S_i
        R += R_i
        S += S_i
    if den == 0 or num == 0:
        return (np.nan, np.nan, np.nan)
    or_mh = num / den
    var = PR / (2 * R ** 2) + PSQR / (2 * R * S) + QS / (2 * S ** 2)
    se = np.sqrt(var)
    return (float(or_mh), float(np.exp(np.log(or_mh) - 1.959964 * se)),
            float(np.exp(np.log(or_mh) + 1.959964 * se)))
