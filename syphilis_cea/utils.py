"""Numerical, formatting, and probability-distribution helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.patches import Ellipse


def dollar_fmt(x, _):
    return f"${x:,.0f}"


def millions_fmt(x, _):
    return f"${x * 1e-6:,.1f}M"


def std2(lo, hi):
    return (hi - lo) / 4.0


CPI = 585.10 / 494.629  # healthcare CPI 2019 -> 2025


def pvf(t: float, r: float) -> float:
    if t <= 0:
        return 0.0
    if r == 0:
        return float(t)
    return (1.0 - (1.0 + r) ** (-t)) / r


def _interval_containing_mean(m: float, lo: float, hi: float) -> Tuple[float, float]:
    """Return a valid open interval containing an interior probability mean."""

    eps = 1e-12
    lo = float(np.clip(lo, eps, 1.0 - eps))
    hi = float(np.clip(hi, eps, 1.0 - eps))
    if hi <= lo:
        width = max(abs(hi - lo), 1e-4)
        lo = max(eps, m - width / 2.0)
        hi = min(1.0 - eps, m + width / 2.0)

    if lo < m < hi:
        return lo, hi

    width = max(hi - lo, 0.10 * min(m, 1.0 - m), 1e-5)
    lo = min(lo, max(eps, m - width / 2.0))
    hi = max(hi, min(1.0 - eps, m + width / 2.0))
    if lo >= m:
        lo = max(eps, m - width)
    if hi <= m:
        hi = min(1.0 - eps, m + width)
    if not lo < m < hi:
        # Last-resort symmetric interval that is guaranteed to contain m.
        half = max(min(m, 1.0 - m) * 0.5, 1e-8)
        lo = max(eps, m - half)
        hi = min(1.0 - eps, m + half)
    return lo, hi


def beta_ab(m: float, lo: float, hi: float) -> Tuple[float, float]:
    """Convert an approximate 95% interval to beta shape parameters.

    ``beta_ab`` is defined only for interior means. Exact 0 and 1 are point
    masses and should be sampled with :func:`draw_beta`. Unlike the former
    implementation, this function never floors alpha and beta independently;
    therefore ``alpha / (alpha + beta)`` always equals the requested mean.
    """

    m = float(m)
    if not 0.0 < m < 1.0:
        raise ValueError("A beta distribution requires an interior mean; use draw_beta for 0 or 1")

    lo, hi = _interval_containing_mean(m, lo, hi)
    variance = max(((hi - lo) / 3.92) ** 2, 1e-14)
    max_variance = m * (1.0 - m)
    variance = min(variance, max_variance * (1.0 - 1e-10))
    concentration = m * (1.0 - m) / variance - 1.0
    concentration = max(concentration, 1e-6)
    alpha = m * concentration
    beta = (1.0 - m) * concentration
    return float(alpha), float(beta)


def draw_beta(
    rng: np.random.Generator,
    n: int,
    mean: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Draw a probability while preserving exact endpoint settings.

    Exact means of 0 and 1 return deterministic arrays. Interior means use a
    beta distribution whose expected value is exactly ``mean``.
    """

    mean = float(mean)
    if not 0.0 <= mean <= 1.0:
        raise ValueError(f"Probability mean must be in [0, 1]; got {mean}")
    if n < 1:
        raise ValueError("n must be at least 1")
    if mean == 0.0:
        return np.zeros(n, dtype=float)
    if mean == 1.0:
        return np.ones(n, dtype=float)
    alpha, beta = beta_ab(mean, lo, hi)
    return rng.beta(alpha, beta, size=n).astype(float)



def draw_probability(
    rng: np.random.Generator,
    n: int,
    mean: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Semantic alias for :func:`draw_beta`."""
    return draw_beta(rng, n, mean, lo, hi)

def gamma_ab(mu: float, sd: float) -> Tuple[float, float]:
    mu_f = float(mu)
    sd_f = float(sd)
    if mu_f <= 0.0 or sd_f <= 0.0:
        raise ValueError("Gamma shape parameters require positive mean and SD")
    return (mu_f / sd_f) ** 2, sd_f ** 2 / mu_f


def draw_gamma(
    rng: np.random.Generator,
    n: int,
    mean: float,
    sd: float,
) -> np.ndarray:
    """Draw a non-negative quantity while preserving exact zero settings.

    A zero mean is a deterministic point mass at zero. A positive mean with
    zero standard deviation is deterministic at the requested mean.
    """

    mean_f = float(mean)
    sd_f = float(sd)
    if mean_f < 0.0 or sd_f < 0.0:
        raise ValueError("Gamma mean and SD must be non-negative")
    if n < 1:
        raise ValueError("n must be at least 1")
    if mean_f == 0.0:
        return np.zeros(n, dtype=float)
    if sd_f == 0.0:
        return np.full(n, mean_f, dtype=float)
    shape, scale = gamma_ab(mean_f, sd_f)
    return rng.gamma(shape, scale, size=n).astype(float)


def lnorm_ms(m: float, lo: float, hi: float) -> Tuple[float, float]:
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo * 1.01)
    return np.log(max(float(m), 1e-12)), (np.log(hi) - np.log(lo)) / 3.92


def safe_icer(cost, effect):
    """Return cost/effect only where the incremental effect is positive.

    Negative and zero effects require dominance/NMB interpretation rather than
    an epsilon-adjusted ratio, so those entries are returned as ``NaN``.
    """

    cost_arr = np.asarray(cost, dtype=float)
    effect_arr = np.asarray(effect, dtype=float)
    result = np.full(np.broadcast(cost_arr, effect_arr).shape, np.nan, dtype=float)
    np.divide(cost_arr, effect_arr, out=result, where=effect_arr > 0.0)
    if result.ndim == 0:
        return float(result)
    return result


def ratio_of_means(cost, effect) -> float:
    """Conventional summary ICER: mean incremental cost / mean effect."""

    mean_cost = float(np.mean(np.asarray(cost, dtype=float)))
    mean_effect = float(np.mean(np.asarray(effect, dtype=float)))
    return float(safe_icer(mean_cost, mean_effect))


def summarize(a) -> dict:
    values = np.asarray(a, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "mean": np.nan,
            "median": np.nan,
            "95% CrI lo": np.nan,
            "95% CrI hi": np.nan,
        }
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "95% CrI lo": float(np.percentile(values, 2.5)),
        "95% CrI hi": float(np.percentile(values, 97.5)),
    }


def ci_ellipse(ax, x, y, ec="steelblue"):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 5:
        return
    mu = np.array([x.mean(), y.mean()])
    cov = np.cov(x, y)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    angle = np.arctan2(evec[1, 0], evec[0, 0])
    width = 2 * np.sqrt(max(ev[0], 0) * 5.991)
    height = 2 * np.sqrt(max(ev[1], 0) * 5.991)
    ax.add_patch(
        Ellipse(
            mu,
            width,
            height,
            angle=np.rad2deg(angle),
            edgecolor=ec,
            facecolor="none",
            lw=2,
            zorder=5,
        )
    )
