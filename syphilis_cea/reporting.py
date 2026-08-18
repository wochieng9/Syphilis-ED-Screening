"""Decision-analytic summaries, EVPI/EVPPI, and OWSA tables."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .deterministic import _det_icost
from .utils import safe_icer


def _evppi_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale_x", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale_poly", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def compute_evppi(
    df: pd.DataFrame,
    wtp: float,
    perspective: str,
    param_groups: Dict[str, List[str]],
    *,
    n_splits: int = 5,
    random_state: int = 731,
) -> Tuple[float, Dict[str, float]]:
    """Cross-fitted regression EVPPI by exogenous parameter group.

    Predictions are generated out of sample using K-fold cross-fitting, avoiding
    the upward bias from fitting and evaluating the conditional-NMB model on the
    same PSA iterations. Callers must pass raw input-draw columns only; derived
    costs and outcomes should not be included in parameter groups.
    """
    dal_col = f"dal_{perspective}"
    cost_col = f"ic_{perspective}"
    nmb = (float(wtp) * df[dal_col] - df[cost_col]).to_numpy(dtype=float)
    finite_nmb = np.isfinite(nmb)
    if not finite_nmb.all():
        raise ValueError("NMB contains non-finite values")

    evpi_total = float(np.mean(np.maximum(nmb, 0.0)) - max(float(nmb.mean()), 0.0))
    results: Dict[str, float] = {}

    for label, columns in param_groups.items():
        valid_columns = [column for column in columns if column in df.columns]
        if not valid_columns:
            results[label] = 0.0
            continue
        X = df[valid_columns].to_numpy(dtype=float)
        variable = np.nanstd(X, axis=0) > 1e-10
        X = X[:, variable]
        if X.shape[1] == 0 or len(X) < 4:
            results[label] = 0.0
            continue

        splits = min(max(int(n_splits), 2), len(X))
        kfold = KFold(n_splits=splits, shuffle=True, random_state=int(random_state))
        predictions = np.empty(len(X), dtype=float)
        for train_index, test_index in kfold.split(X):
            model = _evppi_pipeline()
            model.fit(X[train_index], nmb[train_index])
            predictions[test_index] = model.predict(X[test_index])

        value = float(
            np.mean(np.maximum(predictions, 0.0))
            - max(float(predictions.mean()), 0.0)
        )
        results[label] = float(np.clip(value, 0.0, evpi_total))

    return evpi_total, results


def decision_status(cost: float, effect: float) -> str:
    """Classify a deterministic incremental cost/effect pair without ratios."""

    c = float(cost)
    e = float(effect)
    if not np.isfinite(c) or not np.isfinite(e):
        return "Undefined"
    if e > 0.0 and c < 0.0:
        return "Dominant"
    if e < 0.0 and c > 0.0:
        return "Dominated"
    if e > 0.0 and c >= 0.0:
        return "More effective, more costly"
    if e < 0.0 and c <= 0.0:
        return "Less effective, less costly"
    if e == 0.0 and c < 0.0:
        return "Cost-saving, no health difference"
    if e == 0.0 and c > 0.0:
        return "More costly, no health difference"
    return "No difference"


def ce_quadrant_table(dal: np.ndarray, ic: np.ndarray) -> pd.DataFrame:
    """Partition PSA iterations into the four cost-effectiveness quadrants."""
    dal_arr = np.asarray(dal, dtype=float)
    ic_arr = np.asarray(ic, dtype=float)
    valid = np.isfinite(dal_arr) & np.isfinite(ic_arr)
    dal_arr, ic_arr = dal_arr[valid], ic_arr[valid]
    n = len(dal_arr)
    if n == 0:
        return pd.DataFrame(columns=["Quadrant", "N", "%", "Interpretation"])

    q1 = int(np.sum((dal_arr > 0.0) & (ic_arr < 0.0)))
    q2 = int(np.sum((dal_arr > 0.0) & (ic_arr >= 0.0)))
    q3 = int(np.sum((dal_arr <= 0.0) & (ic_arr > 0.0)))
    q4 = int(np.sum((dal_arr <= 0.0) & (ic_arr <= 0.0)))
    return pd.DataFrame(
        [
            {
                "Quadrant": "Dominant",
                "N": q1,
                "%": f"{100 * q1 / n:.1f}%",
                "Interpretation": "More effective and less costly",
            },
            {
                "Quadrant": "More effective, more costly",
                "N": q2,
                "%": f"{100 * q2 / n:.1f}%",
                "Interpretation": "Cost-effectiveness depends on the selected WTP",
            },
            {
                "Quadrant": "Dominated",
                "N": q3,
                "%": f"{100 * q3 / n:.1f}%",
                "Interpretation": "Less effective and more costly",
            },
            {
                "Quadrant": "Less effective, less costly",
                "N": q4,
                "%": f"{100 * q4 / n:.1f}%",
                "Interpretation": "Decision depends on savings versus health loss",
            },
        ]
    )


def evpi_curve(
    dal: np.ndarray,
    ic: np.ndarray,
    wtp_max: int = 200_000,
    step: int = 2_000,
) -> Tuple[np.ndarray, np.ndarray]:
    thresholds = np.arange(0, int(wtp_max) + int(step), int(step))
    evpi = np.zeros(len(thresholds), dtype=float)
    for index, threshold in enumerate(thresholds):
        nmb = threshold * np.asarray(dal, dtype=float) - np.asarray(ic, dtype=float)
        evpi[index] = np.mean(np.maximum(nmb, 0.0)) - max(float(np.mean(nmb)), 0.0)
    return thresholds, evpi


def psa_convergence(
    dal: np.ndarray, ic: np.ndarray, step: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """Ratio-of-cumulative-means ICER; non-positive effects are NaN."""
    if len(dal) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    actual_step = min(max(int(step), 1), len(dal))
    ns = np.arange(actual_step, len(dal) + 1, actual_step)
    if ns[-1] != len(dal):
        ns = np.append(ns, len(dal))
    rolling = np.asarray(
        [safe_icer(float(np.mean(ic[:n])), float(np.mean(dal[:n]))) for n in ns],
        dtype=float,
    )
    return ns, rolling


def _fmt_param_value(x):
    if isinstance(x, dict):
        return "; ".join(
            f"{key}={value:.3g}"
            if isinstance(value, (int, float, np.number))
            else f"{key}={value}"
            for key, value in x.items()
        )
    if isinstance(x, (int, float, np.number)):
        return f"{float(x):.3g}"
    return str(x)


def _owsa_value_label(kwargs: dict):
    if len(kwargs) == 1:
        return _fmt_param_value(next(iter(kwargs.values())))
    return _fmt_param_value(kwargs)


def owsa_table(base_kw: dict, param_ranges: Dict[str, Tuple]) -> pd.DataFrame:
    """One-way sensitivity table with dominance-safe ICERs."""
    ic_base, dal_base, _, _ = _det_icost(**base_kw)
    base_icer = safe_icer(ic_base, dal_base)
    rows = []
    for label, (low_kwargs, high_kwargs) in param_ranges.items():
        ic_low, dal_low, _, _ = _det_icost(**{**base_kw, **low_kwargs})
        ic_high, dal_high, _, _ = _det_icost(**{**base_kw, **high_kwargs})
        icer_low = safe_icer(ic_low, dal_low)
        icer_high = safe_icer(ic_high, dal_high)
        finite = [value for value in (icer_low, icer_high) if np.isfinite(value)]
        if len(finite) == 2:
            minimum, maximum = min(finite), max(finite)
            range_value = maximum - minimum
        elif len(finite) == 1:
            minimum = maximum = finite[0]
            range_value = np.nan
        else:
            minimum = maximum = range_value = np.nan
        rows.append(
            {
                "Parameter": label,
                "Base ICER": base_icer,
                "ICER (low)": icer_low,
                "ICER (high)": icer_high,
                "ICER min": minimum,
                "ICER max": maximum,
                "Range": range_value,
                "Low param value": _owsa_value_label(low_kwargs),
                "High param value": _owsa_value_label(high_kwargs),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Range", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def owsa_nmb_table(
    base_kw: dict,
    param_ranges: Dict[str, Tuple],
    wtp: float,
    perspective: str = "hs",
) -> pd.DataFrame:
    def _nmb(overrides):
        ic_hs, dal_hs, ic_soc, dal_soc = _det_icost(**{**base_kw, **overrides})
        effect = dal_soc if perspective == "soc" else dal_hs
        cost = ic_soc if perspective == "soc" else ic_hs
        return float(wtp) * effect - cost

    base_nmb = _nmb({})
    rows = []
    for label, (low_kwargs, high_kwargs) in param_ranges.items():
        low_nmb = _nmb(low_kwargs)
        high_nmb = _nmb(high_kwargs)
        rows.append(
            {
                "Parameter": label,
                "Base NMB": base_nmb,
                "NMB (low param)": low_nmb,
                "NMB (high param)": high_nmb,
                "NMB min": min(low_nmb, high_nmb),
                "NMB max": max(low_nmb, high_nmb),
                "Range": abs(high_nmb - low_nmb),
                "Low param value": _owsa_value_label(low_kwargs),
                "High param value": _owsa_value_label(high_kwargs),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Range", ascending=False)
        .reset_index(drop=True)
    )
