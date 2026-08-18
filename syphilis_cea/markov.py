"""Infant lifetime Markov model, state occupancy, and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .config import (
    MarkovConfig,
    markov_point_parameters,
    normalize_complicated_state_probabilities,
)
from .parameters import LongTermCare, lt_qx
from .utils import pvf


@dataclass(frozen=True)
class MarkovResult:
    """Vectorized lifetime outcomes for averted congenital-syphilis cases."""

    dalys: np.ndarray
    medical_costs: np.ndarray
    long_term_care_costs: np.ndarray
    yld: np.ndarray
    yll: np.ndarray

    @property
    def total_costs(self) -> np.ndarray:
        return self.medical_costs + self.long_term_care_costs


def _infant_markov_lifetime(
    n_cs_comp,
    n_cs_uncomp,
    mk,
    r_disc,
    T,
    ltc: LongTermCare | None = None,
    psa_ltc: dict | None = None,
    include_cs_yll: bool = True,
    q_progress: float | None = None,
) -> MarkovResult:
    """Vectorized Healthy/Mild/Severe/Dead lifetime Markov model.

    Initial complicated-CS probabilities are projected to the probability
    simplex for every PSA draw. Background deaths determine occupancy but are
    not charged as congenital-syphilis YLL. End ages are inclusive, matching
    labels such as "through age 21".
    """

    n_comp = np.atleast_1d(np.asarray(n_cs_comp, dtype=float))
    n_uncomp = np.atleast_1d(np.asarray(n_cs_uncomp, dtype=float))
    if n_comp.shape != n_uncomp.shape:
        raise ValueError("n_cs_comp and n_cs_uncomp must have the same shape")
    n = int(n_comp.size)
    if q_progress is None:
        raise ValueError("q_progress must be supplied explicitly")
    q = float(np.clip(q_progress, 0.0, 1.0))
    horizon = max(int(T), 1)

    def arr(key: str, default=0.0) -> np.ndarray:
        value = mk.get(key, default)
        out = np.asarray(value, dtype=float)
        if out.ndim == 0:
            out = np.full(n, float(out), dtype=float)
        return np.broadcast_to(out, (n,)).astype(float, copy=False)

    dw_m = arr("dw_mild")
    dw_s = arr("dw_severe")
    c_m = arr("cost_mild_ann")
    c_s = arr("cost_sev_ann")
    mu_x_m = arr("mu_excess_mild") if include_cs_yll else np.zeros(n)
    mu_x_s = arr("mu_excess_severe") if include_cs_yll else np.zeros(n)

    if ltc is not None:
        def ltc_arr(key: str, point: float) -> np.ndarray:
            if psa_ltc is None:
                return np.full(n, float(point), dtype=float)
            return np.broadcast_to(np.asarray(psa_ltc[key], dtype=float), (n,))

        p_sped_s = ltc_arr("p_sped_severe", ltc.p_sped_severe)
        p_sped_m = ltc_arr("p_sped_mild", ltc.p_sped_mild)
        c_sped = ltc_arr("cost_sped_ann", ltc.cost_sped_ann)
        c_cg_s = ltc_arr("cost_cg_severe_ann", ltc.cost_cg_severe_ann)
        c_cg_m = ltc_arr("cost_cg_mild_ann", ltc.cost_cg_mild_ann)
        sped_start, sped_end = int(ltc.sped_start_age), int(ltc.sped_end_age)
        caregiver_end = int(ltc.caregiver_end_age)
    else:
        p_sped_s = p_sped_m = c_sped = c_cg_s = c_cg_m = np.zeros(n)
        sped_start, sped_end, caregiver_end = 1, 0, -1

    def run(initial: np.ndarray):
        states = np.asarray(initial, dtype=float).copy()
        if states.shape != (n, 4):
            raise ValueError("Initial Markov states must have shape (N, 4)")
        if not np.allclose(states.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("Initial Markov state probabilities must sum to one")

        yld = np.zeros(n)
        yll = np.zeros(n)
        medical_cost = np.zeros(n)
        ltc_cost = np.zeros(n)

        for age in range(horizon):
            discount = (1.0 + float(r_disc)) ** (-age)
            mu = float(lt_qx(age))

            yld += (states[:, 1] * dw_m + states[:, 2] * dw_s) * discount
            mx_m = np.minimum(mu_x_m, np.maximum(1.0 - mu, 0.0))
            mx_s = np.minimum(mu_x_s, np.maximum(1.0 - mu, 0.0))
            progression = np.minimum(q, np.maximum(1.0 - mu - mx_m, 0.0))
            excess_deaths = states[:, 1] * mx_m + states[:, 2] * mx_s
            yll += excess_deaths * discount * pvf(max(horizon - age, 0), float(r_disc))

            medical_cost += (
                states[:, 1] * c_m + states[:, 2] * c_s
            ) * discount
            if sped_start <= age <= sped_end:
                ltc_cost += c_sped * (
                    states[:, 1] * p_sped_m + states[:, 2] * p_sped_s
                ) * discount
            if age <= caregiver_end:
                ltc_cost += (
                    states[:, 1] * c_cg_m + states[:, 2] * c_cg_s
                ) * discount

            new = np.zeros_like(states)
            new[:, 0] += states[:, 0] * (1.0 - mu)
            new[:, 3] += states[:, 0] * mu

            remain_mild = np.maximum(1.0 - mu - progression - mx_m, 0.0)
            new[:, 1] += states[:, 1] * remain_mild
            new[:, 2] += states[:, 1] * progression
            new[:, 3] += states[:, 1] * (mu + mx_m)

            remain_severe = np.maximum(1.0 - mu - mx_s, 0.0)
            new[:, 2] += states[:, 2] * remain_severe
            new[:, 3] += states[:, 2] * (mu + mx_s)
            new[:, 3] += states[:, 3]

            total = new.sum(axis=1)
            states = new / np.maximum(total[:, None], 1e-15)

        return yld + yll, medical_cost, ltc_cost, yld, yll

    healthy_c, mild_c, severe_c = normalize_complicated_state_probabilities(
        arr("p_mild_cs_comp"), arr("p_severe_cs_comp")
    )
    comp_initial = np.stack(
        [healthy_c, mild_c, severe_c, np.zeros(n)], axis=1
    )
    d_comp, med_comp, ltc_comp, yld_comp, yll_comp = run(comp_initial)

    mild_u = np.clip(arr("p_mild_cs_uncomp"), 0.0, 1.0)
    uncomp_initial = np.stack(
        [1.0 - mild_u, mild_u, np.zeros(n), np.zeros(n)], axis=1
    )
    d_uncomp, med_uncomp, ltc_uncomp, yld_uncomp, yll_uncomp = run(uncomp_initial)

    return MarkovResult(
        dalys=n_comp * d_comp + n_uncomp * d_uncomp,
        medical_costs=n_comp * med_comp + n_uncomp * med_uncomp,
        long_term_care_costs=n_comp * ltc_comp + n_uncomp * ltc_uncomp,
        yld=n_comp * yld_comp + n_uncomp * yld_uncomp,
        yll=n_comp * yll_comp + n_uncomp * yll_uncomp,
    )


def infant_markov_point_estimate(
    n_cs_comp: float,
    n_cs_uncomp: float,
    config: MarkovConfig,
    r_disc: float,
    T: int,
    *,
    ltc: LongTermCare | None = None,
    include_cs_yll: bool = True,
    q_progress: float,
) -> tuple[float, float, float, float]:
    result = _infant_markov_lifetime(
        np.asarray([n_cs_comp]),
        np.asarray([n_cs_uncomp]),
        markov_point_parameters(config),
        r_disc,
        T,
        ltc=ltc,
        include_cs_yll=include_cs_yll,
        q_progress=q_progress,
    )
    return (
        float(result.dalys[0]),
        float(result.total_costs[0]),
        float(result.yld[0]),
        float(result.yll[0]),
    )


def markov_state_occupancy(
    config: MarkovConfig,
    q_progress: float,
    T: int,
    *,
    complicated: bool,
) -> np.ndarray:
    """Return annual state occupancy for one case."""

    if complicated:
        initial = np.array(
            [
                config.p_healthy_cs_comp,
                config.p_mild_cs_comp,
                config.p_severe_cs_comp,
                0.0,
            ],
            dtype=float,
        )
    else:
        initial = np.array(
            [config.p_healthy_cs_uncomp, config.p_mild_cs_uncomp, 0.0, 0.0],
            dtype=float,
        )
    history = [initial.copy()]
    state = initial.copy()
    q = float(np.clip(q_progress, 0.0, 1.0))
    for age in range(max(int(T) - 1, 0)):
        mu = float(lt_qx(age))
        mx_m = min(config.mu_excess_mild, max(1.0 - mu, 0.0))
        mx_s = min(config.mu_excess_severe, max(1.0 - mu, 0.0))
        progression = min(q, max(1.0 - mu - mx_m, 0.0))
        new = np.zeros(4)
        new[0] += state[0] * (1.0 - mu)
        new[3] += state[0] * mu
        new[1] += state[1] * max(1.0 - mu - progression - mx_m, 0.0)
        new[2] += state[1] * progression
        new[3] += state[1] * (mu + mx_m)
        new[2] += state[2] * max(1.0 - mu - mx_s, 0.0)
        new[3] += state[2] * (mu + mx_s)
        new[3] += state[3]
        state = new / max(new.sum(), 1e-15)
        history.append(state.copy())
    return np.asarray(history)


def simulate_markov_occupancy(
    config: MarkovConfig,
    q_progress: float,
    T: int,
    complicated: bool,
) -> np.ndarray:
    """Stable public alias used by figure builders."""

    return markov_state_occupancy(
        config, q_progress, T, complicated=complicated
    )


def implied_lifetime_prog(
    q: float,
    T: int,
    mu_excess_mild: float = 0.0,
) -> float:
    """Cumulative mild-to-severe progression with competing mortality."""

    state_mild = 1.0
    progressed = 0.0
    q = float(np.clip(q, 0.0, 1.0))
    excess = float(np.clip(mu_excess_mild, 0.0, 1.0))
    for age in range(max(int(T), 1)):
        mu = float(lt_qx(age))
        available = max(1.0 - mu - excess, 0.0)
        p_progress = min(q, available)
        progressed += state_mild * p_progress
        state_mild *= max(1.0 - mu - excess - p_progress, 0.0)
    return float(progressed)


def calibrate_q_progress(
    target_prob: float,
    T: int,
    tol: float = 1e-8,
    mu_excess_mild: float = 0.0,
) -> float:
    """Solve for annual progression matching a lifetime target."""

    target = float(target_prob)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_prob must be in [0, 1]")
    if int(T) < 1:
        raise ValueError("T must be at least one year")
    if target == 0.0:
        return 0.0
    attainable = implied_lifetime_prog(0.999, int(T), mu_excess_mild)
    if target > attainable + tol:
        raise ValueError(
            f"Target progression {target:.3f} exceeds attainable {attainable:.3f}"
        )
    return float(
        brentq(
            lambda q: implied_lifetime_prog(q, int(T), mu_excess_mild) - target,
            0.0,
            0.999,
            xtol=tol,
            maxiter=300,
        )
    )
