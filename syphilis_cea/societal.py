"""Maternal morbidity and productivity-loss modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import DEFAULT_MARKOV_CONFIG, MarkovConfig
from .parameters import DW_P, MATERNAL_WEIGHTED_EARNINGS
from .utils import pvf


@dataclass(frozen=True)
class MaternalMorbidity:
    """Maternal morbidity from untreated syphilis."""

    p_cardio: float = 0.050
    dw_cardio: float = 0.070
    dur_cardio: float = 5.0
    cost_cardio: float = 18_500
    cost_cardio_sd: float = 4_500

    p_neuro: float = 0.100
    dw_neuro: float = 0.440
    dur_neuro: float = 8.0
    cost_neuro: float = 32_000
    cost_neuro_sd: float = 8_000

    p_hosp: float = 0.120
    cost_hosp: float = 4_200
    cost_hosp_sd: float = 1_100
    dur_hosp_days: float = 3.2

    p_cardio_lo: float = 0.020
    p_cardio_hi: float = 0.100
    p_neuro_lo: float = 0.040
    p_neuro_hi: float = 0.200
    p_hosp_lo: float = 0.070
    p_hosp_hi: float = 0.200


@dataclass(frozen=True)
class ProductivityLoss:
    """Human-capital productivity losses averted by screening."""

    bereavement_days: float = 30.0
    bereavement_days_sd: float = 10.0
    wage_penalty_mild: float = 0.10
    wage_penalty_mild_sd: float = 0.04
    wage_penalty_severe: float = 0.40
    wage_penalty_severe_sd: float = 0.10
    friction_period_days: float = 0.0
    caregiver_hrs_wk: float = 10.0
    caregiver_hrs_wk_sd: float = 4.0
    caregiver_wage_frac: float = 0.50


def _prod_loss_per_case(
    pl: ProductivityLoss,
    psa_pl: dict,
    n_sb: np.ndarray,
    n_nnd: np.ndarray,
    n_cs_comp: np.ndarray,
    n_cs_uncomp: np.ndarray,
    mk: dict,
    r: float,
    LE: float,
    earnings: float = MATERNAL_WEIGHTED_EARNINGS,
) -> np.ndarray:
    """Vectorized productivity savings using PSA draws.

    The friction-period cap is applied consistently to bereavement, infant
    earnings, patient wage losses, and caregiver opportunity costs. The former
    PSA implementation omitted the caregiver cap and therefore disagreed with
    the deterministic path.
    """
    daily_wage = float(earnings) / 260.0
    friction = max(float(pl.friction_period_days), 0.0)

    bereavement_days = np.asarray(psa_pl["bereavement_days"], dtype=float)
    if friction > 0.0:
        bereavement_days = np.minimum(bereavement_days, friction)
    bereavement = (n_sb + n_nnd) * bereavement_days * daily_wage

    pv_infant = earnings * pvf(max(LE - 20.0, 0.0), r) * (1.0 + r) ** (-20)
    if friction > 0.0:
        pv_infant = min(pv_infant, friction * daily_wage)
    infant_earnings = n_nnd * pv_infant

    pv_working = earnings * pvf(45.0, r) * (1.0 + r) ** (-20)
    if friction > 0.0:
        pv_working = min(pv_working, friction * daily_wage)

    cs_mild_loss = (
        n_cs_comp * np.asarray(mk["p_mild_cs_comp"], dtype=float)
        + n_cs_uncomp * np.asarray(mk["p_mild_cs_uncomp"], dtype=float)
    ) * np.asarray(psa_pl["wage_penalty_mild"], dtype=float) * pv_working
    cs_severe_loss = (
        n_cs_comp
        * np.asarray(mk["p_severe_cs_comp"], dtype=float)
        * np.asarray(psa_pl["wage_penalty_severe"], dtype=float)
        * pv_working
    )

    caregiver_annual = (
        np.asarray(psa_pl["caregiver_hrs_wk"], dtype=float)
        * 52.0
        * earnings
        / 2080.0
        * pl.caregiver_wage_frac
    )
    caregiver_pv = caregiver_annual * pvf(18.0, r)
    if friction > 0.0:
        caregiver_pv = np.minimum(caregiver_pv, friction * daily_wage)
    caregiver_total = n_cs_comp * caregiver_pv

    return bereavement + infant_earnings + cs_mild_loss + cs_severe_loss + caregiver_total


def _prod_loss_det(
    pl: ProductivityLoss,
    n_sb,
    n_nnd,
    n_cs_comp,
    n_cs_uncomp,
    r: float,
    LE: float,
    earnings: float = MATERNAL_WEIGHTED_EARNINGS,
    markov_config: MarkovConfig = DEFAULT_MARKOV_CONFIG,
) -> float:
    """Deterministic productivity-loss savings with explicit Markov inputs."""
    daily_wage = float(earnings) / 260.0
    friction = max(float(pl.friction_period_days), 0.0)
    bereavement_days = min(pl.bereavement_days, friction) if friction > 0.0 else pl.bereavement_days
    bereavement = (n_sb + n_nnd) * bereavement_days * daily_wage

    pv_infant = earnings * pvf(max(LE - 20.0, 0.0), r) * (1.0 + r) ** (-20)
    if friction > 0.0:
        pv_infant = min(pv_infant, friction * daily_wage)
    infant_earnings = n_nnd * pv_infant

    pv_working = earnings * pvf(45.0, r) * (1.0 + r) ** (-20)
    if friction > 0.0:
        pv_working = min(pv_working, friction * daily_wage)

    cs_mild = (
        n_cs_comp * markov_config.p_mild_cs_comp
        + n_cs_uncomp * markov_config.p_mild_cs_uncomp
    ) * pl.wage_penalty_mild * pv_working
    cs_severe = (
        n_cs_comp
        * markov_config.p_severe_cs_comp
        * pl.wage_penalty_severe
        * pv_working
    )

    caregiver_annual = (
        pl.caregiver_hrs_wk * 52.0 * earnings / 2080.0 * pl.caregiver_wage_frac
    )
    caregiver_pv = caregiver_annual * pvf(18.0, r)
    if friction > 0.0:
        caregiver_pv = min(caregiver_pv, friction * daily_wage)
    caregiver_total = n_cs_comp * caregiver_pv

    return float(
        bereavement + infant_earnings + cs_mild + cs_severe + caregiver_total
    )


def _mat_morb_dalys(
    mm: MaternalMorbidity,
    n_maternal_tx: np.ndarray,
    p_eff_delta,
    r: float,
    psa_mm: dict,
    dw_hosp: np.ndarray | None = None,
    include_hosp_yld: bool = True,
    return_components: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Maternal morbidity DALYs and costs averted.

    ``p_eff_delta`` is retained for backward-compatible call signatures; the
    already-computed ``n_maternal_tx`` is the operative quantity.
    """
    del p_eff_delta
    p_late_latent = 0.30
    n_cardio = n_maternal_tx * p_late_latent * psa_mm["p_cardio"]
    n_neuro = n_maternal_tx * p_late_latent * psa_mm["p_neuro"]
    n_hosp = n_maternal_tx * psa_mm["p_hosp"]

    daly_cardio = n_cardio * mm.dw_cardio * pvf(mm.dur_cardio, r)
    daly_neuro = n_neuro * mm.dw_neuro * pvf(mm.dur_neuro, r)
    if include_hosp_yld:
        if dw_hosp is None:
            dw_hosp = np.full_like(n_maternal_tx, DW_P["mat_hosp"]["m"], dtype=float)
        hosp_duration = max(float(mm.dur_hosp_days), 0.0) / 365.25
        daly_hosp = n_hosp * dw_hosp * pvf(hosp_duration, r)
    else:
        daly_hosp = np.zeros_like(n_maternal_tx, dtype=float)

    cost_cardio = n_cardio * psa_mm["cost_cardio"] * pvf(mm.dur_cardio, r)
    cost_neuro = n_neuro * psa_mm["cost_neuro"] * pvf(mm.dur_neuro, r)
    cost_hosp = n_hosp * psa_mm["cost_hosp"]

    total_daly = daly_cardio + daly_neuro + daly_hosp
    total_cost = cost_cardio + cost_neuro + cost_hosp
    if return_components:
        return total_daly, total_cost, {
            "mat_cardio_dal": daly_cardio,
            "mat_neuro_dal": daly_neuro,
            "mat_hosp_dal": daly_hosp,
        }
    return total_daly, total_cost


def _mat_morb_det(
    mm: MaternalMorbidity,
    n_maternal_tx: float,
    r: float,
    include_hosp_yld: bool = True,
) -> Tuple[float, float]:
    p_late_latent = 0.30
    n_cardio = n_maternal_tx * p_late_latent * mm.p_cardio
    n_neuro = n_maternal_tx * p_late_latent * mm.p_neuro
    n_hosp = n_maternal_tx * mm.p_hosp
    daly_hosp = 0.0
    if include_hosp_yld:
        hosp_duration = max(float(mm.dur_hosp_days), 0.0) / 365.25
        daly_hosp = n_hosp * DW_P["mat_hosp"]["m"] * pvf(hosp_duration, r)
    daly = (
        n_cardio * mm.dw_cardio * pvf(mm.dur_cardio, r)
        + n_neuro * mm.dw_neuro * pvf(mm.dur_neuro, r)
        + daly_hosp
    )
    cost = (
        n_cardio * mm.cost_cardio * pvf(mm.dur_cardio, r)
        + n_neuro * mm.cost_neuro * pvf(mm.dur_neuro, r)
        + n_hosp * mm.cost_hosp
    )
    return float(daly), float(cost)
