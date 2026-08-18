"""Shared screening cascade, program costs, and outcome savings."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .config import DEFAULT_TREATMENT_COST_POLICY, TreatmentCostPolicy
from .parameters import (
    CS_UNCOMP_OBS_LOS_FRAC,
    P_CS_UNCOMP_OBSERVED,
    P_JH_REACTION,
    P_LBW_GIVEN_PRETERM,
    P_PEN_ALLERGY,
)


def _cost_value(costs, name: str):
    if isinstance(costs, Mapping):
        return np.asarray(costs[name], dtype=float)
    return np.asarray(getattr(costs, name), dtype=float)


def screening_cascade_counts(
    n_screened,
    p_act,
    p_sf,
    sens,
    spec,
    p_adeq,
    tx_completion,
    p_trepo_sf,
    p_ux_sf,
    treat_fp: bool,
) -> dict:
    """Return active, serofast, and false-positive cascade counts.

    Scalars and arrays are supported. ``tx_completion`` is the average
    completion rate among incrementally screened patients; callers using the
    gestational model should obtain it from
    ``outcomes.incremental_tx_completion_rate``.
    """
    n = np.maximum(np.asarray(n_screened, dtype=float), 0.0)
    p_act_arr = np.clip(np.asarray(p_act, dtype=float), 0.0, 1.0)
    p_sf_arr = np.clip(np.asarray(p_sf, dtype=float), 0.0, 1.0)
    p_sn = np.maximum(1.0 - p_act_arr - p_sf_arr, 0.0)

    n_tp_detected = n * p_act_arr * sens
    n_tp_treated = n_tp_detected * p_adeq * tx_completion
    n_sf_detected = n * p_sf_arr * p_trepo_sf
    n_sf_treated = n_sf_detected * p_ux_sf
    n_fp_detected = n * p_sn * (1.0 - spec)
    n_fp_treated = n_fp_detected * p_adeq * tx_completion if treat_fp else np.zeros_like(n_fp_detected)

    return {
        "n_screened": n,
        "n_tp_detected": n_tp_detected,
        "n_tp_treated": n_tp_treated,
        "n_sf_detected": n_sf_detected,
        "n_sf_treated": n_sf_treated,
        "n_fp_detected": n_fp_detected,
        "n_fp_treated": n_fp_treated,
        "n_confirmatory": n_tp_detected + n_sf_detected + n_fp_detected,
        "n_treated_total": n_tp_treated + n_sf_treated + n_fp_treated,
    }


def program_cost_components(
    cascade: Mapping[str, object],
    costs,
    policy: TreatmentCostPolicy = DEFAULT_TREATMENT_COST_POLICY,
) -> dict:
    """Calculate a shared CEA/BIA program-cost cascade."""
    n_screened = np.asarray(cascade["n_screened"], dtype=float)
    n_tp_detected = np.asarray(cascade["n_tp_detected"], dtype=float)
    n_sf_detected = np.asarray(cascade["n_sf_detected"], dtype=float)
    n_fp_detected = np.asarray(cascade["n_fp_detected"], dtype=float)
    n_tp_treated = np.asarray(cascade["n_tp_treated"], dtype=float)
    n_sf_treated = np.asarray(cascade["n_sf_treated"], dtype=float)
    n_fp_treated = np.asarray(cascade["n_fp_treated"], dtype=float)

    poc = _cost_value(costs, "poc")
    rpr = _cost_value(costs, "rpr")
    fta = _cost_value(costs, "fta")
    staff = _cost_value(costs, "staff")
    pen = _cost_value(costs, "pen")
    soc_work = _cost_value(costs, "soc_work")
    sf_wu = _cost_value(costs, "sf_wu")
    desens = _cost_value(costs, "desens")
    jh_obs = _cost_value(costs, "jh_obs")
    followup = _cost_value(costs, "followup")

    cost_screening_tests = n_screened * (poc + rpr)
    cost_confirmatory_tp = n_tp_detected * fta
    cost_confirmatory_sf = n_sf_detected * fta
    cost_confirmatory_fp = n_fp_detected * fta
    cost_confirmatory = cost_confirmatory_tp + cost_confirmatory_sf + cost_confirmatory_fp
    cost_staff = n_screened * staff

    unit_treatment = pen + soc_work
    cost_tx_tp = n_tp_treated * unit_treatment
    cost_tx_sf = n_sf_treated * unit_treatment
    cost_tx_fp = n_fp_treated * unit_treatment
    cost_treatment = cost_tx_tp + cost_tx_sf + cost_tx_fp
    cost_sf_workup = n_sf_detected * sf_wu

    # Desensitization and serologic follow-up are charged to every treated
    # patient. JH allocation is configurable because the source model does not
    # resolve whether it applies only to active infection.
    common_adjacent_unit = P_PEN_ALLERGY * desens + followup
    jh_unit = P_JH_REACTION * jh_obs
    cost_adjacent_tp = n_tp_treated * (common_adjacent_unit + jh_unit)
    if policy.jh_scope == "all_treated":
        cost_adjacent_sf = n_sf_treated * (common_adjacent_unit + jh_unit)
        cost_adjacent_fp = n_fp_treated * (common_adjacent_unit + jh_unit)
    else:
        cost_adjacent_sf = n_sf_treated * common_adjacent_unit
        cost_adjacent_fp = n_fp_treated * common_adjacent_unit
    cost_tx_adjacent = cost_adjacent_tp + cost_adjacent_sf + cost_adjacent_fp

    total = (
        cost_screening_tests
        + cost_confirmatory
        + cost_staff
        + cost_treatment
        + cost_sf_workup
        + cost_tx_adjacent
    )
    cost_sf_total = (
        cost_confirmatory_sf + cost_sf_workup + cost_tx_sf + cost_adjacent_sf
    )

    return {
        "cost_screening_tests": cost_screening_tests,
        "cost_confirmatory": cost_confirmatory,
        "cost_confirmatory_tp": cost_confirmatory_tp,
        "cost_confirmatory_sf": cost_confirmatory_sf,
        "cost_confirmatory_fp": cost_confirmatory_fp,
        "cost_staff": cost_staff,
        "cost_tx_tp": cost_tx_tp,
        "cost_tx_sf": cost_tx_sf,
        "cost_tx_fp": cost_tx_fp,
        "cost_treatment": cost_treatment,
        "cost_sf_workup": cost_sf_workup,
        "cost_adjacent_tp": cost_adjacent_tp,
        "cost_adjacent_sf": cost_adjacent_sf,
        "cost_adjacent_fp": cost_adjacent_fp,
        "cost_tx_adjacent": cost_tx_adjacent,
        "cost_sf_total": cost_sf_total,
        "program_cost": total,
    }


def _serofast_cost(
    n_screened,
    p_sf,
    p_trepo,
    p_ux,
    rpr,
    sf_wu,
    pen,
    soc_work,
    fta=None,
    desens=None,
    jh_obs=None,
    followup=None,
    policy: TreatmentCostPolicy = DEFAULT_TREATMENT_COST_POLICY,
):
    """Serofast-specific cost retained for compatibility and detail displays."""
    del rpr  # screening RPR is charged in the main test cost
    n_sf = np.asarray(n_screened, dtype=float) * p_sf * p_trepo
    n_sf_treated = n_sf * p_ux
    fta_cost = 0.0 if fta is None else np.asarray(fta, dtype=float)
    total = n_sf * (fta_cost + sf_wu) + n_sf_treated * (pen + soc_work)
    if desens is not None and followup is not None:
        adjacent_unit = P_PEN_ALLERGY * np.asarray(desens, dtype=float) + np.asarray(followup, dtype=float)
        if policy.jh_scope == "all_treated" and jh_obs is not None:
            adjacent_unit = adjacent_unit + P_JH_REACTION * np.asarray(jh_obs, dtype=float)
        total = total + n_sf_treated * adjacent_unit
    return total



def treatment_adjacent_unit_cost(
    costs,
    policy: TreatmentCostPolicy = DEFAULT_TREATMENT_COST_POLICY,
    *,
    active_infection: bool = True,
):
    """Expected adjacent direct cost per treated patient."""

    unit = P_PEN_ALLERGY * _cost_value(costs, "desens") + _cost_value(costs, "followup")
    if policy.jh_scope == "all_treated" or active_infection:
        unit = unit + P_JH_REACTION * _cost_value(costs, "jh_obs")
    return unit


def acute_outcome_savings(d, costs) -> dict:
    """Detailed acute medical savings used by the BIA display."""

    n_iufd = np.asarray(d["iufd_subset"], dtype=float)
    n_sb_other = np.maximum(np.asarray(d["stillbirth"], dtype=float) - n_iufd, 0.0)
    n_pt = np.asarray(d["preterm"], dtype=float)
    n_lbw = np.asarray(d["lbw"], dtype=float)
    n_comp = np.asarray(d["cs_comp"], dtype=float)
    n_uncomp = np.asarray(d["cs_uncomp"], dtype=float)
    preterm = n_pt * np.maximum(_cost_value(costs, "preterm") - _cost_value(costs, "term_del"), 0.0)
    overlap = P_LBW_GIVEN_PRETERM * np.minimum(n_pt / np.maximum(n_lbw, 1e-9), 1.0)
    lbw = n_lbw * _cost_value(costs, "lbw_hs") * np.clip(1.0 - overlap, 0.0, 1.0)
    cs_comp = n_comp * (_cost_value(costs, "cs_wu") + _cost_value(costs, "nicu"))
    cs_uncomp = n_uncomp * (
        _cost_value(costs, "cs_wu")
        + P_CS_UNCOMP_OBSERVED * CS_UNCOMP_OBS_LOS_FRAC * _cost_value(costs, "nicu")
    )
    iufd = n_iufd * _cost_value(costs, "iufd")
    stillbirth_other = n_sb_other * _cost_value(costs, "sb")
    neonatal_death = np.asarray(d["neonatal_death"], dtype=float) * _cost_value(costs, "nnd")
    total = cs_comp + cs_uncomp + preterm + lbw + iufd + stillbirth_other + neonatal_death
    return {
        "cs_comp": cs_comp,
        "cs_uncomp": cs_uncomp,
        "preterm": preterm,
        "lbw": lbw,
        "iufd": iufd,
        "stillbirth_other": stillbirth_other,
        "neonatal_death": neonatal_death,
        "total": total,
    }

def outcome_savings_components(d, costs) -> dict:
    """Direct medical savings shared by CEA and BIA."""
    n_iufd = np.asarray(d["iufd_subset"], dtype=float)
    n_stillbirth = np.asarray(d["stillbirth"], dtype=float)
    n_sb_other = np.maximum(n_stillbirth - n_iufd, 0.0)
    n_nnd = np.asarray(d["neonatal_death"], dtype=float)
    n_lbw = np.asarray(d["lbw"], dtype=float)
    n_preterm = np.asarray(d["preterm"], dtype=float)
    n_cs_comp = np.asarray(d["cs_comp"], dtype=float)
    n_cs_uncomp = np.asarray(d["cs_uncomp"], dtype=float)

    preterm_net = np.maximum(_cost_value(costs, "preterm") - _cost_value(costs, "term_del"), 0.0)
    overlap_fraction = P_LBW_GIVEN_PRETERM * np.minimum(
        n_preterm / np.maximum(n_lbw, 1e-9), 1.0
    )
    lbw_unique_fraction = np.clip(1.0 - overlap_fraction, 0.0, 1.0)
    cs_uncomp_cost = _cost_value(costs, "cs_wu") + (
        P_CS_UNCOMP_OBSERVED * CS_UNCOMP_OBS_LOS_FRAC * _cost_value(costs, "nicu")
    )

    sav_cs = n_cs_comp * (_cost_value(costs, "cs_wu") + _cost_value(costs, "nicu")) + n_cs_uncomp * cs_uncomp_cost
    sav_preterm = n_preterm * preterm_net
    sav_lbw = n_lbw * _cost_value(costs, "lbw_hs") * lbw_unique_fraction
    sav_stillbirth = n_iufd * _cost_value(costs, "iufd") + n_sb_other * _cost_value(costs, "sb")
    sav_nnd = n_nnd * _cost_value(costs, "nnd")
    total = sav_cs + sav_preterm + sav_lbw + sav_stillbirth + sav_nnd

    return {
        "sav_cs": sav_cs,
        "sav_preterm": sav_preterm,
        "sav_lbw": sav_lbw,
        "sav_sb": sav_stillbirth,
        "sav_nnd": sav_nnd,
        "medical_savings": total,
    }


def _icost(
    d,
    costs,
    sf_cost,
    sc_b,
    sc_e,
    p_act,
    p_sf,
    p_trepo_sf,
    p_id,
    sens,
    spec,
    p_adeq,
    tx_eff,
    treat_fp,
    cohort,
    mk_med_cost_saving,
    *,
    mk_ltc_cost_saving=None,
    mat_cost_saving=None,
    prod_loss_saving=None,
    program_cost_override=None,
    policy: TreatmentCostPolicy = DEFAULT_TREATMENT_COST_POLICY,
):
    """Return health-sector and societal incremental costs.

    The corrected perspective mapping follows resource type rather than the
    module in which a cost happens to be calculated. Health-sector costs include
    screening/treatment, acute medical offsets, infant medical sequelae costs,
    and maternal direct medical costs. Societal costs additionally include
    special education, paid caregiving, and productivity losses. All modeled
    health effects are counted under both perspectives.

    Passing ``program_cost_override`` routes the complete shared CEA/BIA
    screening-and-treatment cascade directly into this calculation.
    """

    del p_trepo_sf
    extra = max(float(sc_e) - float(sc_b), 0.0) * float(p_id) * float(cohort)
    p_sf_arr = np.asarray(p_sf, dtype=float)
    p_seronegative = np.maximum(1.0 - float(p_act) - p_sf_arr, 0.0)
    n_tp_detected = extra * float(p_act) * float(sens)
    n_fp_detected = extra * p_seronegative * (1.0 - float(spec))
    n_tp_treated = n_tp_detected * float(p_adeq) * float(tx_eff)
    n_fp_treated = (
        n_fp_detected * float(p_adeq) * float(tx_eff)
        if treat_fp
        else np.zeros_like(n_fp_detected, dtype=float)
    )

    program_non_serofast = (
        extra * (_cost_value(costs, "poc") + _cost_value(costs, "rpr") + _cost_value(costs, "staff"))
        + (n_tp_detected + n_fp_detected) * _cost_value(costs, "fta")
        + (n_tp_treated + n_fp_treated) * (_cost_value(costs, "pen") + _cost_value(costs, "soc_work"))
        + n_tp_treated * treatment_adjacent_unit_cost(costs, policy, active_infection=True)
        + n_fp_treated * treatment_adjacent_unit_cost(costs, policy, active_infection=False)
    )
    program_cost = (
        np.asarray(program_cost_override, dtype=float)
        if program_cost_override is not None
        else program_non_serofast + np.asarray(sf_cost, dtype=float)
    )
    acute_savings = acute_outcome_savings(d, costs)["total"]

    health_sector = (
        program_cost
        - acute_savings
        - np.asarray(mk_med_cost_saving, dtype=float)
    )
    if mat_cost_saving is not None:
        health_sector = health_sector - np.asarray(mat_cost_saving, dtype=float)

    societal = health_sector.copy()
    if mk_ltc_cost_saving is not None:
        societal = societal - np.asarray(mk_ltc_cost_saving, dtype=float)
    if prod_loss_saving is not None:
        societal = societal - np.asarray(prod_loss_saving, dtype=float)
    return np.asarray(health_sector, dtype=float), np.asarray(societal, dtype=float)
