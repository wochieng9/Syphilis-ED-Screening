"""Budget-impact population funnel, shared cascade, and scenario runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .costing import (
    outcome_savings_components,
    program_cost_components,
    screening_cascade_counts,
)
from .outcomes import (
    _deterministic_outcome_inputs,
    arm_outcome_delta,
    incremental_tx_completion_rate,
)
from .parameters import Costs


@dataclass(frozen=True)
class BIAPopulation:
    """Population funnel inputs for the budget-impact analysis.

    ``covered_lives`` is treated as enrolled payer membership by default, so
    payer share is not applied a second time. Set ``apply_payer_fraction=True``
    only when ``covered_lives`` is a broader catchment population rather than
    plan enrollment.
    """

    covered_lives: float = 100_000
    frac_repro_female: float = 0.135
    pregnancy_rate: float = 0.085
    p_ed_visit: float = 0.45
    p_unscreened: float = 0.35
    payer_fraction: float = 0.40
    apply_payer_fraction: bool = False


BIA_SCENARIOS: Dict[str, dict] = {
    "Conservative": dict(
        pop=BIAPopulation(p_ed_visit=0.35, p_unscreened=0.25, payer_fraction=0.35, apply_payer_fraction=True),
        t_half=2.5,
        t_ninety=4.5,
    ),
    "Base case": dict(
        pop=BIAPopulation(p_ed_visit=0.45, p_unscreened=0.35, payer_fraction=0.40, apply_payer_fraction=True),
        t_half=1.5,
        t_ninety=3.0,
    ),
    "Optimistic": dict(
        pop=BIAPopulation(p_ed_visit=0.55, p_unscreened=0.50, payer_fraction=0.45, apply_payer_fraction=True),
        t_half=1.0,
        t_ninety=2.0,
    ),
}


def common_population_funnel(pop: BIAPopulation) -> dict:
    n_repro = pop.covered_lives * pop.frac_repro_female
    n_pregnant = n_repro * pop.pregnancy_rate
    n_ed_all = n_pregnant * pop.p_ed_visit
    payer_multiplier = pop.payer_fraction if pop.apply_payer_fraction else 1.0
    n_ed_payer = n_ed_all * payer_multiplier
    n_eligible = n_ed_payer * pop.p_unscreened
    return {
        "n_repro": n_repro,
        "n_pregnant": n_pregnant,
        "n_ed_all": n_ed_all,
        "n_ed_payer": n_ed_payer,
        "n_eligible": n_eligible,
        "payer_multiplier": payer_multiplier,
    }


def sigmoid_coverage_at_t(t, t_half, t_ninety, sc_uc, sc_e) -> float:
    if t_ninety <= t_half or sc_e <= sc_uc:
        return float(sc_e)
    k = np.log(9.0) / (t_ninety - t_half)
    uptake = 1.0 / (1.0 + np.exp(-k * (t - t_half)))
    coverage = sc_uc + (sc_e - sc_uc) * uptake
    return float(np.clip(coverage, min(sc_uc, sc_e), max(sc_uc, sc_e)))


def sigmoid_ramp(t_half, t_ninety, n_years, sc_uc, sc_e) -> Dict[int, float]:
    if int(n_years) < 1:
        raise ValueError("n_years must be at least 1")
    return {
        year: sigmoid_coverage_at_t(year, t_half, t_ninety, sc_uc, sc_e)
        for year in range(1, int(n_years) + 1)
    }


def bia_population_funnel(
    pop: BIAPopulation,
    year: int,
    ramp: Dict[int, float],
    sc_uc: float,
    p_id: float,
) -> dict:
    base = common_population_funnel(pop)
    eff_cov = float(ramp.get(year, ramp[max(ramp)]))
    n_intr = base["n_eligible"] * eff_cov * p_id
    n_uc = base["n_eligible"] * sc_uc * p_id
    n_incremental = float(max(n_intr - n_uc, 0.0))
    return {
        "year": int(year),
        "n_repro": base["n_repro"],
        "n_pregnant": base["n_pregnant"],
        "n_ed_all": base["n_ed_all"],
        "n_ed": base["n_ed_payer"],
        "n_unscreened": base["n_eligible"],
        "n_eligible": base["n_eligible"],
        "n_intr": n_intr,
        "n_uc": n_uc,
        "n_incremental": n_incremental,
        "eff_coverage": eff_cov,
    }


def bia_screening_cascade(
    n_screened: float,
    p_act: float,
    p_sf: float,
    sens: float,
    spec: float,
    p_adeq: float,
    tx_eff: float,
    p_trepo_sf: float,
    p_ux_sf: float,
    treat_fp: bool,
) -> dict:
    """Compatibility wrapper around the shared screening cascade."""
    raw = screening_cascade_counts(
        float(n_screened),
        p_act,
        p_sf,
        sens,
        spec,
        p_adeq,
        tx_eff,
        p_trepo_sf,
        p_ux_sf,
        treat_fp,
    )
    return {key: float(np.asarray(value)) for key, value in raw.items()}


def bia_annual_impact(
    funnel: dict,
    co: Costs,
    p_act: float,
    p_id: float,
    sc_uc: float,
    sens: float,
    spec: float,
    p_adeq: float,
    tx_eff: Optional[float],
    prop_symp: float,
    prop_late: Optional[float],
    p_sf: float,
    p_trepo_sf: float,
    p_ux_sf: float,
    treat_fp: bool,
    sc_target: Optional[float] = None,
) -> dict:
    """Within-horizon program costs and medical savings for one BIA year."""
    n_incremental = float(funnel["n_incremental"])
    final_target = float(
        funnel["eff_coverage"] if sc_target is None else sc_target
    )
    tx_completion_incremental = incremental_tx_completion_rate(
        float(sc_uc),
        float(funnel["eff_coverage"]),
        None if tx_eff is None else float(tx_eff),
        sc_target=final_target,
    )
    cascade_raw = screening_cascade_counts(
        n_incremental,
        p_act,
        p_sf,
        sens,
        spec,
        p_adeq,
        tx_completion_incremental,
        p_trepo_sf,
        p_ux_sf,
        treat_fp,
    )
    cascade = {key: float(np.asarray(value)) for key, value in cascade_raw.items()}
    program_raw = program_cost_components(cascade_raw, co)
    program = {key: float(np.asarray(value)) for key, value in program_raw.items()}

    br_mean, ur_mean, rr_mean = _deterministic_outcome_inputs()
    _, _, delta_raw = arm_outcome_delta(
        cohort=float(funnel["n_eligible"]),
        p_act=float(p_act),
        p_id=float(p_id),
        sc_b=float(sc_uc),
        sc_e=float(funnel["eff_coverage"]),
        sens=float(sens),
        p_adeq=float(p_adeq),
        tx_completion_override=(None if tx_eff is None else float(tx_eff)),
        prop_symp=float(prop_symp),
        prop_late_override=(None if prop_late is None else float(prop_late)),
        br=br_mean,
        ur=ur_mean,
        rr=rr_mean,
        sc_target=final_target,
    )
    delta = {key: float(np.asarray(value)) for key, value in delta_raw.items()}
    savings_raw = outcome_savings_components(
        {key: np.asarray([value], dtype=float) for key, value in delta.items()}, co
    )
    savings = {
        key: float(np.asarray(value).reshape(-1)[0])
        for key, value in savings_raw.items()
    }
    net = program["program_cost"] - savings["medical_savings"]

    return {
        "year": funnel["year"],
        "eff_coverage": funnel["eff_coverage"],
        "tx_completion_incremental": tx_completion_incremental,
        "n_incremental": cascade["n_screened"],
        "n_tp_detected": cascade["n_tp_detected"],
        "n_tp_treated": cascade["n_tp_treated"],
        "n_sf_detected": cascade["n_sf_detected"],
        "n_sf_treated": cascade["n_sf_treated"],
        "n_fp_detected": cascade["n_fp_detected"],
        "n_fp_treated": cascade["n_fp_treated"],
        "n_treated_total": cascade["n_treated_total"],
        "n_cs_averted": delta["cs_comp"] + delta["cs_uncomp"],
        "n_preterm_averted": delta["preterm"],
        "n_lbw_averted": delta["lbw"],
        "n_sb_averted": delta["stillbirth"],
        "n_nnd_averted": delta["neonatal_death"],
        "cost_screening_tests": program["cost_screening_tests"],
        "cost_confirmatory": program["cost_confirmatory"],
        "cost_staff": program["cost_staff"],
        "cost_tx_tp": program["cost_tx_tp"],
        "cost_tx_sf": program["cost_tx_sf"],
        "cost_tx_fp": program["cost_tx_fp"],
        "cost_treatment": program["cost_treatment"],
        "cost_sf_workup": program["cost_sf_workup"],
        "cost_tx_adjacent": program["cost_tx_adjacent"],
        "program_cost": program["program_cost"],
        "sav_cs": savings["sav_cs"],
        "sav_preterm": savings["sav_preterm"],
        "sav_lbw": savings["sav_lbw"],
        "sav_sb": savings["sav_sb"],
        "sav_nnd": savings["sav_nnd"],
        "medical_savings": savings["medical_savings"],
        "net_impact": net,
    }


def run_bia_scenario(
    pop: BIAPopulation,
    t_half: float,
    t_ninety: float,
    n_years: int,
    co: Costs,
    sc_e: float,
    sc_uc: float,
    p_act: float,
    p_id: float,
    sens: float,
    spec: float,
    p_adeq: float,
    tx_eff: Optional[float],
    prop_symp: float,
    prop_late: Optional[float],
    p_sf: float,
    p_trepo_sf: float,
    p_ux_sf: float,
    treat_fp: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ramp = sigmoid_ramp(t_half, t_ninety, int(n_years), sc_uc, sc_e)
    impact_rows = []
    funnel_rows = []
    cumulative = 0.0
    for year in range(1, int(n_years) + 1):
        funnel = bia_population_funnel(pop, year, ramp, sc_uc, p_id)
        impact = bia_annual_impact(
            funnel,
            co,
            p_act=p_act,
            p_id=p_id,
            sc_uc=sc_uc,
            sc_target=sc_e,
            sens=sens,
            spec=spec,
            p_adeq=p_adeq,
            tx_eff=tx_eff,
            prop_symp=prop_symp,
            prop_late=prop_late,
            p_sf=p_sf,
            p_trepo_sf=p_trepo_sf,
            p_ux_sf=p_ux_sf,
            treat_fp=treat_fp,
        )
        cumulative += impact["net_impact"]
        impact["cumulative_net"] = cumulative
        impact_rows.append(impact)
        funnel_rows.append(funnel)
    return pd.DataFrame(impact_rows), pd.DataFrame(funnel_rows)
