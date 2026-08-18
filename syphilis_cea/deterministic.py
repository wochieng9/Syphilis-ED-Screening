"""Deterministic cost-effectiveness and threshold-analysis helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import DEFAULT_MARKOV_CONFIG, MarkovConfig, markov_point_parameters
from .costing import _icost, program_cost_components, screening_cascade_counts
from .markov import _infant_markov_lifetime, calibrate_q_progress
from .outcomes import (
    _dalys_non_cs,
    _deterministic_outcome_inputs,
    arm_outcome_delta,
    incremental_screening_fraction,
    incremental_treatment_completion_mass,
    incremental_tx_completion_rate,
)
from .parameters import DW_P, Costs, LongTermCare
from .societal import MaternalMorbidity, ProductivityLoss, _mat_morb_det, _prod_loss_det


def _det_icost(
    p_act,
    p_sf,
    p_id,
    sc_b,
    sc_e,
    sens,
    spec,
    p_adeq,
    prop_symp,
    prop_late,
    p_trepo_sf,
    p_ux_sf,
    r,
    LE,
    inc_lbw,
    inc_mat,
    cohort,
    mm: MaternalMorbidity | None = None,
    pl: ProductivityLoss | None = None,
    ltc: LongTermCare | None = None,
    inc_sb_yll: bool = True,
    inc_cs_yll: bool = True,
    inc_misc_yld: bool = True,
    inc_mat_hosp_yld: bool = True,
    inc_preterm_yld: bool = True,
    tx_eff_override: float | None = None,
    treat_fp: bool = False,
    markov_config: MarkovConfig = DEFAULT_MARKOV_CONFIG,
) -> Tuple[float, float, float, float]:
    """Deterministic means for threshold and OWSA calculations.

    All Markov inputs are explicit and immutable. ``prop_late=None`` uses the
    source gestational-stratum values; a float is an explicit common override.
    """
    co = Costs()
    br_mean, ur_mean, rr_mean = _deterministic_outcome_inputs()
    comparator, intervention, delta_arr = arm_outcome_delta(
        cohort=float(cohort),
        p_act=float(p_act),
        p_id=float(p_id),
        sc_b=float(sc_b),
        sc_e=float(sc_e),
        sens=float(sens),
        p_adeq=float(p_adeq),
        tx_completion_override=(None if tx_eff_override is None else float(tx_eff_override)),
        prop_symp=float(prop_symp),
        prop_late_override=(None if prop_late is None else float(prop_late)),
        br=br_mean,
        ur=ur_mean,
        rr=rr_mean,
    )
    del comparator, intervention
    delta = {key: float(np.asarray(value)) for key, value in delta_arr.items()}

    point_dw = {key: np.asarray([value["m"]], dtype=float) for key, value in DW_P.items()}
    delta_vector = {key: np.asarray([value], dtype=float) for key, value in delta.items()}
    dal_non_cs = float(
        _dalys_non_cs(
            delta_vector,
            point_dw,
            float(r),
            float(LE),
            bool(inc_lbw),
            bool(inc_mat),
            inc_sb_yll=bool(inc_sb_yll),
            inc_misc_yld=bool(inc_misc_yld),
            inc_preterm_yld=bool(inc_preterm_yld),
        )[0]
    )

    T = max(int(LE), 1)
    q_progress = calibrate_q_progress(
        markov_config.q_progress_target,
        T,
        mu_excess_mild=markov_config.mu_excess_mild,
    )
    markov_result = _infant_markov_lifetime(
        delta_vector["cs_comp"],
        delta_vector["cs_uncomp"],
        markov_point_parameters(markov_config),
        float(r),
        T,
        ltc=ltc,
        include_cs_yll=bool(inc_cs_yll),
        q_progress=q_progress,
    )

    coverage_gap = incremental_screening_fraction(float(sc_b), float(sc_e))
    tx_mass = incremental_treatment_completion_mass(
        float(sc_b),
        float(sc_e),
        None if tx_eff_override is None else float(tx_eff_override),
    )
    tx_completion_incremental = incremental_tx_completion_rate(
        float(sc_b),
        float(sc_e),
        None if tx_eff_override is None else float(tx_eff_override),
    )
    n_incremental_screened = float(cohort) * float(p_id) * coverage_gap
    cascade = screening_cascade_counts(
        n_incremental_screened,
        float(p_act),
        float(p_sf),
        float(sens),
        float(spec),
        float(p_adeq),
        tx_completion_incremental,
        float(p_trepo_sf),
        float(p_ux_sf),
        bool(treat_fp),
    )
    program = program_cost_components(cascade, co)

    n_maternal_tx = (
        float(cohort)
        * float(p_act)
        * float(p_id)
        * float(sens)
        * float(p_adeq)
        * tx_mass
    )
    mm_dal = 0.0
    mm_cost = 0.0
    if mm is not None:
        mm_dal, mm_cost = _mat_morb_det(
            mm, n_maternal_tx, float(r), include_hosp_yld=bool(inc_mat_hosp_yld)
        )

    prod_saving = 0.0
    if pl is not None:
        prod_saving = _prod_loss_det(
            pl,
            delta["stillbirth"],
            delta["neonatal_death"],
            delta["cs_comp"],
            delta["cs_uncomp"],
            float(r),
            float(LE),
            markov_config=markov_config,
        )

    health_sector, societal = _icost(
        delta_vector,
        co,
        np.asarray([program["cost_sf_total"]], dtype=float),
        float(sc_b),
        float(sc_e),
        float(p_act),
        float(p_sf),
        float(p_trepo_sf),
        float(p_id),
        float(sens),
        float(spec),
        float(p_adeq),
        tx_completion_incremental,
        bool(treat_fp),
        float(cohort),
        markov_result.medical_costs,
        mk_ltc_cost_saving=markov_result.long_term_care_costs,
        mat_cost_saving=(np.asarray([mm_cost]) if mm is not None else None),
        prod_loss_saving=(np.asarray([prod_saving]) if pl is not None else None),
        program_cost_override=np.asarray([program["program_cost"]], dtype=float),
    )

    # Perspective changes resource valuation, not which health gains count.
    # Maternal morbidity health effects therefore enter both denominators.
    dal_hs = dal_non_cs + float(markov_result.dalys[0]) + float(mm_dal)
    dal_soc = dal_hs
    return (
        float(np.asarray(health_sector).reshape(-1)[0]),
        float(dal_hs),
        float(np.asarray(societal).reshape(-1)[0]),
        float(dal_soc),
    )


def nmb_surface(
    prev_grid,
    tx_grid,
    p_sf,
    p_id,
    sc_b,
    sc_e,
    sens,
    spec,
    prop_symp,
    prop_late,
    p_trepo_sf,
    p_ux_sf,
    r,
    LE,
    inc_lbw,
    inc_mat,
    cohort,
    wtp,
    mm=None,
    pl=None,
    ltc=None,
    societal=False,
    inc_sb_yll=True,
    inc_cs_yll=True,
    inc_misc_yld=True,
    inc_mat_hosp_yld=True,
    inc_preterm_yld=True,
    tx_eff_override=None,
    treat_fp=False,
    markov_config: MarkovConfig = DEFAULT_MARKOV_CONFIG,
    p_sf_ratio: Optional[float] = None,
) -> np.ndarray:
    """Net-monetary-benefit surface over prevalence and treatment initiation."""
    grid = np.zeros((len(tx_grid), len(prev_grid)), dtype=float)
    for i, p_adeq in enumerate(tx_grid):
        for j, p_act in enumerate(prev_grid):
            p_sf_current = (
                float(np.clip(float(p_sf_ratio) * float(p_act), 0.0, 1.0))
                if p_sf_ratio is not None
                else float(p_sf)
            )
            ic_hs, dal_hs, ic_soc, dal_soc = _det_icost(
                p_act,
                p_sf_current,
                p_id,
                sc_b,
                sc_e,
                sens,
                spec,
                p_adeq,
                prop_symp,
                prop_late,
                p_trepo_sf,
                p_ux_sf,
                r,
                LE,
                inc_lbw,
                inc_mat,
                cohort,
                mm,
                pl,
                ltc,
                inc_sb_yll=inc_sb_yll,
                inc_cs_yll=inc_cs_yll,
                inc_misc_yld=inc_misc_yld,
                inc_mat_hosp_yld=inc_mat_hosp_yld,
                inc_preterm_yld=inc_preterm_yld,
                tx_eff_override=tx_eff_override,
                treat_fp=treat_fp,
                markov_config=markov_config,
            )
            cost = ic_soc if societal else ic_hs
            effect = dal_soc if societal else dal_hs
            grid[i, j] = float(wtp) * effect - cost
    return grid
