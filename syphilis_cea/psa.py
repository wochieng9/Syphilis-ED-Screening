"""Probabilistic sensitivity-analysis draws and simulation engine."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import DEFAULT_MARKOV_CONFIG, MarkovConfig, normalize_complicated_state_probabilities
from .costing import _icost, program_cost_components, screening_cascade_counts
from .markov import _infant_markov_lifetime, calibrate_q_progress
from .outcomes import (
    _dalys_non_cs,
    arm_outcome_delta,
    incremental_screening_fraction,
    incremental_treatment_completion_mass,
    incremental_tx_completion_rate,
)
from .parameters import BASE_BETA, DW_P, TX_RR, UNT_ABS, Costs, LongTermCare
from .randomness import spawn_named_generators
from .societal import (
    MaternalMorbidity,
    ProductivityLoss,
    _mat_morb_dalys,
    _prod_loss_per_case,
)
from .utils import draw_gamma, draw_probability, lnorm_ms, ratio_of_means, safe_icer, summarize


def _draw_all(N: int, rng: np.random.Generator, costs: Costs):
    br = {
        key: rng.beta(value["a"], value["b"], size=N).astype(float)
        for key, value in BASE_BETA.items()
    }
    ur = {
        key: rng.beta(value * 1000.0, (1.0 - value) * 1000.0, size=N).astype(float)
        for key, value in UNT_ABS.items()
    }
    rr = {}
    for key, value in TX_RR.items():
        mu, sigma = lnorm_ms(value["rr"], value["lo"], value["hi"])
        rr[key] = rng.lognormal(mu, sigma, size=N).astype(float)
    dw = {
        key: draw_probability(rng, N, value["m"], value["lo"], value["hi"])
        for key, value in DW_P.items()
    }

    cost_dict = asdict(costs)
    cs = {}
    for key, value in cost_dict.items():
        if key.endswith("_sd"):
            continue
        sd = float(cost_dict.get(f"{key}_sd", 0.0))
        cs[key] = draw_gamma(rng, N, float(value), sd)
    return br, ur, rr, dw, cs


def _draw_infant_mk(N: int, rng: np.random.Generator, config: MarkovConfig) -> dict:
    mk = {
        "p_severe_cs_comp": draw_probability(
            rng,
            N,
            config.p_severe_cs_comp,
            config.p_severe_cs_comp_lo,
            config.p_severe_cs_comp_hi,
        ),
        "p_mild_cs_comp": draw_probability(
            rng,
            N,
            config.p_mild_cs_comp,
            config.p_mild_cs_comp_lo,
            config.p_mild_cs_comp_hi,
        ),
        "p_mild_cs_uncomp": draw_probability(
            rng,
            N,
            config.p_mild_cs_uncomp,
            config.p_mild_cs_uncomp_lo,
            config.p_mild_cs_uncomp_hi,
        ),
        "dw_mild": draw_probability(
            rng, N, config.dw_mild, config.dw_mild_lo, config.dw_mild_hi
        ),
        "dw_severe": draw_probability(
            rng, N, config.dw_severe, config.dw_severe_lo, config.dw_severe_hi
        ),
        "mu_excess_mild": draw_probability(
            rng,
            N,
            config.mu_excess_mild,
            config.mu_excess_mild_lo,
            config.mu_excess_mild_hi,
        ),
        "mu_excess_severe": draw_probability(
            rng,
            N,
            config.mu_excess_severe,
            config.mu_excess_severe_lo,
            config.mu_excess_severe_hi,
        ),
    }
    _, mild, severe = normalize_complicated_state_probabilities(
        mk["p_mild_cs_comp"], mk["p_severe_cs_comp"]
    )
    mk["p_mild_cs_comp"] = mild
    mk["p_severe_cs_comp"] = severe

    mk["cost_mild_ann"] = draw_gamma(
        rng, N, config.cost_mild_ann, config.cost_mild_ann_sd
    )
    mk["cost_sev_ann"] = draw_gamma(
        rng, N, config.cost_severe_ann, config.cost_severe_ann_sd
    )
    return mk


def _draw_ltc(N: int, rng: np.random.Generator, ltc: LongTermCare) -> dict:
    draws = {
        "p_sped_severe": draw_probability(
            rng, N, ltc.p_sped_severe, ltc.p_sped_severe_lo, ltc.p_sped_severe_hi
        ),
        "p_sped_mild": draw_probability(
            rng, N, ltc.p_sped_mild, ltc.p_sped_mild_lo, ltc.p_sped_mild_hi
        ),
    }
    for key, mean, sd in (
        ("cost_sped_ann", ltc.cost_sped_ann, ltc.cost_sped_sd),
        ("cost_cg_severe_ann", ltc.cost_cg_severe_ann, ltc.cost_cg_severe_sd),
        ("cost_cg_mild_ann", ltc.cost_cg_mild_ann, ltc.cost_cg_mild_sd),
    ):
        draws[key] = draw_gamma(rng, N, mean, sd)
    return draws


def _draw_mat_morb(N: int, rng: np.random.Generator, mm: MaternalMorbidity) -> dict:
    draws = {
        "p_cardio": draw_probability(rng, N, mm.p_cardio, mm.p_cardio_lo, mm.p_cardio_hi),
        "p_neuro": draw_probability(rng, N, mm.p_neuro, mm.p_neuro_lo, mm.p_neuro_hi),
        "p_hosp": draw_probability(rng, N, mm.p_hosp, mm.p_hosp_lo, mm.p_hosp_hi),
    }
    for key, mean, sd in (
        ("cost_cardio", mm.cost_cardio, mm.cost_cardio_sd),
        ("cost_neuro", mm.cost_neuro, mm.cost_neuro_sd),
        ("cost_hosp", mm.cost_hosp, mm.cost_hosp_sd),
    ):
        draws[key] = draw_gamma(rng, N, mean, sd)
    return draws


def _draw_serofast(
    N: int,
    rng: np.random.Generator,
    p_sf: float,
    p_trepo_sf: float,
    p_ux_sf: float,
) -> dict:
    p_sf_f = float(np.clip(p_sf, 0.0, 1.0))
    lo_sf = max(p_sf_f * 0.50, 0.0)
    hi_sf = min(max(p_sf_f * 1.50, p_sf_f + 1e-6), 1.0)
    return {
        "p_sf": draw_probability(rng, N, p_sf_f, lo_sf, hi_sf),
        "p_trepo_sf": draw_probability(rng, N, p_trepo_sf, 0.70, 1.0),
        "p_ux_sf": draw_probability(
            rng,
            N,
            p_ux_sf,
            max(float(p_ux_sf) - 0.15, 0.0),
            min(float(p_ux_sf) + 0.15, 1.0),
        ),
    }


def _draw_prod_loss(N: int, rng: np.random.Generator, pl: ProductivityLoss) -> dict:
    """Draw productivity inputs with bounded wage-penalty probabilities."""
    draws = {}
    for key, mean, sd in (
        ("bereavement_days", pl.bereavement_days, pl.bereavement_days_sd),
        ("caregiver_hrs_wk", pl.caregiver_hrs_wk, pl.caregiver_hrs_wk_sd),
    ):
        draws[key] = draw_gamma(rng, N, mean, sd)
    for key, mean, sd in (
        ("wage_penalty_mild", pl.wage_penalty_mild, pl.wage_penalty_mild_sd),
        ("wage_penalty_severe", pl.wage_penalty_severe, pl.wage_penalty_severe_sd),
    ):
        draws[key] = draw_probability(
            rng, N, mean, max(mean - 1.96 * sd, 0.0), min(mean + 1.96 * sd, 1.0)
        )
    return draws

def _zero_draws(N: int, keys) -> dict:
    return {key: np.zeros(N, dtype=float) for key in keys}


def run_psa(
    N,
    seed,
    cohort,
    p_act,
    p_sf,
    p_id,
    sc_b,
    sc_e,
    sens,
    spec,
    p_adeq,
    p_tx_override,
    p_trepo_sf,
    p_ux_sf,
    prop_symp,
    prop_late,
    r,
    LE,
    inc_lbw,
    inc_mat,
    inc_sb_yll,
    inc_cs_yll,
    inc_misc_yld,
    inc_mat_hosp_yld,
    inc_preterm_yld,
    treat_fp,
    vsl,
    use_mat_morb,
    use_prod_loss,
    use_friction,
    mm_p_cardio,
    mm_p_neuro,
    mm_p_hosp,
    mm_cost_cardio,
    mm_cost_neuro,
    mm_cost_hosp,
    pl_bereavement_days,
    pl_wage_mild,
    pl_wage_severe,
    pl_caregiver_hrs,
    pl_caregiver_wage_frac,
    mk_p_sev,
    mk_p_mc,
    mk_p_mu,
    mk_c_mild,
    mk_c_sev,
    mk_q_target,
    mk_mu_x_mild,
    mk_mu_x_sev,
    use_ltc,
    ltc_p_sped_sev,
    ltc_p_sped_mid,
    ltc_cost_sped,
    ltc_cost_cg_sv,
    ltc_cost_cg_ml,
    ltc_sped_start,
    ltc_sped_end,
    ltc_cg_end,
):
    """Run the corrected vectorized PSA.

    ``prop_late=None`` uses gestational-stratum values; a float is an explicit
    common override. All random parameter groups use independent named streams,
    so disabled-module settings cannot perturb active results.
    """
    N = int(N)
    cohort_f = float(cohort)
    if N < 1:
        raise ValueError("N must be at least 1")

    markov_config = MarkovConfig(
        p_severe_cs_comp=float(mk_p_sev),
        p_mild_cs_comp=float(mk_p_mc),
        p_mild_cs_uncomp=float(mk_p_mu),
        dw_mild=DEFAULT_MARKOV_CONFIG.dw_mild,
        dw_mild_lo=DEFAULT_MARKOV_CONFIG.dw_mild_lo,
        dw_mild_hi=DEFAULT_MARKOV_CONFIG.dw_mild_hi,
        dw_severe=DEFAULT_MARKOV_CONFIG.dw_severe,
        dw_severe_lo=DEFAULT_MARKOV_CONFIG.dw_severe_lo,
        dw_severe_hi=DEFAULT_MARKOV_CONFIG.dw_severe_hi,
        cost_mild_ann=float(mk_c_mild),
        cost_severe_ann=float(mk_c_sev),
        q_progress_target=float(mk_q_target),
        mu_excess_mild=float(mk_mu_x_mild),
        mu_excess_severe=float(mk_mu_x_sev),
    )
    T = max(int(LE), 1)
    q_progress = calibrate_q_progress(
        markov_config.q_progress_target, T,
        mu_excess_mild=markov_config.mu_excess_mild,
    )

    streams = spawn_named_generators(int(seed))
    br, ur, rr, dw, cs = _draw_all(N, streams["clinical"], Costs())
    mk = _draw_infant_mk(N, streams["markov"], markov_config)

    mm = MaternalMorbidity(
        p_cardio=float(mm_p_cardio),
        p_neuro=float(mm_p_neuro),
        p_hosp=float(mm_p_hosp),
        cost_cardio=float(mm_cost_cardio),
        cost_neuro=float(mm_cost_neuro),
        cost_hosp=float(mm_cost_hosp),
    )
    pl = ProductivityLoss(
        bereavement_days=float(pl_bereavement_days),
        wage_penalty_mild=float(pl_wage_mild),
        wage_penalty_severe=float(pl_wage_severe),
        caregiver_hrs_wk=float(pl_caregiver_hrs),
        caregiver_wage_frac=float(pl_caregiver_wage_frac),
        friction_period_days=90.0 if use_friction else 0.0,
    )
    ltc = (
        LongTermCare(
            p_sped_severe=float(ltc_p_sped_sev),
            p_sped_mild=float(ltc_p_sped_mid),
            cost_sped_ann=float(ltc_cost_sped),
            cost_cg_severe_ann=float(ltc_cost_cg_sv),
            cost_cg_mild_ann=float(ltc_cost_cg_ml),
            sped_start_age=int(ltc_sped_start),
            sped_end_age=int(ltc_sped_end),
            caregiver_end_age=int(ltc_cg_end),
        )
        if use_ltc
        else None
    )

    psa_ltc = (
        _draw_ltc(N, streams["long_term_care"], ltc)
        if ltc is not None
        else _zero_draws(
            N,
            (
                "p_sped_severe",
                "p_sped_mild",
                "cost_sped_ann",
                "cost_cg_severe_ann",
                "cost_cg_mild_ann",
            ),
        )
    )
    psa_mm = (
        _draw_mat_morb(N, streams["maternal"], mm)
        if use_mat_morb
        else _zero_draws(
            N,
            ("p_cardio", "p_neuro", "p_hosp", "cost_cardio", "cost_neuro", "cost_hosp"),
        )
    )
    psa_pl = (
        _draw_prod_loss(N, streams["productivity"], pl)
        if use_prod_loss
        else _zero_draws(
            N,
            ("bereavement_days", "caregiver_hrs_wk", "wage_penalty_mild", "wage_penalty_severe"),
        )
    )
    psa_sf = _draw_serofast(
        N, streams["serofast"], float(p_sf), float(p_trepo_sf), float(p_ux_sf)
    )

    comparator, intervention, delta = arm_outcome_delta(
        cohort=cohort_f,
        p_act=float(p_act),
        p_id=float(p_id),
        sc_b=float(sc_b),
        sc_e=float(sc_e),
        sens=float(sens),
        p_adeq=float(p_adeq),
        tx_completion_override=(None if p_tx_override is None else float(p_tx_override)),
        prop_symp=float(prop_symp),
        prop_late_override=(None if prop_late is None else float(prop_late)),
        br=br,
        ur=ur,
        rr=rr,
    )
    comp_means = {key: float(np.mean(value)) for key, value in comparator.items()}
    intr_means = {key: float(np.mean(value)) for key, value in intervention.items()}

    markov_result = _infant_markov_lifetime(
        delta["cs_comp"],
        delta["cs_uncomp"],
        mk,
        float(r),
        T,
        ltc=ltc,
        psa_ltc=(psa_ltc if ltc is not None else None),
        include_cs_yll=bool(inc_cs_yll),
        q_progress=q_progress,
    )

    coverage_gap = incremental_screening_fraction(float(sc_b), float(sc_e))
    tx_mass = incremental_treatment_completion_mass(
        float(sc_b),
        float(sc_e),
        None if p_tx_override is None else float(p_tx_override),
    )
    tx_completion_incremental = incremental_tx_completion_rate(
        float(sc_b),
        float(sc_e),
        None if p_tx_override is None else float(p_tx_override),
    )
    n_incremental_screened = np.full(
        N, cohort_f * float(p_id) * coverage_gap, dtype=float
    )
    n_maternal_tx = np.full(
        N,
        cohort_f
        * float(p_act)
        * float(p_id)
        * float(sens)
        * float(p_adeq)
        * tx_mass,
        dtype=float,
    )

    cascade = screening_cascade_counts(
        n_incremental_screened,
        float(p_act),
        psa_sf["p_sf"],
        float(sens),
        float(spec),
        float(p_adeq),
        tx_completion_incremental,
        psa_sf["p_trepo_sf"],
        psa_sf["p_ux_sf"],
        bool(treat_fp),
    )
    program = program_cost_components(cascade, cs)
    sf_cost = np.asarray(program["cost_sf_total"], dtype=float)

    if use_mat_morb:
        mm_dal, mm_cost, mm_components = _mat_morb_dalys(
            mm,
            n_maternal_tx,
            tx_mass,
            float(r),
            psa_mm,
            dw_hosp=dw.get("mat_hosp"),
            include_hosp_yld=bool(inc_mat_hosp_yld),
            return_components=True,
        )
    else:
        mm_dal = np.zeros(N, dtype=float)
        mm_cost = np.zeros(N, dtype=float)
        mm_components = {
            "mat_cardio_dal": np.zeros(N, dtype=float),
            "mat_neuro_dal": np.zeros(N, dtype=float),
            "mat_hosp_dal": np.zeros(N, dtype=float),
        }

    if use_prod_loss:
        prod_sav = _prod_loss_per_case(
            pl,
            psa_pl,
            np.asarray(delta["stillbirth"], dtype=float),
            np.asarray(delta["neonatal_death"], dtype=float),
            np.asarray(delta["cs_comp"], dtype=float),
            np.asarray(delta["cs_uncomp"], dtype=float),
            mk,
            float(r),
            float(LE),
        )
    else:
        prod_sav = np.zeros(N, dtype=float)

    dal_non_cs_hs, non_cs_components = _dalys_non_cs(
        delta,
        dw,
        float(r),
        float(LE),
        bool(inc_lbw),
        bool(inc_mat),
        inc_sb_yll=bool(inc_sb_yll),
        inc_misc_yld=bool(inc_misc_yld),
        inc_preterm_yld=bool(inc_preterm_yld),
        return_components=True,
    )
    # Perspective changes resource valuation, not the set of health outcomes.
    # Maternal morbidity health gains enter both health-sector and societal DALYs.
    dal_hs = dal_non_cs_hs + markov_result.dalys + mm_dal
    soc_daly_increment = np.zeros(N, dtype=float)
    dal_soc = dal_hs.copy()

    ic_hs, ic_soc = _icost(
        delta,
        cs,
        sf_cost,
        float(sc_b),
        float(sc_e),
        float(p_act),
        psa_sf["p_sf"],
        psa_sf["p_trepo_sf"],
        float(p_id),
        float(sens),
        float(spec),
        float(p_adeq),
        tx_completion_incremental,
        bool(treat_fp),
        cohort_f,
        markov_result.medical_costs,
        mk_ltc_cost_saving=markov_result.long_term_care_costs,
        mat_cost_saving=(mm_cost if use_mat_morb else None),
        prod_loss_saving=(prod_sav if use_prod_loss else None),
        program_cost_override=np.asarray(program["program_cost"], dtype=float),
    )

    vsl_nb = (
        float(vsl)
        * (np.asarray(delta["stillbirth"]) + np.asarray(delta["neonatal_death"]))
        - ic_hs
    )

    icer_hs = safe_icer(ic_hs, dal_hs)
    icer_soc = safe_icer(ic_soc, dal_soc)
    df = pd.DataFrame(
        {
            "dal_hs": np.asarray(dal_hs, dtype=float),
            "dal_soc": np.asarray(dal_soc, dtype=float),
            "ic_hs": np.asarray(ic_hs, dtype=float),
            "ic_soc": np.asarray(ic_soc, dtype=float),
            "icer_hs": np.asarray(icer_hs, dtype=float),
            "icer_soc": np.asarray(icer_soc, dtype=float),
            "vsl_nb": np.asarray(vsl_nb, dtype=float),
            "sf_cost": sf_cost,
            "program_cost": np.asarray(program["program_cost"], dtype=float),
            "mk_dal": markov_result.dalys,
            "mk_yld": markov_result.yld,
            "mk_yll": markov_result.yll,
            "mk_med_cst": markov_result.medical_costs,
            "mk_ltc_cst": markov_result.long_term_care_costs,
            "mk_cst": markov_result.total_costs,
            "mm_dal": np.asarray(mm_dal, dtype=float),
            "soc_daly_increment": np.asarray(soc_daly_increment, dtype=float),
            "mm_cost": np.asarray(mm_cost, dtype=float),
            "prod_sav": np.asarray(prod_sav, dtype=float),
            "tx_completion_incremental": np.full(N, tx_completion_incremental),
            **{f"dal_{key}": np.asarray(value, dtype=float) for key, value in non_cs_components.items()},
            **{f"dal_{key}": np.asarray(value, dtype=float) for key, value in mm_components.items()},
            **{f"rr_{key}": value for key, value in rr.items()},
            **{f"ur_{key}": value for key, value in ur.items()},
            **{f"br_{key}": value for key, value in br.items()},
            **{f"dw_{key}": value for key, value in dw.items()},
            **{f"co_{key}": value for key, value in cs.items()},
            **{f"mkp_{key}": value for key, value in mk.items()},
            **{f"mmp_{key}": value for key, value in psa_mm.items()},
            **{f"plp_{key}": value for key, value in psa_pl.items()},
            **{f"ltcp_{key}": value for key, value in psa_ltc.items()},
            **{f"d_{key}": np.asarray(value, dtype=float) for key, value in delta.items()},
            **{f"sf_{key}": value for key, value in psa_sf.items()},
        }
    )
    df["d_cs_total"] = np.asarray(delta["cs_comp"] + delta["cs_uncomp"], dtype=float)
    df["d_sb_nnd_total"] = np.asarray(
        delta["stillbirth"] + delta["neonatal_death"], dtype=float
    )
    df["n_maternal_tx"] = n_maternal_tx

    smry = {
        "inc_cost_hs": summarize(ic_hs),
        "inc_cost_soc": summarize(ic_soc),
        "dalys_hs": summarize(dal_hs),
        "dalys_soc": summarize(dal_soc),
        "dalys_soc_increment": summarize(soc_daly_increment),
        "dalys_non_cs": summarize(dal_non_cs_hs),
        "dalys_markov": summarize(markov_result.dalys),
        "dalys_markov_yld": summarize(markov_result.yld),
        "dalys_markov_yll": summarize(markov_result.yll),
        "dalys_mat_morb": summarize(mm_dal),
        **{f"dalys_{key}": summarize(value) for key, value in non_cs_components.items()},
        **{f"dalys_{key}": summarize(value) for key, value in mm_components.items()},
        "icer_hs": summarize(icer_hs),
        "icer_soc": summarize(icer_soc),
        "icer_hs_ratio_of_means": ratio_of_means(ic_hs, dal_hs),
        "icer_soc_ratio_of_means": ratio_of_means(ic_soc, dal_soc),
        "vsl_nb": summarize(vsl_nb),
        "p_cost_saving_hs": float(np.mean(ic_hs < 0.0)),
        "p_dominant_hs": float(np.mean((ic_hs < 0.0) & (dal_hs > 0.0))),
        "p_cost_saving_soc": float(np.mean(ic_soc < 0.0)),
        "p_dominant_soc": float(np.mean((ic_soc < 0.0) & (dal_soc > 0.0))),
        "sf_cost": summarize(sf_cost),
        "mk_cst": summarize(markov_result.total_costs),
        "mk_med_cst": summarize(markov_result.medical_costs),
        "mk_ltc_cst": summarize(markov_result.long_term_care_costs),
        "prod_sav": summarize(prod_sav),
        **{f"d_{key}": summarize(value) for key, value in delta.items()},
        "d_cs_total": summarize(delta["cs_comp"] + delta["cs_uncomp"]),
        "d_sb_nnd_total": summarize(delta["stillbirth"] + delta["neonatal_death"]),
        "n_maternal_tx": summarize(n_maternal_tx),
        "comp_means": comp_means,
        "intr_means": intr_means,
        "tx_completion_incremental": tx_completion_incremental,
    }
    return df, smry, comp_means, intr_means
