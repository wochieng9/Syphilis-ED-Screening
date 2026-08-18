from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from syphilis_cea.bia import (
    BIAPopulation,
    bia_annual_impact,
    bia_population_funnel,
    common_population_funnel,
)
from syphilis_cea.config import DEFAULT_MARKOV_CONFIG, MarkovConfig, markov_point_parameters
from syphilis_cea.costing import (
    _serofast_cost,
    outcome_savings_components,
    program_cost_components,
    screening_cascade_counts,
)
from syphilis_cea.deterministic import _det_icost, nmb_surface
from syphilis_cea.markov import (
    _infant_markov_lifetime,
    calibrate_q_progress,
    markov_state_occupancy,
)
from syphilis_cea.outcomes import (
    _deterministic_outcome_inputs,
    arm_outcome_delta,
    incremental_treatment_completion_mass,
    incremental_tx_completion_rate,
)
from syphilis_cea.parameters import (
    GES_STRATA,
    INFANT_MK,
    Costs,
    LongTermCare,
    ges_eff,
    joint_usual_care_treatment_completion,
)
from syphilis_cea.psa import _draw_infant_mk, run_psa
from syphilis_cea.reporting import ce_quadrant_table, compute_evppi
from syphilis_cea.societal import ProductivityLoss, _prod_loss_det, _prod_loss_per_case
from syphilis_cea.utils import draw_probability, safe_icer


def base_psa_kwargs(**overrides):
    sc_uc, _, _ = ges_eff()
    values = dict(
        N=60,
        seed=2025,
        cohort=1_000,
        p_act=0.0075,
        p_sf=0.0015,
        p_id=0.85,
        sc_b=sc_uc,
        sc_e=0.90,
        sens=0.98,
        spec=0.98,
        p_adeq=0.85,
        p_tx_override=None,
        p_trepo_sf=0.95,
        p_ux_sf=0.20,
        prop_symp=0.38,
        prop_late=None,
        r=0.035,
        LE=78.0,
        inc_lbw=True,
        inc_mat=True,
        inc_sb_yll=True,
        inc_cs_yll=True,
        inc_misc_yld=True,
        inc_mat_hosp_yld=True,
        inc_preterm_yld=True,
        treat_fp=False,
        vsl=13_700_000.0,
        use_mat_morb=True,
        use_prod_loss=True,
        use_friction=False,
        mm_p_cardio=0.05,
        mm_p_neuro=0.10,
        mm_p_hosp=0.12,
        mm_cost_cardio=18_500.0,
        mm_cost_neuro=32_000.0,
        mm_cost_hosp=4_200.0,
        pl_bereavement_days=30.0,
        pl_wage_mild=0.10,
        pl_wage_severe=0.40,
        pl_caregiver_hrs=10.0,
        pl_caregiver_wage_frac=0.50,
        mk_p_sev=0.35,
        mk_p_mc=0.40,
        mk_p_mu=0.06,
        mk_c_mild=8_500.0,
        mk_c_sev=26_000.0,
        mk_q_target=0.20,
        mk_mu_x_mild=0.0,
        mk_mu_x_sev=0.0,
        use_ltc=True,
        ltc_p_sped_sev=0.85,
        ltc_p_sped_mid=0.35,
        ltc_cost_sped=14_000.0,
        ltc_cost_cg_sv=32_000.0,
        ltc_cost_cg_ml=4_500.0,
        ltc_sped_start=3,
        ltc_sped_end=21,
        ltc_cg_end=18,
    )
    values.update(overrides)
    return values


def base_det_kwargs(**overrides):
    sc_uc, _, _ = ges_eff()
    values = dict(
        p_act=0.0075,
        p_sf=0.0015,
        p_id=0.85,
        sc_b=sc_uc,
        sc_e=0.90,
        sens=0.98,
        spec=0.98,
        p_adeq=0.85,
        prop_symp=0.38,
        prop_late=None,
        p_trepo_sf=0.95,
        p_ux_sf=0.20,
        r=0.035,
        LE=78.0,
        inc_lbw=True,
        inc_mat=True,
        cohort=10_000,
        tx_eff_override=None,
        treat_fp=False,
    )
    values.update(overrides)
    return values


def test_probability_endpoints_are_point_masses_not_half_probability():
    rng = np.random.default_rng(17)
    assert np.array_equal(draw_probability(rng, 100, 0.0, 0.0, 0.15), np.zeros(100))
    assert np.array_equal(draw_probability(rng, 100, 1.0, 0.70, 1.0), np.ones(100))

    df, _, _, _ = run_psa(**base_psa_kwargs(p_sf=0.0, p_trepo_sf=1.0, p_ux_sf=0.0))
    assert (df["sf_p_sf"] == 0.0).all()
    assert (df["sf_p_trepo_sf"] == 1.0).all()
    assert (df["sf_p_ux_sf"] == 0.0).all()
    assert (df["sf_cost"] == 0.0).all()


def test_disabled_module_inputs_do_not_change_active_psa_draws_or_results():
    disabled = dict(use_mat_morb=False, use_prod_loss=False, use_ltc=False)
    df_a, _, _, _ = run_psa(**base_psa_kwargs(**disabled))
    df_b, _, _, _ = run_psa(
        **base_psa_kwargs(
            **disabled,
            mm_p_cardio=0.14,
            mm_p_neuro=0.24,
            mm_p_hosp=0.29,
            mm_cost_cardio=49_000.0,
            mm_cost_neuro=79_000.0,
            mm_cost_hosp=14_000.0,
            pl_bereavement_days=89.0,
            pl_wage_mild=0.29,
            pl_wage_severe=0.59,
            pl_caregiver_hrs=29.0,
            ltc_cost_sped=39_000.0,
            ltc_cost_cg_sv=79_000.0,
            ltc_cost_cg_ml=19_500.0,
        )
    )
    columns = [
        "dal_hs",
        "ic_hs",
        "sf_cost",
        "d_cs_total",
        "rr_cs_any",
        "co_poc",
        "mkp_p_mild_cs_comp",
    ]
    for column in columns:
        assert np.array_equal(df_a[column].to_numpy(), df_b[column].to_numpy()), column


def test_gestational_treatment_reach_uses_joint_aggregation():
    sc_b, tx_mean, _ = ges_eff()
    sc_e = 0.90
    expected = sum(
        values["w"] * (sc_e - values["p_uc"]) * values["p_tx"]
        for values in GES_STRATA.values()
    )
    observed = incremental_treatment_completion_mass(sc_b, sc_e, None)
    naive = (sc_e - sc_b) * tx_mean
    assert observed == pytest.approx(expected)
    assert observed != pytest.approx(naive)
    assert joint_usual_care_treatment_completion() == pytest.approx(
        sum(v["w"] * v["p_uc"] * v["p_tx"] for v in GES_STRATA.values())
    )
    assert incremental_tx_completion_rate(sc_b, sc_e, None) == pytest.approx(
        observed / (sc_e - sc_b)
    )


def test_markov_probabilities_are_simplex_safe_and_occupancy_conserves_mass():
    with pytest.raises(ValueError):
        MarkovConfig(p_severe_cs_comp=0.60, p_mild_cs_comp=0.65)

    rng = np.random.default_rng(2)
    draws = _draw_infant_mk(2_000, rng, DEFAULT_MARKOV_CONFIG)
    total = draws["p_severe_cs_comp"] + draws["p_mild_cs_comp"]
    assert np.all(total <= 1.0 + 1e-12)
    assert np.all(total >= 0.0)

    q = calibrate_q_progress(DEFAULT_MARKOV_CONFIG.q_progress_target, 78)
    occupancy = markov_state_occupancy(
        DEFAULT_MARKOV_CONFIG, q, 78, complicated=True
    )
    assert np.allclose(occupancy.sum(axis=1), 1.0)


def test_mutating_deprecated_global_does_not_change_deterministic_results():
    before = _det_icost(**base_det_kwargs())
    snapshot = copy.deepcopy(INFANT_MK)
    try:
        INFANT_MK["p_severe_cs_comp"]["m"] = 0.01
        INFANT_MK["p_mild_cs_comp"]["m"] = 0.01
        INFANT_MK["cost_mild_ann"]["mu"] = 1.0
        after = _det_icost(**base_det_kwargs())
    finally:
        INFANT_MK.clear()
        INFANT_MK.update(snapshot)
    assert before == pytest.approx(after)


def test_friction_cost_psa_and_deterministic_productivity_paths_agree():
    pl = ProductivityLoss(friction_period_days=90.0)
    mk = {
        "p_mild_cs_comp": np.array([0.40]),
        "p_mild_cs_uncomp": np.array([0.06]),
        "p_severe_cs_comp": np.array([0.35]),
    }
    psa_pl = {
        "bereavement_days": np.array([pl.bereavement_days]),
        "wage_penalty_mild": np.array([pl.wage_penalty_mild]),
        "wage_penalty_severe": np.array([pl.wage_penalty_severe]),
        "caregiver_hrs_wk": np.array([pl.caregiver_hrs_wk]),
    }
    vector = _prod_loss_per_case(
        pl,
        psa_pl,
        np.array([1.2]),
        np.array([0.7]),
        np.array([2.5]),
        np.array([1.1]),
        mk,
        0.035,
        78.0,
    )[0]
    scalar = _prod_loss_det(
        pl, 1.2, 0.7, 2.5, 1.1, 0.035, 78.0, markov_config=DEFAULT_MARKOV_CONFIG
    )
    assert vector == pytest.approx(scalar)


def test_serofast_cost_cascade_is_identical_in_cea_and_bia_paths():
    co = Costs()
    n = 1_000.0
    p_sf = 0.002
    p_trepo = 0.95
    p_ux = 0.20
    cascade = screening_cascade_counts(
        n,
        p_act=0.0075,
        p_sf=p_sf,
        sens=0.98,
        spec=0.98,
        p_adeq=0.85,
        tx_completion=0.77,
        p_trepo_sf=p_trepo,
        p_ux_sf=p_ux,
        treat_fp=False,
    )
    shared = float(np.asarray(program_cost_components(cascade, co)["cost_sf_total"]))
    cea = float(
        np.asarray(
            _serofast_cost(
                n,
                p_sf,
                p_trepo,
                p_ux,
                co.rpr,
                co.sf_wu,
                co.pen,
                co.soc_work,
                fta=co.fta,
                desens=co.desens,
                jh_obs=co.jh_obs,
                followup=co.followup,
            )
        )
    )
    assert cea == pytest.approx(shared)


def test_bia_default_denominator_does_not_apply_payer_share_twice():
    enrolled = BIAPopulation(covered_lives=500_000, payer_fraction=0.40)
    catchment = BIAPopulation(
        covered_lives=500_000, payer_fraction=0.40, apply_payer_fraction=True
    )
    f_enrolled = common_population_funnel(enrolled)
    f_catchment = common_population_funnel(catchment)
    expected_all_ed = (
        500_000 * enrolled.frac_repro_female * enrolled.pregnancy_rate
        * enrolled.p_ed_visit
    )
    assert f_enrolled["n_ed_payer"] == pytest.approx(expected_all_ed)
    assert f_catchment["n_ed_payer"] == pytest.approx(expected_all_ed * 0.40)


def test_bia_outcomes_use_actual_coverage_arms_without_double_applying_gap():
    sc_b, _, _ = ges_eff()
    pop = BIAPopulation(covered_lives=100_000)
    funnel = bia_population_funnel(pop, 1, {1: 0.70}, sc_b, 0.85)
    impact = bia_annual_impact(
        funnel,
        Costs(),
        p_act=0.0075,
        p_id=0.85,
        sc_uc=sc_b,
        sens=0.98,
        spec=0.98,
        p_adeq=0.85,
        tx_eff=None,
        prop_symp=0.38,
        prop_late=None,
        p_sf=0.0015,
        p_trepo_sf=0.95,
        p_ux_sf=0.20,
        treat_fp=False,
        sc_target=0.90,
    )
    br, ur, rr = _deterministic_outcome_inputs()
    _, _, direct = arm_outcome_delta(
        cohort=funnel["n_eligible"],
        p_act=0.0075,
        p_id=0.85,
        sc_b=sc_b,
        sc_e=0.70,
        sens=0.98,
        p_adeq=0.85,
        tx_completion_override=None,
        prop_symp=0.38,
        prop_late_override=None,
        br=br,
        ur=ur,
        rr=rr,
        sc_target=0.90,
    )
    assert impact["n_cs_averted"] == pytest.approx(
        float(direct["cs_comp"] + direct["cs_uncomp"])
    )
    assert impact["n_sb_averted"] == pytest.approx(float(direct["stillbirth"]))


def test_direct_medical_maternal_costs_and_health_effects_enter_health_sector():
    base = _det_icost(**base_det_kwargs())
    with_mm = _det_icost(
        **base_det_kwargs(
            mm=__import__("syphilis_cea.societal", fromlist=["MaternalMorbidity"]).MaternalMorbidity()
        )
    )
    assert with_mm[0] < base[0]  # direct medical savings reduce HS cost
    assert with_mm[1] > base[1]  # maternal health gains count in HS effects
    assert with_mm[1] == pytest.approx(with_mm[3])


def test_long_term_nonmedical_care_changes_societal_not_health_sector_cost():
    no_ltc = _det_icost(**base_det_kwargs())
    with_ltc = _det_icost(**base_det_kwargs(ltc=LongTermCare()))
    assert with_ltc[0] == pytest.approx(no_ltc[0])
    assert with_ltc[2] < no_ltc[2]



def test_psa_cost_identities_follow_explicit_perspective_policy_and_shared_cascade():
    df, _, _, _ = run_psa(**base_psa_kwargs(N=80))
    delta = {
        key: df[f"d_{key}"].to_numpy()
        for key in (
            "preterm",
            "lbw",
            "stillbirth",
            "neonatal_death",
            "cs_comp",
            "cs_uncomp",
            "iufd_subset",
        )
    }
    cost_draws = {
        key.removeprefix("co_"): df[key].to_numpy()
        for key in df.columns
        if key.startswith("co_")
    }
    acute = np.asarray(
        outcome_savings_components(delta, cost_draws)["medical_savings"],
        dtype=float,
    )
    expected_hs = (
        df["program_cost"].to_numpy()
        - acute
        - df["mk_med_cst"].to_numpy()
        - df["mm_cost"].to_numpy()
    )
    expected_soc = (
        expected_hs
        - df["mk_ltc_cst"].to_numpy()
        - df["prod_sav"].to_numpy()
    )
    assert np.allclose(df["ic_hs"], expected_hs)
    assert np.allclose(df["ic_soc"], expected_soc)
    assert np.array_equal(df["dal_hs"].to_numpy(), df["dal_soc"].to_numpy())

def test_inclusive_long_term_care_end_age_includes_terminal_cycle():
    config = MarkovConfig(
        p_severe_cs_comp=0.0,
        p_mild_cs_comp=1.0,
        p_mild_cs_uncomp=0.0,
        cost_mild_ann=0.0,
        cost_mild_ann_sd=0.0,
        cost_severe_ann=0.0,
        cost_severe_ann_sd=0.0,
        q_progress_target=0.0,
    )
    ltc = LongTermCare(
        p_sped_mild=1.0,
        p_sped_severe=0.0,
        cost_sped_ann=100.0,
        cost_cg_mild_ann=0.0,
        cost_cg_severe_ann=0.0,
        sped_start_age=0,
        sped_end_age=0,
        caregiver_end_age=0,
    )
    result = _infant_markov_lifetime(
        np.array([1.0]),
        np.array([0.0]),
        markov_point_parameters(config),
        0.0,
        1,
        ltc=ltc,
        q_progress=0.0,
    )
    assert result.long_term_care_costs[0] == pytest.approx(100.0)


def test_linked_serofast_threshold_surface_preserves_ratio():
    kwargs = base_det_kwargs(cohort=1_000)
    prevalence = 0.01
    treatment = 0.80
    surface = nmb_surface(
        np.array([prevalence]),
        np.array([treatment]),
        p_sf=0.123,  # ignored when p_sf_ratio is supplied
        p_id=kwargs["p_id"],
        sc_b=kwargs["sc_b"],
        sc_e=kwargs["sc_e"],
        sens=kwargs["sens"],
        spec=kwargs["spec"],
        prop_symp=kwargs["prop_symp"],
        prop_late=kwargs["prop_late"],
        p_trepo_sf=kwargs["p_trepo_sf"],
        p_ux_sf=kwargs["p_ux_sf"],
        r=kwargs["r"],
        LE=kwargs["LE"],
        inc_lbw=kwargs["inc_lbw"],
        inc_mat=kwargs["inc_mat"],
        cohort=kwargs["cohort"],
        wtp=100_000,
        p_sf_ratio=0.20,
    )
    ic, dal, _, _ = _det_icost(
        **base_det_kwargs(
            cohort=1_000,
            p_act=prevalence,
            p_sf=0.20 * prevalence,
            p_adeq=treatment,
        )
    )
    assert surface[0, 0] == pytest.approx(100_000 * dal - ic)


def test_icers_do_not_replace_nonpositive_effects_with_epsilon():
    values = safe_icer(np.array([100.0, -100.0, 50.0]), np.array([-1.0, 0.0, 2.0]))
    assert np.isnan(values[0])
    assert np.isnan(values[1])
    assert values[2] == pytest.approx(25.0)


def test_quadrant_two_is_not_labeled_cost_effective_without_wtp():
    table = ce_quadrant_table(np.array([1.0]), np.array([1.0]))
    assert table.loc[1, "Quadrant"] == "More effective, more costly"
    assert "WTP" in table.loc[1, "Interpretation"]


def test_evppi_is_cross_fitted_and_bounded_by_evpi():
    rng = np.random.default_rng(5)
    n = 120
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "dal_hs": 1.0 + 0.2 * x + rng.normal(scale=0.05, size=n),
            "ic_hs": 20_000.0 - 5_000.0 * x + rng.normal(scale=1_000.0, size=n),
            "raw_x": x,
        }
    )
    evpi, values = compute_evppi(
        df, 100_000.0, "hs", {"Raw input": ["raw_x"]}, n_splits=5
    )
    assert 0.0 <= values["Raw input"] <= evpi + 1e-9


def test_zero_positive_quantity_means_remain_exact_zero_in_psa():
    from syphilis_cea.utils import draw_gamma

    rng = np.random.default_rng(99)
    assert np.array_equal(draw_gamma(rng, 50, 0.0, 5.0), np.zeros(50))
    assert np.array_equal(draw_gamma(rng, 50, 7.0, 0.0), np.full(50, 7.0))

    config = MarkovConfig(
        cost_mild_ann=0.0,
        cost_mild_ann_sd=2_500.0,
        cost_severe_ann=0.0,
        cost_severe_ann_sd=7_500.0,
    )
    draws = _draw_infant_mk(50, np.random.default_rng(100), config)
    assert (draws["cost_mild_ann"] == 0.0).all()
    assert (draws["cost_sev_ann"] == 0.0).all()


def test_static_bia_presets_apply_their_declared_payer_shares():
    from syphilis_cea.bia import BIA_SCENARIOS

    for scenario in BIA_SCENARIOS.values():
        pop = scenario["pop"]
        assert pop.apply_payer_fraction is True
        funnel = common_population_funnel(pop)
        expected = (
            pop.covered_lives
            * pop.frac_repro_female
            * pop.pregnancy_rate
            * pop.p_ed_visit
            * pop.payer_fraction
        )
        assert funnel["n_ed_payer"] == pytest.approx(expected)
