import numpy as np

from syphilis_cea.bia import BIAPopulation, run_bia_scenario
from syphilis_cea.markov import calibrate_q_progress, implied_lifetime_prog
from syphilis_cea.outcomes import _incremental_screening_outcome_delta
from syphilis_cea.parameters import Costs, ges_eff
from syphilis_cea.psa import run_psa


def test_q_progress_calibration_matches_target():
    q = calibrate_q_progress(0.20, 78)
    assert abs(implied_lifetime_prog(q, 78) - 0.20) < 1e-4


def test_outcome_delta_scales_linearly_with_incremental_screening():
    kwargs = dict(
        p_act=0.0075,
        sens=0.98,
        p_adeq=0.85,
        tx_eff=ges_eff()[1],
        prop_symp=0.38,
        prop_late=ges_eff()[2],
    )
    d1 = _incremental_screening_outcome_delta(n_screened=1_000, **kwargs)
    d2 = _incremental_screening_outcome_delta(n_screened=2_000, **kwargs)
    for key in d1:
        assert np.isclose(float(d2[key]), 2.0 * float(d1[key]))


def test_bia_scenario_smoke():
    sc_uc, tx_eff, prop_late = ges_eff()
    impact, funnel = run_bia_scenario(
        pop=BIAPopulation(),
        t_half=1.5,
        t_ninety=3.0,
        n_years=3,
        co=Costs(),
        sc_e=0.90,
        sc_uc=sc_uc,
        p_act=0.0075,
        p_id=0.85,
        sens=0.98,
        spec=0.98,
        p_adeq=0.85,
        tx_eff=tx_eff,
        prop_symp=0.38,
        prop_late=prop_late,
        p_sf=0.0015,
        p_trepo_sf=0.95,
        p_ux_sf=0.20,
        treat_fp=False,
    )
    assert len(impact) == 3
    assert len(funnel) == 3
    assert impact["n_incremental"].ge(0).all()
    assert np.isfinite(impact["net_impact"]).all()


def test_psa_smoke_and_summary_contract():
    sc_uc, _, prop_late = ges_eff()
    df, smry, comp, intr = run_psa(
        N=30,
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
        prop_late=prop_late,
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
    assert len(df) == 30
    assert {"dal_hs", "dal_soc", "ic_hs", "ic_soc", "icer_hs", "icer_soc"} <= set(df.columns)
    assert "inc_cost_hs" in smry and "d_cs_total" in smry
    assert set(comp) == set(intr)
