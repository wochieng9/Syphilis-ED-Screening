"""Streamlit entry point for the corrected modular syphilis screening CEA.

The presentation layer contains no model equations. All calculations are routed
through the importable ``syphilis_cea`` package and immutable configuration
objects.
"""

from __future__ import annotations

from dataclasses import asdict
import io

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st

from syphilis_cea.bia import (
    BIAPopulation,
    common_population_funnel,
    run_bia_scenario,
    sigmoid_ramp,
)
from syphilis_cea.config import (
    DEFAULT_MARKOV_CONFIG,
    DEFAULT_NATURAL_HISTORY_REFERENCE,
    MarkovConfig,
)
from syphilis_cea.deterministic import _det_icost, nmb_surface
from syphilis_cea.figures import (
    fig_ce_plane,
    fig_ceac,
    fig_convergence,
    fig_evpi,
    fig_evppi_bar,
    fig_markov_daly_dist,
    fig_markov_states,
    fig_nmb_surface,
    fig_prod_loss_breakdown,
    fig_tornado,
    fig_tornado_nmb,
)
from syphilis_cea.markov import calibrate_q_progress, implied_lifetime_prog
from syphilis_cea.outcomes import (
    incremental_tx_completion_rate,
)
from syphilis_cea.parameters import (
    BLS_ANNUAL_EARNINGS,
    GES_STRATA,
    MATERNAL_AGE_DIST,
    MATERNAL_WEIGHTED_EARNINGS,
    PRESETS,
    Costs,
    LongTermCare,
    ges_eff,
    joint_usual_care_treatment_completion,
)
from syphilis_cea.psa import run_psa as _run_psa
from syphilis_cea.reporting import (
    ce_quadrant_table,
    compute_evppi,
    decision_status,
    owsa_nmb_table,
    owsa_table,
)
from syphilis_cea.societal import MaternalMorbidity, ProductivityLoss
from syphilis_cea.utils import safe_icer


run_psa = st.cache_data(show_spinner=False)(_run_psa)


def fmt_money(value: float, digits: int = 0) -> str:
    if not np.isfinite(value):
        return "Undefined"
    return f"${value:,.{digits}f}"


def fmt_icer(cost: float, effect: float) -> str:
    status = decision_status(float(cost), float(effect))
    if status == "Dominant":
        return "Dominant"
    if status == "Dominated":
        return "Dominated"
    value = safe_icer(float(cost), float(effect))
    return f"${value:,.0f}/DALY" if np.isfinite(value) else status


def figure_download(fig, filename: str, label: str) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    st.download_button(label, buffer, filename, "image/png")


def summary_frame(summary: dict, perspective: str) -> pd.DataFrame:
    cost_key = "inc_cost_hs" if perspective == "hs" else "inc_cost_soc"
    daly_key = "dalys_hs" if perspective == "hs" else "dalys_soc"
    rows = []
    for label, key, currency in (
        ("Incremental cost", cost_key, True),
        ("DALYs averted", daly_key, False),
    ):
        values = summary[key]
        rows.append(
            {
                "Metric": label,
                "Mean": fmt_money(values["mean"]) if currency else f"{values['mean']:,.2f}",
                "Median": fmt_money(values["median"]) if currency else f"{values['median']:,.2f}",
                "95% CrI": (
                    f"{fmt_money(values['95% CrI lo'])} to {fmt_money(values['95% CrI hi'])}"
                    if currency
                    else f"{values['95% CrI lo']:,.2f} to {values['95% CrI hi']:,.2f}"
                ),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Syphilis ED Screening CEA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model settings")
    preset_name = st.selectbox("Scenario preset", list(PRESETS), index=2)
    preset = PRESETS[preset_name]

    def preset_value(name: str, default):
        return preset.get(name, default)

    st.subheader("Population and epidemiology")
    cohort = int(st.number_input("Eligible cohort size", 1_000, 500_000, 100_000, 1_000))
    p_act = float(
        st.slider(
            "Active syphilis prevalence",
            0.001,
            0.030,
            float(preset_value("p_act", 0.0075)),
            0.0005,
            format="%.4f",
        )
    )

    linked_serofast = st.checkbox(
        "Link serofast prevalence to 20% of active prevalence", value=True
    )
    linked_value = float(np.clip(0.20 * p_act, 0.0, 1.0 - p_act))
    if linked_serofast:
        p_sf = linked_value
        st.caption(f"Serofast prevalence: {p_sf:.4f} ({p_sf:.2%})")
    else:
        p_sf = float(
            st.slider(
                "Serofast / prior-treated prevalence",
                0.0001,
                0.0300,
                float(np.clip(linked_value, 0.0001, 0.0300)),
                0.0001,
                format="%.4f",
            )
        )

    st.subheader("ED operations")
    p_id = float(st.slider("P(pregnancy identified)", 0.50, 1.00, 0.85, 0.01))
    sc_e = float(
        st.slider(
            "Enhanced screening coverage",
            0.80,
            1.00,
            max(float(preset_value("sc_e", 0.90)), 0.80),
            0.01,
        )
    )
    sens = float(st.slider("Treponemal-screen sensitivity", 0.85, 1.00, 0.98, 0.01))
    spec = float(st.slider("Treponemal-screen specificity", 0.85, 1.00, 0.98, 0.01))

    st.subheader("Treatment cascade")
    p_adeq = float(
        st.slider(
            "P(appropriate treatment initiated | true positive)",
            0.30,
            1.00,
            float(preset_value("p_adeq", 0.85)),
            0.01,
            help=(
                "This parameter is multiplied by a separate treatment-completion "
                "probability. Interpret it as appropriate initiation/linkage, not "
                "as completed adequate treatment."
            ),
        )
    )
    override_tx = st.checkbox("Override gestational-stratum treatment completion")
    p_tx = (
        float(st.slider("Treatment completion override", 0.30, 1.00, 0.77, 0.01))
        if override_tx
        else None
    )
    p_trepo_sf = float(st.slider("P(treponemal+ | serofast)", 0.70, 1.00, 0.95, 0.01))
    p_ux_sf = float(
        st.slider("P(unnecessarily treated | serofast detected)", 0.00, 0.60, 0.20, 0.01)
    )
    treat_fp = st.checkbox("Treat seronegative false positives", value=False)

    st.subheader("Clinical structure")
    prop_symp = float(st.slider("Proportion CS complicated", 0.10, 0.70, 0.38, 0.01))
    sc_uc_eff, tx_mean_display, prop_late_display = ges_eff()
    override_prop_late = st.checkbox(
        f"Override source gestational prop_late values (weighted mean {prop_late_display:.2f})"
    )
    prop_late = (
        float(st.slider("Common prop_late override", 0.00, 1.00, prop_late_display, 0.01))
        if override_prop_late
        else None
    )
    st.caption(
        "The source file is internally ambiguous about whether prop_late denotes "
        "late IUFD or early loss. The source stratum values are retained unless overridden."
    )

    st.subheader("Infant Markov model")
    with st.expander("Sequelae, costs, and progression", expanded=False):
        p_sev_ui = float(
            st.slider(
                "P(severe sequelae | complicated CS)",
                0.00,
                0.90,
                DEFAULT_MARKOV_CONFIG.p_severe_cs_comp,
                0.01,
            )
        )
        mild_max = float(max(1.0 - p_sev_ui, 0.0))
        p_mc_ui = float(
            st.slider(
                "P(mild sequelae | complicated CS)",
                0.00,
                mild_max,
                float(min(DEFAULT_MARKOV_CONFIG.p_mild_cs_comp, mild_max)),
                0.01,
            )
        )
        p_mu_ui = float(
            st.slider(
                "P(mild sequelae | uncomplicated CS)",
                0.00,
                0.50,
                DEFAULT_MARKOV_CONFIG.p_mild_cs_uncomp,
                0.01,
            )
        )
        c_mild_ui = float(
            st.number_input(
                "Annual medical cost - mild sequelae",
                0,
                100_000,
                int(DEFAULT_MARKOV_CONFIG.cost_mild_ann),
                500,
            )
        )
        c_sev_ui = float(
            st.number_input(
                "Annual medical cost - severe sequelae",
                0,
                200_000,
                int(DEFAULT_MARKOV_CONFIG.cost_severe_ann),
                1_000,
            )
        )
        q_target = float(
            st.slider(
                "Target lifetime mild-to-severe progression",
                0.00,
                0.60,
                DEFAULT_MARKOV_CONFIG.q_progress_target,
                0.01,
            )
        )
        mu_x_mild = float(
            st.slider("Annual excess mortality - mild", 0.000, 0.020, 0.000, 0.001)
        )
        mu_x_severe = float(
            st.slider("Annual excess mortality - severe", 0.000, 0.050, 0.000, 0.001)
        )

    markov_config = DEFAULT_MARKOV_CONFIG.updated(
        p_severe_cs_comp=p_sev_ui,
        p_mild_cs_comp=p_mc_ui,
        p_mild_cs_uncomp=p_mu_ui,
        cost_mild_ann=c_mild_ui,
        cost_severe_ann=c_sev_ui,
        q_progress_target=q_target,
        mu_excess_mild=mu_x_mild,
        mu_excess_severe=mu_x_severe,
    )

    st.subheader("Long-term nonmedical care")
    use_ltc = st.checkbox("Include special education and paid caregiving", value=True)
    with st.expander("Long-term care inputs", expanded=False):
        ltc_p_sped_sev = float(st.slider("P(special education | severe)", 0.00, 1.00, 0.85, 0.05))
        ltc_p_sped_mild = float(st.slider("P(special education | mild)", 0.00, 1.00, 0.35, 0.05))
        ltc_cost_sped = float(st.number_input("Annual special-education cost", 0, 100_000, 14_000, 500))
        ltc_sped_start = int(st.number_input("Special-education start age", 0, 10, 3, 1))
        ltc_sped_end = int(st.number_input("Special-education end age (inclusive)", 0, 30, 21, 1))
        ltc_cost_cg_sev = float(st.number_input("Annual paid caregiving - severe", 0, 150_000, 32_000, 1_000))
        ltc_cost_cg_mild = float(st.number_input("Annual paid caregiving - mild", 0, 50_000, 4_500, 500))
        ltc_cg_end = int(st.number_input("Paid-caregiving end age (inclusive)", 0, 30, 18, 1))

    ltc = (
        LongTermCare(
            p_sped_severe=ltc_p_sped_sev,
            p_sped_mild=ltc_p_sped_mild,
            cost_sped_ann=ltc_cost_sped,
            cost_cg_severe_ann=ltc_cost_cg_sev,
            cost_cg_mild_ann=ltc_cost_cg_mild,
            sped_start_age=ltc_sped_start,
            sped_end_age=max(ltc_sped_end, ltc_sped_start),
            caregiver_end_age=ltc_cg_end,
        )
        if use_ltc
        else None
    )

    st.subheader("Maternal morbidity")
    use_mat_morb = st.checkbox("Include maternal morbidity", value=True)
    with st.expander("Maternal morbidity inputs", expanded=False):
        mm_p_cardio = float(st.slider("P(cardiovascular event | untreated late latent)", 0.00, 0.20, 0.05, 0.01))
        mm_p_neuro = float(st.slider("P(neurosyphilis | untreated late latent)", 0.00, 0.30, 0.10, 0.01))
        mm_p_hosp = float(st.slider("P(pregnancy hospitalization | active infection)", 0.00, 0.40, 0.12, 0.01))
        mm_cost_cardio = float(st.number_input("Annual cardiovascular treatment cost", 0, 100_000, 18_500, 500))
        mm_cost_neuro = float(st.number_input("Annual neurosyphilis treatment cost", 0, 150_000, 32_000, 1_000))
        mm_cost_hosp = float(st.number_input("Pregnancy hospitalization cost", 0, 30_000, 4_200, 200))

    mm = (
        MaternalMorbidity(
            p_cardio=mm_p_cardio,
            p_neuro=mm_p_neuro,
            p_hosp=mm_p_hosp,
            cost_cardio=mm_cost_cardio,
            cost_neuro=mm_cost_neuro,
            cost_hosp=mm_cost_hosp,
        )
        if use_mat_morb
        else None
    )

    st.subheader("Productivity")
    use_prod_loss = st.checkbox("Include productivity losses", value=True)
    use_friction = st.checkbox("Use 90-day friction-cost cap", value=False)
    with st.expander("Productivity inputs", expanded=False):
        pl_bereavement_days = float(st.slider("Lost workdays per bereavement", 0, 120, 30, 1))
        pl_wage_mild = float(st.slider("Wage penalty - mild", 0.00, 0.50, 0.10, 0.01))
        pl_wage_severe = float(st.slider("Wage penalty - severe", 0.00, 0.80, 0.40, 0.01))
        pl_caregiver_hrs = float(st.slider("Additional caregiver hours/week", 0, 60, 10, 1))
        pl_caregiver_wage_frac = float(st.slider("Caregiver wage fraction", 0.00, 1.00, 0.50, 0.05))

    pl = (
        ProductivityLoss(
            bereavement_days=pl_bereavement_days,
            wage_penalty_mild=pl_wage_mild,
            wage_penalty_severe=pl_wage_severe,
            caregiver_hrs_wk=pl_caregiver_hrs,
            caregiver_wage_frac=pl_caregiver_wage_frac,
            friction_period_days=90.0 if use_friction else 0.0,
        )
        if use_prod_loss
        else None
    )

    st.subheader("DALYs and PSA")
    r_disc = float(st.number_input("Discount rate", 0.0, 0.08, 0.035, 0.005, format="%.3f"))
    LE = float(st.number_input("Life expectancy at birth", 60.0, 90.0, 78.0, 1.0))
    inc_lbw = st.checkbox("Include LBW YLD", value=True)
    inc_mat = st.checkbox("Include maternal grief YLD", value=True)
    inc_sb_yll = st.checkbox("Include stillbirth YLL", value=True)
    inc_cs_yll = st.checkbox("Include CS excess-mortality YLL", value=True)
    inc_misc_yld = st.checkbox("Include miscarriage grief YLD", value=True)
    inc_mat_hosp_yld = st.checkbox("Include maternal hospitalization YLD", value=True)
    inc_preterm_yld = st.checkbox("Include preterm infant YLD", value=True)
    vsl = float(st.number_input("VSL reference value", 0, 30_000_000, 13_700_000, 500_000))
    N_iter = int(st.number_input("PSA iterations", 500, 100_000, 10_000, 500))
    seed = int(st.number_input("Random seed", 0, 999_999, 2025, 1))
    wtp_max = int(st.number_input("Maximum WTP shown", 50_000, 500_000, 200_000, 10_000))

    st.subheader("Budget impact")
    bia_covered = float(st.number_input("Covered lives", 10_000, 10_000_000, 500_000, 50_000))
    bia_frac_rf = float(st.slider("Reproductive-age female fraction", 0.05, 0.25, 0.135, 0.005))
    bia_preg_rate = float(st.slider("Annual pregnancy rate", 0.04, 0.15, 0.085, 0.005))
    bia_p_ed = float(st.slider("P(ED visit during pregnancy)", 0.20, 0.70, 0.45, 0.05))
    bia_p_unscreened = float(st.slider("P(not previously screened | ED)", 0.10, 0.80, 0.35, 0.05))
    apply_payer_fraction = st.checkbox(
        "Covered lives are a broad catchment; apply payer share",
        value=False,
        help=(
            "Leave this off when Covered lives is plan enrollment. Turn it on "
            "only when Covered lives is a broader catchment population."
        ),
    )
    bia_payer_fraction = float(
        st.slider(
            "Payer share of ED volume",
            0.05,
            1.00,
            0.40,
            0.05,
            disabled=not apply_payer_fraction,
        )
    )
    bia_years = int(st.slider("BIA horizon", 1, 10, 5, 1))
    half_max = max(float(bia_years) - 0.5, 0.5)
    bia_t_half = float(st.slider("Year at 50% of coverage gain", 0.5, half_max, min(1.5, half_max), 0.5))
    ninety_min = bia_t_half + 0.5
    ninety_max = max(float(bia_years) + 2.0, ninety_min)
    bia_t_ninety = float(
        st.slider("Year at 90% of coverage gain", ninety_min, ninety_max, min(bia_t_half + 1.5, ninety_max), 0.5)
    )


# ---------------------------------------------------------------------------
# Run the model
# ---------------------------------------------------------------------------
q_progress = calibrate_q_progress(
    markov_config.q_progress_target,
    max(int(LE), 1),
    mu_excess_mild=markov_config.mu_excess_mild,
)
implied_progression = implied_lifetime_prog(
    q_progress, max(int(LE), 1), markov_config.mu_excess_mild
)

with st.spinner("Running probabilistic sensitivity analysis..."):
    df_psa, smry, comp_means, intr_means = run_psa(
        N=N_iter,
        seed=seed,
        cohort=cohort,
        p_act=p_act,
        p_sf=p_sf,
        p_id=p_id,
        sc_b=sc_uc_eff,
        sc_e=sc_e,
        sens=sens,
        spec=spec,
        p_adeq=p_adeq,
        p_tx_override=p_tx,
        p_trepo_sf=p_trepo_sf,
        p_ux_sf=p_ux_sf,
        prop_symp=prop_symp,
        prop_late=prop_late,
        r=r_disc,
        LE=LE,
        inc_lbw=inc_lbw,
        inc_mat=inc_mat,
        inc_sb_yll=inc_sb_yll,
        inc_cs_yll=inc_cs_yll,
        inc_misc_yld=inc_misc_yld,
        inc_mat_hosp_yld=inc_mat_hosp_yld,
        inc_preterm_yld=inc_preterm_yld,
        treat_fp=treat_fp,
        vsl=vsl,
        use_mat_morb=use_mat_morb,
        use_prod_loss=use_prod_loss,
        use_friction=use_friction,
        mm_p_cardio=mm_p_cardio,
        mm_p_neuro=mm_p_neuro,
        mm_p_hosp=mm_p_hosp,
        mm_cost_cardio=mm_cost_cardio,
        mm_cost_neuro=mm_cost_neuro,
        mm_cost_hosp=mm_cost_hosp,
        pl_bereavement_days=pl_bereavement_days,
        pl_wage_mild=pl_wage_mild,
        pl_wage_severe=pl_wage_severe,
        pl_caregiver_hrs=pl_caregiver_hrs,
        pl_caregiver_wage_frac=pl_caregiver_wage_frac,
        mk_p_sev=markov_config.p_severe_cs_comp,
        mk_p_mc=markov_config.p_mild_cs_comp,
        mk_p_mu=markov_config.p_mild_cs_uncomp,
        mk_c_mild=markov_config.cost_mild_ann,
        mk_c_sev=markov_config.cost_severe_ann,
        mk_q_target=markov_config.q_progress_target,
        mk_mu_x_mild=markov_config.mu_excess_mild,
        mk_mu_x_sev=markov_config.mu_excess_severe,
        use_ltc=use_ltc,
        ltc_p_sped_sev=ltc_p_sped_sev,
        ltc_p_sped_mid=ltc_p_sped_mild,
        ltc_cost_sped=ltc_cost_sped,
        ltc_cost_cg_sv=ltc_cost_cg_sev,
        ltc_cost_cg_ml=ltc_cost_cg_mild,
        ltc_sped_start=ltc_sped_start,
        ltc_sped_end=max(ltc_sped_end, ltc_sped_start),
        ltc_cg_end=ltc_cg_end,
    )

mean_cost_hs = smry["inc_cost_hs"]["mean"]
mean_cost_soc = smry["inc_cost_soc"]["mean"]
mean_daly_hs = smry["dalys_hs"]["mean"]
mean_daly_soc = smry["dalys_soc"]["mean"]

st.title("ED universal syphilis screening")
st.caption(
    f"Usual-care screening {sc_uc_eff:.1%} to enhanced {sc_e:.1%} | "
    f"joint incremental treatment completion {smry['tx_completion_incremental']:.1%} | "
    f"q_progress {q_progress:.5f} (implied lifetime progression {implied_progression:.1%})"
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Health-sector result", fmt_icer(mean_cost_hs, mean_daly_hs))
k2.metric("Societal result", fmt_icer(mean_cost_soc, mean_daly_soc))
k3.metric("Mean HS incremental cost", fmt_money(mean_cost_hs))
k4.metric("Mean societal cost", fmt_money(mean_cost_soc))
k5.metric("Mean DALYs averted", f"{mean_daly_hs:,.1f}")
k6.metric("P(dominant, HS)", f"{smry['p_dominant_hs']:.1%}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_cea, tab_soc, tab_bia, tab_threshold, tab_markov, tab_assumptions = st.tabs(
    [
        "Standard CEA",
        "Societal",
        "Budget impact",
        "Thresholds",
        "Infant Markov",
        "Assumptions and corrections",
    ]
)

with tab_cea:
    st.subheader("Health-sector cost-effectiveness")
    st.info(
        "Health-sector results include all modeled health effects and direct medical "
        "costs: screening/treatment, acute outcome costs, infant medical sequelae care, "
        "and maternal direct medical care. Education, paid caregiving, and productivity "
        "are valued only in the societal result."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = fig_ce_plane(df_psa["dal_hs"].to_numpy(), df_psa["ic_hs"].to_numpy(), "CE plane - health sector")
        st.pyplot(fig, width="stretch")
        figure_download(fig, "ce_plane_health_sector.png", "Download CE plane")
    with col2:
        fig = fig_convergence(df_psa["dal_hs"].to_numpy(), df_psa["ic_hs"].to_numpy())
        st.pyplot(fig, width="stretch")

    fig = fig_ceac(
        df_psa["dal_hs"].to_numpy(),
        df_psa["ic_hs"].to_numpy(),
        df_psa["dal_soc"].to_numpy(),
        df_psa["ic_soc"].to_numpy(),
        wtp_max=wtp_max,
    )
    st.pyplot(fig, width="stretch")
    figure_download(fig, "ceac.png", "Download CEAC")

    left, right = st.columns([3, 2])
    with left:
        fig = fig_evpi(df_psa["dal_hs"].to_numpy(), df_psa["ic_hs"].to_numpy(), wtp_max)
        st.pyplot(fig, width="stretch")
    with right:
        st.dataframe(
            ce_quadrant_table(df_psa["dal_hs"].to_numpy(), df_psa["ic_hs"].to_numpy()),
            hide_index=True,
            width="stretch",
        )

    st.subheader("PSA summary")
    st.dataframe(summary_frame(smry, "hs"), hide_index=True, width="stretch")

    st.subheader("Clinical outcomes averted")
    outcome_rows = []
    for label, key in (
        ("Congenital syphilis - total", "d_cs_total"),
        ("Complicated congenital syphilis", "d_cs_comp"),
        ("Uncomplicated congenital syphilis", "d_cs_uncomp"),
        ("Stillbirth", "d_stillbirth"),
        ("Neonatal death", "d_neonatal_death"),
        ("Preterm birth", "d_preterm"),
        ("Low birth weight", "d_lbw"),
        ("Miscarriage", "d_miscarriage"),
    ):
        values = smry[key]
        outcome_rows.append(
            {
                "Outcome": label,
                "Mean": values["mean"],
                "95% CrI low": values["95% CrI lo"],
                "95% CrI high": values["95% CrI hi"],
            }
        )
    st.dataframe(pd.DataFrame(outcome_rows), hide_index=True, width="stretch")

    st.subheader("One-way sensitivity analysis")
    base_det = dict(
        p_act=p_act,
        p_sf=p_sf,
        p_id=p_id,
        sc_b=sc_uc_eff,
        sc_e=sc_e,
        sens=sens,
        spec=spec,
        p_adeq=p_adeq,
        prop_symp=prop_symp,
        prop_late=prop_late,
        p_trepo_sf=p_trepo_sf,
        p_ux_sf=p_ux_sf,
        r=r_disc,
        LE=LE,
        inc_lbw=inc_lbw,
        inc_mat=inc_mat,
        cohort=cohort,
        mm=mm,
        pl=pl,
        ltc=ltc,
        inc_sb_yll=inc_sb_yll,
        inc_cs_yll=inc_cs_yll,
        inc_misc_yld=inc_misc_yld,
        inc_mat_hosp_yld=inc_mat_hosp_yld,
        inc_preterm_yld=inc_preterm_yld,
        tx_eff_override=p_tx,
        treat_fp=treat_fp,
        markov_config=markov_config,
    )
    sf_ratio = p_sf / max(p_act, 1e-12)
    prevalence_low = {"p_act": 0.005}
    prevalence_high = {"p_act": 0.030}
    if linked_serofast:
        prevalence_low["p_sf"] = 0.005 * sf_ratio
        prevalence_high["p_sf"] = 0.030 * sf_ratio
    ranges = {
        "Active prevalence": (prevalence_low, prevalence_high),
        "Treatment initiation": ({"p_adeq": 0.50}, {"p_adeq": 0.95}),
        "Pregnancy identification": ({"p_id": 0.65}, {"p_id": 0.98}),
        "Enhanced coverage": (
            {"sc_e": max(max(v["p_uc"] for v in GES_STRATA.values()), sc_uc_eff)},
            {"sc_e": 0.98},
        ),
        "Sensitivity": ({"sens": 0.90}, {"sens": 1.00}),
        "Discount rate": ({"r": 0.00}, {"r": 0.05}),
        "CS complicated fraction": ({"prop_symp": 0.20}, {"prop_symp": 0.60}),
        "Unnecessary serofast treatment": ({"p_ux_sf": 0.00}, {"p_ux_sf": 0.50}),
    }
    owsa = owsa_table(base_det, ranges)
    base_icer = float(owsa["Base ICER"].iloc[0])
    fig = fig_tornado(owsa, base_icer)
    st.pyplot(fig, width="stretch")
    st.dataframe(owsa, hide_index=True, width="stretch")

    nmb_wtp = int(
        st.selectbox(
            "WTP for NMB OWSA",
            [50_000, 100_000, 150_000, 200_000],
            index=1,
            format_func=lambda x: f"${x/1000:.0f}K/DALY",
        )
    )
    owsa_nmb = owsa_nmb_table(base_det, ranges, nmb_wtp, "hs")
    base_nmb = float(owsa_nmb["Base NMB"].iloc[0])
    fig = fig_tornado_nmb(owsa_nmb, base_nmb, nmb_wtp, "hs")
    st.pyplot(fig, width="stretch")

    st.subheader("Expected value of partial perfect information")
    st.caption("Cross-fitted polynomial regression; groups contain raw uncertain inputs only.")
    evppi_wtp = int(
        st.selectbox(
            "WTP for EVPPI",
            [50_000, 100_000, 150_000, 200_000],
            index=1,
            format_func=lambda x: f"${x/1000:.0f}K/DALY",
        )
    )
    evppi_perspective = st.radio("EVPPI perspective", ["Health sector", "Societal"], horizontal=True)
    groups = {
        "Treatment relative risks": [c for c in df_psa if c.startswith("rr_")],
        "Untreated outcome risks": [c for c in df_psa if c.startswith("ur_")],
        "Background risks": [c for c in df_psa if c.startswith("br_")],
        "Direct cost inputs": [c for c in df_psa if c.startswith("co_")],
        "Disability weights": [c for c in df_psa if c.startswith("dw_")],
        "Infant Markov inputs": [c for c in df_psa if c.startswith("mkp_")],
        "Maternal morbidity inputs": [c for c in df_psa if c.startswith("mmp_")],
        "Productivity inputs": [c for c in df_psa if c.startswith("plp_")],
        "Long-term care inputs": [c for c in df_psa if c.startswith("ltcp_")],
        "Serofast inputs": ["sf_p_sf", "sf_p_trepo_sf", "sf_p_ux_sf"],
    }
    if st.button("Compute EVPPI"):
        evpi, evppi = compute_evppi(
            df_psa,
            evppi_wtp,
            "hs" if evppi_perspective == "Health sector" else "soc",
            groups,
        )
        fig = fig_evppi_bar(
            evppi,
            evpi,
            evppi_wtp,
            "hs" if evppi_perspective == "Health sector" else "soc",
        )
        st.pyplot(fig, width="stretch")
        st.dataframe(
            pd.DataFrame(
                {
                    "Group": list(evppi),
                    "EVPPI": list(evppi.values()),
                    "% of EVPI": [v / max(evpi, 1e-12) for v in evppi.values()],
                }
            ).sort_values("EVPPI", ascending=False),
            hide_index=True,
            width="stretch",
        )

    st.download_button(
        "Download PSA iterations (CSV)",
        df_psa.to_csv(index=False).encode(),
        "syphilis_cea_psa.csv",
        "text/csv",
    )

with tab_soc:
    st.subheader("Societal perspective")
    st.info(
        "The societal result uses the same modeled health effects as the health-sector "
        "result and additionally values special education, paid caregiving, and "
        "productivity losses. Maternal direct medical costs and health effects are "
        "already included in the health-sector result."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = fig_ce_plane(df_psa["dal_soc"].to_numpy(), df_psa["ic_soc"].to_numpy(), "CE plane - societal")
        st.pyplot(fig, width="stretch")
    with col2:
        st.dataframe(summary_frame(smry, "soc"), hide_index=True, width="stretch")
        st.metric("Mean productivity saving", fmt_money(smry["prod_sav"]["mean"]))
        st.metric(
            "Mean non-health long-term-care saving (societal only)",
            fmt_money(smry["mk_ltc_cst"]["mean"]),
        )
        st.metric("Mean maternal direct medical saving", fmt_money(df_psa["mm_cost"].mean()))

    if use_prod_loss:
        fig = fig_prod_loss_breakdown(df_psa)
        st.pyplot(fig, width="stretch")

    st.subheader("VSL reference analysis")
    st.warning("This is a separate benefit-cost framework and is not included in either ICER.")
    vsl_values = smry["vsl_nb"]
    st.metric("Mean VSL net benefit", fmt_money(vsl_values["mean"]))
    st.caption(
        f"95% CrI: {fmt_money(vsl_values['95% CrI lo'])} to {fmt_money(vsl_values['95% CrI hi'])}"
    )

with tab_bia:
    st.subheader("Budget impact")
    population = BIAPopulation(
        covered_lives=bia_covered,
        frac_repro_female=bia_frac_rf,
        pregnancy_rate=bia_preg_rate,
        p_ed_visit=bia_p_ed,
        p_unscreened=bia_p_unscreened,
        payer_fraction=bia_payer_fraction,
        apply_payer_fraction=apply_payer_fraction,
    )
    funnel_base = common_population_funnel(population)
    st.caption(
        "Payer share is applied once when the catchment checkbox is selected; otherwise "
        "covered lives is treated as enrolled membership. "
        f"Annual eligible patients: {funnel_base['n_eligible']:,.1f}."
    )
    impact, funnel = run_bia_scenario(
        pop=population,
        t_half=bia_t_half,
        t_ninety=bia_t_ninety,
        n_years=bia_years,
        co=Costs(),
        sc_e=sc_e,
        sc_uc=sc_uc_eff,
        p_act=p_act,
        p_id=p_id,
        sens=sens,
        spec=spec,
        p_adeq=p_adeq,
        tx_eff=p_tx,
        prop_symp=prop_symp,
        prop_late=prop_late,
        p_sf=p_sf,
        p_trepo_sf=p_trepo_sf,
        p_ux_sf=p_ux_sf,
        treat_fp=treat_fp,
    )
    cumulative_net = float(impact["cumulative_net"].iloc[-1])
    cumulative_program = float(impact["program_cost"].sum())
    cumulative_savings = float(impact["medical_savings"].sum())
    member_denominator = bia_covered * (
        bia_payer_fraction if apply_payer_fraction else 1.0
    )
    pmpm = cumulative_net / max(member_denominator * 12 * bia_years, 1.0)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Program cost", fmt_money(cumulative_program))
    b2.metric("Acute medical savings", fmt_money(cumulative_savings))
    b3.metric("Net budget impact", fmt_money(cumulative_net))
    b4.metric("PMPM", fmt_money(pmpm, 4))

    ramp = sigmoid_ramp(bia_t_half, bia_t_ninety, bia_years, sc_uc_eff, sc_e)
    ramp_df = pd.DataFrame({"Year": list(ramp), "Coverage": list(ramp.values())})
    st.line_chart(ramp_df.set_index("Year"))

    display = impact[
        [
            "year",
            "eff_coverage",
            "tx_completion_incremental",
            "n_incremental",
            "n_tp_treated",
            "n_sf_treated",
            "n_cs_averted",
            "program_cost",
            "medical_savings",
            "net_impact",
            "cumulative_net",
        ]
    ].copy()
    st.dataframe(display, hide_index=True, width="stretch")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(impact["year"] - 0.2, impact["program_cost"] / 1e6, 0.4, label="Program cost")
    ax.bar(impact["year"] + 0.2, impact["medical_savings"] / 1e6, 0.4, label="Medical savings")
    ax.plot(impact["year"], impact["cumulative_net"] / 1e6, "o--", label="Cumulative net")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.2f}M"))
    ax.set_xlabel("Year")
    ax.set_title("Annual budget impact")
    ax.grid(alpha=0.15)
    ax.legend()
    st.pyplot(fig, width="stretch")

    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        impact.to_excel(writer, sheet_name="Impact", index=False)
        funnel.to_excel(writer, sheet_name="Population funnel", index=False)
    st.download_button(
        "Download BIA workbook",
        workbook.getvalue(),
        "syphilis_bia.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab_threshold:
    st.subheader("Net monetary benefit thresholds")
    t1, t2, t3 = st.columns(3)
    with t1:
        threshold_wtp = int(
            st.selectbox(
                "WTP threshold",
                [50_000, 100_000, 150_000, 200_000],
                index=1,
                format_func=lambda x: f"${x/1000:.0f}K/DALY",
                key="threshold_wtp",
            )
        )
    with t2:
        prev_max = float(st.slider("Maximum prevalence shown", 0.01, 0.06, 0.04, 0.005))
    with t3:
        threshold_societal = st.checkbox("Societal NMB", value=False)

    prevalence_grid = np.arange(0.001, prev_max + 0.0001, 0.001)
    treatment_grid = np.arange(0.40, 1.001, 0.04)
    surface = nmb_surface(
        prevalence_grid,
        treatment_grid,
        p_sf,
        p_id,
        sc_uc_eff,
        sc_e,
        sens,
        spec,
        prop_symp,
        prop_late,
        p_trepo_sf,
        p_ux_sf,
        r_disc,
        LE,
        inc_lbw,
        inc_mat,
        cohort,
        threshold_wtp,
        mm=mm,
        pl=pl,
        ltc=ltc,
        societal=threshold_societal,
        inc_sb_yll=inc_sb_yll,
        inc_cs_yll=inc_cs_yll,
        inc_misc_yld=inc_misc_yld,
        inc_mat_hosp_yld=inc_mat_hosp_yld,
        inc_preterm_yld=inc_preterm_yld,
        tx_eff_override=p_tx,
        treat_fp=treat_fp,
        markov_config=markov_config,
        p_sf_ratio=(sf_ratio if linked_serofast else None),
    )
    fig = fig_nmb_surface(prevalence_grid, treatment_grid, surface, threshold_wtp)
    st.pyplot(fig, width="stretch")

    fixed_nmb = []
    fixed_icer = []
    for prevalence in prevalence_grid:
        sf_current = sf_ratio * prevalence if linked_serofast else p_sf
        ic_hs, dal_hs, ic_soc, dal_soc = _det_icost(
            **{
                **base_det,
                "p_act": float(prevalence),
                "p_sf": float(sf_current),
            }
        )
        cost = ic_soc if threshold_societal else ic_hs
        effect = dal_soc if threshold_societal else dal_hs
        fixed_nmb.append(threshold_wtp * effect - cost)
        fixed_icer.append(safe_icer(cost, effect))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(prevalence_grid * 100, fixed_nmb)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Active prevalence (%)")
    ax.set_ylabel("NMB")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x/1e6:,.1f}M"))
    ax.set_title("NMB at current treatment initiation")
    ax.grid(alpha=0.15)
    st.pyplot(fig, width="stretch")

    nonnegative = np.flatnonzero(np.asarray(fixed_nmb) >= 0.0)
    if len(nonnegative):
        st.success(f"First modeled nonnegative NMB: {prevalence_grid[nonnegative[0]]:.2%} prevalence")
    else:
        st.warning("NMB remains negative across the modeled prevalence range.")

with tab_markov:
    st.subheader("Infant lifetime Markov model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(healthy | complicated)", f"{markov_config.p_healthy_cs_comp:.1%}")
    m2.metric("P(mild | complicated)", f"{markov_config.p_mild_cs_comp:.1%}")
    m3.metric("P(severe | complicated)", f"{markov_config.p_severe_cs_comp:.1%}")
    m4.metric("Calibrated annual progression", f"{q_progress:.5f}")
    fig = fig_markov_states(r_disc, LE, markov_config, q_progress)
    st.pyplot(fig, width="stretch")
    fig = fig_markov_daly_dist(df_psa)
    st.pyplot(fig, width="stretch")

    st.dataframe(
        pd.DataFrame(
            [
                {"Component": "Markov YLD", **smry["dalys_markov_yld"]},
                {"Component": "Markov excess-mortality YLL", **smry["dalys_markov_yll"]},
                {"Component": "Markov medical cost saving", **smry["mk_med_cst"]},
                {"Component": "Long-term nonmedical cost saving", **smry["mk_ltc_cst"]},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

with tab_assumptions:
    st.subheader("Implemented corrections")
    st.markdown(
        """
- Exact 0 and 1 probability settings are deterministic point masses rather than malformed beta draws.
- PSA parameter groups use independent named random streams; disabled modules cannot perturb active draws.
- Screening coverage and treatment completion are aggregated jointly by gestational stratum.
- Complicated-CS initial states are constrained to a probability simplex.
- Markov parameters are immutable and explicitly passed; the deprecated global snapshot is never read by calculations.
- Friction-cost caps are identical in deterministic and PSA productivity paths.
- CEA and BIA share one serofast/treatment-adjacent cost cascade.
- EVPPI uses raw inputs only and out-of-fold predictions.
- Linked serofast prevalence remains linked throughout threshold analysis.
- ICERs are not calculated for nonpositive incremental effects; NMB and dominance classifications are used instead.
- Long-term-care end ages are inclusive.
- BIA denominator semantics are explicit: covered lives defaults to enrolled membership; payer share is optional for a broader catchment.
- Maternal direct medical costs and health effects enter the health-sector result; education, paid caregiving, and productivity enter societal costs.
        """
    )

    st.subheader("Perspective definitions")
    st.markdown(
        """
**Health sector:** screening/treatment costs, acute medical offsets, infant medical sequelae costs, maternal direct medical costs, and all modeled health effects.

**Societal:** the same health effects and direct medical costs, plus special education, paid caregiving, and productivity effects. VSL remains a separate reference benefit-cost analysis.
        """
    )

    st.subheader("Source-defined uncertainties retained")
    st.warning(
        "The code cannot resolve several model-definition questions from the source alone. "
        "They are retained and made explicit rather than silently changed."
    )
    st.markdown(
        """
- `prop_late` is retained exactly as supplied by gestational stratum. The source describes the values as early-loss fractions in one place and labels them as P(IUFD >=28 weeks | stillbirth) elsewhere.
- Preterm/LBW and other marginal endpoints retain the source overlap adjustment; a fully mutually exclusive outcome tree requires additional clinical specification.
- The 0.92 uncomplicated-CS observation fraction is retained despite tension with the accompanying 48-72-hour narrative.
- `p_adeq` must be interpreted as appropriate treatment initiation/linkage because the model separately multiplies by treatment completion.
- The maternal neurosyphilis probability is implemented conditional on untreated late latency, matching the equation rather than the contradictory former UI label.
        """
    )

    st.subheader("Gestational strata")
    strata_table = pd.DataFrame(GES_STRATA).T.rename(
        columns={
            "w": "Cohort weight",
            "p_uc": "Usual-care coverage",
            "p_tx": "Treatment completion",
            "prop_late": "Source prop_late",
        }
    )
    st.dataframe(strata_table, width="stretch")
    st.caption(
        f"Separate weighted means (display only): coverage {sc_uc_eff:.3f}, completion {tx_mean_display:.3f}. "
        f"Joint usual-care treated reach: {joint_usual_care_treatment_completion():.3f}. "
        f"Joint incremental completion at current coverage: "
        f"{incremental_tx_completion_rate(sc_uc_eff, sc_e, p_tx):.3f}."
    )

    st.subheader("Natural-history reference - diagnostic only")
    st.json(asdict(DEFAULT_NATURAL_HISTORY_REFERENCE))
    st.caption("These values do not enter the computational model.")

    st.subheader("Markov configuration")
    st.json(markov_config.as_legacy_dict(q_progress=q_progress))

    with st.expander("Earnings inputs"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Age band": list(BLS_ANNUAL_EARNINGS),
                    "Annual earnings": list(BLS_ANNUAL_EARNINGS.values()),
                    "Maternal age weight": [MATERNAL_AGE_DIST[key] for key in BLS_ANNUAL_EARNINGS],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Weighted annual earnings: {fmt_money(MATERNAL_WEIGHTED_EARNINGS)}")
