"""Model constants, parameter tables, and parameter dataclasses."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import DEFAULT_MARKOV_CONFIG
from .utils import CPI, std2

LIFE_TABLE_QX: List[float] = [
    0.00586,  0.00042,  0.000272, 0.000225, 0.000184, 0.000157, 0.00014,  0.000128,
    0.000122, 0.000123, 0.000129, 0.000138, 0.000164, 0.00022,  0.00031,  0.000446,
    0.000637, 0.000868, 0.0011,   0.00127,  0.001373, 0.001488, 0.001605, 0.001714,
    0.001835, 0.001963, 0.002082, 0.002202, 0.00233,  0.002457, 0.002574, 0.002683,
    0.002787, 0.002881, 0.002974, 0.003074, 0.003175, 0.003295, 0.003444, 0.003608,
    0.00378,  0.003958, 0.004144, 0.004337, 0.00454,  0.004774, 0.005064, 0.005399,
    0.005796, 0.006214, 0.006671, 0.007167, 0.007736, 0.008351, 0.009035, 0.00977,
    0.010567, 0.011398, 0.012291, 0.013224, 0.014267, 0.015353, 0.016484, 0.017617,
    0.018759, 0.019914, 0.021104, 0.022423, 0.023847, 0.025357, 0.02705,  0.02897,
    0.031188, 0.033754, 0.036747, 0.040563, 0.044308, 0.048498, 0.053229, 0.058778,
    0.064617, 0.070947, 0.077834, 0.085686, 0.094809, 0.10509,  0.116592, 0.129306,
    0.142732, 0.157638, 0.174458, 0.193027, 0.21293,  0.232657, 0.251826, 0.270943,
    0.289756, 0.307998, 0.325393, 0.341662, 0.358746, 0.376683, 0.395517, 0.415293,
    0.436058, 0.45786,  0.480753, 0.504791, 0.530031, 0.556532,
]

def lt_qx(age: int) -> float:
    if age < 0: return 0.0
    if age < len(LIFE_TABLE_QX): return LIFE_TABLE_QX[age]
    return 0.6

BLS_ANNUAL_EARNINGS: Dict[str, float] = {
    "16–24": 33_280,   # $640/wk × 52
    "25–34": 57_200,   # $1,100/wk
    "35–44": 65_520,   # $1,260/wk
    "45–54": 64_480,   # $1,240/wk
    "55–64": 60_320,   # $1,160/wk
    "65+":   49_400,   # $950/wk
}

MATERNAL_AGE_DIST: Dict[str, float] = {
    "16–24": 0.28,
    "25–34": 0.52,
    "35–44": 0.18,
    "45–54": 0.02,
    "55–64": 0.00,
    "65+":   0.00,
}

MATERNAL_WEIGHTED_EARNINGS: float = sum(
    MATERNAL_AGE_DIST[b] * BLS_ANNUAL_EARNINGS[b] for b in MATERNAL_AGE_DIST
)

# Gestational-age strata retained exactly as defined in the source model.
# ``prop_late`` remains a model-defined stratum value because the source does
# not establish whether it is a late-IUFD fraction or an early-loss fraction.
# The corrected engine uses these values jointly with screening and treatment
# completion instead of multiplying separately weighted averages.
GES_STRATA = {
    "<14w":   dict(w=0.20, p_uc=0.08, p_tx=0.95, prop_late=0.80),
    "14-27w": dict(w=0.35, p_uc=0.35, p_tx=0.88, prop_late=0.55),
    "28-36w": dict(w=0.30, p_uc=0.58, p_tx=0.72, prop_late=0.25),
    ">=37w":  dict(w=0.15, p_uc=0.78, p_tx=0.38, prop_late=0.05),
}


def ges_eff(strata: dict = None) -> Tuple[float, float, float]:
    """Return descriptive weighted means for UI display.

    Outcome calculations use the full joint stratum distribution and therefore
    do not multiply these separate means.
    """

    s = strata or GES_STRATA
    weights = sum(float(v["w"]) for v in s.values())
    if not np.isclose(weights, 1.0):
        raise ValueError(f"Gestational-stratum weights must sum to 1; got {weights}")
    return (
        sum(v["w"] * v["p_uc"] for v in s.values()),
        sum(v["w"] * v["p_tx"] for v in s.values()),
        sum(v["w"] * v["prop_late"] for v in s.values()),
    )


def joint_usual_care_treatment_completion(strata: dict = None) -> float:
    """Return E[p_usual-care screening * p_treatment completion]."""
    s = strata or GES_STRATA
    return float(sum(v["w"] * v["p_uc"] * v["p_tx"] for v in s.values()))


# Deprecated compatibility snapshot. The computational model never reads or
# mutates this object; use MarkovConfig/DEFAULT_MARKOV_CONFIG instead.
INFANT_MK = DEFAULT_MARKOV_CONFIG.as_legacy_dict()

BASE_BETA = {
    "preterm":        dict(a=1040, b=8960),
    "lbw":            dict(a=850,  b=9150),
    "stillbirth":     dict(a=55,   b=9945),
    "neonatal_death": dict(a=36,   b=9964),
    "miscarriage":    dict(a=1500, b=8500),
}

UNT_ABS = dict(
    preterm=0.232, lbw=0.234, stillbirth=0.264,
    miscarriage=0.149, neonatal_death=0.162, cs_any=0.360,
)

TX_RR = {
    "preterm":        dict(rr=0.48, lo=0.39, hi=0.58),
    "lbw":            dict(rr=0.50, lo=0.42, hi=0.59),
    "stillbirth":     dict(rr=0.21, lo=0.10, hi=0.35),
    "neonatal_death": dict(rr=0.20, lo=0.13, hi=0.32),
    "cs_any":         dict(rr=0.03, lo=0.02, hi=0.07),
    # No syphilis-specific miscarriage/early-fetal-loss RR was identified.
    # Conservative placeholder: assumed no stronger than the weakest measured
    # adverse-outcome treatment effect in an independent syphilis-in-pregnancy
    # cohort; the 95% interval upper bound reaches 1.0 (i.e. allows "no effect").
    # This understates program benefit. UPDATE if a syphilis-specific early-loss
    # relative risk becomes available.
    "miscarriage":    dict(rr=0.90, lo=0.80, hi=1.00),
}

DW_P = {
    "lbw":     dict(m=0.106, lo=0.035, hi=0.159, dur=0.25),
    # Acute infant morbidity. These are intentionally conservative defaults and
    # are exposed through the DALY component table so they can be audited.
    "preterm": dict(m=0.049, lo=0.020, hi=0.090, dur=0.25),
    # Maternal grief / acute maternal morbidity components.
    "mat_sb":  dict(m=0.740, lo=0.600, hi=0.800, dur=1.00),
    "mat_nnd": dict(m=0.658, lo=0.528, hi=0.768, dur=1.00),
    "miscarriage_grief": dict(m=0.110, lo=0.050, hi=0.200, dur=14.0 / 365.25),
    "mat_hosp": dict(m=0.133, lo=0.051, hi=0.264, dur=None),  # duration comes from MaternalMorbidity.dur_hosp_days
}

@dataclass
class Costs:
    poc:       float = 50.00;          poc_sd:       float = 10.00
    soc_work:  float = 500.00;         soc_work_sd:  float = 125.00
    rpr:       float = 9.82  * CPI;    rpr_sd:       float = std2(6.71,    26.85)  * CPI
    fta:       float = 31.07 * CPI;    fta_sd:       float = std2(20.14,   53.71)  * CPI
    pen:       float = 20.0;           pen_sd:       float = 4.0
    sf_wu:     float = 75.0;           sf_wu_sd:     float = 25.0
    staff:     float = 30.0;           staff_sd:     float = 10.0
    iufd:      float = 13_049 * CPI;   iufd_sd:      float = std2(10_742, 20_141) * CPI
    preterm:   float = 37_780 * CPI;   preterm_sd:   float = std2(26_855, 53_709) * CPI
    term_del:  float = 13_828 * CPI;   term_del_sd:  float = std2(6_714,  26_855) * CPI
    sb:        float = 141_792 * CPI;  sb_sd:        float = std2(120_846,201_410)* CPI
    nnd:       float = 189_784 * CPI;  nnd_sd:       float = std2(147_701,268_547)* CPI
    lbw_hs:    float = 64_086;         lbw_hs_sd:    float = std2(60_205,  67_891)
    nicu:      float = 50_000.0;       nicu_sd:      float = 10_000.0
    cs_wu:     float = 1_643.68 * CPI; cs_wu_sd:     float = std2(939.91,2_685.47)* CPI
    desens:    float = 4_000.0;        desens_sd:    float = 1_000.0
    jh_obs:    float = 1573.0;         jh_obs_sd:    float = 523.44  # Jarisch-Herxheimer observation cost
    followup:  float = 120.0;          followup_sd:  float = 40.0

P_PEN_ALLERGY: float = 0.10

P_JH_REACTION: float = 0.24

P_LBW_GIVEN_PRETERM: float = 0.70

P_CS_UNCOMP_OBSERVED: float = 0.85

CS_UNCOMP_OBS_LOS_FRAC: float = 0.92

@dataclass
class LongTermCare:
    """
    Direct long-term care costs for children with congenital syphilis sequelae.
    Distinct from ProductivityLoss (which captures caregiver *opportunity* costs).
    These are out-of-pocket / payer-facing direct expenditures.

    Special education
    -----------------
    IDEA Part B covers ages 3–21. Incremental cost above regular education
    estimated from SEEP (2003) and Chambers et al. (2010): ~$10K–$20K/yr
    additional, higher for severe developmental disability.
    Source: Chambers, J.G. et al. (2010). Special Education Expenditure Project.

    Direct caregiver costs
    ----------------------
    Paid home care, respite care, adaptive equipment, therapy co-pays.
    Distinct from informal caregiver time (modelled in ProductivityLoss).
    Genworth Cost of Care Survey (2023) and AHRQ estimates used as anchors.
    Most intensive in early childhood; modelled through age caregiver_end_age.
    """
    # Special education — incremental cost above regular schooling
    p_sped_severe:      float = 0.85    # P(special ed | severe CS sequelae)
    p_sped_severe_lo:   float = 0.70
    p_sped_severe_hi:   float = 0.95
    p_sped_mild:        float = 0.35    # P(special ed | mild CS sequelae)
    p_sped_mild_lo:     float = 0.15
    p_sped_mild_hi:     float = 0.55
    cost_sped_ann:      float = 14_000  # Incremental annual special ed cost ($2023)
    cost_sped_sd:       float = 4_000
    sped_start_age:     int   = 3       # IDEA Part B eligibility
    sped_end_age:       int   = 21      # Inclusive IDEA coverage ceiling

    # Direct paid caregiver / support costs
    cost_cg_severe_ann: float = 32_000  # Paid home care + respite, severe sequelae
    cost_cg_severe_sd:  float = 9_000
    cost_cg_mild_ann:   float = 4_500   # Therapy co-pays + adaptive equipment, mild
    cost_cg_mild_sd:    float = 1_800
    caregiver_end_age:  int   = 18      # Inclusive end age for direct care

PRESETS = {
    "Custom": {},
    "High-burden urban ED":    dict(p_act=0.015,  sc_e=0.92, p_adeq=0.80),
    "Moderate-burden (base)":  dict(p_act=0.0075, sc_e=0.90, p_adeq=0.85),
    "Low-prevalence rural ED": dict(p_act=0.001,  sc_e=0.85, p_adeq=0.75),
    "Best-case operations":    dict(p_act=0.010,  sc_e=0.95, p_adeq=0.95),
}
