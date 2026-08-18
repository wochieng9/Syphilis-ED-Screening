"""Shared gestational-stratum clinical outcome engine and non-CS DALYs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from .parameters import BASE_BETA, DW_P, GES_STRATA, TX_RR, UNT_ABS
from .utils import pvf


def _mean_beta_param(name: str) -> float:
    p = BASE_BETA[name]
    return p["a"] / (p["a"] + p["b"])


def _deterministic_outcome_inputs() -> Tuple[dict, dict, dict]:
    """Return point-estimate background, untreated, and RR inputs."""
    br_mean = {k: _mean_beta_param(k) for k in BASE_BETA}
    ur_mean = {k: float(v) for k, v in UNT_ABS.items()}
    rr_mean = {k: float(v["rr"]) for k, v in TX_RR.items()}
    return br_mean, ur_mean, rr_mean


def calibrated_usual_care_coverages(
    target_mean: float,
    strata: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Calibrate stratum coverages to a caller-supplied weighted mean.

    A common log-odds shift preserves the ordering and relative heterogeneity of
    the source ``p_uc`` values while honoring ``target_mean``. At the source
    weighted mean, the original stratum values are returned (up to numerical
    precision).
    """
    s = strata or GES_STRATA
    target = float(np.clip(target_mean, 0.0, 1.0))
    names = list(s)
    weights = np.asarray([float(s[name]["w"]) for name in names], dtype=float)
    weights = weights / weights.sum()
    base = np.asarray([float(s[name]["p_uc"]) for name in names], dtype=float)

    if target <= 0.0:
        values = np.zeros_like(base)
    elif target >= 1.0:
        values = np.ones_like(base)
    elif np.isclose(np.dot(weights, base), target, atol=1e-14):
        values = base
    else:
        clipped = np.clip(base, 1e-12, 1.0 - 1e-12)
        logits = np.log(clipped / (1.0 - clipped))
        low, high = -40.0, 40.0
        for _ in range(120):
            mid = (low + high) / 2.0
            shifted = 1.0 / (1.0 + np.exp(-(logits + mid)))
            if float(np.dot(weights, shifted)) < target:
                low = mid
            else:
                high = mid
        values = 1.0 / (1.0 + np.exp(-(logits + (low + high) / 2.0)))

    return {name: float(value) for name, value in zip(names, values)}



def intervention_coverages(
    sc_b: float,
    sc_current: float,
    sc_target: float | None = None,
    *,
    strata: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Return stratum coverages along a coherent implementation path.

    Each stratum moves from its own usual-care coverage to the common final
    target. ``sc_current`` is the weighted average coverage at the current
    implementation stage. This guarantees that current coverage equal to the
    baseline reproduces usual care exactly and prevents an implementation ramp
    from reducing screening in high-baseline strata.
    """

    s = strata or GES_STRATA
    baseline = calibrated_usual_care_coverages(float(sc_b), s)
    target = float(sc_current if sc_target is None else sc_target)
    current = float(sc_current)
    denominator = target - float(sc_b)
    if abs(denominator) <= 1e-15:
        fraction = 0.0 if abs(current - float(sc_b)) <= 1e-15 else 1.0
    else:
        fraction = float(np.clip((current - float(sc_b)) / denominator, 0.0, 1.0))
    return {
        name: float(np.clip(baseline[name] + fraction * (target - baseline[name]), 0.0, 1.0))
        for name in s
    }


@dataclass(frozen=True)
class GestationalReach:
    """Joint gestational-stratum screening and treatment reach.

    ``delta_treated_reach`` is E[sc_e * p_tx] - E[sc_b * p_tx].
    It is deliberately not calculated as (E[sc_e] - E[sc_b]) * E[p_tx].
    """

    delta_screening: float
    baseline_treated_reach: float
    intervention_treated_reach: float
    delta_treated_reach: float
    incremental_tx_completion: float


def gestational_reach(
    sc_b: float,
    sc_current: float,
    sc_target: float | None = None,
    tx_eff_override: Optional[float] = None,
    prop_late_override: Optional[float] = None,
    *,
    strata: Mapping[str, Mapping[str, float]] | None = None,
) -> GestationalReach:
    """Return joint reach quantities for a comparator/intervention pair.

    ``sc_target`` is the final common target used to map an intermediate BIA
    average coverage back to stratum-specific coverages.
    """

    del prop_late_override
    s = strata or GES_STRATA
    baseline = calibrated_usual_care_coverages(float(sc_b), s)
    intervention = intervention_coverages(
        float(sc_b), float(sc_current), sc_target, strata=s
    )
    delta_screening = max(float(sc_current) - float(sc_b), 0.0)
    baseline_mass = 0.0
    intervention_mass = 0.0
    for name, values in s.items():
        tx = (
            float(tx_eff_override)
            if tx_eff_override is not None
            else float(values["p_tx"])
        )
        weight = float(values["w"])
        baseline_mass += weight * baseline[name] * tx
        intervention_mass += weight * intervention[name] * tx
    delta_mass = max(intervention_mass - baseline_mass, 0.0)
    incremental_rate = (
        float(np.clip(delta_mass / delta_screening, 0.0, 1.0))
        if delta_screening > 0.0
        else 0.0
    )
    return GestationalReach(
        delta_screening=float(delta_screening),
        baseline_treated_reach=float(baseline_mass),
        intervention_treated_reach=float(intervention_mass),
        delta_treated_reach=float(delta_mass),
        incremental_tx_completion=incremental_rate,
    )

def incremental_screening_fraction(sc_b: float, sc_e: float) -> float:
    """Incremental screened fraction of the eligible cohort, before ``p_id``."""
    return max(float(sc_e) - float(sc_b), 0.0)


def incremental_treatment_completion_mass(
    sc_b: float,
    sc_e: float,
    tx_completion_override: Optional[float] = None,
    *,
    sc_target: float | None = None,
    strata: Mapping[str, Mapping[str, float]] | None = None,
    positive_only: bool = True,
) -> float:
    """Return weighted incremental screening x treatment-completion mass.

    This is ``E[(coverage_intervention - coverage_usual_care) * p_tx]``.
    It replaces the logically incorrect product ``E[coverage gap] * E[p_tx]``.
    ``positive_only`` is used for program treatment counts; arm outcome
    comparisons retain the full signed difference through separate arm runs.
    """
    s = strata or GES_STRATA
    baseline = calibrated_usual_care_coverages(sc_b, s)
    intervention = intervention_coverages(sc_b, sc_e, sc_target, strata=s)
    total = 0.0
    for name, values in s.items():
        gap = intervention[name] - baseline[name]
        if positive_only:
            gap = max(gap, 0.0)
        tx = float(tx_completion_override) if tx_completion_override is not None else float(values["p_tx"])
        total += float(values["w"]) * gap * tx
    return float(total)


def incremental_tx_completion_rate(
    sc_b: float,
    sc_e: float,
    tx_completion_override: Optional[float] = None,
    *,
    sc_target: float | None = None,
    strata: Mapping[str, Mapping[str, float]] | None = None,
) -> float:
    """Average treatment completion among incrementally screened patients."""
    gap = incremental_screening_fraction(sc_b, sc_e)
    if gap <= 0.0:
        return 0.0
    mass = incremental_treatment_completion_mass(
        sc_b, sc_e, tx_completion_override,
        sc_target=sc_target, strata=strata, positive_only=True
    )
    return float(np.clip(mass / gap, 0.0, 1.0))


def _arm_counts_single(
    sc,
    p_act,
    p_id,
    sens,
    p_adeq,
    p_tx,
    cohort,
    br,
    ur,
    rr,
    prop_symp,
    prop_late,
):
    """Unstratified arm calculation used inside the joint stratum engine."""
    p_eff = sc * p_id * sens * p_adeq * p_tx

    def mix(unt, rr_key):
        untreated = np.asarray(unt, dtype=float)
        treated = np.minimum(untreated * rr[rr_key], 1.0)
        return p_eff * treated + (1.0 - p_eff) * untreated

    sb_syph = mix(ur["stillbirth"], "stillbirth")
    neo_syph = mix(ur["neonatal_death"], "neonatal_death")
    cs_syph = mix(ur["cs_any"], "cs_any")
    pt_syph = mix(ur["preterm"], "preterm")
    lbw_syph = mix(ur["lbw"], "lbw")
    misc_syph = mix(ur["miscarriage"], "miscarriage")

    sb_rate = p_act * sb_syph + (1.0 - p_act) * np.asarray(br["stillbirth"], dtype=float)
    neo_rate = p_act * neo_syph + (1.0 - p_act) * np.asarray(br["neonatal_death"], dtype=float)
    cs_rate = p_act * cs_syph
    pt_rate = p_act * pt_syph + (1.0 - p_act) * np.asarray(br["preterm"], dtype=float)
    lbw_rate = p_act * lbw_syph + (1.0 - p_act) * np.asarray(br["lbw"], dtype=float)
    misc_rate = p_act * misc_syph + (1.0 - p_act) * np.asarray(br["miscarriage"], dtype=float)

    def cnt(x):
        return np.maximum(x, 0.0) * cohort

    return {
        "preterm": cnt(pt_rate),
        "lbw": cnt(lbw_rate),
        "stillbirth": cnt(sb_rate),
        "miscarriage": cnt(misc_rate),
        "neonatal_death": cnt(neo_rate),
        "cs_comp": cnt(cs_rate * prop_symp),
        "cs_uncomp": cnt(cs_rate * (1.0 - prop_symp)),
        "iufd_subset": cnt(sb_rate * prop_late),
    }


def _arm_counts(
    sc,
    p_act,
    p_id,
    sens,
    p_adeq,
    p_tx,
    cohort,
    br,
    ur,
    rr,
    prop_symp,
    prop_late,
    *,
    baseline: bool = False,
    strata: Mapping[str, Mapping[str, float]] | None = None,
    coverage_by_stratum: Mapping[str, float] | None = None,
):
    """Arm outcome counts aggregated from gestational strata.

    ``baseline=True`` interprets ``sc`` as the desired weighted usual-care
    coverage and calibrates the source stratum coverages to that mean. The
    intervention can be supplied as a stratum-specific coverage map; otherwise
    ``sc`` is treated as a common coverage value.

    ``p_tx=None`` uses stratum-specific treatment completion. A numeric value is
    an explicit override. ``prop_late=None`` uses the source stratum values; a
    numeric value is an explicit override.
    """
    s = strata or GES_STRATA
    baseline_cov = calibrated_usual_care_coverages(float(sc), s) if baseline else None
    total: dict[str, np.ndarray] = {}

    for name, values in s.items():
        if baseline_cov is not None:
            coverage = baseline_cov[name]
        elif coverage_by_stratum is not None:
            coverage = float(coverage_by_stratum[name])
        else:
            coverage = float(sc)
        tx = float(p_tx) if p_tx is not None else float(values["p_tx"])
        late = float(prop_late) if prop_late is not None else float(values["prop_late"])
        result = _arm_counts_single(
            coverage,
            p_act,
            p_id,
            sens,
            p_adeq,
            tx,
            float(cohort) * float(values["w"]),
            br,
            ur,
            rr,
            prop_symp,
            late,
        )
        for key, value in result.items():
            total[key] = np.asarray(value, dtype=float) if key not in total else total[key] + value
    return total


def _arm(
    sc,
    p_act,
    p_id,
    sens,
    p_adeq,
    p_tx,
    cohort,
    br,
    ur,
    rr,
    prop_symp,
    prop_late,
    *,
    baseline: bool = False,
):
    """Vectorized arm wrapper retained for compatibility."""
    return _arm_counts(
        sc,
        p_act,
        p_id,
        sens,
        p_adeq,
        p_tx,
        cohort,
        br,
        ur,
        rr,
        prop_symp,
        prop_late,
        baseline=baseline,
    )


def arm_outcome_delta(
    cohort: float,
    p_act: float,
    p_id: float,
    sc_b: float,
    sc_e: float,
    sens: float,
    p_adeq: float,
    tx_completion_override: Optional[float],
    prop_symp: float,
    prop_late_override: Optional[float],
    br,
    ur,
    rr,
    sc_target: Optional[float] = None,
) -> tuple[dict, dict, dict]:
    """Return comparator, intervention, and comparator-minus-intervention counts."""
    comparator = _arm_counts(
        sc_b,
        p_act,
        p_id,
        sens,
        p_adeq,
        tx_completion_override,
        cohort,
        br,
        ur,
        rr,
        prop_symp,
        prop_late_override,
        baseline=True,
    )
    intervention_cov = intervention_coverages(
        float(sc_b), float(sc_e), sc_target
    )
    intervention = _arm_counts(
        sc_e,
        p_act,
        p_id,
        sens,
        p_adeq,
        tx_completion_override,
        cohort,
        br,
        ur,
        rr,
        prop_symp,
        prop_late_override,
        baseline=False,
        coverage_by_stratum=intervention_cov,
    )
    delta = {key: comparator[key] - intervention[key] for key in comparator}
    return comparator, intervention, delta



def compare_stratified_arms(
    *,
    sc_b: float,
    sc_e: float,
    p_act: float,
    p_id: float,
    sens: float,
    p_adeq: float,
    cohort: float,
    br,
    ur,
    rr,
    prop_symp: float,
    tx_eff_override: Optional[float] = None,
    prop_late_override: Optional[float] = None,
):
    """Return comparator, intervention, delta, and joint reach metadata."""

    comparator, intervention, delta = arm_outcome_delta(
        cohort=float(cohort),
        p_act=float(p_act),
        p_id=float(p_id),
        sc_b=float(sc_b),
        sc_e=float(sc_e),
        sens=float(sens),
        p_adeq=float(p_adeq),
        tx_completion_override=tx_eff_override,
        prop_symp=float(prop_symp),
        prop_late_override=prop_late_override,
        br=br,
        ur=ur,
        rr=rr,
        sc_target=float(sc_e),
    )
    reach = gestational_reach(
        sc_b=float(sc_b),
        sc_current=float(sc_e),
        sc_target=float(sc_e),
        tx_eff_override=tx_eff_override,
        prop_late_override=prop_late_override,
    )
    return comparator, intervention, delta, reach

def _incremental_screening_outcome_delta(
    n_screened,
    p_act,
    sens,
    p_adeq,
    tx_eff=None,
    prop_symp=0.38,
    prop_late=None,
    sc_b=None,
    sc_e=None,
    sc_target=None,
):
    """Outcomes averted per incrementally screened cohort.

    This compatibility helper compares zero versus full screening using the
    same joint gestational engine. New CEA/BIA code should use
    :func:`arm_outcome_delta` with the actual eligible cohort and coverage arms.
    """
    del sc_b, sc_e, sc_target
    br_mean, ur_mean, rr_mean = _deterministic_outcome_inputs()
    _, _, delta = arm_outcome_delta(
        cohort=float(n_screened),
        p_act=float(p_act),
        p_id=1.0,
        sc_b=0.0,
        sc_e=1.0,
        sens=float(sens),
        p_adeq=float(p_adeq),
        tx_completion_override=tx_eff,
        prop_symp=float(prop_symp),
        prop_late_override=prop_late,
        br=br_mean,
        ur=ur_mean,
        rr=rr_mean,
    )
    return delta


def _dalys_non_cs(
    d,
    dw,
    r,
    LE,
    inc_lbw,
    inc_mat,
    inc_sb_yll: bool = True,
    inc_misc_yld: bool = True,
    inc_preterm_yld: bool = True,
    return_components: bool = False,
):
    """Non-CS DALYs averted, decomposed for auditability."""
    af = lambda t: pvf(t, r)
    ref = np.asarray(d["neonatal_death"], dtype=float)
    zero = np.zeros_like(ref, dtype=float)

    nnd_yll = ref * af(LE)
    sb_base = np.asarray(d.get("iufd_subset", d["stillbirth"]), dtype=float)
    stillbirth_yll = sb_base * af(LE) if inc_sb_yll else zero.copy()
    lbw_yld = (
        np.asarray(d["lbw"], dtype=float) * dw["lbw"] * af(DW_P["lbw"]["dur"])
        if inc_lbw
        else zero.copy()
    )
    preterm_yld = (
        np.asarray(d["preterm"], dtype=float)
        * dw["preterm"]
        * af(DW_P["preterm"]["dur"])
        if inc_preterm_yld
        else zero.copy()
    )
    miscarriage_yld = (
        np.asarray(d["miscarriage"], dtype=float)
        * dw["miscarriage_grief"]
        * af(DW_P["miscarriage_grief"]["dur"])
        if inc_misc_yld
        else zero.copy()
    )
    mat_sb_grief = zero.copy()
    mat_nnd_grief = zero.copy()
    if inc_mat:
        mat_sb_grief = (
            np.asarray(d["stillbirth"], dtype=float)
            * dw["mat_sb"]
            * af(DW_P["mat_sb"]["dur"])
        )
        mat_nnd_grief = (
            np.asarray(d["neonatal_death"], dtype=float)
            * dw["mat_nnd"]
            * af(DW_P["mat_nnd"]["dur"])
        )

    components = {
        "nnd_yll": nnd_yll,
        "stillbirth_yll": stillbirth_yll,
        "lbw_yld": lbw_yld,
        "preterm_yld": preterm_yld,
        "miscarriage_yld": miscarriage_yld,
        "mat_sb_grief_yld": mat_sb_grief,
        "mat_nnd_grief_yld": mat_nnd_grief,
    }
    total = sum(components.values())
    if return_components:
        return total.astype(float), {k: v.astype(float) for k, v in components.items()}
    return total.astype(float)
