"""Immutable configuration objects and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MarkovConfig:
    """Point estimates and uncertainty ranges for the infant Markov model.

    The object is immutable so deterministic and probabilistic calculations do
    not depend on process-wide mutable state. Complicated congenital-syphilis
    state probabilities must lie on the probability simplex.
    """

    p_severe_cs_comp: float = 0.35
    p_severe_cs_comp_lo: float = 0.20
    p_severe_cs_comp_hi: float = 0.50

    p_mild_cs_comp: float = 0.40
    p_mild_cs_comp_lo: float = 0.25
    p_mild_cs_comp_hi: float = 0.55

    p_mild_cs_uncomp: float = 0.06
    p_mild_cs_uncomp_lo: float = 0.02
    p_mild_cs_uncomp_hi: float = 0.14

    dw_mild: float = 0.110
    dw_mild_lo: float = 0.050
    dw_mild_hi: float = 0.210

    dw_severe: float = 0.390
    dw_severe_lo: float = 0.260
    dw_severe_hi: float = 0.530

    cost_mild_ann: float = 8_500.0
    cost_mild_ann_sd: float = 2_500.0
    cost_severe_ann: float = 26_000.0
    cost_severe_ann_sd: float = 7_500.0

    q_progress_target: float = 0.20

    mu_excess_mild: float = 0.0
    mu_excess_mild_lo: float = 0.0
    mu_excess_mild_hi: float = 0.003

    mu_excess_severe: float = 0.0
    mu_excess_severe_lo: float = 0.0
    mu_excess_severe_hi: float = 0.015

    def __post_init__(self) -> None:
        probability_fields = (
            "p_severe_cs_comp",
            "p_severe_cs_comp_lo",
            "p_severe_cs_comp_hi",
            "p_mild_cs_comp",
            "p_mild_cs_comp_lo",
            "p_mild_cs_comp_hi",
            "p_mild_cs_uncomp",
            "p_mild_cs_uncomp_lo",
            "p_mild_cs_uncomp_hi",
            "dw_mild",
            "dw_mild_lo",
            "dw_mild_hi",
            "dw_severe",
            "dw_severe_lo",
            "dw_severe_hi",
            "q_progress_target",
            "mu_excess_mild",
            "mu_excess_mild_lo",
            "mu_excess_mild_hi",
            "mu_excess_severe",
            "mu_excess_severe_lo",
            "mu_excess_severe_hi",
        )
        for name in probability_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {value}")

        if self.p_severe_cs_comp + self.p_mild_cs_comp > 1.0 + 1e-12:
            raise ValueError(
                "p_severe_cs_comp + p_mild_cs_comp must not exceed 1. "
                "Use conditional or normalized probabilities."
            )

        for prefix in (
            "p_severe_cs_comp",
            "p_mild_cs_comp",
            "p_mild_cs_uncomp",
            "dw_mild",
            "dw_severe",
            "mu_excess_mild",
            "mu_excess_severe",
        ):
            lo = float(getattr(self, f"{prefix}_lo"))
            mean = float(getattr(self, prefix))
            hi = float(getattr(self, f"{prefix}_hi"))
            if lo > hi:
                raise ValueError(f"{prefix}_lo must not exceed {prefix}_hi")
            # User-selected means may move outside the original uncertainty
            # interval. Draw helpers widen the working interval while preserving
            # the selected mean, so only bound ordering is validated here.

        for name in (
            "cost_mild_ann",
            "cost_mild_ann_sd",
            "cost_severe_ann",
            "cost_severe_ann_sd",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def p_healthy_cs_comp(self) -> float:
        return 1.0 - self.p_severe_cs_comp - self.p_mild_cs_comp

    @property
    def p_healthy_cs_uncomp(self) -> float:
        return 1.0 - self.p_mild_cs_uncomp

    def updated(self, **changes: Any) -> "MarkovConfig":
        """Return a validated copy with selected fields replaced."""
        return replace(self, **changes)

    def as_legacy_dict(self, q_progress: float | None = None) -> Dict[str, Any]:
        """Return the former dictionary representation for tables and exports."""
        data: Dict[str, Any] = {
            "p_severe_cs_comp": {
                "m": self.p_severe_cs_comp,
                "lo": self.p_severe_cs_comp_lo,
                "hi": self.p_severe_cs_comp_hi,
            },
            "p_mild_cs_comp": {
                "m": self.p_mild_cs_comp,
                "lo": self.p_mild_cs_comp_lo,
                "hi": self.p_mild_cs_comp_hi,
            },
            "p_mild_cs_uncomp": {
                "m": self.p_mild_cs_uncomp,
                "lo": self.p_mild_cs_uncomp_lo,
                "hi": self.p_mild_cs_uncomp_hi,
            },
            "dw_mild": {"m": self.dw_mild, "lo": self.dw_mild_lo, "hi": self.dw_mild_hi},
            "dw_severe": {
                "m": self.dw_severe,
                "lo": self.dw_severe_lo,
                "hi": self.dw_severe_hi,
            },
            "cost_mild_ann": {"mu": self.cost_mild_ann, "sd": self.cost_mild_ann_sd},
            "cost_sev_ann": {"mu": self.cost_severe_ann, "sd": self.cost_severe_ann_sd},
            "q_progress_target": self.q_progress_target,
            "mu_excess_mild": {
                "m": self.mu_excess_mild,
                "lo": self.mu_excess_mild_lo,
                "hi": self.mu_excess_mild_hi,
            },
            "mu_excess_severe": {
                "m": self.mu_excess_severe,
                "lo": self.mu_excess_severe_lo,
                "hi": self.mu_excess_severe_hi,
            },
        }
        if q_progress is not None:
            data["q_progress"] = float(q_progress)
        return data


DEFAULT_MARKOV_CONFIG = MarkovConfig()


@dataclass(frozen=True)
class NaturalHistoryReference:
    """Diagnostic-only reference values; these are not model inputs."""

    cs_early_cure_rate: float = 0.95
    cs_late_manifest_rate: float = 0.20
    cs_neuro_disorder_rate: float = 0.20


DEFAULT_NATURAL_HISTORY_REFERENCE = NaturalHistoryReference()


@dataclass(frozen=True)
class TreatmentCostPolicy:
    """Protocol policy for allocating treatment-adjacent costs.

    ``jh_scope='all_treated'`` preserves the source model's BIA assumption and
    is now used consistently by CEA and BIA. ``'active_only'`` is available for
    scenario analysis because the source does not resolve whether the
    Jarisch-Herxheimer observation cost should apply to serofast and false-
    positive treatment.
    """

    jh_scope: str = "all_treated"

    def __post_init__(self) -> None:
        if self.jh_scope not in {"all_treated", "active_only"}:
            raise ValueError("jh_scope must be 'all_treated' or 'active_only'")


DEFAULT_TREATMENT_COST_POLICY = TreatmentCostPolicy()


def normalize_complicated_state_probabilities(
    p_mild: np.ndarray | float,
    p_severe: np.ndarray | float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project complicated-CS probabilities onto the simplex.

    Valid inputs pass through unchanged. Invalid non-negative pairs are scaled
    proportionally so mild + severe = 1, preventing cohort mass above one.
    Negative values are clipped to zero.
    """

    mild = np.maximum(np.asarray(p_mild, dtype=float), 0.0)
    severe = np.maximum(np.asarray(p_severe, dtype=float), 0.0)
    total = mild + severe
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, 1e-15), 1.0)
    mild = mild * scale
    severe = severe * scale
    healthy = np.maximum(1.0 - mild - severe, 0.0)
    return healthy, mild, severe


def markov_point_parameters(config: MarkovConfig, n: int = 1) -> Dict[str, np.ndarray]:
    """Create vector-shaped point parameters for the shared Markov engine."""

    if n < 1:
        raise ValueError("n must be at least 1")
    return {
        "p_severe_cs_comp": np.full(n, config.p_severe_cs_comp, dtype=float),
        "p_mild_cs_comp": np.full(n, config.p_mild_cs_comp, dtype=float),
        "p_mild_cs_uncomp": np.full(n, config.p_mild_cs_uncomp, dtype=float),
        "dw_mild": np.full(n, config.dw_mild, dtype=float),
        "dw_severe": np.full(n, config.dw_severe, dtype=float),
        "cost_mild_ann": np.full(n, config.cost_mild_ann, dtype=float),
        "cost_sev_ann": np.full(n, config.cost_severe_ann, dtype=float),
        "mu_excess_mild": np.full(n, config.mu_excess_mild, dtype=float),
        "mu_excess_severe": np.full(n, config.mu_excess_severe, dtype=float),
    }
