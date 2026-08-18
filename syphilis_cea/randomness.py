"""Named, independent random-number streams for reproducible PSA."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


DEFAULT_STREAM_NAMES = (
    "clinical",
    "markov",
    "long_term_care",
    "maternal",
    "productivity",
    "serofast",
)


def spawn_named_generators(
    seed: int,
    names: Iterable[str] = DEFAULT_STREAM_NAMES,
) -> Dict[str, np.random.Generator]:
    """Spawn one stable independent generator per named parameter group.

    The sequence position for each group is fixed by ``names``. A disabled
    module can therefore be skipped without shifting any other module's draws.
    """

    ordered = tuple(names)
    if len(set(ordered)) != len(ordered):
        raise ValueError("Random-stream names must be unique")
    children = np.random.SeedSequence(int(seed)).spawn(len(ordered))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(ordered, children)
    }
