"""One channel profile generator used at every N in this experiment.

The paper's default eight-channel profile has |h_i| running from 0.5 to
1.5, so |h_i|^2 spans [0.25, 2.25].  To sweep N we keep that dynamic
range fixed and space |h_i| linearly inside it, so a larger N packs more
channels into the same spread rather than widening it.  The power cap and
the declared rate floor scale with N so that per-channel power and the
fraction of the ceiling being demanded stay constant.
"""

import numpy as np

H_ABS_LO = 0.5
H_ABS_HI = 1.5
POWER_PER_CHANNEL = 5.0      # P_total = 5 N, matching P_total = 40 at N = 8
RATE_PER_CHANNEL = 1.25      # tau = 1.25 N bits, i.e. 62.5% of the 2 N ceiling


def h_abs2(n: int) -> list[float]:
    return list(np.linspace(H_ABS_LO, H_ABS_HI, n) ** 2)


def p_total(n: int) -> float:
    return POWER_PER_CHANNEL * n


def tau_bits(n: int) -> float:
    return RATE_PER_CHANNEL * n
