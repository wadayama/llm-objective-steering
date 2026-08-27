"""A reference optimum computed independently of the optimizer under test.

The optimizer estimates MI by Monte Carlo and finds an allocation by
projected gradient ascent. To check its optimality certificate we need a
answer that shares neither of those. So:

  * MI is evaluated by tensor-product Gauss--Hermite quadrature, which is
    deterministic -- no sampling noise at all.
  * the optimum is found from the KKT condition rather than by iterating.
    For "minimize total power subject to SR >= tau" with each I_i(P_i)
    concave and increasing, stationarity requires the marginal MI per unit
    power to be equal across all channels that carry more than the floor.
    Sweeping that common marginal value m and bisecting until the sum rate
    hits tau therefore lands on the optimum directly.
"""

import math

import numpy as np
import torch

from core import N, T_VAL, H_ABS2_DEFAULT, LAMBDA_MIN, QPSK_CONST

NATS2BITS = 1.0 / math.log(2)


def mi_quadrature(a: torch.Tensor, n_nodes: int = 48) -> torch.Tensor:
    """MI [nats] at effective amplitude a, by Gauss--Hermite quadrature.

    Same integrand as the Monte Carlo estimator, evaluated exactly instead
    of sampled. Differentiable, so dI/da comes from autograd.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    x = torch.tensor(x, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64) / math.sqrt(2.0 * math.pi)
    # 2-D product rule over the real and imaginary noise components
    z = torch.stack(torch.meshgrid(x, x, indexing="ij"), dim=-1).reshape(-1, 2)
    wt = (w[:, None] * w[None, :]).reshape(-1)

    const = QPSK_CONST.to(torch.float64)
    M = const.shape[0]
    total = torch.zeros((), dtype=torch.float64)
    for k in range(M):
        diffs = const[k].unsqueeze(0) - const              # (M,2)
        a_diffs = a * diffs
        sq = (a_diffs ** 2).sum(dim=1)                     # (M,)
        cross = a_diffs @ z.T                              # (M,Q)
        expo = -sq.unsqueeze(1) / (2 * T_VAL) - cross / math.sqrt(T_VAL)
        total = total + (torch.logsumexp(expo, dim=0) * wt).sum()
    return math.log(M) - total / M


def mi_and_slope(P: float, h_abs2: float, n_nodes: int = 48):
    """I(P) [nats] and dI/dP at power P on a channel with gain h_abs2."""
    Pt = torch.tensor(max(P, 1e-12), dtype=torch.float64, requires_grad=True)
    a = math.sqrt(h_abs2) * torch.sqrt(Pt)
    I = mi_quadrature(a, n_nodes)
    (g,) = torch.autograd.grad(I, Pt)
    return float(I.detach()), float(g)


def _power_for_marginal(m: float, h_abs2: float, lo=1e-6, hi=400.0) -> float:
    """P such that dI/dP = m, or the floor if even there the slope is below m."""
    p_floor = LAMBDA_MIN ** 2
    if mi_and_slope(p_floor, h_abs2)[1] <= m:
        return p_floor
    if mi_and_slope(hi, h_abs2)[1] >= m:
        return hi
    lo, hi = p_floor, hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mi_and_slope(mid, h_abs2)[1] > m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def min_power_for_rate(tau_bits: float, h_abs2=None, n_nodes: int = 48) -> dict:
    """Least total power achieving sum rate >= tau_bits, from the KKT condition."""
    h = np.array(H_ABS2_DEFAULT if h_abs2 is None else h_abs2, dtype=np.float64)
    tau_nats = tau_bits / NATS2BITS

    def sum_rate(m):
        P = np.array([_power_for_marginal(m, hi) for hi in h])
        I = np.array([mi_and_slope(p, hi)[0] for p, hi in zip(P, h)])
        return float(I.sum()), P, I

    lo, hi = 1e-9, 10.0            # marginal value: small m -> lots of power
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        sr, _, _ = sum_rate(mid)
        if sr > tau_nats:
            lo = mid               # can afford a higher marginal (less power)
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    sr, P, I = sum_rate(m)
    return {"tau_bits": tau_bits, "marginal": m, "powers": P.tolist(),
            "total_power": float(P.sum()), "sum_rate_bits": sr * NATS2BITS,
            "mi_bits": (I * NATS2BITS).tolist()}
