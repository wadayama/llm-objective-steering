"""Water-filling versus the true discrete-input optimum, both by quadrature.

Water-filling is the familiar Gaussian-input reference. Under QPSK the MI
saturates, so the allocation it prescribes over-serves channels that are
already at the ceiling. This quantifies the gap without simulation: MI is
evaluated by Gauss-Hermite quadrature and the true optimum is located from
the KKT condition, exactly as in verify_certificate.py.
"""

import numpy as np

from core import N, H_ABS2_DEFAULT, P_TOTAL_DEFAULT, LAMBDA_MIN
from reference import mi_and_slope, NATS2BITS


def water_filling(h_abs2, p_total, sigma2=1.0):
    """P_i = [1/nu - sigma^2/|h_i|^2]_+ with nu set so the budget is spent."""
    h = np.asarray(h_abs2, dtype=np.float64)
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        nu = 0.5 * (lo + hi)
        P = np.clip(1.0 / nu - sigma2 / h, 0.0, None)
        if P.sum() > p_total:
            lo = nu
        else:
            hi = nu
    nu = 0.5 * (lo + hi)
    return np.clip(1.0 / nu - sigma2 / h, 0.0, None)


def max_sum_rate(h_abs2, p_total):
    """Largest achievable sum rate: equalize dI/dP across active channels."""
    h = np.asarray(h_abs2, dtype=np.float64)

    def alloc(m):
        P = []
        for hi in h:
            lo, hi_b = LAMBDA_MIN ** 2, 400.0
            if mi_and_slope(lo, hi)[1] <= m:
                P.append(lo); continue
            if mi_and_slope(hi_b, hi)[1] >= m:
                P.append(hi_b); continue
            for _ in range(50):
                mid = 0.5 * (lo + hi_b)
                if mi_and_slope(mid, hi)[1] > m:
                    lo = mid
                else:
                    hi_b = mid
            P.append(0.5 * (lo + hi_b))
        return np.array(P)

    lo, hi_m = 1e-9, 10.0
    for _ in range(60):
        m = 0.5 * (lo + hi_m)
        if alloc(m).sum() > p_total:
            lo = m
        else:
            hi_m = m
    P = alloc(0.5 * (lo + hi_m))
    return P


def sum_rate(P, h_abs2):
    return sum(mi_and_slope(p, hi)[0] for p, hi in zip(P, h_abs2)) * NATS2BITS


if __name__ == "__main__":
    h = np.array(H_ABS2_DEFAULT)
    Pt = P_TOTAL_DEFAULT
    P_wf = water_filling(h, Pt)
    P_opt = max_sum_rate(h, Pt)
    r_wf, r_opt = sum_rate(P_wf, h), sum_rate(P_opt, h)
    print(f"budget {Pt}, N={N}")
    print(f"water-filling   : sum rate {r_wf:.3f} bits, power {P_wf.sum():.2f}")
    print(f"discrete optimum: sum rate {r_opt:.3f} bits, power {P_opt.sum():.2f}")
    print(f"shortfall       : {r_opt - r_wf:.3f} bits ({100*(r_opt-r_wf)/r_opt:.1f}%)")
    print("water-filling P :", np.round(P_wf, 2))
    print("optimal P       :", np.round(P_opt, 2))
