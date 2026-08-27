"""Does the reported optimality-gap bound actually bound the gap?

We hold a declaration fixed, run System 1 alone (no LLM), and at every dual
interval compare what the certificate reports against the true distance to
the optimum, which reference.py computes independently by quadrature and
the KKT condition.
"""

import json
import math

import numpy as np
import torch

from core import (N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
                  QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
                  optimize_step, measure_mi, kkt_residuals, classify_status,
                  STATUS_TRANSIENT)
from reference import min_power_for_rate, NATS2BITS

STEPS = 250
DUAL_EVERY = 10
N_MC_GRAD = 8000
N_MC_STATE = 20_000
TAU = 10.0
device = torch.device("cpu")


def run(tau_bits=TAU, seed=42):
    al = AugLagState()
    al.set_constraint(tau_bits)
    # the optimizer enforces tau + margin, so that is the problem whose
    # optimum the certificate should be measured against
    ref = min_power_for_rate(al.tau_eff)
    p_star = ref["total_power"]
    print(f"reference optimum for SR >= {al.tau_eff:.2f} bits: "
          f"total power {p_star:.3f}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_re = torch.tensor(np.sqrt(np.array(H_ABS2_DEFAULT)), dtype=torch.float32)
    lam = torch.ones(N) * math.sqrt(P_TOTAL_DEFAULT / N)
    spec = ObjectiveSpec(family="min_power").validate(P_TOTAL_DEFAULT)
    status = STATUS_TRANSIENT
    res = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
    lam_iv = lam.clone()
    rows = []

    for step in range(STEPS):
        if step % DUAL_EVERY == 0 and step > 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            sr = float(mib.sum())
            al.dual_update(sr, mib)
            res = kkt_residuals(sr, lam, lam_iv, al, mi_bits_meas=mib)
            status = classify_status(res, al, status)
            lam_iv = lam.clone()

            power = float((lam ** 2).sum())
            # objective actually optimized: J = -sum(P) / P_TOTAL_DEFAULT
            true_gap = (power - p_star) / P_TOTAL_DEFAULT
            rows.append(dict(step=step, status=status, sum_rate=sr,
                             power=power, true_gap=true_gap,
                             gap_bound=res["gap_bound"], mu=al.mu,
                             r_stat=res["r_stat"], r_feas=res["r_feas"],
                             r_comp=res["r_comp"]))
        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            P_TOTAL_DEFAULT, est, al=al)

    return ref, rows


if __name__ == "__main__":
    ref, rows = run()
    print(f"\n{'step':>5}{'status':>11}{'SR':>8}{'power':>8}"
          f"{'true gap':>10}{'bound':>9}{'r_comp':>9}{'holds':>7}")
    for r in rows:
        holds = "yes" if r["gap_bound"] >= r["true_gap"] else "NO"
        print(f"{r['step']:>5}{r['status']:>11}{r['sum_rate']:>8.2f}"
              f"{r['power']:>8.2f}{r['true_gap']:>10.4f}"
              f"{r['gap_bound']:>9.4f}{r['r_comp']:>9.4f}{holds:>7}")

    settled = [r for r in rows if r["status"] == "CONVERGED"]
    print(f"\nheld at every interval: "
          f"{all(r['gap_bound'] >= r['true_gap'] for r in rows)}")
    if settled:
        print(f"once CONVERGED ({len(settled)} intervals): "
              f"true gap {min(r['true_gap'] for r in settled):+.4f} .. "
              f"{max(r['true_gap'] for r in settled):+.4f}, "
              f"bound {min(r['gap_bound'] for r in settled):.4f} .. "
              f"{max(r['gap_bound'] for r in settled):.4f}")
    json.dump({"reference": ref, "rows": rows},
              open("results/verify_certificate.json", "w"), indent=1)
