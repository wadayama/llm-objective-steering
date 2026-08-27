"""System 1 alone under a sustained INFEASIBLE declaration (no LLM).

exp06 observed that phi-4's C0 episode ends at SR = 8.32 bits, well BELOW
the 13.66 bits it started from -- more infeasibility, less rate. If that is
a property of System 1 rather than of the model, it must reproduce with the
declaration held fixed and no LLM in the loop.

Arms:
  P40   : the demanded 15 bits at the default budget (infeasible forever)
  P70   : the same declaration at a budget where 15 bits is reachable
  P40-feasible : a reachable demand at the default budget (control)
"""

import json
import math

import numpy as np
import torch

from common import (STEPS, CALL_EVERY, TRACK_EVERY, N_MC_GRAD, N_MC_STATE,
                    N_MC_FINAL, RESULTS, device, status_line)
from core import (N, T_VAL, H_ABS2_DEFAULT, QPSKMonteCarloEstimator,
                  ObjectiveSpec, AugLagState, optimize_step, measure_mi,
                  kkt_residuals, classify_status, STATUS_TRANSIENT)
from steering import NATS2BITS


def run(tau: float, p_total: float, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_re = torch.tensor(np.sqrt(np.array(H_ABS2_DEFAULT)), dtype=torch.float32)
    lam = torch.ones(N) * math.sqrt(p_total / N)
    al = AugLagState()
    al.set_constraint(tau)
    spec = ObjectiveSpec(family="min_power").validate(p_total)
    status = STATUS_TRANSIENT
    res = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
    lam_iv = lam.clone()
    ts = {"steps": [], "sr": [], "power": [], "mu": [], "rho": [], "status": []}

    for step in range(STEPS):
        if step % CALL_EVERY == 0 and step > 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            al.dual_update(float(mib.sum()), mib)
            res = kkt_residuals(float(mib.sum()), lam, lam_iv, al, mi_bits_meas=mib)
            status = classify_status(res, al, status)
            lam_iv = lam.clone()
        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, al=al)
        if step % TRACK_EVERY == 0 or step == STEPS - 1:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            ts["steps"].append(step)
            ts["sr"].append(float(mib.sum()))
            ts["power"].append(float((lam ** 2).sum()))
            ts["mu"].append(float(al.mu))
            ts["rho"].append(float(al.rho))
            ts["status"].append(status)

    mi = measure_mi(lam, h_re, est, N_MC_FINAL, device)
    mib = mi * NATS2BITS
    return {"tau": tau, "p_total": p_total, "seed": seed, "ts": ts,
            "final_sr": float(mib.sum()), "final_power": float((lam ** 2).sum()),
            "final_mi_bits": mib.tolist(), "final_mu": float(al.mu),
            "final_rho": float(al.rho)}


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    arms = [("P40-infeasible", 15.0, 40.0),
            ("P70-feasible", 15.0, 70.0),
            ("P40-feasible", 10.0, 40.0)]
    out = {}
    print(f"{'arm':<16}{'startSR':>9}{'minSR':>8}{'finalSR':>9}"
          f"{'finalP':>8}{'mu':>7}{'rho':>7}  status")
    for name, tau, p in arms:
        r = run(tau, p)
        out[name] = r
        ts = r["ts"]
        seen = " ".join(sorted(set(ts["status"]), key=ts["status"].index))
        print(f"{name:<16}{ts['sr'][0]:>9.2f}{min(ts['sr']):>8.2f}"
              f"{r['final_sr']:>9.2f}{r['final_power']:>8.2f}"
              f"{r['final_mu']:>7.1f}{r['final_rho']:>7.1f}  {seen}")
    json.dump(out, open(RESULTS / "verify_stiffness.json", "w"),
              indent=1, default=str)
    print(f"\nsaved {RESULTS / 'verify_stiffness.json'}")
