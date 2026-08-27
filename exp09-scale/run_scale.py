"""Does the architecture still hold up as the number of channels grows?

System 1 alone, no LLM: one declaration ("minimize total power subject to
SR >= tau") is held fixed and enforced for 600 steps at N = 8, 16, 32, 64.
For every N we ask the three questions the paper asks at N = 8:

  enforcement -- is the declared floor ever violated?
  optimality  -- how far is the settled allocation from the optimum that
                 reference.py computes independently, by quadrature and
                 the KKT condition?
  certificate -- does the reported indicator stay at or above the true gap?

and we time the optimizer so the cost of scaling is on the record.
"""

import argparse
import json
import math
import time

import numpy as np
import torch

import core
import scale_profile as prof

STEPS = 600
DUAL_EVERY = 10
N_MC_GRAD = 8000
N_MC_STATE = 20_000
device = torch.device("cpu")


def run_one(n: int, seed: int, p_star: float, tau: float,
            per_channel: bool = True):
    """One run at size n. Returns per-interval rows and s/step."""
    from core import (T_VAL, QPSKMonteCarloEstimator, ObjectiveSpec,
                      AugLagState, optimize_step, measure_mi, kkt_residuals,
                      classify_status, STATUS_TRANSIENT)
    from reference import NATS2BITS

    p_total = prof.p_total(n)
    al = AugLagState()
    al.per_channel_violation = per_channel
    al.set_constraint(tau)

    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_re = torch.tensor(np.sqrt(np.array(prof.h_abs2(n))), dtype=torch.float32)
    lam = torch.ones(n) * math.sqrt(p_total / n)
    spec = ObjectiveSpec(family="min_power").validate(p_total)
    status = STATUS_TRANSIENT
    lam_iv = lam.clone()
    rows = []

    t_opt = 0.0
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
            rows.append(dict(step=step, status=status, sum_rate=sr,
                             power=power,
                             true_gap=(power - p_star) / p_total,
                             gap_bound=res["gap_bound"], mu=al.mu))
        t0 = time.perf_counter()
        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, al=al)
        t_opt += time.perf_counter() - t0

    return rows, t_opt / STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="8,16,32,64")
    ap.add_argument("--seeds", type=str,
                    default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--out", type=str, default="results/scale.json")
    ap.add_argument("--violation", type=str, default="per-channel",
                    choices=["per-channel", "extensive"],
                    help="units the sum-rate violation is measured in")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    from reference import min_power_for_rate

    summary = []
    for n in sizes:
        core.N = n                       # every default_factory reads this
        tau = prof.tau_bits(n)
        # the optimizer enforces tau + margin, so that is the level whose
        # optimum the certificate should be measured against
        per_channel = args.violation == "per-channel"
        probe = core.AugLagState()
        probe.set_constraint(tau)
        tau_eff = probe.tau_eff
        t0 = time.time()
        ref = min_power_for_rate(tau_eff, h_abs2=prof.h_abs2(n))
        p_star = ref["total_power"]
        t_ref = time.time() - t0
        print(f"N={n:3d}  tau={tau:.1f} (eff {tau_eff:.2f})  "
              f"P*={p_star:.4f}  [reference took {t_ref:.0f} s]", flush=True)

        per_seed = []
        for seed in seeds:
            rows, t_step = run_one(n, seed, p_star, tau, per_channel)
            settled = [r for r in rows if r["status"] == "CONVERGED"]
            viol = sum(1 for r in rows if r["sum_rate"] < tau)
            per_seed.append(dict(
                seed=seed,
                n_intervals=len(rows),
                n_settled=len(settled),
                violations=viol,
                optimistic=sum(1 for r in rows
                               if r["gap_bound"] < r["true_gap"]),
                settled_power=(float(np.mean([r["power"] for r in settled]))
                               if settled else None),
                tail_power=float(np.mean([r["power"]
                                          for r in rows[len(rows) // 2:]])),
                tail_power_std=float(np.std([r["power"]
                                             for r in rows[len(rows) // 2:]])),
                min_sum_rate=min(r["sum_rate"] for r in rows[len(rows) // 2:]),
                final_sum_rate=rows[-1]["sum_rate"],
                sec_per_step=t_step,
            ))
            print(f"   seed {seed}: power {per_seed[-1]['settled_power']}, "
                  f"{viol} violations, {per_seed[-1]['optimistic']} optimistic,"
                  f" {t_step*1e3:.1f} ms/step", flush=True)

        pw = np.array([r["tail_power"] for r in per_seed])
        # scatter within a run, averaged over seeds: an oscillating run has a
        # large one even when its seed-to-seed mean looks respectable
        wobble = float(np.mean([r["tail_power_std"] for r in per_seed]))
        rec = dict(
            n=n, violation=args.violation, steps=STEPS,
            tau=tau, tau_eff=tau_eff, p_star=p_star,
            p_total=prof.p_total(n),
            seeds=seeds,
            power_mean=float(pw.mean()), power_std=float(pw.std(ddof=1)),
            within_run_std=wobble,
            min_sum_rate=min(r["min_sum_rate"] for r in per_seed),
            rel_error_pct=float(100 * abs(pw.mean() - p_star) / p_star),
            total_intervals=sum(r["n_intervals"] for r in per_seed),
            total_violations=sum(r["violations"] for r in per_seed),
            total_optimistic=sum(r["optimistic"] for r in per_seed),
            ms_per_step=float(np.mean([r["sec_per_step"]
                                       for r in per_seed]) * 1e3),
            reference_seconds=t_ref,
            per_seed=per_seed,
        )
        summary.append(rec)
        print(f"   --> N={n}: power {rec['power_mean']:.3f} "
              f"+/- {rec['power_std']:.3f} vs P*={p_star:.3f} "
              f"({rec['rel_error_pct']:.2f}%), "
              f"{rec['total_violations']}/{rec['total_intervals']} violations, "
              f"{rec['ms_per_step']:.1f} ms/step\n", flush=True)

    json.dump(summary, open(args.out, "w"), indent=1)
    print("=== scale summary ===")
    print(f"{'N':>4}{'P*':>10}{'tail power':>18}{'excess %':>10}"
          f"{'wobble':>9}{'viol':>10}{'optimistic':>12}{'ms/step':>9}")
    for r in summary:
        print(f"{r['n']:>4}{r['p_star']:>10.3f}"
              f"{r['power_mean']:>12.3f}+/-{r['power_std']:<5.3f}"
              f"{r['rel_error_pct']:>10.2f}{r['within_run_std']:>9.2f}"
              f"{r['total_violations']:>6}/{r['total_intervals']:<4}"
              f"{r['total_optimistic']:>8}/{r['total_intervals']:<5}"
              f"{r['ms_per_step']:>9.1f}")


if __name__ == "__main__":
    main()
