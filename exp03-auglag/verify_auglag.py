#!/usr/bin/env python3
"""Headless verification of the augmented-Lagrangian System 1 (no LLM).

S1  servo   : solve  min power  s.t.  sum rate >= 10 bits  from a cold start
              at P_total = 40. Expect convergence to the boundary
              (SR ~ 10.3 = tau + margin, power ~ optimal ~17) with the
              KKT residuals decaying and status reaching CONVERGED.
S2  disturb : after convergence, shuffle the channel gains at step 150.
              Expect a KKT-residual spike (status DISTURBED) followed by
              autonomous re-convergence -- the "disturbance detector".

Run:  uv run python verify_auglag.py
Outputs: results/verify_s1.pdf, results/verify_s2.pdf, results/verify_auglag.json
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (
    N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
    QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
    optimize_step, measure_mi, kkt_residuals, classify_status,
    STATUS_TRANSIENT,
)

SEED = 42
N_MC_GRAD = 8000
N_MC_STATE = 20_000
N_MC_FINAL = 100_000
DUAL_EVERY = 10
TAU_BITS = 10.0
NATS2BITS = 1.0 / math.log(2)

device = torch.device("cpu")
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def run(steps: int, h_schedule=None, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
    h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)

    p_total = P_TOTAL_DEFAULT
    lam = torch.ones(N) * math.sqrt(p_total / N)
    spec = ObjectiveSpec(family="min_power")
    al = AugLagState()
    al.set_constraint(TAU_BITS)
    status = STATUS_TRANSIENT

    log = {k: [] for k in ("step", "sr", "power", "mu", "rho",
                           "r_stat", "r_feas", "gap", "status")}
    lam_interval = lam.clone()

    for step in range(steps):
        if h_schedule is not None:
            new_h = h_schedule(step)
            if new_h is not None:
                h_abs2 = new_h
                h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)

        if step % DUAL_EVERY == 0 and step > 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            sr_bits = float(mi.sum() * NATS2BITS)
            al.dual_update(sr_bits)
            res = kkt_residuals(sr_bits, lam, lam_interval, al)
            status = classify_status(res, al, status)
            lam_interval = lam.clone()
            log["step"].append(step)
            log["sr"].append(sr_bits)
            log["power"].append(float((lam ** 2).sum()))
            log["mu"].append(al.mu)
            log["rho"].append(al.rho)
            log["r_stat"].append(res["r_stat"])
            log["r_feas"].append(res["r_feas"])
            log["gap"].append(res["gap_bound"])
            log["status"].append(status)

        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, al=al)

    mi = measure_mi(lam, h_re, est, N_MC_FINAL, device)
    log["final_sr_bits"] = float(mi.sum() * NATS2BITS)
    log["final_power"] = float((lam ** 2).sum())
    log["final_mu"] = al.mu
    return log


def plot_run(log, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    steps = log["step"]
    axes[0].plot(steps, log["sr"], "b-", label="sum rate")
    axes[0].axhline(TAU_BITS, color="red", linestyle="--", linewidth=1,
                    label=f"tau = {TAU_BITS:.0f} bits")
    axes[0].set_ylabel("Sum rate [bits]")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, log["power"], "g-", label="total power")
    axes[1].set_ylabel("Total power")
    ax1b = axes[1].twinx()
    ax1b.plot(steps, log["mu"], "m--", label="mu")
    ax1b.set_ylabel("Multiplier $\\mu$", color="m")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[2].semilogy(steps, np.maximum(log["r_feas"], 1e-4), label="feasibility")
    axes[2].semilogy(steps, np.maximum(log["r_stat"], 1e-4), label="stationarity")
    axes[2].semilogy(steps, np.maximum(log["gap"], 1e-4), label="gap bound")
    # mark status per interval
    for s, st in zip(steps, log["status"]):
        if st == "DISTURBED":
            axes[2].axvline(s, color="orange", alpha=0.4, linewidth=2)
        elif st == "CONVERGED":
            axes[2].axvline(s, color="green", alpha=0.08)
    axes[2].set_ylabel("KKT residuals")
    axes[2].set_xlabel("Iteration")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    t0 = time.time()

    print("=== S1: min power s.t. SR >= 10 bits (cold start) ===")
    log1 = run(steps=250)
    conv_at = next((s for s, st in zip(log1["step"], log1["status"])
                    if st == "CONVERGED"), None)
    print(f"  final: SR = {log1['final_sr_bits']:.3f} bits, "
          f"power = {log1['final_power']:.2f}, mu = {log1['final_mu']:.3f}")
    print(f"  first CONVERGED at step {conv_at}")
    print(f"  statuses: {log1['status']}")
    plot_run(log1, RESULTS / "verify_s1.pdf",
             "S1: AL servo — min power s.t. SR ≥ 10 bits")

    print("=== S2: gain shuffle at step 150 ===")
    rng = np.random.default_rng(7)
    h_shuffled = np.array(H_ABS2_DEFAULT)[rng.permutation(N)]

    def sched(step):
        return h_shuffled if step == 150 else None

    log2 = run(steps=320, h_schedule=sched)
    print(f"  final: SR = {log2['final_sr_bits']:.3f} bits, "
          f"power = {log2['final_power']:.2f}")
    print(f"  statuses: {log2['status']}")
    plot_run(log2, RESULTS / "verify_s2.pdf",
             "S2: disturbance — gains shuffled at iter 150")

    with open(RESULTS / "verify_auglag.json", "w") as f:
        json.dump({"S1": log1, "S2": log2,
                   "config": {"tau": TAU_BITS, "dual_every": DUAL_EVERY,
                              "seed": SEED}}, f, indent=1)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
