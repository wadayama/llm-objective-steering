#!/usr/bin/env python3
"""Headless verification of the extended System 1 (no LLM involved).

Scenarios
  S1  sanity        : weighted_sum + equal weights at P=40 reproduces the
                      conference-version operating point (sum rate ~13.8 bits).
  S2  MI fairness   : at P_total=12 (unsaturated regime), compare
                      weighted_sum(equal) vs alpha_fair(alpha=5) vs soft_min(beta=10).
                      Fair objectives should equalize per-channel MI.
  S3  equal power   : power_target with Pi_i = P/N vs weighted_sum(equal).
                      power_target should drive all P_i to P/N exactly.
  S4  sphere vs ball: budget step 15 -> 40 mid-run. Sphere projection should
                      adapt instantly; ball projection only via slow gradient growth.

Run:  uv run python verify_system1.py
Outputs: results/*.pdf, results/verify_results.json
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
    N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT, LAMBDA_MIN,
    QPSKMonteCarloEstimator, ObjectiveSpec, optimize_step,
    jain_index, measure_mi,
)

SEED = 42
N_MC_GRAD = 6000
N_MC_TRACK = 20_000
N_MC_FINAL = 100_000
STEPS = 200
TRACK_EVERY = 10
NATS2BITS = 1.0 / math.log(2)

device = torch.device("cpu")
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def run_trajectory(spec: ObjectiveSpec, p_total: float, steps: int = STEPS,
                   projection: str = "sphere", lam0: torch.Tensor | None = None,
                   p_schedule=None, seed: int = SEED):
    """Run System 1 alone and record trajectories.

    p_schedule: optional callable step -> p_total override (for S4 budget step).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_re = torch.tensor(np.sqrt(np.array(H_ABS2_DEFAULT)), dtype=torch.float32)

    if lam0 is None:
        lam = torch.ones(N) * math.sqrt(p_total / N)
    else:
        lam = lam0.clone()

    spec.validate(p_total)
    ts_steps, ts_mi, ts_pow = [], [], []
    for s in range(steps):
        pt = p_schedule(s) if p_schedule is not None else p_total
        spec.validate(pt)
        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device, pt,
                            est, projection=projection)
        if s % TRACK_EVERY == 0 or s == steps - 1:
            mi = measure_mi(lam, h_re, est, N_MC_TRACK, device)
            ts_steps.append(s)
            ts_mi.append(mi)
            ts_pow.append((lam ** 2).numpy().copy())

    mi_final = measure_mi(lam, h_re, est, N_MC_FINAL, device)
    powers = (lam ** 2).numpy()
    return {
        "mi_final_bits": (mi_final * NATS2BITS).tolist(),
        "powers_final": powers.tolist(),
        "sum_rate_bits": float(mi_final.sum() * NATS2BITS),
        "min_mi_bits": float(mi_final.min() * NATS2BITS),
        "jain_mi": jain_index(mi_final),
        "jain_power": jain_index(powers),
        "power_std": float(powers.std()),
        "total_power": float(powers.sum()),
        "ts_steps": ts_steps,
        "ts_mi_bits": [(m * NATS2BITS).tolist() for m in ts_mi],
        "ts_pow": [p.tolist() for p in ts_pow],
    }


def main():
    t0 = time.time()
    out = {}

    # ---- S1: sanity at the conference operating point ----
    print("=== S1: sanity (weighted_sum, equal, P=40, sphere) ===")
    r = run_trajectory(ObjectiveSpec(family="weighted_sum"), P_TOTAL_DEFAULT)
    out["S1_ws_equal_P40"] = r
    print(f"  sum rate = {r['sum_rate_bits']:.3f} bits (conference B0: 13.846)")
    print(f"  min MI   = {r['min_mi_bits']:.3f} bits, Jain(MI) = {r['jain_mi']:.4f}")

    # ---- S2: MI fairness in the unsaturated regime ----
    print("=== S2: MI fairness at P_total=12 ===")
    P_LOW = 12.0
    s2 = {}
    for tag, spec in [
        ("weighted_sum", ObjectiveSpec(family="weighted_sum")),
        ("alpha_fair", ObjectiveSpec(family="alpha_fair", alpha=5.0)),
        ("soft_min", ObjectiveSpec(family="soft_min", beta=10.0)),
    ]:
        r = run_trajectory(spec, P_LOW)
        s2[tag] = r
        mi = np.array(r["mi_final_bits"])
        print(f"  {tag:13s}: sum={r['sum_rate_bits']:6.3f}  min={r['min_mi_bits']:.3f}  "
              f"spread={mi.max() - mi.min():.3f}  Jain(MI)={r['jain_mi']:.4f}")
    out["S2_fairness_P12"] = s2

    # ---- S3: equalize power ----
    print("=== S3: equalize power at P_total=40 ===")
    tg = np.ones(N) * (P_TOTAL_DEFAULT / N)
    s3 = {}
    for tag, spec in [
        ("weighted_sum", ObjectiveSpec(family="weighted_sum")),
        ("power_target", ObjectiveSpec(family="power_target", targets=tg)),
    ]:
        r = run_trajectory(spec, P_TOTAL_DEFAULT)
        s3[tag] = r
        p = np.array(r["powers_final"])
        print(f"  {tag:13s}: P = [" + ", ".join(f"{v:5.2f}" for v in p) + "]"
              f"  std={r['power_std']:.3f}  Jain(P)={r['jain_power']:.4f}")
    out["S3_equal_power_P40"] = s3

    # ---- S4: sphere vs ball on a budget step 15 -> 40 ----
    print("=== S4: budget step 15 -> 40, sphere vs ball ===")
    STEP_AT = 100

    def sched(s):
        return 15.0 if s < STEP_AT else 40.0

    s4 = {}
    for proj in ("sphere", "ball"):
        lam0 = torch.ones(N) * math.sqrt(15.0 / N)
        r = run_trajectory(ObjectiveSpec(family="weighted_sum"), 15.0,
                           steps=STEPS, projection=proj, lam0=lam0,
                           p_schedule=sched)
        s4[proj] = r
        # sum rate right after the step and at the end
        ts = np.array(r["ts_steps"])
        sr = np.array([sum(m) for m in r["ts_mi_bits"]])
        after = sr[ts >= STEP_AT]
        print(f"  {proj:6s}: sum rate at step {STEP_AT + TRACK_EVERY} = {after[1]:.3f}, "
              f"final = {r['sum_rate_bits']:.3f}, total power = {r['total_power']:.2f}")
    out["S4_budget_step"] = s4

    # ---- plots ----
    make_plots(out)

    with open(RESULTS / "verify_results.json", "w") as f:
        json.dump({
            "config": {"seed": SEED, "n_mc_grad": N_MC_GRAD, "steps": STEPS,
                       "h_abs2": H_ABS2_DEFAULT, "lambda_min": LAMBDA_MIN},
            "results": out,
        }, f, indent=1)
    print(f"done in {time.time() - t0:.1f}s -> {RESULTS}")


def make_plots(out):
    x = np.arange(N)

    # S2: fairness comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s2 = out["S2_fairness_P12"]
    wd = 0.27
    labels = [("weighted_sum", "weighted_sum (equal w)"),
              ("alpha_fair", r"alpha_fair ($\alpha$=5)"),
              ("soft_min", r"soft_min ($\beta$=10)")]
    for j, (tag, lab) in enumerate(labels):
        axes[0].bar(x + (j - 1) * wd, s2[tag]["mi_final_bits"], wd, label=lab)
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("MI [bits]")
    axes[0].set_title("S2: Per-channel MI at $P_{total}$=12")
    axes[0].set_xticks(x)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    for j, (tag, lab) in enumerate(labels):
        axes[1].bar(x + (j - 1) * wd, s2[tag]["powers_final"], wd, label=lab)
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Power $P_i$")
    axes[1].set_title("S2: Power allocation at $P_{total}$=12")
    axes[1].set_xticks(x)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "s2_fairness.pdf")
    plt.close(fig)

    # S3: power equalization
    fig, ax = plt.subplots(figsize=(6.5, 4))
    s3 = out["S3_equal_power_P40"]
    ax.bar(x - 0.2, s3["weighted_sum"]["powers_final"], 0.4, label="weighted_sum (equal w)")
    ax.bar(x + 0.2, s3["power_target"]["powers_final"], 0.4, label="power_target ($\\Pi_i$=P/N)")
    ax.axhline(P_TOTAL_DEFAULT / N, color="k", linestyle="--", linewidth=1, label="P/N = 5")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Power $P_i$")
    ax.set_title("S3: Equalize power at $P_{total}$=40")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "s3_equal_power.pdf")
    plt.close(fig)

    # S4: sphere vs ball budget step
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s4 = out["S4_budget_step"]
    for proj, color in [("sphere", "tab:blue"), ("ball", "tab:red")]:
        ts = s4[proj]["ts_steps"]
        sr = [sum(m) for m in s4[proj]["ts_mi_bits"]]
        tp = [sum(p) for p in s4[proj]["ts_pow"]]
        axes[0].plot(ts, sr, color=color, label=proj)
        axes[1].plot(ts, tp, color=color, label=proj)
    for ax, ylab, title in [(axes[0], "Sum rate [bits]", "S4: Sum rate"),
                            (axes[1], "Total power", "S4: Total transmit power")]:
        ax.axvline(100, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylab)
        ax.set_title(title + " (budget 15$\\to$40 at iter 100)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "s4_sphere_vs_ball.pdf")
    plt.close(fig)

    # S1: convergence
    fig, ax = plt.subplots(figsize=(6.5, 4))
    r = out["S1_ws_equal_P40"]
    ax.plot(r["ts_steps"], [sum(m) for m in r["ts_mi_bits"]], "b-")
    ax.axhline(16.0, color="green", linestyle="--", linewidth=1, label="2N = 16 bits")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum rate [bits]")
    ax.set_title("S1: Convergence (weighted_sum, equal, P=40)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "s1_convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
