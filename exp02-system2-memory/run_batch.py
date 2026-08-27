#!/usr/bin/env python3
"""
exp02-system2-memory: does giving System 2 a memory reduce servo oscillation?
==============================================================================

Ablation over the history-window size K in {0, 3, 8} on a servo-type policy
that the memoryless conference-version System 2 handled poorly (oscillating
P_total): "Minimize total transmit power while keeping the total data rate
above 10 bits."

Protocol (per K, per seed):
  - Start at the default operating point (weighted_sum equal, P_total = 40).
  - Run 300 System-1 steps; call the LLM every 10 steps (30 calls).
  - Before each call the state is measured (20k MC samples) and, for K > 0,
    the last K (action, state) records plus the measured effect of the most
    recent action are included in the prompt.
  - EMA smoothing (alpha = 0.5 on P_total and on within-family parameters)
    is identical across all K, so the only ablated variable is the memory.

Metrics (computed on the second half, steps >= 150):
  - violation rate: fraction of tracked steps with sum rate < 10 bits
  - mean total power (lower is better subject to the constraint)
  - std of the applied budget P_total (oscillation amplitude)
  - number of direction reversals in the LLM's P_total commands

Run:  uv run python run_batch.py            (full: 3 K x 3 seeds)
      uv run python run_batch.py --smoke    (quick pipeline check)
Requires LM Studio with a loaded model on localhost:1234.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

from core import (
    N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
    QPSKMonteCarloEstimator, ObjectiveSpec, optimize_step, measure_mi,
)
from steering import (
    ControlHistory, build_user_message, compute_kpis, ema_merge_spec,
    parse_response, system_prompt, NATS2BITS,
)

# ---- experiment configuration ----
POLICY = "Minimize total transmit power while keeping the total data rate above 10 bits"
KS = [0, 3, 8]
SEEDS = [42, 43, 44]
STEPS = 300
CALL_EVERY = 10
TRACK_EVERY = 5
SECOND_HALF = 150          # metrics window start (step index)
TARGET_BITS = 10.0
N_MC_GRAD = 8000
N_MC_STATE = 20_000
N_MC_FINAL = 100_000
EMA_ALPHA_P = 0.5
EMA_ALPHA_PARAM = 0.5
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_TIMEOUT = 120.0
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2048

device = torch.device("cpu")
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def run_episode(client, model_name, K: int, seed: int, steps: int = STEPS,
                verbose: bool = False, variant: str = "V0",
                ema_p: float | None = EMA_ALPHA_P,
                ema_param: float | None = EMA_ALPHA_PARAM,
                slew_frac: float | None = None,
                actuator: str = "ema") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
    h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)

    p_total = P_TOTAL_DEFAULT
    lam = torch.ones(N) * math.sqrt(p_total / N)
    spec = ObjectiveSpec()  # weighted_sum, equal
    hist = ControlHistory(K)
    sys_prompt = system_prompt(with_history=(K > 0), variant=variant,
                               actuator=actuator)

    log = {
        "ts_steps": [], "ts_sum_rate": [], "ts_total_power": [], "ts_budget": [],
        "calls": [],  # per call: step, p_cmd, p_applied, family, latency, parse_error
    }
    call_no = 0
    parse_errors = 0

    for step in range(steps):
        if step % CALL_EVERY == 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            powers = (lam ** 2).numpy()
            kpis = compute_kpis(mi, powers, p_total)
            msg = build_user_message(
                h_abs2, mi, powers, p_total, spec.describe(), POLICY,
                history_block=hist.render(kpis),
            )
            if verbose and call_no == 2:
                print("---- sample user message (call 2) ----")
                print(msg)
                print("---------------------------------------")
            t0 = time.time()
            p_cmd = None
            predicted = None
            family = spec.family
            err = False
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": msg}],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content.strip()
                spec_new, p_cmd, predicted, _ = parse_response(raw)
                if p_cmd is not None:
                    p_new = (ema_p * p_cmd + (1 - ema_p) * p_total
                             if ema_p is not None else p_cmd)
                    if slew_frac is not None:
                        max_delta = slew_frac * p_total
                        p_new = min(max(p_new, p_total - max_delta),
                                    p_total + max_delta)
                    p_total = p_new
                merge_alpha = ema_param if ema_param is not None else 1.0
                spec = ema_merge_spec(spec_new, spec, merge_alpha, p_total)
                family = spec.family
            except Exception as e:
                parse_errors += 1
                err = True
                if verbose:
                    print(f"    call {call_no} error: {str(e)[:80]}")
            latency = time.time() - t0
            hist.add(call_no, spec.describe(), p_cmd, kpis, predicted=predicted)
            log["calls"].append({
                "step": step, "p_cmd": p_cmd, "p_applied": p_total,
                "family": family, "latency_s": latency, "error": err,
                "predicted": predicted,
            })
            call_no += 1

        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, projection="sphere")

        if step % TRACK_EVERY == 0 or step == steps - 1:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            log["ts_steps"].append(step)
            log["ts_sum_rate"].append(float(mi.sum() * NATS2BITS))
            log["ts_total_power"].append(float((lam ** 2).sum()))
            log["ts_budget"].append(p_total)

    mi_final = measure_mi(lam, h_re, est, N_MC_FINAL, device)
    log["final_sum_rate_bits"] = float(mi_final.sum() * NATS2BITS)
    log["final_total_power"] = float((lam ** 2).sum())
    log["parse_errors"] = parse_errors
    log["K"] = K
    log["seed"] = seed
    log["variant"] = variant
    return log


def episode_metrics(log: dict) -> dict:
    ts = np.array(log["ts_steps"])
    sr = np.array(log["ts_sum_rate"])
    tp = np.array(log["ts_total_power"])
    bg = np.array(log["ts_budget"])
    mask = ts >= SECOND_HALF

    cmds = [c["p_cmd"] for c in log["calls"] if c["p_cmd"] is not None]
    reversals = 0
    if len(cmds) >= 3:
        deltas = np.diff(np.array(cmds))
        deltas = deltas[np.abs(deltas) > 1e-9]
        reversals = int(np.sum(np.diff(np.sign(deltas)) != 0))

    return {
        "violation_rate": float(np.mean(sr[mask] < TARGET_BITS)),
        "mean_power": float(np.mean(tp[mask])),
        "mean_sum_rate": float(np.mean(sr[mask])),
        "budget_std": float(np.std(bg[mask])),
        "cmd_reversals": reversals,
        "parse_errors": log["parse_errors"],
    }


def make_plots(all_logs: dict, ks: list[int], seeds: list[int]):
    fig, axes = plt.subplots(2, len(ks), figsize=(4.2 * len(ks), 6.5),
                             sharex=True, sharey="row")
    if len(ks) == 1:
        axes = axes.reshape(2, 1)
    for j, K in enumerate(ks):
        for seed in seeds:
            log = all_logs[(K, seed)]
            axes[0, j].plot(log["ts_steps"], log["ts_budget"], alpha=0.8,
                            label=f"seed {seed}")
            axes[1, j].plot(log["ts_steps"], log["ts_sum_rate"], alpha=0.8)
        axes[0, j].set_title(f"K = {K}" + (" (memoryless)" if K == 0 else ""))
        axes[0, j].grid(True, alpha=0.3)
        axes[1, j].axhline(TARGET_BITS, color="red", linestyle="--",
                           linewidth=1, label=f"target {TARGET_BITS:.0f} bits")
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Budget $P_{total}$")
    axes[1, 0].set_ylabel("Sum rate [bits]")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(f'Servo policy: "{POLICY}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "trajectories.pdf")
    plt.close(fig)

    # metrics summary bars
    names = [("violation_rate", "Constraint violation rate\n(2nd half)"),
             ("mean_power", "Mean total power\n(2nd half)"),
             ("budget_std", "Budget std\n(oscillation)"),
             ("cmd_reversals", "P_total command\nreversals")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    x = np.arange(len(ks))
    for ax, (key, title) in zip(axes, names):
        means, spreads = [], []
        for K in ks:
            vals = [episode_metrics(all_logs[(K, s)])[key] for s in seeds]
            means.append(np.mean(vals))
            spreads.append((np.max(vals) - np.min(vals)) / 2)
        ax.bar(x, means, 0.55, yerr=spreads, capsize=4, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={K}" for K in ks])
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "metrics_summary.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline check: K=3, 1 seed, 40 steps")
    args = ap.parse_args()

    client = OpenAI(base_url=LLM_BASE_URL, api_key="lm-studio", timeout=LLM_TIMEOUT)
    models = client.models.list()
    if not models.data:
        sys.exit("No model loaded in LM Studio")
    model_name = models.data[0].id
    print(f"model: {model_name}")
    print(f"policy: {POLICY}")

    t0 = time.time()
    if args.smoke:
        log = run_episode(client, model_name, K=3, seed=42, steps=40, verbose=True)
        lat = [c["latency_s"] for c in log["calls"]]
        print(f"smoke OK: {len(log['calls'])} calls, "
              f"mean latency {np.mean(lat):.1f}s, parse errors {log['parse_errors']}")
        print(f"final: sum_rate={log['final_sum_rate_bits']:.3f} bits, "
              f"power={log['final_total_power']:.2f}")
        return

    all_logs = {}
    for K in KS:
        for seed in SEEDS:
            epfile = RESULTS / f"episode_K{K}_s{seed}.json"
            if epfile.exists():
                with open(epfile) as f:
                    log = json.load(f)
                print(f"=== K={K} seed={seed} (checkpoint found, skipping) ===", flush=True)
            else:
                print(f"=== K={K} seed={seed} ===", flush=True)
                log = run_episode(client, model_name, K=K, seed=seed)
                with open(epfile, "w") as f:
                    json.dump(log, f)
            all_logs[(K, seed)] = log
            m = episode_metrics(log)
            print(f"    violation={m['violation_rate']:.2f}  "
                  f"mean_power={m['mean_power']:.2f}  "
                  f"budget_std={m['budget_std']:.2f}  "
                  f"reversals={m['cmd_reversals']}  "
                  f"parse_err={m['parse_errors']}", flush=True)

    make_plots(all_logs, KS, SEEDS)

    out = {
        "experiment": "exp02-system2-memory",
        "policy": POLICY,
        "config": {
            "KS": KS, "seeds": SEEDS, "steps": STEPS, "call_every": CALL_EVERY,
            "target_bits": TARGET_BITS, "ema_alpha_p": EMA_ALPHA_P,
            "ema_alpha_param": EMA_ALPHA_PARAM, "n_mc_grad": N_MC_GRAD,
            "n_mc_state": N_MC_STATE, "model": model_name,
            "temperature": LLM_TEMPERATURE, "h_abs2": H_ABS2_DEFAULT,
        },
        "metrics": {f"K{K}_seed{s}": episode_metrics(all_logs[(K, s)])
                    for K in KS for s in SEEDS},
        "logs": {f"K{K}_seed{s}": all_logs[(K, s)] for K in KS for s in SEEDS},
    }
    with open(RESULTS / "results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
