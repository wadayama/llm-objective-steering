#!/usr/bin/env python3
"""
exp03: augmented-Lagrangian System 1 + declaring System 2, end to end.
======================================================================

Same servo policy and protocol as exp02 ("Minimize total transmit power
while keeping the total data rate above 10 bits"; 300 steps, LLM every 10,
K=8 history, 3 seeds), but now the LLM can DECLARE the constraint and
System 1 enforces it via the PHR augmented Lagrangian, reporting certified
status signals back.

Conditions:
  semantic -- status signals explained in the system prompt (meaning + how to act)
  plain    -- same signals in the user message, no explanation

Run:  uv run python run_batch.py           (both conditions, 3 seeds)
      uv run python run_batch.py --smoke   (semantic, 1 seed, 60 steps)
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
    QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
    optimize_step, measure_mi, kkt_residuals, classify_status,
    STATUS_TRANSIENT,
)
from steering import (
    ControlHistory, build_user_message, compute_kpis, ema_merge_spec,
    parse_response, system_prompt, NATS2BITS,
)

POLICY = "Minimize total transmit power while keeping the total data rate above 10 bits"
CONDITIONS = ["semantic", "plain"]
SEEDS = [42, 43, 44]
K_HIST = 8
STEPS = 300
CALL_EVERY = 10
TRACK_EVERY = 5
SECOND_HALF = 150
TARGET_BITS = 10.0
N_MC_GRAD = 8000
N_MC_STATE = 20_000
N_MC_FINAL = 100_000
EMA_ALPHA_P = 0.5
EMA_ALPHA_PARAM = 0.5
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_TIMEOUT = 120.0

device = torch.device("cpu")
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def status_line(status, res, al):
    c = (f"sum_rate >= {al.tau_bits:.1f} bits" if al.tau_bits is not None
         else "(none)")
    return (f"{status} | gap_bound = {res['gap_bound']:.3f} bits | "
            f"constraint {c} | shadow_price mu = {al.mu:.3f}")


def run_episode(client, model_name, condition: str, seed: int,
                steps: int = STEPS, verbose: bool = False,
                policy: str = POLICY) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
    h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)

    p_total = P_TOTAL_DEFAULT
    lam = torch.ones(N) * math.sqrt(p_total / N)
    spec = ObjectiveSpec()
    al = AugLagState()
    status = STATUS_TRANSIENT
    res = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
    hist = ControlHistory(K_HIST)
    sys_prompt = system_prompt(condition)
    lam_interval = lam.clone()

    log = {"ts_steps": [], "ts_sum_rate": [], "ts_total_power": [],
           "ts_budget": [], "ts_mu": [], "ts_status": [], "calls": []}
    call_no = 0
    parse_errors = 0

    for step in range(steps):
        if step % CALL_EVERY == 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            powers = (lam ** 2).numpy()
            sr_bits = float(mi.sum() * NATS2BITS)

            # ---- System 1 dual update + certified signals ----
            if step > 0:
                al.dual_update(sr_bits)
                res = kkt_residuals(sr_bits, lam, lam_interval, al)
                status = classify_status(res, al, status)
                lam_interval = lam.clone()

            kpis = compute_kpis(mi, powers, p_total)
            cdesc = (f"sum_rate>={al.tau_bits:.1f}" if al.tau_bits is not None
                     else "(none)")
            msg = build_user_message(
                h_abs2, mi, powers, p_total, spec.describe(), cdesc,
                status_line(status, res, al), policy,
                history_block=hist.render(kpis),
            )
            if verbose and call_no == 2:
                print("---- sample user message (call 2) ----")
                print(msg)
                print("---------------------------------------")

            t0 = time.time()
            changed = {"objective": False, "constraint": False, "p_total": False}
            err = False
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": msg}],
                    temperature=0.0, max_tokens=2048,
                )
                raw = resp.choices[0].message.content.strip()
                spec_new, tau_new, p_cmd, _ = parse_response(raw)
                changed["objective"] = spec_new.family != spec.family
                old_tau = al.tau_bits
                if tau_new != old_tau:
                    al.set_constraint(tau_new)
                    changed["constraint"] = True
                if p_cmd is not None:
                    p_new = EMA_ALPHA_P * p_cmd + (1 - EMA_ALPHA_P) * p_total
                    changed["p_total"] = abs(p_new - p_total) > 1e-6
                    p_total = p_new
                spec = ema_merge_spec(spec_new, spec, EMA_ALPHA_PARAM, p_total)
            except Exception as e:
                parse_errors += 1
                err = True
                if verbose:
                    print(f"    call {call_no} error: {str(e)[:80]}")
            latency = time.time() - t0
            hist.add(call_no, spec.describe(), cdesc, kpis, status)
            log["calls"].append({
                "step": step, "family": spec.family, "tau": al.tau_bits,
                "p_applied": p_total, "status": status, "mu": al.mu,
                "changed": changed, "latency_s": latency, "error": err,
            })
            call_no += 1

        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, al=al)

        if step % TRACK_EVERY == 0 or step == steps - 1:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            log["ts_steps"].append(step)
            log["ts_sum_rate"].append(float(mi.sum() * NATS2BITS))
            log["ts_total_power"].append(float((lam ** 2).sum()))
            log["ts_budget"].append(p_total)
            log["ts_mu"].append(al.mu)
            log["ts_status"].append(status)

    mi_final = measure_mi(lam, h_re, est, N_MC_FINAL, device)
    log["final_sum_rate_bits"] = float(mi_final.sum() * NATS2BITS)
    log["final_total_power"] = float((lam ** 2).sum())
    log["parse_errors"] = parse_errors
    log["condition"] = condition
    log["seed"] = seed
    return log


def episode_metrics(log: dict) -> dict:
    ts = np.array(log["ts_steps"])
    sr = np.array(log["ts_sum_rate"])
    tp = np.array(log["ts_total_power"])
    mask = ts >= SECOND_HALF
    meddling = sum(1 for c in log["calls"]
                   if c["step"] >= SECOND_HALF and any(c["changed"].values()))
    return {
        "violation_rate": float(np.mean(sr[mask] < TARGET_BITS)),
        "hard_violation_rate": float(np.mean(sr[mask] < TARGET_BITS - 0.25)),
        "mean_shortfall_bits": float(np.mean(np.maximum(0, TARGET_BITS - sr[mask]))),
        "mean_power": float(np.mean(tp[mask])),
        "mean_sum_rate": float(np.mean(sr[mask])),
        "meddling_actions_2nd_half": meddling,
        "parse_errors": log["parse_errors"],
    }


def make_plots(all_logs):
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4.6 * len(CONDITIONS), 6.5),
                             sharex=True, sharey="row")
    for j, c in enumerate(CONDITIONS):
        for seed in SEEDS:
            log = all_logs[(c, seed)]
            axes[0, j].plot(log["ts_steps"], log["ts_total_power"], alpha=0.8,
                            label=f"seed {seed}")
            axes[1, j].plot(log["ts_steps"], log["ts_sum_rate"], alpha=0.8)
        axes[0, j].set_title(f"{c}", fontsize=10)
        axes[0, j].grid(True, alpha=0.3)
        axes[1, j].axhline(TARGET_BITS, color="red", linestyle="--", linewidth=1)
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Total power")
    axes[1, 0].set_ylabel("Sum rate [bits]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'exp03 (augmented Lagrangian): "{POLICY}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "e2e_trajectories.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    client = OpenAI(base_url=LLM_BASE_URL, api_key="lm-studio", timeout=LLM_TIMEOUT)
    models = client.models.list()
    if not models.data:
        sys.exit("No model loaded in LM Studio")
    model_name = models.data[0].id
    print(f"model: {model_name}")

    if args.smoke:
        log = run_episode(client, model_name, "semantic", 42, steps=60, verbose=True)
        print(f"smoke OK: parse errors {log['parse_errors']}")
        print(f"final: SR={log['final_sum_rate_bits']:.3f} bits, "
              f"power={log['final_total_power']:.2f}")
        print("calls:", [(c["family"], c["tau"], c["status"]) for c in log["calls"]])
        return

    t0 = time.time()
    all_logs = {}
    for c in CONDITIONS:
        for seed in SEEDS:
            epfile = RESULTS / f"episode_{c}_s{seed}.json"
            if epfile.exists():
                with open(epfile) as f:
                    all_logs[(c, seed)] = json.load(f)
                print(f"=== {c} seed={seed} (checkpoint found, skipping) ===", flush=True)
                continue
            print(f"=== {c} seed={seed} ===", flush=True)
            log = run_episode(client, model_name, c, seed)
            with open(epfile, "w") as f:
                json.dump(log, f)
            all_logs[(c, seed)] = log
            m = episode_metrics(log)
            print(f"    viol={m['violation_rate']:.2f}  hard={m['hard_violation_rate']:.2f}  "
                  f"shortfall={m['mean_shortfall_bits']:.3f}  P={m['mean_power']:.2f}  "
                  f"SR={m['mean_sum_rate']:.2f}  meddling={m['meddling_actions_2nd_half']}",
                  flush=True)

    make_plots(all_logs)
    out = {
        "experiment": "exp03-auglag-e2e",
        "policy": POLICY,
        "conditions": CONDITIONS,
        "metrics": {f"{c}_seed{s}": episode_metrics(all_logs[(c, s)])
                    for c in CONDITIONS for s in SEEDS},
        "logs": {f"{c}_seed{s}": all_logs[(c, s)] for c in CONDITIONS for s in SEEDS},
    }
    with open(RESULTS / "results_e2e.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
