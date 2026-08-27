#!/usr/bin/env python3
"""
exp02b: prompt-improvement ablation at fixed K=8.

Variants (same policy, model, seeds, and EMA as exp02):
  V0: exp02 prompt (history only)          -- reuses episode_K8_s*.json
  V1: V0 + control literacy: actuator (EMA) disclosure, hard-constraint
      priority with safety margin, measurement-noise deadband,
      anti-oscillation guidance
  V2: V1 + explicit feedback procedure: in-context gain estimation and a
      mandatory "predicted_sum_rate" field whose error is fed back via the
      history ("you predicted X, measured Y")

Run:  uv run python run_prompt_ablation.py           (V1+V2, 3 seeds each)
      uv run python run_prompt_ablation.py --smoke   (V2, 1 seed, 40 steps)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

from run_batch import (
    LLM_BASE_URL, LLM_TIMEOUT, POLICY, SEEDS, TARGET_BITS,
    episode_metrics, run_episode,
)

K_FIXED = 8
VARIANTS = ["V0", "V1", "V2"]
RESULTS = Path(__file__).parent / "results"


def episode_file(variant: str, seed: int) -> Path:
    if variant == "V0":  # exp02's K=8 episodes
        return RESULTS / f"episode_K{K_FIXED}_s{seed}.json"
    return RESULTS / f"episode_{variant}_K{K_FIXED}_s{seed}.json"


def prediction_stats(log: dict) -> dict:
    """Calibration of predicted_sum_rate vs the measurement at the next call."""
    calls = log["calls"]
    ts = dict(zip(log["ts_steps"], log["ts_sum_rate"]))
    pairs = []
    for a, b in zip(calls, calls[1:]):
        if a.get("predicted") is not None and b["step"] in ts:
            pairs.append((a["predicted"], ts[b["step"]]))
    if not pairs:
        return {"n": 0}
    pred = np.array([p for p, _ in pairs])
    meas = np.array([m for _, m in pairs])
    return {"n": len(pairs),
            "mae": float(np.mean(np.abs(pred - meas))),
            "bias": float(np.mean(pred - meas))}


def make_plots(all_logs: dict):
    # trajectories: budget (top) and sum rate (bottom) per variant column
    fig, axes = plt.subplots(2, len(VARIANTS), figsize=(4.2 * len(VARIANTS), 6.5),
                             sharex=True, sharey="row")
    titles = {"V0": "V0: history only", "V1": "V1: +control literacy",
              "V2": "V2: +procedure & prediction"}
    for j, v in enumerate(VARIANTS):
        for seed in SEEDS:
            log = all_logs[(v, seed)]
            axes[0, j].plot(log["ts_steps"], log["ts_budget"], alpha=0.8,
                            label=f"seed {seed}")
            axes[1, j].plot(log["ts_steps"], log["ts_sum_rate"], alpha=0.8)
        axes[0, j].set_title(titles[v], fontsize=10)
        axes[0, j].grid(True, alpha=0.3)
        axes[1, j].axhline(TARGET_BITS, color="red", linestyle="--", linewidth=1,
                           label=f"target {TARGET_BITS:.0f} bits")
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Budget $P_{total}$")
    axes[1, 0].set_ylabel("Sum rate [bits]")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(f'K={K_FIXED}, servo policy: "{POLICY}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "prompt_trajectories.pdf")
    plt.close(fig)

    names = [("violation_rate", "Constraint violation rate\n(2nd half)"),
             ("mean_power", "Mean total power\n(2nd half)"),
             ("mean_sum_rate", "Mean sum rate [bits]\n(2nd half)"),
             ("cmd_reversals", "P_total command\nreversals")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    x = np.arange(len(VARIANTS))
    for ax, (key, title) in zip(axes, names):
        means, spreads = [], []
        for v in VARIANTS:
            vals = [episode_metrics(all_logs[(v, s)])[key] for s in SEEDS]
            means.append(np.mean(vals))
            spreads.append((np.max(vals) - np.min(vals)) / 2)
        ax.bar(x, means, 0.55, yerr=spreads, capsize=4, color="steelblue")
        if key == "mean_sum_rate":
            ax.axhline(TARGET_BITS, color="red", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANTS)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "prompt_metrics.pdf")
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
        log = run_episode(client, model_name, K=K_FIXED, seed=42, steps=40,
                          verbose=True, variant="V2")
        preds = [c.get("predicted") for c in log["calls"]]
        print(f"smoke OK: {len(log['calls'])} calls, predictions: {preds}, "
              f"parse errors {log['parse_errors']}")
        print(f"final: sum_rate={log['final_sum_rate_bits']:.3f} bits, "
              f"power={log['final_total_power']:.2f}")
        return

    t0 = time.time()
    all_logs = {}
    for v in VARIANTS:
        for seed in SEEDS:
            epfile = episode_file(v, seed)
            if epfile.exists():
                with open(epfile) as f:
                    all_logs[(v, seed)] = json.load(f)
                print(f"=== {v} seed={seed} (checkpoint found, skipping) ===", flush=True)
                continue
            print(f"=== {v} seed={seed} ===", flush=True)
            log = run_episode(client, model_name, K=K_FIXED, seed=seed, variant=v)
            with open(epfile, "w") as f:
                json.dump(log, f)
            all_logs[(v, seed)] = log
            m = episode_metrics(log)
            ps = prediction_stats(log)
            extra = f"  pred_mae={ps['mae']:.2f} bias={ps['bias']:+.2f}" if ps["n"] else ""
            print(f"    violation={m['violation_rate']:.2f}  "
                  f"mean_power={m['mean_power']:.2f}  "
                  f"mean_SR={m['mean_sum_rate']:.2f}  "
                  f"reversals={m['cmd_reversals']}{extra}", flush=True)

    make_plots(all_logs)

    out = {
        "experiment": "exp02b-prompt-ablation",
        "policy": POLICY,
        "K": K_FIXED,
        "variants": VARIANTS,
        "metrics": {f"{v}_seed{s}": episode_metrics(all_logs[(v, s)])
                    for v in VARIANTS for s in SEEDS},
        "prediction_stats": {f"{v}_seed{s}": prediction_stats(all_logs[(v, s)])
                             for v in VARIANTS for s in SEEDS},
        "logs": {f"{v}_seed{s}": all_logs[(v, s)] for v in VARIANTS for s in SEEDS},
    }
    with open(RESULTS / "results_prompt.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
