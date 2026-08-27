#!/usr/bin/env python3
"""
exp02c: is the EMA actuator still needed once System 2 has memory?
===================================================================

2x2 factorial {EMA on/off} x {slew-rate limit on/off} at fixed K=8 with the
V1 (control-literacy) prompt, whose actuator paragraph is rewritten to match
each condition truthfully. Plus one bonus arm re-running the V2 prompt
without EMA, to test whether V2's optimistic prediction bias in exp02b was
an artifact of the undisclosed-in-effect smoothing.

Conditions (3 seeds each):
  ema        EMA a=0.5, no slew           -- reuses exp02b V1 episodes
  direct     no EMA, no slew              -- commands apply verbatim
  slew       no EMA, +/-30% budget change cap per call
  ema_slew   EMA a=0.5 AND +/-30% cap
  v2_direct  V2 prompt, no EMA, no slew   -- prediction-calibration check

Run:  uv run python run_c_ablation.py           (all missing episodes)
      uv run python run_c_ablation.py --smoke   (direct, 1 seed, 40 steps)
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
from run_prompt_ablation import prediction_stats

K_FIXED = 8
SLEW = 0.3
CONDS = {
    # name: (variant, ema_p, ema_param, slew_frac, actuator)
    "ema":       ("V1", 0.5, 0.5, None, "ema"),
    "direct":    ("V1", None, None, None, "direct"),
    "slew":      ("V1", None, None, SLEW, "slew"),
    "ema_slew":  ("V1", 0.5, 0.5, SLEW, "ema_slew"),
    "v2_direct": ("V2", None, None, None, "direct"),
}
RESULTS = Path(__file__).parent / "results"


def episode_file(cond: str, seed: int) -> Path:
    if cond == "ema":  # exp02b V1 episodes are exactly this condition
        return RESULTS / f"episode_V1_K{K_FIXED}_s{seed}.json"
    return RESULTS / f"episode_C_{cond}_s{seed}.json"


def make_plots(all_logs: dict):
    conds = list(CONDS.keys())
    titles = {"ema": "EMA / no cap (=V1)", "direct": "direct / no cap",
              "slew": "direct / 30% cap", "ema_slew": "EMA / 30% cap",
              "v2_direct": "V2 prompt, direct"}
    fig, axes = plt.subplots(2, len(conds), figsize=(3.6 * len(conds), 6.5),
                             sharex=True, sharey="row")
    for j, c in enumerate(conds):
        for seed in SEEDS:
            log = all_logs[(c, seed)]
            axes[0, j].plot(log["ts_steps"], log["ts_budget"], alpha=0.8,
                            label=f"seed {seed}")
            axes[1, j].plot(log["ts_steps"], log["ts_sum_rate"], alpha=0.8)
        axes[0, j].set_title(titles[c], fontsize=9)
        axes[0, j].grid(True, alpha=0.3)
        axes[1, j].axhline(TARGET_BITS, color="red", linestyle="--", linewidth=1)
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Budget $P_{total}$")
    axes[1, 0].set_ylabel("Sum rate [bits]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'K={K_FIXED}, servo policy: "{POLICY}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "c_trajectories.pdf")
    plt.close(fig)

    names = [("violation_rate", "Constraint violation rate\n(2nd half)"),
             ("mean_sum_rate", "Mean sum rate [bits]\n(2nd half)"),
             ("mean_power", "Mean total power\n(2nd half)"),
             ("cmd_reversals", "P_total command\nreversals")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    x = np.arange(len(conds))
    for ax, (key, title) in zip(axes, names):
        means, spreads = [], []
        for c in conds:
            vals = [episode_metrics(all_logs[(c, s)])[key] for s in SEEDS]
            means.append(np.mean(vals))
            spreads.append((np.max(vals) - np.min(vals)) / 2)
        ax.bar(x, means, 0.55, yerr=spreads, capsize=4, color="steelblue")
        if key == "mean_sum_rate":
            ax.axhline(TARGET_BITS, color="red", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=20, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "c_metrics.pdf")
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
        variant, ema_p, ema_param, slew, actuator = CONDS["direct"]
        log = run_episode(client, model_name, K=K_FIXED, seed=42, steps=40,
                          verbose=True, variant=variant, ema_p=ema_p,
                          ema_param=ema_param, slew_frac=slew, actuator=actuator)
        print(f"smoke OK: parse errors {log['parse_errors']}, "
              f"final sum_rate={log['final_sum_rate_bits']:.3f}, "
              f"power={log['final_total_power']:.2f}")
        cmds = [(c["p_cmd"], round(c["p_applied"], 1)) for c in log["calls"]]
        print("(cmd, applied):", cmds)
        return

    t0 = time.time()
    all_logs = {}
    for cond, (variant, ema_p, ema_param, slew, actuator) in CONDS.items():
        for seed in SEEDS:
            epfile = episode_file(cond, seed)
            if epfile.exists():
                with open(epfile) as f:
                    all_logs[(cond, seed)] = json.load(f)
                print(f"=== {cond} seed={seed} (checkpoint found, skipping) ===", flush=True)
                continue
            print(f"=== {cond} seed={seed} ===", flush=True)
            log = run_episode(client, model_name, K=K_FIXED, seed=seed,
                              variant=variant, ema_p=ema_p, ema_param=ema_param,
                              slew_frac=slew, actuator=actuator)
            with open(epfile, "w") as f:
                json.dump(log, f)
            all_logs[(cond, seed)] = log
            m = episode_metrics(log)
            ps = prediction_stats(log)
            extra = f"  pred_mae={ps['mae']:.2f} bias={ps['bias']:+.2f}" if ps["n"] else ""
            print(f"    violation={m['violation_rate']:.2f}  "
                  f"mean_power={m['mean_power']:.2f}  "
                  f"mean_SR={m['mean_sum_rate']:.2f}  "
                  f"reversals={m['cmd_reversals']}{extra}", flush=True)

    make_plots(all_logs)

    out = {
        "experiment": "exp02c-ema-ablation",
        "policy": POLICY,
        "K": K_FIXED,
        "conditions": {c: {"variant": v[0], "ema_p": v[1], "ema_param": v[2],
                           "slew_frac": v[3], "actuator": v[4]}
                       for c, v in CONDS.items()},
        "metrics": {f"{c}_seed{s}": episode_metrics(all_logs[(c, s)])
                    for c in CONDS for s in SEEDS},
        "prediction_stats": {f"{c}_seed{s}": prediction_stats(all_logs[(c, s)])
                             for c in CONDS for s in SEEDS},
        "logs": {f"{c}_seed{s}": all_logs[(c, s)] for c in CONDS for s in SEEDS},
    }
    with open(RESULTS / "results_c.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
