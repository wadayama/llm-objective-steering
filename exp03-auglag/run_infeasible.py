#!/usr/bin/env python3
"""
exp03b: INFEASIBLE-signal scenario — where signal semantics should matter.

Policy demands sum rate >= 15 bits, which is unreachable at the default
budget cap (max ~13.85 bits at P_total = 40). System 1 will drive mu to
mu_max and report INFEASIBLE. The question is what System 2 does:
  - semantic condition: told that INFEASIBLE means "regulation cannot fix
    this; raise P_total, relax the constraint, or change objective"
  - plain condition: sees the same INFEASIBLE label with no explanation

1 seed per condition, 200 steps. Descriptive analysis (call-by-call actions).

Run:  uv run python run_infeasible.py
"""

import json
import sys
import time
from pathlib import Path

from openai import OpenAI

from run_batch import LLM_BASE_URL, LLM_TIMEOUT, run_episode

POLICY_INF = ("Minimize total transmit power while keeping the total data "
              "rate above 15 bits")
RESULTS = Path(__file__).parent / "results"


def main():
    client = OpenAI(base_url=LLM_BASE_URL, api_key="lm-studio", timeout=LLM_TIMEOUT)
    models = client.models.list()
    if not models.data:
        sys.exit("No model loaded in LM Studio")
    model_name = models.data[0].id
    print(f"model: {model_name}")
    print(f"policy: {POLICY_INF}")

    t0 = time.time()
    out = {}
    for cond in ["semantic", "plain"]:
        epfile = RESULTS / f"episode_inf_{cond}_s42.json"
        if epfile.exists():
            with open(epfile) as f:
                log = json.load(f)
            print(f"=== {cond} (checkpoint found) ===")
        else:
            print(f"=== {cond} ===", flush=True)
            log = run_episode(client, model_name, cond, seed=42, steps=200,
                              policy=POLICY_INF)
            with open(epfile, "w") as f:
                json.dump(log, f)
        out[cond] = log
        print(f"  final: SR={log['final_sum_rate_bits']:.2f} bits, "
              f"power={log['final_total_power']:.2f}, parse_err={log['parse_errors']}")
        for c in log["calls"]:
            print(f"    step {c['step']:3d}: {c['status']:10s} fam={c['family']:12s} "
                  f"tau={c['tau']} P_cap={c['p_applied']:.1f} mu={c['mu']:.2f}")

    with open(RESULTS / "results_infeasible.json", "w") as f:
        json.dump({"policy": POLICY_INF, "logs": out}, f, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
