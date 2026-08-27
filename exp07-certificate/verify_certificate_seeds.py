"""The certificate check of verify_certificate.py, repeated over seeds.

Same protocol, one run per seed, reporting the settled power and the
gap-indicator margin as mean +/- std across seeds rather than from a
single run.  No LLM is involved: this audits System 1 alone.
"""

import argparse
import json

import numpy as np

from verify_certificate import run, TAU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--out", type=str, default="results/verify_certificate_seeds.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    per_seed = []
    for seed in seeds:
        ref, rows = run(TAU, seed=seed)
        settled = [r for r in rows if r["status"] == "CONVERGED"]
        rec = dict(
            seed=seed,
            p_star=ref["total_power"],
            n_intervals=len(rows),
            n_settled=len(settled),
            holds_all=all(r["gap_bound"] >= r["true_gap"] for r in rows),
            n_violations=sum(1 for r in rows if r["gap_bound"] < r["true_gap"]),
            settled_power_mean=float(np.mean([r["power"] for r in settled])) if settled else None,
            settled_power_std=float(np.std([r["power"] for r in settled])) if settled else None,
            settled_true_gap_max=max((r["true_gap"] for r in settled), default=None),
            settled_bound_min=min((r["gap_bound"] for r in settled), default=None),
            settled_bound_max=max((r["gap_bound"] for r in settled), default=None),
            final_power=rows[-1]["power"],
        )
        per_seed.append(rec)
        print(f"seed {seed}: settled power {rec['settled_power_mean']:.3f} "
              f"+/- {rec['settled_power_std']:.3f}, "
              f"{rec['n_violations']}/{rec['n_intervals']} intervals optimistic",
              flush=True)

    pw = np.array([r["settled_power_mean"] for r in per_seed])
    p_star = per_seed[0]["p_star"]
    summary = dict(
        seeds=seeds,
        p_star=p_star,
        power_across_seeds_mean=float(pw.mean()),
        power_across_seeds_std=float(pw.std(ddof=1)),
        rel_error_pct=float(100 * abs(pw.mean() - p_star) / p_star),
        total_intervals=sum(r["n_intervals"] for r in per_seed),
        total_settled=sum(r["n_settled"] for r in per_seed),
        total_optimistic=sum(r["n_violations"] for r in per_seed),
        settled_true_gap_max=max(r["settled_true_gap_max"] for r in per_seed),
        settled_bound_min=min(r["settled_bound_min"] for r in per_seed),
        settled_bound_max=max(r["settled_bound_max"] for r in per_seed),
    )
    print("\n--- across seeds ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    json.dump({"summary": summary, "per_seed": per_seed}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
