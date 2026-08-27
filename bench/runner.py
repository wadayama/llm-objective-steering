#!/usr/bin/env python3
"""
Benchmark v0.1 runner.

Modes per scenario:
  translation -- one LLM call against a fixed synthetic state; the returned
                 declaration is scored by the scenario's check()
  episode     -- full closed loop (LLM every 10 steps, K=8 history, EMA,
                 augmented Lagrangian), plus an ORACLE run (System 1 alone
                 with the scenario's hand-written optimal declaration, no
                 LLM) used as the regret reference

Usage:
  uv run python runner.py                 # run everything (checkpointed)
  uv run python runner.py --only S1,T4    # subset
  uv run python runner.py --report        # rebuild scoreboard from checkpoints
Outputs: results/bench_<id>_*.json, results/scoreboard.json, results/report.md
"""

import argparse
import urllib.request
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI

from core import (
    N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
    QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
    optimize_step, measure_mi, kkt_residuals, classify_status,
    STATUS_TRANSIENT, STATUS_CONVERGED, STATUS_INFEASIBLE,
)
from steering import (
    ControlHistory, build_user_message, compute_kpis, ema_merge_spec,
    parse_response, system_prompt, NATS2BITS,
)
from scenarios import SCENARIOS, TRANSLATION_STATE

CALL_EVERY = 10
TRACK_EVERY = 5
K_HIST = 8
N_MC_GRAD = 8000
N_MC_STATE = 20_000
N_MC_FINAL = 100_000
EMA_ALPHA_P = 0.5
EMA_ALPHA_PARAM = 0.5
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_TIMEOUT = 120.0

device = torch.device("cpu")
RESULTS = Path(__file__).parent / "results"


# =====================================================================
# LLM plumbing
# =====================================================================
def make_client(model_id: str | None = None):
    client = OpenAI(base_url=LLM_BASE_URL, api_key="lm-studio", timeout=LLM_TIMEOUT)
    if model_id:
        return client, model_id
    # /v1/models lists embeddings and unloaded models too, so its first entry is
    # not necessarily the model under test. LM Studio's own endpoint carries
    # type and load state; fall back to the OpenAI-compatible list without it.
    try:
        url = LLM_BASE_URL.rsplit("/v1", 1)[0] + "/api/v0/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            for m in json.loads(r.read()).get("data", []):
                if m.get("state") == "loaded" and m.get("type") != "embeddings":
                    return client, m["id"]
    except Exception:
        pass
    models = client.models.list()
    if not models.data:
        sys.exit("No model loaded in LM Studio")
    return client, models.data[0].id


def llm_call(client, model, sys_prompt, msg):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": msg}],
        temperature=0.0, max_tokens=2048)
    return parse_response(resp.choices[0].message.content.strip())


# =====================================================================
# Translation scenarios
# =====================================================================
def run_translation(client, model, sc) -> dict:
    st = TRANSLATION_STATE
    msg = build_user_message(
        np.array(H_ABS2_DEFAULT), st["mi_nats"], st["powers"], st["p_total"],
        "weighted_sum(w=[0.12, ...])", "(none)",
        "CONVERGED | gap_bound = 0.050 bits | constraint (none) | shadow_price mu = 0.000",
        sc["policy"])
    try:
        spec, tau, tau_ch, p_cmd, reasoning = llm_call(
            client, model, system_prompt("semantic"), msg)
        spec.validate(st["p_total"])
        ok, note = sc["check"](spec, tau, tau_ch)
        return {"ok": bool(ok), "note": note,
                "declaration": {"family": spec.family, "tau": tau,
                                "tau_ch": tau_ch, "p_cmd": p_cmd},
                "reasoning": reasoning[:200]}
    except Exception as e:
        return {"ok": False, "note": f"parse/call error: {str(e)[:100]}",
                "declaration": None, "reasoning": ""}


# =====================================================================
# Episode machinery
# =====================================================================
def apply_declaration(spec_new, tau, tau_ch, p_cmd, spec_old, al, p_total):
    """Guardrail + EMA application, mirroring the exp05 controller."""
    rejected = (spec_new.family == "min_power" and tau is None and tau_ch is None)
    if rejected:
        return spec_old, p_total, True
    if p_cmd is not None:
        p_total = EMA_ALPHA_P * p_cmd + (1 - EMA_ALPHA_P) * p_total
    spec = ema_merge_spec(spec_new, spec_old, EMA_ALPHA_PARAM, p_total)
    al.set_constraint(tau)
    al.set_channel_constraint(tau_ch)
    return spec, p_total, False


def oracle_spec(oracle) -> tuple[ObjectiveSpec, float | None, float | None]:
    spec = ObjectiveSpec(family=oracle["family"])
    if "targets" in oracle:
        spec.targets = np.asarray(oracle["targets"], dtype=np.float64)
    return spec, oracle.get("tau"), oracle.get("tau_ch")


def run_episode(sc, seed: int, mode: str, client=None, model=None) -> dict:
    """mode: 'llm' (closed loop) or 'oracle' (System 1 with oracle declaration)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
    h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)
    env_schedule = sc.get("env_schedule", {})

    p_total = P_TOTAL_DEFAULT
    lam = torch.ones(N) * math.sqrt(p_total / N)
    al = AugLagState()
    status = STATUS_TRANSIENT
    res = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
    hist = ControlHistory(K_HIST)
    sys_prompt = system_prompt("semantic")

    if mode == "oracle":
        spec, tau, tau_ch = oracle_spec(sc["oracle"])
        spec.validate(p_total)
        al.set_constraint(tau)
        al.set_channel_constraint(tau_ch)
    else:
        spec = ObjectiveSpec()

    log = {"ts_steps": [], "ts_sum_rate": [], "ts_min_mi": [],
           "ts_total_power": [], "ts_status": [], "calls": [],
           "parse_errors": 0, "rejections": 0}
    lam_iv = lam.clone()
    call_no = 0

    for step in range(sc["steps"]):
        if step in env_schedule:
            h_abs2 = np.array(env_schedule[step], dtype=np.float64)
            h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)

        if step % CALL_EVERY == 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            sr = float(mib.sum())
            if step > 0:
                al.dual_update(sr, mib)
                res = kkt_residuals(sr, lam, lam_iv, al, mi_bits_meas=mib)
                status = classify_status(res, al, status)
                lam_iv = lam.clone()

            if mode == "llm":
                kpis = compute_kpis(mi, (lam ** 2).numpy(), p_total)
                parts = []
                if al.tau_bits is not None:
                    parts.append(f"sum_rate>={al.tau_bits:.1f}")
                if al.tau_ch_bits is not None:
                    parts.append(f"each_MI>={al.tau_ch_bits:.2f}")
                cdesc = "; ".join(parts) if parts else "(none)"
                sline = (f"{status} | gap_bound = {res['gap_bound']:.3f} bits | "
                         f"constraint {cdesc} | shadow_price mu = "
                         f"{al.mu + float(al.mu_ch.max()):.3f}")
                msg = build_user_message(
                    h_abs2, mi, (lam ** 2).numpy(), p_total, spec.describe(),
                    cdesc, sline, sc["policy"], history_block=hist.render(kpis))
                family_before = spec.family
                try:
                    spec_new, tau, tau_ch, p_cmd, _ = llm_call(
                        client, model, sys_prompt, msg)
                    spec, p_total, rej = apply_declaration(
                        spec_new, tau, tau_ch, p_cmd, spec, al, p_total)
                    log["rejections"] += int(rej)
                except Exception:
                    log["parse_errors"] += 1
                hist.add(call_no, spec.describe(), cdesc, kpis, status)
                log["calls"].append({"step": step, "family": spec.family,
                                     "switched": spec.family != family_before,
                                     "tau": al.tau_bits, "tau_ch": al.tau_ch_bits,
                                     "p_total": p_total, "status": status})
                call_no += 1

        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, al=al)

        if step % TRACK_EVERY == 0 or step == sc["steps"] - 1:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            log["ts_steps"].append(step)
            log["ts_sum_rate"].append(float(mib.sum()))
            log["ts_min_mi"].append(float(mib.min()))
            log["ts_total_power"].append(float((lam ** 2).sum()))
            log["ts_status"].append(status)

    mi = measure_mi(lam, h_re, est, N_MC_FINAL, device)
    mib = mi * NATS2BITS
    log["final_sum_rate"] = float(mib.sum())
    log["final_min_mi"] = float(mib.min())
    log["final_mi_bits"] = mib.tolist()
    log["final_power"] = float((lam ** 2).sum())
    log["mode"] = mode
    log["seed"] = seed
    return log


# =====================================================================
# Metrics
# =====================================================================
def episode_metrics(sc, log_llm: dict, log_oracle: dict | None) -> dict:
    ts = np.array(log_llm["ts_steps"])
    sr = np.array(log_llm["ts_sum_rate"])
    mn = np.array(log_llm["ts_min_mi"])
    tp = np.array(log_llm["ts_total_power"])
    half = sc["steps"] // 2
    mask = ts >= half
    statuses = log_llm["ts_status"]

    m = {
        "final_sum_rate": log_llm["final_sum_rate"],
        "final_power": log_llm["final_power"],
        "parse_errors": log_llm["parse_errors"],
        "rejections": log_llm["rejections"],
    }
    tau = sc.get("oracle", {}).get("tau")
    tau_ch = sc.get("oracle", {}).get("tau_ch")
    m["violation_rate"] = (float(np.mean(sr[mask] < tau)) if tau is not None else 0.0)
    m["ch_violation_rate"] = (float(np.mean(mn[mask] < tau_ch)) if tau_ch is not None else 0.0)
    m["power_regret"] = (float(np.mean(tp[mask]) - np.mean(
        np.array(log_oracle["ts_total_power"])[np.array(log_oracle["ts_steps"]) >= half]))
        if log_oracle is not None else None)

    # settling: first tracked step in CONVERGED whose remainder is >=80% CONVERGED
    m["settling_step"] = None
    for i, st in enumerate(statuses):
        if st == STATUS_CONVERGED:
            rest = statuses[i:]
            if rest.count(STATUS_CONVERGED) / len(rest) >= 0.8:
                m["settling_step"] = int(ts[i])
                break

    # infeasibility scenario
    m["reached_infeasible"] = STATUS_INFEASIBLE in statuses

    # disturbance scenario
    if sc.get("env_schedule"):
        t_dist = min(sc["env_schedule"].keys())
        post = ts >= t_dist + 20
        m["violation_rate_post"] = (float(np.mean(sr[ts >= t_dist + 50] < tau))
                                    if tau is not None else 0.0)
        post_statuses = [s for t, s in zip(ts, statuses) if t >= t_dist + 20]
        m["recovered_after_disturbance"] = (
            post_statuses.count(STATUS_CONVERGED) / max(len(post_statuses), 1) >= 0.5)

    # switching stability
    calls = log_llm["calls"]
    settle_call = 5
    m["family_switches_after_settle"] = sum(
        1 for c in calls[settle_call:] if c["switched"])
    m["weak_channels_off"] = bool(
        np.array(log_llm["final_mi_bits"])[:4].max() < 0.4) if sc["id"].startswith("X1") else None
    return m


# =====================================================================
# Orchestration
# =====================================================================
def ckpt(name: str) -> Path:
    return RESULTS / f"bench_{name}.json"


def run_scenario(sc, client, model) -> dict:
    if sc["kind"] == "translation":
        f = ckpt(sc["id"])
        if f.exists():
            return json.load(open(f))
        out = run_translation(client, model, sc)
        out["id"] = sc["id"]
        out["category"] = sc["category"]
        out["expected_fail"] = sc.get("expected_fail", False)
        json.dump(out, open(f, "w"), indent=1)
        return out

    # episode scenario: oracle once (seed 42) + llm per seed
    of = ckpt(f"{sc['id']}_oracle")
    if of.exists():
        log_oracle = json.load(open(of))
    else:
        print(f"    oracle run...", flush=True)
        log_oracle = run_episode(sc, 42, "oracle")
        json.dump(log_oracle, open(of, "w"))

    per_seed = []
    for seed in sc["seeds"]:
        lf = ckpt(f"{sc['id']}_llm_s{seed}")
        if lf.exists():
            log_llm = json.load(open(lf))
        else:
            print(f"    llm run seed={seed}...", flush=True)
            log_llm = run_episode(sc, seed, "llm", client, model)
            json.dump(log_llm, open(lf, "w"))
        m = episode_metrics(sc, log_llm, log_oracle)
        m["seed"] = seed
        m["ok"] = bool(sc["pass_fn"](m))
        per_seed.append(m)

    out = {"id": sc["id"], "category": sc["category"],
           "expected_fail": sc.get("expected_fail", False),
           "ok": all(m["ok"] for m in per_seed),
           "per_seed": per_seed,
           "oracle_final_power": log_oracle["final_power"],
           "oracle_final_sum_rate": log_oracle["final_sum_rate"]}
    json.dump(out, open(ckpt(f"{sc['id']}_summary"), "w"), indent=1)
    return out


def scoreboard(results: list[dict]):
    lines = ["# Benchmark v0.1 scoreboard", ""]
    lines.append("| id | category | result | detail |")
    lines.append("|---|---|---|---|")
    n_pass = n_fail = n_xfail = 0
    for r in results:
        if r.get("expected_fail"):
            tag = "XFAIL (known gap)"
            n_xfail += 1
        elif r["ok"]:
            tag = "PASS"
            n_pass += 1
        else:
            tag = "**FAIL**"
            n_fail += 1
        if "note" in r:
            detail = r["note"]
        else:
            ms = r["per_seed"]
            detail = "; ".join(
                f"s{m['seed']}: viol={m.get('violation_rate', 0):.2f} "
                f"chviol={m.get('ch_violation_rate', 0):.2f} "
                f"regret={m.get('power_regret')}"
                for m in ms)
        lines.append(f"| {r['id']} | {r['category']} | {tag} | {detail[:110]} |")
    lines.append("")
    lines.append(f"**{n_pass} pass, {n_fail} fail, {n_xfail} expected-fail "
                 f"(of {len(results)})**")
    report = "\n".join(lines)
    (RESULTS / "report.md").write_text(report)
    json.dump(results, open(RESULTS / "scoreboard.json", "w"), indent=1, default=str)
    print(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated scenario id prefixes")
    ap.add_argument("--model", type=str, default=None,
                    help="LM Studio model id (default: the loaded one)")
    ap.add_argument("--out", type=str, default="results",
                    help="results subdirectory; use a fresh one per model so "
                         "checkpoints from another model are never reused")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds overriding every episode "
                         "scenario's own list; per-seed checkpoints are keyed "
                         "by seed, so previously run seeds are reused")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
        for sc in SCENARIOS:
            if sc["kind"] == "episode":
                sc["seeds"] = seeds

    global RESULTS
    RESULTS = Path(__file__).parent / args.out
    RESULTS.mkdir(exist_ok=True)

    client, model = make_client(args.model)
    print(f"model: {model}")
    print(f"results: {RESULTS}")

    todo = SCENARIOS
    if args.only:
        prefixes = [p.strip() for p in args.only.split(",")]
        todo = [s for s in SCENARIOS
                if any(s["id"].startswith(p) for p in prefixes)]

    t0 = time.time()
    results = []
    for sc in todo:
        print(f"=== {sc['id']} ({sc['category']}) ===", flush=True)
        r = run_scenario(sc, client, model)
        results.append(r)
        state = ("XFAIL" if r.get("expected_fail")
                 else ("PASS" if r["ok"] else "FAIL"))
        print(f"    -> {state}", flush=True)
    scoreboard(results)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
