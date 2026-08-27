"""exp06-a closed loop: the I1 arbitration scenario under C0 / C1 / C2.

C0 reproduces bench v0.2 exactly (same protocol, same prompt), so the arm is
a control on the earlier finding rather than a fresh claim. C1 adds an
imperative block about INFEASIBLE (information). C2 adds "escalate" to the
grammar and rejects an unchanged declaration under INFEASIBLE (structure).
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from common import (POLICY, STEPS, CALL_EVERY, TRACK_EVERY, K_HIST,
                    N_MC_GRAD, N_MC_STATE, N_MC_FINAL, EMA_ALPHA_P,
                    EMA_ALPHA_PARAM, TAU_DEMANDED, RESOLVED_SR, CONDITIONS,
                    RESULTS, device, make_client, status_line,
                    requests_no_relief)
from core import (N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
                  QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
                  optimize_step, measure_mi, kkt_residuals, classify_status,
                  STATUS_TRANSIENT, STATUS_INFEASIBLE)
from steering import (ControlHistory, build_user_message, compute_kpis,
                      ema_merge_spec, parse_response, system_prompt, NATS2BITS)


def run_episode(client, model, condition: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
    h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)

    p_total = P_TOTAL_DEFAULT
    lam = torch.ones(N) * math.sqrt(p_total / N)
    al = AugLagState()
    status = STATUS_TRANSIENT
    res = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
    hist = ControlHistory(K_HIST)
    sys_prompt = system_prompt(condition)
    spec = ObjectiveSpec()

    log = {"condition": condition, "seed": seed, "model": model,
           "ts_steps": [], "ts_sum_rate": [], "ts_total_power": [],
           "ts_status": [], "calls": [], "parse_errors": 0,
           "rejections": 0, "n_escalations": 0}
    lam_iv = lam.clone()
    call_no = 0

    for step in range(STEPS):
        if step % CALL_EVERY == 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            sr = float(mib.sum())
            if step > 0:
                al.dual_update(sr, mib)
                res = kkt_residuals(sr, lam, lam_iv, al, mi_bits_meas=mib)
                status = classify_status(res, al, status)
                lam_iv = lam.clone()

            kpis = compute_kpis(mi, (lam ** 2).numpy(), p_total)
            cdesc, sline, mu = status_line(status, res, al)
            msg = build_user_message(h_abs2, mi, (lam ** 2).numpy(), p_total,
                                     spec.describe(), cdesc, sline, POLICY,
                                     history_block=hist.render(kpis))
            rej_note = None
            try:
                (spec_new, tau, tau_ch, p_cmd,
                 reasoning, esc) = parse_response(_call(client, model,
                                                        sys_prompt, msg))
                if esc is not None:
                    log["n_escalations"] += 1

                # exp05 guardrail: unconstrained min_power is not a sentence.
                rejected = (spec_new.family == "min_power"
                            and tau is None and tau_ch is None)
                if rejected:
                    rej_note = "min_power without a constraint"

                # exp06 C2 structural rule: under a certified INFEASIBLE, a
                # declaration that asks for no relief at all is not a
                # sentence either -- it cannot resolve what the optimizer has
                # already proven unresolvable.
                if (not rejected and condition in ("C2", "C3")
                        and status == STATUS_INFEASIBLE
                        and requests_no_relief(spec_new, tau, p_cmd, esc,
                                               spec, al, p_total)):
                    rejected = True
                    rej_note = "declaration asks for no relief under INFEASIBLE"

                if rejected:
                    log["rejections"] += 1
                else:
                    if condition in ("C3", "A3") and esc is not None:
                        # An escalation is a discrete request, not a servo
                        # command, so it bypasses the EMA that exp02c put on
                        # the budget knob and applies in full.
                        p_total = esc
                    elif p_cmd is not None:
                        p_total = EMA_ALPHA_P * p_cmd + (1 - EMA_ALPHA_P) * p_total
                    spec = ema_merge_spec(spec_new, spec, EMA_ALPHA_PARAM, p_total)
                    al.set_constraint(tau)
                    al.set_channel_constraint(tau_ch)
            except Exception as e:
                log["parse_errors"] += 1
                rej_note = f"parse error: {type(e).__name__}"

            hist.add(call_no, spec.describe()
                     + ("" if rej_note is None else f" [REJECTED: {rej_note}]"),
                     cdesc, kpis, status)
            log["calls"].append({"step": step, "family": spec.family,
                                 "tau": al.tau_bits, "p_total": p_total,
                                 "status": status, "mu": mu,
                                 "rejected": rej_note})
            call_no += 1

        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, al=al)

        if step % TRACK_EVERY == 0 or step == STEPS - 1:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            log["ts_steps"].append(step)
            log["ts_sum_rate"].append(float(mib.sum()))
            log["ts_total_power"].append(float((lam ** 2).sum()))
            log["ts_status"].append(status)

    mi = measure_mi(lam, h_re, est, N_MC_FINAL, device)
    mib = mi * NATS2BITS
    log["final_sum_rate"] = float(mib.sum())
    log["final_power"] = float((lam ** 2).sum())
    ps = [c["p_total"] for c in log["calls"]]
    log["max_p_total"] = max(ps)
    changed = [c["step"] for c in log["calls"]
               if abs(c["p_total"] - P_TOTAL_DEFAULT) > 1e-6]
    log["first_budget_change_step"] = changed[0] if changed else None
    log["reached_infeasible"] = STATUS_INFEASIBLE in log["ts_status"]
    log["resolved"] = (log["reached_infeasible"]
                       and log["final_sum_rate"] >= RESOLVED_SR)
    return log


def _call(client, model, sys_prompt, msg):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": msg}],
        temperature=0.0, max_tokens=2048)
    return (r.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()

    client, model = make_client(args.model)
    tag = args.tag or model.replace("/", "-")
    seeds = [int(x) for x in args.seeds.split(",")]
    conds = [c.strip() for c in args.conditions.split(",")]
    RESULTS.mkdir(exist_ok=True)

    print(f"### {model}  conditions={conds} seeds={seeds}", flush=True)
    print(f"{'cond':<5}{'seed':<6}{'resolved':<10}{'finalSR':>9}"
          f"{'maxP':>7}{'1stChg':>8}{'esc':>5}{'rej':>5}{'err':>5}  status seen",
          flush=True)
    rows = []
    t_all = time.time()
    for cond in conds:
        for seed in seeds:
            f = RESULTS / f"arb_{tag}_{cond}_s{seed}.json"
            if f.exists():
                log = json.load(open(f))
            else:
                log = run_episode(client, model, cond, seed)
                json.dump(log, open(f, "w"), indent=1, default=str)
            rows.append(log)
            seen = " ".join(sorted(set(log["ts_status"]),
                                   key=log["ts_status"].index))
            print(f"{cond:<5}{seed:<6}{str(log['resolved']):<10}"
                  f"{log['final_sum_rate']:>9.2f}{log['max_p_total']:>7.1f}"
                  f"{str(log['first_budget_change_step']):>8}"
                  f"{log['n_escalations']:>5}{log['rejections']:>5}"
                  f"{log['parse_errors']:>5}  {seen}", flush=True)

    print("\n--- summary ---", flush=True)
    for cond in conds:
        rs = [r for r in rows if r["condition"] == cond]
        n = sum(r["resolved"] for r in rs)
        print(f"{cond}: resolved {n}/{len(rs)}  "
              f"mean finalSR {np.mean([r['final_sum_rate'] for r in rs]):.2f}  "
              f"mean maxP {np.mean([r['max_p_total'] for r in rs]):.1f}",
              flush=True)
    print(f"done in {(time.time() - t_all)/60:.1f} min", flush=True)
    json.dump(rows, open(RESULTS / f"arb_summary_{tag}.json", "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
