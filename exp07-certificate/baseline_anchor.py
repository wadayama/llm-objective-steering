"""Does putting an LLM in the loop cost anything when the policy is the classic one?

Baseline: System 1 alone with equal weights -- the objective the optimizer
would have been given if nobody were steering it.
Steered: the same optimizer, but the objective comes from the LLM reading a
plain throughput policy.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# the benchmark next door supplies the prompt and the parser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from core import (N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
                  QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
                  optimize_step, measure_mi, kkt_residuals, classify_status,
                  STATUS_TRANSIENT)
from steering import (ControlHistory, build_user_message, compute_kpis,
                      ema_merge_spec, parse_response, system_prompt, NATS2BITS)

POLICY = "Maximize total throughput"
STEPS, CALL_EVERY, K_HIST = 250, 10, 8
N_MC_GRAD, N_MC_STATE, N_MC_FINAL = 8000, 20_000, 100_000
dev = torch.device("cpu")


def episode(mode, client=None, model=None, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h = np.array(H_ABS2_DEFAULT)
    h_re = torch.tensor(np.sqrt(h), dtype=torch.float32)
    lam = torch.ones(N) * math.sqrt(P_TOTAL_DEFAULT / N)
    al = AugLagState(); spec = ObjectiveSpec()   # equal weights by default
    status, hist = STATUS_TRANSIENT, ControlHistory(K_HIST)
    res = {"r_stat":1.0,"r_feas":0.0,"r_comp":0.0,"gap_bound":1.0}
    lam_iv = lam.clone(); errs = 0
    for step in range(STEPS):
        if mode == "llm" and step % CALL_EVERY == 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, dev); mib = mi * NATS2BITS
            if step > 0:
                al.dual_update(float(mib.sum()), mib)
                res = kkt_residuals(float(mib.sum()), lam, lam_iv, al, mi_bits_meas=mib)
                status = classify_status(res, al, status); lam_iv = lam.clone()
            kpis = compute_kpis(mi, (lam**2).numpy(), P_TOTAL_DEFAULT)
            sline = (f"{status} | gap_bound = {res['gap_bound']:.3f} bits | "
                     f"constraint (none) | shadow_price mu = 0.000")
            msg = build_user_message(h, mi, (lam**2).numpy(), P_TOTAL_DEFAULT,
                                     spec.describe(), "(none)", sline, POLICY,
                                     history_block=hist.render(kpis))
            try:
                r = client.chat.completions.create(model=model,
                    messages=[{"role":"system","content":system_prompt("semantic")},
                              {"role":"user","content":msg}],
                    temperature=0.0, max_tokens=2048)
                spec_new, tau, tau_ch, p_cmd, _ = parse_response(
                    (r.choices[0].message.content or "").strip())
                spec = ema_merge_spec(spec_new, spec, 0.5, P_TOTAL_DEFAULT)
                al.set_constraint(tau); al.set_channel_constraint(tau_ch)
            except Exception:
                errs += 1
            hist.add(step // CALL_EVERY, spec.describe(), "(none)", kpis, status)
        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, dev,
                            P_TOTAL_DEFAULT, est, al=al)
    mi = measure_mi(lam, h_re, est, N_MC_FINAL, dev)
    return float((mi * NATS2BITS).sum()), float((lam**2).sum()), spec.describe(), errs


if __name__ == "__main__":
    from openai import OpenAI
    model = sys.argv[1]
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", timeout=600.0)
    sr0, p0, d0, _ = episode("s1")
    print(f"System 1 alone (equal weights): SR {sr0:.3f} bits, power {p0:.2f}")
    sr1, p1, d1, e = episode("llm", client, model)
    print(f"LLM-steered ({model})        : SR {sr1:.3f} bits, power {p1:.2f}")
    print(f"  declaration at the end: {d1}   (parse errors: {e})")
    print(f"  difference: {sr1 - sr0:+.3f} bits")
    json.dump({"s1": {"sr": sr0, "power": p0}, "llm": {"sr": sr1, "power": p1,
               "model": model, "declaration": d1, "parse_errors": e}},
              open("results/baseline_anchor.json", "w"), indent=1)
