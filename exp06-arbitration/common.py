"""Shared plumbing for exp06-a: the I1 arbitration scenario under C0/C1/C2.

The scenario is bench's I1: a policy that cannot be met inside the default
budget, so System 1 certifies INFEASIBLE and only System 2 can resolve it by
changing the declaration. bench v0.2 found that 15B and 4B models never do.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from core import (
    N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
    QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
    optimize_step, measure_mi, kkt_residuals, classify_status,
    STATUS_TRANSIENT, STATUS_INFEASIBLE,
)
from steering import (
    ControlHistory, build_user_message, compute_kpis, ema_merge_spec,
    parse_response, system_prompt, NATS2BITS,
)

# --- protocol (identical to bench I1 so the C0 arm reproduces v0.2) ---
POLICY = ("Minimize total transmit power while keeping the total "
          "data rate above 15 bits")
STEPS = 250
CALL_EVERY = 10
TRACK_EVERY = 5
K_HIST = 8
N_MC_GRAD = 8000
N_MC_STATE = 20_000
N_MC_FINAL = 100_000
EMA_ALPHA_P = 0.5
EMA_ALPHA_PARAM = 0.5
TAU_DEMANDED = 15.0
RESOLVED_SR = 14.5     # bench I1 pass threshold

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_TIMEOUT = 300.0
CONDITIONS = ("C0", "C1", "C2", "C3")

device = torch.device("cpu")
RESULTS = Path(__file__).parent / "results"


def make_client(model_id=None):
    import sys
    import urllib.request
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key="lm-studio", timeout=LLM_TIMEOUT)
    if model_id:
        return client, model_id
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


def status_line(status, res, al):
    parts = []
    if al.tau_bits is not None:
        parts.append(f"sum_rate>={al.tau_bits:.1f}")
    if al.tau_ch_bits is not None:
        parts.append(f"each_MI>={al.tau_ch_bits:.2f}")
    cdesc = "; ".join(parts) if parts else "(none)"
    mu = al.mu + float(al.mu_ch.max())
    line = (f"{status} | gap_bound = {res['gap_bound']:.3f} bits | "
            f"constraint {cdesc} | shadow_price mu = {mu:.3f}")
    return cdesc, line, mu


def requests_no_relief(spec_new, tau, p_cmd, esc, spec_cur, al, p_total) -> bool:
    """True when a declaration cannot possibly clear a certified infeasibility.

    The first version of this rule compared the declaration to the previous
    one and rejected an identical re-emission. That was wrong: P_total is
    EMA-smoothed, so repeating the SAME budget request keeps moving the cap
    toward it and is real progress. Asking again is not a no-op.

    What actually cannot help is a declaration that asks for no relief at
    all: no budget above the current cap, no relaxation of the demanded
    rate, and the same objective family.
    """
    asked = max([v for v in (p_cmd, esc) if v is not None], default=None)
    more_budget = asked is not None and asked > p_total + 1e-6
    cur_tau = al.tau_bits
    relaxed = (tau is not None and cur_tau is not None
               and tau < cur_tau - 1e-6)
    same_family = spec_new.family == spec_cur.family
    return not (more_budget or relaxed or not same_family)


def build_infeasible_snapshot(seed: int = 42, max_steps: int = 150):
    """Run System 1 alone under the oracle declaration until it certifies
    INFEASIBLE, then freeze the state. Used by the one-shot probes so every
    model is asked about exactly the same certified situation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = QPSKMonteCarloEstimator()
    h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
    h_re = torch.tensor(np.sqrt(h_abs2), dtype=torch.float32)
    p_total = P_TOTAL_DEFAULT
    lam = torch.ones(N) * math.sqrt(p_total / N)
    al = AugLagState()
    al.set_constraint(TAU_DEMANDED)
    spec = ObjectiveSpec(family="min_power").validate(p_total)
    status = STATUS_TRANSIENT
    res = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
    lam_iv = lam.clone()

    for step in range(max_steps):
        if step % CALL_EVERY == 0 and step > 0:
            mi = measure_mi(lam, h_re, est, N_MC_STATE, device)
            mib = mi * NATS2BITS
            al.dual_update(float(mib.sum()), mib)
            res = kkt_residuals(float(mib.sum()), lam, lam_iv, al, mi_bits_meas=mib)
            status = classify_status(res, al, status)
            lam_iv = lam.clone()
            if status == STATUS_INFEASIBLE:
                cdesc, sline, mu = status_line(status, res, al)
                return dict(step=step, h_abs2=h_abs2, mi=mi,
                            powers=(lam ** 2).numpy(), p_total=p_total,
                            spec_desc=spec.describe(), cdesc=cdesc,
                            status=status, sline=sline, mu=mu,
                            sum_rate=float(mib.sum()))
        lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                            p_total, est, al=al)
    raise RuntimeError("System 1 never certified INFEASIBLE")
