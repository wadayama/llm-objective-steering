"""exp06-a one-shot probes: can the model READ the certificate, and does it ACT?

Both probes show the model the SAME frozen, certified-INFEASIBLE state
(System 1 alone under the oracle declaration, so no model influenced it).

  probe ACT   -- the ordinary declaration request. Does the response raise
                 P_total / escalate, i.e. does it resolve the infeasibility?
  probe READ  -- the same state, but asked in prose: what does this status
                 mean and what can you do about it? Never parsed as a
                 declaration; it measures comprehension, not action.

READ passing while ACT fails is the signature of an action problem;
both failing is the signature of an information problem.
"""

import argparse
import json
import sys
import time

from common import (POLICY, CONDITIONS, RESULTS, TAU_DEMANDED,
                    build_infeasible_snapshot, make_client)
from steering import build_user_message, parse_response, system_prompt

READ_QUESTION = """\
Do NOT answer with JSON this time. In plain prose, answer these two questions:
(1) What does the optimizer status line above tell you about your declared
    constraint, and why is the optimizer unable to fix it by itself?
(2) List every action that is available to YOU to resolve this situation.
"""

# What a correct READ answer has to contain: the idea of giving the system
# more power budget, or of relaxing the demanded rate.
RAISE_HINTS = ["p_total", "budget", "raise", "increase", "more power"]
RELAX_HINTS = ["relax", "lower the constraint", "reduce the constraint",
               "lower the min", "reduce the required", "relax the constraint",
               "lower the demanded", "decrease the constraint"]


def scores(text: str) -> dict:
    t = text.lower()
    return {"mentions_raise_budget": any(h in t for h in RAISE_HINTS),
            "mentions_relax": any(h in t for h in RELAX_HINTS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None, help="label for the output file")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    client, model = make_client(args.model)
    tag = args.tag or model.replace("/", "-")
    snap = build_infeasible_snapshot()
    RESULTS.mkdir(exist_ok=True)

    base_msg = build_user_message(
        snap["h_abs2"], snap["mi"], snap["powers"], snap["p_total"],
        snap["spec_desc"], snap["cdesc"], snap["sline"], POLICY,
        history_block="")

    print(f"### {model}")
    print(f"frozen state: {snap['sline']}")
    print(f"             sum_rate = {snap['sum_rate']:.2f} bits, "
          f"power = {snap['powers'].sum():.2f}\n")

    out = {"model": model, "snapshot": {k: snap[k] for k in
           ("step", "sline", "mu", "sum_rate")}, "probes": []}

    for cond in CONDITIONS:
        sp = system_prompt(cond)
        # ---- ACT ----
        t0 = time.time()
        r = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": sp},
                                   {"role": "user", "content": base_msg}],
            temperature=0.0, max_tokens=args.max_tokens)
        act_raw = (r.choices[0].message.content or "").strip()
        act_dt = time.time() - t0
        act = {"raw": act_raw, "sec": act_dt, "parse_ok": False,
               "p_cmd": None, "escalated": None, "tau": None,
               "family": None, "resolves": False}
        try:
            spec, tau, tau_ch, p_cmd, reasoning, esc = parse_response(act_raw)
            act.update(parse_ok=True, p_cmd=p_cmd, escalated=esc, tau=tau,
                       family=spec.family, reasoning=reasoning[:200])
            # Resolving means changing the declaration in a way that can
            # actually clear the infeasibility: ask for more budget, or
            # relax the demanded rate.
            raised = ((p_cmd is not None and p_cmd > snap["p_total"] + 1e-9)
                      or (esc is not None and esc > snap["p_total"] + 1e-9))
            relaxed = (tau is not None and tau < TAU_DEMANDED - 1e-9)
            act["raised_budget"] = bool(raised)
            act["relaxed_constraint"] = bool(relaxed)
            act["resolves"] = bool(raised or relaxed)
        except Exception as e:
            act["error"] = f"{type(e).__name__}: {e}"

        # ---- READ ----
        t0 = time.time()
        r = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": sp},
                                   {"role": "user",
                                    "content": base_msg + "\n\n" + READ_QUESTION}],
            temperature=0.0, max_tokens=args.max_tokens)
        read_raw = (r.choices[0].message.content or "").strip()
        read = {"raw": read_raw, "sec": time.time() - t0, **scores(read_raw)}

        out["probes"].append({"condition": cond, "act": act, "read": read})
        print(f"[{cond}] ACT  {act_dt:5.1f}s  parse={act['parse_ok']} "
              f"family={act['family']} P_total={act['p_cmd']} "
              f"escalate={act['escalated']} -> "
              f"{'RESOLVES' if act['resolves'] else 'no change'}")
        print(f"[{cond}] READ {read['sec']:5.1f}s  "
              f"mentions_raise_budget={read['mentions_raise_budget']} "
              f"mentions_relax={read['mentions_relax']}")
        print(f"       READ text: {read_raw[:220]}...\n")

    f = RESULTS / f"probe_{tag}.json"
    json.dump(out, open(f, "w"), indent=1, default=str)
    print(f"saved {f}")


if __name__ == "__main__":
    main()
