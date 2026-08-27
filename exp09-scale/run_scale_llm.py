"""Does the declaration layer scale with the number of channels?

System 1's cost grows with N because it optimizes N powers. The question
for the interface is different: how much does the LLM's job grow?

Three of the seven canonical policies name an objective whose parameters
are scalars, so their declaration has the same size at N = 64 as at
N = 8. The other four carry a per-channel vector, so the model must emit
N numbers. This script puts the same seven policies to a model at
N = 8, 16, 32, 64 -- the policy wording unchanged, only the system state
and the grammar's stated vector length differing -- and records, per
declaration, whether it is correct and how many numbers it carried.
"""

import argparse
import json
import time

import numpy as np
from openai import OpenAI

import core
import scale_profile as prof

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_TIMEOUT = 600.0


# ---------------------------------------------------------------------
# The seven canonical policies, with checks written to hold at any N.
# ---------------------------------------------------------------------
def _w(spec):
    return np.asarray(spec.weights, dtype=np.float64)


def _t(spec):
    return np.asarray(spec.targets, dtype=np.float64)


POLICIES = [
    dict(id="T1-throughput",
         policy="Maximize total throughput",
         vector=True,
         check=lambda s, tau, tau_ch, n: (
             s.family == "weighted_sum"
             and float(_w(s).std() / max(_w(s).mean(), 1e-9)) < 0.2)),
    dict(id="T2-fair-rates",
         policy="Ensure all channels achieve similar data rates",
         vector=False,
         check=lambda s, tau, tau_ch, n: s.family in ("soft_min", "alpha_fair")),
    dict(id="T3-equal-power",
         policy="Equalize transmit power across all channels",
         vector=True,
         check=lambda s, tau, tau_ch, n: (
             s.family == "power_target"
             and float(_t(s).std()) < 0.5)),
    dict(id="T4-shutdown-subset",
         policy="Shut down channels 0 to 3",
         vector=True,
         check=lambda s, tau, tau_ch, n: (
             (s.family == "power_target"
              and all(_t(s)[:4] < 0.5) and any(_t(s)[4:] > 1.0))
             or (s.family == "weighted_sum" and all(_w(s)[:4] < 1e-3)))),
    dict(id="T5-prioritize",
         policy="Prioritize channels 6 and 7",
         vector=True,
         # channels 6 and 7 by index, so the check reads the same at any N
         check=lambda s, tau, tau_ch, n: (
             s.family == "weighted_sum"
             and float(_w(s)[6:8].min())
             > 1.5 * float(np.delete(_w(s), [6, 7]).max()))),
    dict(id="T6-servo-decl",
         policy=("Minimize total transmit power while keeping the total "
                 "data rate above 10 bits"),
         vector=False,
         check=lambda s, tau, tau_ch, n: (
             s.family == "min_power" and tau is not None
             and abs(tau - 10.0) < 0.51)),
    dict(id="T7-perch-decl",
         policy=("minimize total power while keeping MI larger than 1.0 bit "
                 "for all the channels"),
         vector=False,
         check=lambda s, tau, tau_ch, n: (
             s.family == "min_power" and tau_ch is not None
             and abs(tau_ch - 1.0) < 0.11)),
]


def make_client(model_id: str | None):
    import urllib.request
    client = OpenAI(base_url=LLM_BASE_URL, api_key="lm-studio",
                    timeout=LLM_TIMEOUT)
    if model_id:
        return client, model_id
    url = LLM_BASE_URL.rsplit("/v1", 1)[0] + "/api/v0/models"
    with urllib.request.urlopen(url, timeout=5) as r:
        for m in json.loads(r.read()).get("data", []):
            if m.get("state") == "loaded" and m.get("type") != "embeddings":
                return client, m["id"]
    raise SystemExit("No model loaded in LM Studio")


def neutral_state(n: int):
    """The state shown to the model: equal powers, MI measured there."""
    import math
    import torch
    from core import T_VAL, QPSKMonteCarloEstimator, measure_mi
    torch.manual_seed(42)
    np.random.seed(42)
    est = QPSKMonteCarloEstimator()
    h2 = np.array(prof.h_abs2(n))
    p_total = prof.p_total(n)
    lam = torch.ones(n) * math.sqrt(p_total / n)
    h_re = torch.tensor(np.sqrt(h2), dtype=torch.float32)
    mi = measure_mi(lam, h_re, est, 20_000, "cpu")
    return h2, np.asarray(mi), (lam ** 2).numpy(), p_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="8,16,32,64")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    client, model = make_client(args.model)
    print(f"model: {model}")

    out = {"model": model, "sizes": {}}
    for n in sizes:
        core.N = n
        import steering
        steering.N = n                       # its own binding, made at import
        sys_prompt = steering.system_prompt("semantic", n)
        h2, mi, powers, p_total = neutral_state(n)

        rows = []
        for pol in POLICIES:
            msg = steering.build_user_message(
                h2, mi, powers, p_total,
                "weighted_sum, equal weights", "none",
                "TRANSIENT", pol["policy"])
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": msg}],
                temperature=0.0, max_tokens=4096)
            dt = time.time() - t0
            raw = (resp.choices[0].message.content or "").strip()
            try:
                spec, tau, tau_ch, _p, _r = steering.parse_response(raw)[:5]
                spec = spec.validate(p_total)
                ok = bool(pol["check"](spec, tau, tau_ch, n))
                note = f"family={spec.family}"
                err = None
            except Exception as e:                       # malformed or wrong length
                ok, note, err = False, "parse/validate failed", str(e)[:120]
            rows.append(dict(id=pol["id"], vector=pol["vector"], ok=ok,
                             note=note, error=err, seconds=dt,
                             reply_chars=len(raw)))
            print(f"  N={n:3d} {pol['id']:<20} {'ok ' if ok else 'FAIL'} "
                  f"{note}  {err or ''}  ({dt:.1f}s, {len(raw)} chars)",
                  flush=True)

        n_ok = sum(1 for r in rows if r["ok"])
        vec = [r for r in rows if r["vector"]]
        sca = [r for r in rows if not r["vector"]]
        out["sizes"][n] = dict(
            rows=rows, score=n_ok, total=len(rows),
            vector_score=sum(1 for r in vec if r["ok"]), vector_total=len(vec),
            scalar_score=sum(1 for r in sca if r["ok"]), scalar_total=len(sca),
            prompt_chars=len(sys_prompt) + len(msg),
            mean_reply_chars=float(np.mean([r["reply_chars"] for r in rows])),
            mean_seconds=float(np.mean([r["seconds"] for r in rows])),
        )
        s = out["sizes"][n]
        print(f"  --> N={n}: {n_ok}/{len(rows)}  "
              f"(vector-valued {s['vector_score']}/{s['vector_total']}, "
              f"scalar {s['scalar_score']}/{s['scalar_total']}), "
              f"reply {s['mean_reply_chars']:.0f} chars, "
              f"{s['mean_seconds']:.1f} s\n", flush=True)

    path = f"results/scale_llm_{args.tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("=== declaration layer vs N ===")
    print(f"{'N':>4}{'all':>8}{'vector':>9}{'scalar':>9}"
          f"{'reply chars':>13}{'s/call':>9}")
    for n in sizes:
        s = out["sizes"][n]
        print(f"{n:>4}{s['score']:>5}/{s['total']:<2}"
              f"{s['vector_score']:>6}/{s['vector_total']:<2}"
              f"{s['scalar_score']:>6}/{s['scalar_total']:<2}"
              f"{s['mean_reply_chars']:>13.0f}{s['mean_seconds']:>9.1f}")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
