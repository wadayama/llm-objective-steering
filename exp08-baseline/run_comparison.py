"""Keyword baseline versus LLM on canonical policies and on paraphrases."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# the benchmark next door supplies the prompt, the parser and the canonical
# policies, so that the baseline is compared against exactly what the models see
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from core import H_ABS2_DEFAULT
from steering import build_user_message, parse_response, system_prompt
from scenarios import SCENARIOS, TRANSLATION_STATE

from keyword_baseline import translate as kw_translate
from policies import PARAPHRASE

TRANS = {sc["id"].split("-")[0]: sc for sc in SCENARIOS
         if sc["kind"] == "translation"}
STATE_LINE = ("CONVERGED | gap_bound = 0.050 bits | constraint (none) | "
              "shadow_price mu = 0.000")


def llm_translate(client, model, policy):
    st = TRANSLATION_STATE
    msg = build_user_message(np.array(H_ABS2_DEFAULT), st["mi_nats"],
                             st["powers"], st["p_total"],
                             "weighted_sum(w=[0.12, ...])", "(none)",
                             STATE_LINE, policy)
    r = client.chat.completions.create(
        model=model, messages=[{"role": "system",
                                "content": system_prompt("semantic")},
                               {"role": "user", "content": msg}],
        temperature=0.0, max_tokens=2048)
    spec, tau, tau_ch, _, _ = parse_response(
        (r.choices[0].message.content or "").strip())
    spec.validate(st["p_total"])
    return spec, tau, tau_ch


def score(base_id, spec, tau, tau_ch):
    ok, note = TRANS[base_id]["check"](spec, tau, tau_ch)
    return bool(ok), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                    timeout=600.0)

    canonical = [(sc["id"].split("-")[0], sc["policy"])
                 for sc in SCENARIOS if sc["kind"] == "translation"]
    sets = {"canonical": canonical, "paraphrase": PARAPHRASE}

    out = {"model": args.model,
           "baseline_md5": hashlib.md5(
               open("keyword_baseline.py", "rb").read()).hexdigest(),
           "sets": {}}

    for name, items in sets.items():
        rows = []
        for base_id, policy in items:
            kb_ok, kb_note = score(base_id, *kw_translate(policy))
            try:
                llm_ok, llm_note = score(base_id,
                                         *llm_translate(client, args.model,
                                                        policy))
            except Exception as e:
                llm_ok, llm_note = False, f"{type(e).__name__}"
            rows.append(dict(base=base_id, policy=policy,
                             keyword=kb_ok, keyword_note=kb_note,
                             llm=llm_ok, llm_note=llm_note))
        kb = sum(r["keyword"] for r in rows)
        lm = sum(r["llm"] for r in rows)
        out["sets"][name] = {"rows": rows, "keyword": kb, "llm": lm,
                             "n": len(rows)}
        print(f"\n### {name}  (n={len(rows)})")
        print(f"{'policy':<62}{'keyword':>9}{'LLM':>6}")
        for r in rows:
            p = r["policy"] if len(r["policy"]) <= 60 else r["policy"][:57] + "..."
            print(f"{p:<62}{'pass' if r['keyword'] else 'FAIL':>9}"
                  f"{'pass' if r['llm'] else 'FAIL':>6}")
        print(f"--> keyword {kb}/{len(rows)}   LLM {lm}/{len(rows)}")

    json.dump(out, open(f"results/compare_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
