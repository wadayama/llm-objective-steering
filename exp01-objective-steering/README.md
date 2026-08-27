# Objective families, not fixed weights

The simpler design lets the LLM set the weights of a linear objective. That
cannot express equalization: for any fixed w, the maximizer of sum_i w_i I_i
is a boundary point of the achievable region, and equal-MI or equal-power
states are in general not maximizers of any static linear weighting. This
experiment measures the difference.

## What is here

- `verify_system1.py` — headless verification of System 1, no LLM involved:
  - S1 sanity: `weighted_sum` with equal weights at P_total = 40 reproduces
    the operating point of the conference version
  - S2 MI fairness: at P_total = 12, the unsaturated regime, compares
    `weighted_sum` (equal), `alpha_fair` and `soft_min`
  - S3 equal power: `power_target` against `weighted_sum`
  - S4 sphere against ball projection under a mid-run budget change
- `run.py` — the objective-steering demo this experiment was built around,
  superseded by `../exp05-integrated-demo`
- `core.py` — System 1

## Running

```bash
uv run python verify_system1.py      # no LLM needed
```

Writes `results/verify_results.json` and four figures.
