# Does the reported optimality indicator actually bound the gap?

Concavity in the power coordinates makes the duality gap a valid optimality
certificate. It does not make the quantity we compute at run time a valid one.
Checking that needs the true optimum, obtained without either of the two
devices under audit: Monte Carlo estimation and gradient ascent.

## What is here

- `reference.py` — the independent optimum. Mutual information by
  tensor-product Gauss--Hermite quadrature, which is deterministic, and the
  optimum located from the KKT condition rather than by iterating: with each
  I_i(P_i) concave and increasing, the marginal MI per unit power is equal
  across active channels at the optimum, so bisecting that common marginal
  until the sum rate meets tau lands on it.
- `verify_certificate.py` — one run, every dual interval compared against it
- `verify_certificate_seeds.py` — the same over ten seeds, reporting the
  settled power and the indicator's margin as mean +/- std
- `waterfilling.py` — the Gaussian-input allocation against the
  discrete-input optimum, both by quadrature
- `baseline_anchor.py` — the LLM asked simply to maximize throughput, against
  the optimizer run alone with no LLM in the loop

## Running

```bash
uv run python verify_certificate.py            # one seed, interval by interval
uv run python verify_certificate_seeds.py      # ten seeds
uv run python waterfilling.py
uv run python baseline_anchor.py               # this one needs LM Studio
```

Only `baseline_anchor.py` calls a model. The rest audit System 1 alone.

## A note on what the check found

Over ten seeds the indicator was at least the true gap at 237 of 240 dual
intervals. The three exceptions all occur at CONVERGED status and are small,
the largest short by 0.0014 in objective units, well inside the Monte Carlo
noise floor of the power itself. A single seed shows none of them, which is
why the ten-seed version exists.
