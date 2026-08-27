# Declaring the constraint instead of regulating it

The constraint moves out of the prompt and into System 1, enforced by a
Powell--Hestenes--Rockafellar augmented Lagrangian. The LLM declares it once;
the optimizer holds it continuously, and the KKT residuals it produces along
the way become the status categories and the shadow price reported back.

## What is here

- `verify_auglag.py` — headless verification, no LLM involved:
  - S1 servo: minimize power subject to a 10-bit sum-rate floor from a cold
    start, expecting convergence to the boundary with the residuals decaying
    and the status reaching CONVERGED
  - S2 disturbance: shuffle the channel gains after convergence, expecting a
    residual spike (DISTURBED) followed by autonomous re-convergence
- `run_batch.py` — the same policy in closed loop with an LLM, the last row of
  the paper's servo ladder
- `run_infeasible.py` — a declaration that cannot be met within the budget,
  which is where INFEASIBLE and the shadow price come from
- `core.py`, `steering.py` — System 1 and 2 for this experiment

## Running

```bash
uv run python verify_auglag.py       # no LLM needed
uv run python run_batch.py
uv run python run_infeasible.py
```
