# Asking the LLM to regulate: how far prompting gets

Before the constraint moved into the optimizer, it was the LLM's job to keep
it satisfied. This experiment establishes what that costs, and it is the
source of the first three rows of the paper's servo ladder.

## What is here

- `run_batch.py` — the history-window ablation, K in {0, 3, 8}, on the policy
  "minimize total transmit power while keeping the total data rate above 10
  bits". K = 0 is memoryless control: each call independently decides to save
  power without knowing it has already done so, and the allocation does not
  oscillate so much as collapse.
- `run_prompt_ablation.py` — control-literacy prompting on top of the history:
  disclosing the actuator's smoothing, hard-constraint priority with a safety
  margin, a measurement-noise deadband, anti-oscillation guidance, and a
  variant that additionally demands a predicted sum rate and feeds the
  prediction error back.
- `run_c_ablation.py` — a 2x2 over {smoothing on/off} x {slew limit on/off},
  asking whether the actuator's damping is still needed once System 2 has a
  memory.
- `core.py`, `steering.py` — System 1 and 2 for this experiment

## Running

```bash
uv run python run_batch.py
uv run python run_prompt_ablation.py
uv run python run_c_ablation.py
```

All three call a model; they need an OpenAI-compatible server on
`localhost:1234`.

## What it shows

Memory stops the collapse and control-literacy prompting improves it further,
but the violation rate has a floor well above zero, and what improvement there
is gets bought with power margin. Declaring the constraint instead —
`../exp03-auglag` — attains a zero violation rate at less power.
