# Arbitration: what an instruction cannot do and a typed field can

INFEASIBLE is the one status the optimizer cannot resolve on its own. It has
already done everything regulation allows, and only a changed declaration
helps. This experiment asks whether the LLM acts on that, and what it takes
to make it act.

## What is here

- `common.py` — the frozen certified-infeasible state and the episode loop
- `run_probe.py` — the one-shot probe: ask each model for a declaration, and
  separately, in prose, what the status means and what actions it leaves
- `run_arbitration.py` — the closed-loop episodes under each condition
- `verify_stiffness.py` — System 1 alone under a declaration held at an
  infeasible level, against the same declaration with a sufficient cap
- `plot_stiffness.py`, `plot_escalation.py` — the figures
- `core.py`, `steering.py` — System 1 and 2 for this experiment

## Conditions

| Id | Prompt |
|---|---|
| `C0` | the benchmark prompt, unchanged |
| `C1` | plus an imperative block: re-emitting the same declaration changes nothing, doing nothing is not an option |
| `C2` | plus the escalation grammar |
| `C3` | escalation grammar and immediate actuation |
| `A1` | the grammar alone, without the rejection rule |
| `A3` | the grammar and immediate actuation, without the rejection rule |

`A1` and `A3` separate the two ingredients that are easily conflated: giving
the action a name and a type, and actuating it in full rather than damping it
like an ordinary budget command.

## Running

```bash
uv run python run_probe.py       --tag qwen3-4b
uv run python run_arbitration.py --tag qwen3-4b --conditions C1,A1,A3 \
                                 --seeds 42,43,44,45,46,47,48,49,50,51
uv run python verify_stiffness.py        # no LLM involved
uv run python plot_stiffness.py [out.pdf]
```

Results are checkpointed per condition and seed, so a rerun only fills gaps.
