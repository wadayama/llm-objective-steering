# Benchmark

The evaluation suite for LLM-steered power allocation. It is run as a
regression test whenever the grammar, the prompt, or the architecture
changes — a CI for the interface.

## Running

```bash
uv run python runner.py                       # everything (checkpointed; needs LM Studio)
uv run python runner.py --only S1,T4          # a subset, by scenario id prefix
uv run python runner.py --model microsoft/phi-4 --out results-phi4
uv run python runner.py --only S1,S2,D1,X1 --seeds 42,43,44,45,46
```

Output goes to `results/report.md` (scoreboard), `results/scoreboard.json`,
and one raw log per scenario in `results/bench_*.json`.

**Use a separate `--out` per model.** Runs are checkpointed by scenario and
seed, so pointing two models at the same directory makes the second reuse
the first one's answers and report them as its own.

## Layout

- `scenarios.py` — the 13 scenarios in six categories
- `runner.py` — the runner (translation and episode modes, plus the oracle)
- `core.py` / `steering.py` — System 1 and System 2, frozen as of exp05

`steering.py` holds the system prompt: the channel model, the objective
families and constraint metrics with their ranges, the four status
categories and what each implies, and five worked examples. It is the same
prompt for every policy and every model.

## Scenarios

| Category | Scenarios | What it checks |
|---|---|---|
| translation (T1–T7) | 7 | one-shot policy text → declaration (T4 is the shutdown regression) |
| servo (S1, S2) | 2 | constraint satisfaction, regret against the oracle, settling time |
| infeasible (I1) | 1 | INFEASIBLE certified → the LLM arbitrates for resource → recovery |
| disturbance (D1) | 1 | abrupt gain change → autonomous re-convergence |
| switching (X1) | 1 | semantic oscillation regression (family switches after settling = 0) |
| expressiveness (E1) | 1 | a requirement the grammar cannot express (expected fail: a known gap, kept on the record) |

## The oracle baseline

Every episode scenario has an oracle run in which a hand-written declaration
is given straight to System 1. The LLM run is scored as **regret** against
it, in power units. The problem is convex in the power coordinates, so that
comparison is exact rather than indicative.

## Notes

E1 is an expected failure and is meant to stay one until the grammar can say
"I cannot express this". Every model tested silently substitutes a fabricated
per-channel MI floor for a decoding-latency requirement, which is evidence
about the grammar rather than about the models.
