# The non-LLM baseline

Does the translation from a natural-language policy to a declaration
actually need a language model? The seven benchmark policies are lexically
regular, so the honest way to ask is to write the rule-based translator that
replaces it, and then to test both on wording neither was tuned for.

## What is here

- `keyword_baseline.py` — a rule-based translator over regular expressions.
  It covers every objective family, parses channel indices and numeric
  thresholds, and handles both constraint metrics. It is a fair attempt, not
  a straw man.
- `policies.py` — `CANONICAL`, the seven benchmark policies verbatim, and
  `PARAPHRASE`, 21 restatements of the same seven intents (three each) in
  wording an operator might plausibly use instead.
- `run_comparison.py` — scores the translator and a model on both sets. The
  prompt, the parser and the pass criteria come from `../bench`, so the two
  are judged by exactly the same standard.

## The ordering matters

`keyword_baseline.py` was written against the canonical seven and **frozen**.
The paraphrases were authored afterwards, without consulting its rules, and
the translator was not revised in response to them. A keyword translator can
always be made to pass a phrasing it was built from; the question is what it
does with a phrasing it was not.

`run_comparison.py` records the translator's MD5 in every result file so that
the freeze is checkable after the fact.

## Running

```bash
uv run python run_comparison.py --model openai/gpt-oss-20b --tag gptoss20b
uv run python run_comparison.py --model microsoft/phi-4    --tag phi4
```

Both flags are required: `--model` is the LM Studio model id, `--tag` names
the result file.

Each paraphrase inherits the pass criterion of the canonical scenario it
restates, so a correct translation means the same thing in both sets.
