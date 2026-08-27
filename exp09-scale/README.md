# Scaling: what happens as the number of channels grows

Eight channels isolate the steering mechanism, but a per-subcarrier allocator
is wider. This experiment sweeps N with System 1 alone, one declaration held
fixed, and asks the three questions the eight-channel study asks: is the
declared floor ever violated, how far is the settled allocation from the
independently computed optimum, and does the reported indicator stay at or
above the true gap.

## Comparability

Rows differ in N and in nothing else. The gain spread stays at
|h_i| in [0.5, 1.5] and is packed with more channels rather than widened;
P_total = 5N and tau = 1.25N hold the per-channel budget and the demanded
fraction of the 2N-bit ceiling constant. At N = 8 this reproduces the
operating point of the rest of the study.

## Running

```bash
uv run python run_scale.py --violation extensive   --out results/scale_extensive.json
uv run python run_scale.py --violation per-channel --out results/scale_perchannel.json
uv run python run_scale_llm.py --tag gptoss20b     # the declaration layer; needs LM Studio
```

Each takes about 25 minutes for N in {8, 16, 32, 64} over ten seeds, of which
roughly eight minutes is the reference optimum.

## What the two `--violation` settings mean

A declared sum rate is *extensive*: a shortfall of a given fraction is N times
more bits at N channels than at one, while the correction it buys is shared
over channels whose individual authority does not grow. `per-channel` divides
the violation by N, which states the same constraint per channel and moves no
boundary — g = 0 is the same set — and changes only the penalty's magnitude.
The two are indistinguishable at N = 8 and are not at N = 64.

## Differences from the other copies

`core.py` carries an `AugLagState.per_channel_violation` flag the other copies
do not, and its sum-rate clip is expressed as a fraction of the 2N ceiling
rather than the 15.5 that eight channels imply. `steering.py` makes the channel
count a template slot and adds one line telling the model how many entries its
vectors must carry, since the worked examples stay eight long.
