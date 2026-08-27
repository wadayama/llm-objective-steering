"""A rule-based translator: the baseline an LLM has to beat.

WRITTEN AGAINST THE SEVEN CANONICAL POLICIES OF THE BENCHMARK AND THEN
FROZEN. The paraphrase set in policies.py was authored afterwards and this
file was not revised in response to it. That ordering is the whole point of
the comparison: a keyword translator can always be made to pass a phrasing
it was built from, so the question is what it does with a phrasing it was
not.

The rules below are a fair attempt, not a straw man: they cover every
family in the grammar, parse channel indices and numeric thresholds, and
handle the two constraint metrics.
"""

import re

import numpy as np

from core import N, P_TOTAL_DEFAULT, ObjectiveSpec

NUM = r"(\d+(?:\.\d+)?)"


def _channel_list(text: str):
    """Channel indices named by 'channels 0 to 3', 'channels 6 and 7', etc."""
    m = re.search(r"channels?\s+" + NUM + r"\s*(?:to|-|through)\s*" + NUM, text)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        return [i for i in range(min(a, b), max(a, b) + 1) if 0 <= i < N]
    m = re.search(r"channels?\s+" + NUM + r"\s*(?:and|,)\s*" + NUM, text)
    if m:
        return [i for i in (int(float(m.group(1))), int(float(m.group(2))))
                if 0 <= i < N]
    idx = [int(float(x)) for x in re.findall(r"channel\s+" + NUM, text)]
    return [i for i in idx if 0 <= i < N]


def translate(policy: str):
    """policy -> (ObjectiveSpec, tau_bits, tau_ch_bits). Never raises."""
    t = policy.lower()
    spec = ObjectiveSpec()
    tau = tau_ch = None

    # --- constraints, recognised before the objective ---
    m = re.search(r"(?:data rate|sum rate|total rate|throughput)\s*"
                  r"(?:above|over|at least|greater than|>=?)\s*" + NUM, t)
    if m:
        tau = float(m.group(1))
    m = re.search(r"mi\s*(?:larger than|greater than|above|at least|>=?)\s*"
                  + NUM, t)
    if m:
        tau_ch = float(m.group(1))

    # --- objective family ---
    if "shut down" in t or "shutdown" in t or "disable" in t:
        chans = _channel_list(t)
        if chans:
            spec.family = "power_target"
            tg = np.ones(N) * (P_TOTAL_DEFAULT / max(N - len(chans), 1))
            tg[chans] = 0.0
            spec.targets = tg
            return spec.validate(P_TOTAL_DEFAULT), tau, tau_ch

    if "prioritize" in t or "priority" in t:
        chans = _channel_list(t)
        if chans:
            spec.family = "weighted_sum"
            w = np.ones(N)
            w[chans] = 4.0
            spec.weights = w / w.sum()
            return spec.validate(P_TOTAL_DEFAULT), tau, tau_ch

    if "equalize transmit power" in t or "equal power" in t or \
       ("equalize" in t and "power" in t):
        spec.family = "power_target"
        spec.targets = np.ones(N) * (P_TOTAL_DEFAULT / N)
        return spec.validate(P_TOTAL_DEFAULT), tau, tau_ch

    if "minimize" in t and "power" in t:
        spec.family = "min_power"
        return spec.validate(P_TOTAL_DEFAULT), tau, tau_ch

    if ("similar" in t and "rate" in t) or "fair" in t or "equalize mi" in t:
        spec.family = "soft_min"
        spec.beta = 10.0
        return spec.validate(P_TOTAL_DEFAULT), tau, tau_ch

    if "throughput" in t or "sum rate" in t or "total data rate" in t:
        spec.family = "weighted_sum"
        spec.weights = np.ones(N) / N
        return spec.validate(P_TOTAL_DEFAULT), tau, tau_ch

    # nothing matched: fall back to the neutral objective
    return spec.validate(P_TOTAL_DEFAULT), tau, tau_ch
