"""System 2 plumbing shared by exp02 scripts: prompt, parsing, control history.

The new piece over exp01 is ControlHistory: a bounded window of past
(action, state) records rendered into the LLM user message, turning the
otherwise memoryless System 2 into a finite-window feedback controller.
"""

import json
import math
from dataclasses import dataclass

import numpy as np

from core import N, OBJECTIVE_FAMILIES, ObjectiveSpec

NATS2BITS = 1.0 / math.log(2)

SYSTEM_PROMPT_TEMPLATE = """\
You are an autonomous controller for a QPSK parallel communication system with N={n} independent sub-channels.
You are called periodically to enforce a user-defined policy.

## System model
Signal on sub-channel i:  Y_i = h_i * lambda_i * X_i + noise
- |h_i|^2: channel gain (fixed physical property; larger = better channel)
- P_i = lambda_i^2: transmit power on channel i, sum(P_i) <= P_total
- MI_i: mutual information (data rate) of channel i, max log(4) ~ 1.386 nats (2 bits)

A gradient-ascent optimizer (System 1) continuously maximizes an objective J
that YOU select. Your job is to translate the user's policy into the right
objective family and parameters.

## Objective families you can choose from
1. "weighted_sum": J = sum_i w_i * MI_i
   - For throughput maximization and channel-priority policies.
   - Params: "weights" = list of 8 non-negative numbers (will be normalized).
   - Equal weights = maximize total sum rate. Higher w_i = more power to channel i.
     w_i = 0 lets channel i's power be drained toward the minimum.
2. "alpha_fair": J = (1/N) sum_i U_alpha(MI_i), U_alpha = x^(1-alpha)/(1-alpha)
   - For balancing fairness vs efficiency on the MI values.
   - Params: "alpha" in [0.5, 20]. alpha=1: proportional fairness;
     larger alpha = stronger fairness (alpha >= 5 approaches max-min).
3. "soft_min": J = -(1/beta) * log sum_i exp(-beta * MI_i)
   - Maximizes (a smooth version of) the MINIMUM MI. Use for strict fairness
     policies like "make all rates equal" / "help the worst channel".
   - Params: "beta" in [1, 50]. Larger beta = closer to the exact minimum (10 is a good default).
4. "power_target": J = -sum_i (P_i - target_i)^2
   - Drives the power allocation itself to explicit per-channel targets.
     Use for policies about the POWER distribution: "equalize transmit power",
     "give channel 0 exactly 10 units", "use only half the budget", etc.
   - Params: "targets" = list of 8 non-negative numbers (sum must be <= P_total;
     it will be scaled down if it exceeds P_total).

## Second control knob
- "P_total" (optional): total power budget in [1, 100] (default 40).
  Reduce it for power-saving policies; raise it for maximum-throughput policies.
  With families 1-3 the optimizer always uses the FULL budget, so power saving
  requires lowering P_total (or using power_target with small targets).

## Your task
Each call, look at the current state and the active policy, then pick the
objective family + parameters that best express the policy's intent.
- Prefer the family that expresses the policy DIRECTLY (e.g. fairness on rates
  -> soft_min or alpha_fair; anything about the power profile -> power_target),
  rather than hand-tuning weighted_sum weights step by step.
- If no policy is set, use weighted_sum with equal weights.
- The state may change between calls (channel gains vary). Adapt accordingly.

Respond ONLY with a JSON object (no markdown fences, no extra text). Examples:
{"objective": {"family": "soft_min", "beta": 10}, "reasoning": "equalize rates"}
{"objective": {"family": "weighted_sum", "weights": [1,1,1,1,1,1,3,3]}, "reasoning": "prioritize ch 6,7"}
{"objective": {"family": "power_target", "targets": [5,5,5,5,5,5,5,5]}, "reasoning": "equal power"}
{"objective": {"family": "alpha_fair", "alpha": 5}, "P_total": 20.0, "reasoning": "fair and power-saving"}
"""

def system_prompt_base(n: int | None = None) -> str:
    """The base prompt with the channel count filled in.

    A plain replace and not .format(): the prompt is full of JSON braces.
    Rendering at call time rather than at import means an experiment that
    rebinds the channel count gets a prompt that agrees with it, instead of
    asking the model for eight numbers and then rejecting the eight it sends.
    """
    return SYSTEM_PROMPT_TEMPLATE.replace("{n}", str(N if n is None else n))


HISTORY_SECTION = """
## Control history (feedback)
The user message includes a "Recent control history" table: your past actions
and the system state observed at the moment of each action, plus the measured
effect of your most recent action. Use it like a feedback controller:
- Verify whether your previous action moved the metrics in the intended direction.
- If the metrics oscillate around a target (repeated overshoot in alternating
  directions), make a SMALLER adjustment or keep the current values.
- If repeated actions produced no effect, that knob is saturated; reconsider.
- When close to the target, prefer small refinements over large corrections.
"""


ACTUATOR_TEXT = {
    "ema": """\
- Your "P_total" command is SMOOTHED before being applied:
  applied = 0.5 * your_command + 0.5 * current_budget.
  A single command moves the budget only HALFWAY toward your value. To reach
  a target budget quickly, either overshoot your command or repeat it on
  consecutive calls until the applied budget matches your intent.
- Objective parameters (weights/targets/alpha/beta) are smoothed the same way
  when you stay within the same family.""",
    "direct": """\
- Your "P_total" command and objective parameters are applied DIRECTLY, with
  no smoothing: what you command is what the system uses until your next
  call. There is no need to overshoot or repeat commands.""",
    "slew": """\
- Your "P_total" command is applied directly (no smoothing), but the applied
  budget can change by at most 30% of its current value per call; a larger
  requested change is truncated to that limit. Split large corrections
  across consecutive calls. Objective parameters are applied directly.""",
    "ema_slew": """\
- Your "P_total" command is SMOOTHED (applied = 0.5 * command + 0.5 *
  current_budget) AND the resulting change is additionally capped at 30% of
  the current budget per call. Large corrections therefore require several
  consecutive calls. Objective parameters are smoothed the same way within
  a family.""",
}

PROCEDURE_STEP3 = {
    "ema": "3. Account for the command smoothing (one command moves the budget halfway).",
    "direct": "3. Your command takes full effect before the next call; no compensation needed.",
    "slew": "3. Corrections larger than the 30% rate limit are truncated; split them across calls.",
    "ema_slew": "3. Account for both the smoothing (half effect) and the 30% rate limit.",
}


def control_section(actuator: str) -> str:
    return f"""
## Actuator characteristics and control discipline
{ACTUATOR_TEXT[actuator]}
- Hard requirements in the policy (e.g. "keep X above Y") take PRIORITY over
  optimization goals (e.g. "minimize ..."): violating a requirement is much
  worse than being mildly suboptimal. First bring the constrained metric
  safely to the required side WITH A MARGIN, and only then optimize the rest.
- Measurements fluctuate by roughly +/-0.3 bits between calls (Monte-Carlo
  noise). Do not react to changes smaller than that, and judge requirements
  with a safety margin rather than at the exact threshold.
- If the history shows you overshooting in alternating directions, halve
  your adjustment size instead of reversing at full amplitude.
"""


def procedure_section(actuator: str) -> str:
    return f"""
## Feedback procedure (follow this every call)
1. From the control history, estimate the local sensitivity (gain) of the
   constrained metric to your knob, e.g.
   g = (change in sum_rate) / (change in applied budget) over recent rows.
2. Compute the error between the target (including your safety margin) and
   the current value, and derive the needed change: delta = error / g.
   If g cannot be estimated yet, probe with a moderate step.
{PROCEDURE_STEP3[actuator]}
4. Add to your JSON response a field "predicted_sum_rate": the sum rate in
   BITS you expect to measure at the next call given your action. Your past
   predictions are compared against measurements in the history; use the
   prediction errors to correct your sensitivity estimate.
"""


PROMPT_VARIANTS = ("V0", "V1", "V2")


def system_prompt(with_history: bool, variant: str = "V0",
                  actuator: str = "ema") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant}")
    if actuator not in ACTUATOR_TEXT:
        raise ValueError(f"unknown actuator: {actuator}")
    s = system_prompt_base() + (HISTORY_SECTION if with_history else "")
    if variant in ("V1", "V2"):
        s += control_section(actuator)
    if variant == "V2":
        s += procedure_section(actuator)
    return s


# =====================================================================
# KPIs
# =====================================================================
def compute_kpis(mi_nats: np.ndarray, powers: np.ndarray, p_total: float) -> dict:
    mi_bits = mi_nats * NATS2BITS
    s = mi_bits.sum()
    jain = float(s ** 2 / (N * (mi_bits ** 2).sum())) if s > 0 else 0.0
    return {
        "sum_rate_bits": float(s),
        "min_mi_bits": float(mi_bits.min()),
        "jain_mi": jain,
        "total_power": float(powers.sum()),
        "p_total": float(p_total),
    }


# =====================================================================
# ControlHistory
# =====================================================================
@dataclass
class HistRecord:
    call_no: int
    objective: str      # short description of the chosen objective
    p_cmd: float | None  # raw P_total command from the LLM (None = not emitted)
    kpis: dict          # state observed at the moment of this action
    predicted: float | None = None  # LLM's predicted sum rate [bits] (V2)


class ControlHistory:
    """Bounded window of past (action, state) records. K=0 disables history."""

    def __init__(self, K: int):
        self.K = K
        self._recs: list[HistRecord] = []

    def add(self, call_no: int, objective: str, p_cmd: float | None, kpis: dict,
            predicted: float | None = None):
        if self.K == 0:
            return
        self._recs.append(HistRecord(call_no, objective, p_cmd, kpis, predicted))
        if len(self._recs) > self.K:
            self._recs = self._recs[-self.K:]

    def clear(self):
        self._recs = []

    def render(self, current_kpis: dict) -> str:
        """Compact table of past actions + explicit effect of the last one."""
        if self.K == 0 or not self._recs:
            return ""
        lines = ["Recent control history (oldest first; state measured at the moment of each action):"]
        lines.append("  call  objective                        P_cmd   sum_rate  min_MI  Jain    budget")
        for r in self._recs:
            k = r.kpis
            pc = f"{r.p_cmd:5.1f}" if r.p_cmd is not None else "  -  "
            lines.append(
                f"  {r.call_no:4d}  {r.objective[:30]:30s}  {pc}   "
                f"{k['sum_rate_bits']:7.3f}  {k['min_mi_bits']:5.3f}  {k['jain_mi']:.4f}  {k['p_total']:5.1f}"
            )
        last = self._recs[-1]
        lk, ck = last.kpis, current_kpis
        lines.append(
            f"Effect of your last action (call {last.call_no} -> now): "
            f"sum_rate {lk['sum_rate_bits']:.3f} -> {ck['sum_rate_bits']:.3f} "
            f"({ck['sum_rate_bits'] - lk['sum_rate_bits']:+.3f} bits), "
            f"total power {lk['total_power']:.2f} -> {ck['total_power']:.2f} "
            f"({ck['total_power'] - lk['total_power']:+.2f})."
        )
        if last.predicted is not None:
            err = ck["sum_rate_bits"] - last.predicted
            lines.append(
                f"You predicted sum_rate = {last.predicted:.3f} bits; "
                f"measured now: {ck['sum_rate_bits']:.3f} bits "
                f"(prediction error {err:+.3f})."
            )
        return "\n".join(lines)


# =====================================================================
# Message building / response parsing (exp01-compatible)
# =====================================================================
def build_user_message(h_abs2, mi_nats, powers, p_total, spec_desc, policy,
                       history_block: str = "") -> str:
    lines = ["Current system state:"]
    lines.append(f"  Total MI = {mi_nats.sum():.4f} nats ({mi_nats.sum() * NATS2BITS:.3f} bits)")
    lines.append(f"  Total Power = {powers.sum():.2f} (budget P_total = {p_total:.1f})")
    lines.append(f"  Current objective: {spec_desc}")
    lines.append("")
    lines.append("  ch  |h_i|^2    MI_i     P_i")
    lines.append("  --  -------  ------  -------")
    for i in range(N):
        lines.append(f"  {i:2d}  {h_abs2[i]:7.4f}  {mi_nats[i]:6.4f}  {powers[i]:7.3f}")
    lines.append("")
    if history_block:
        lines.append(history_block)
        lines.append("")
    if policy:
        lines.append(f"Active policy: {policy}")
    else:
        lines.append("Active policy: (none — use weighted_sum with equal weights)")
    return "\n".join(lines)


def parse_response(raw: str):
    """Returns (ObjectiveSpec, p_total_cmd | None, reasoning). Raises on bad output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    data = json.loads(text)
    obj = data.get("objective", {})
    family = obj.get("family", "weighted_sum")
    if family not in OBJECTIVE_FAMILIES:
        raise ValueError(f"unknown family: {family}")

    spec = ObjectiveSpec(family=family)
    if "weights" in obj:
        w = np.asarray(obj["weights"], dtype=np.float64)
        if len(w) != N:
            raise ValueError(f"expected {N} weights, got {len(w)}")
        spec.weights = w
    if "alpha" in obj:
        spec.alpha = float(obj["alpha"])
    if "beta" in obj:
        spec.beta = float(obj["beta"])
    if "targets" in obj:
        tg = np.asarray(obj["targets"], dtype=np.float64)
        if len(tg) != N:
            raise ValueError(f"expected {N} targets, got {len(tg)}")
        spec.targets = tg

    p_total = data.get("P_total", None)
    if p_total is not None:
        p_total = float(np.clip(float(p_total), 1.0, 100.0))

    predicted = data.get("predicted_sum_rate", None)
    if predicted is not None:
        try:
            predicted = float(predicted)
        except (TypeError, ValueError):
            predicted = None

    return spec, p_total, predicted, data.get("reasoning", "")


def ema_merge_spec(spec_new: ObjectiveSpec, spec_old: ObjectiveSpec,
                   alpha: float, p_total: float) -> ObjectiveSpec:
    """EMA smoothing of parameters within the same family (exp01 behavior)."""
    if spec_new.family == spec_old.family:
        spec_new.weights = alpha * spec_new.weights + (1 - alpha) * spec_old.weights
        spec_new.targets = alpha * spec_new.targets + (1 - alpha) * spec_old.targets
        spec_new.alpha = alpha * spec_new.alpha + (1 - alpha) * spec_old.alpha
        spec_new.beta = alpha * spec_new.beta + (1 - alpha) * spec_old.beta
    return spec_new.validate(p_total)
