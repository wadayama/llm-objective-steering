"""System 2 plumbing for exp03: constraint declaration + optimizer status signals.

New over exp02: the LLM can DECLARE constraints (enforced by System 1 via the
PHR augmented Lagrangian) and receives certified status signals (status
category, gap bound, shadow price). Two prompt conditions:
  semantic -- signals come with meaning and behavioral guidance
  plain    -- signals appear in the user message with no explanation
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
that YOU select, subject to constraints that YOU declare. Constraints are
enforced by the optimizer itself (augmented Lagrangian): once declared, you
do NOT need to regulate them call-by-call.

## Objective families
1. "weighted_sum": J = sum_i w_i * MI_i  (throughput / priorities; params: "weights", {n} numbers)
2. "alpha_fair": fairness on MI values (params: "alpha" in [0.5, 20])
3. "soft_min": maximize the minimum MI (params: "beta" in [1, 50], 10 typical)
4. "power_target": drive powers to explicit targets (params: "targets", {n} numbers)
5. "min_power": MINIMIZE the TOTAL power of ALL channels together. Use for
   system-wide power-saving policies, and ALWAYS together with a declared
   constraint that protects performance — an unconstrained min_power would
   drive every channel to zero, so the system REJECTS it.
   Do NOT use min_power to shut down a SUBSET of channels: for that, use
   "power_target" with 0 for the channels to disable (or "weighted_sum"
   with zero weights for them).

## Constraints you can declare
"constraints": [{"metric": "sum_rate", "min": <bits>},
                {"metric": "min_channel_mi", "min": <bits>}]
- "sum_rate": holds the TOTAL sum rate >= min (in BITS).
- "min_channel_mi": holds EVERY channel's individual MI >= min (in BITS,
  at most 1.95). Use for policies like "each channel must achieve at least
  X bits" / "keep MI larger than X for all channels".
- Both metrics may be declared together. Declare once; the optimizer
  enforces them continuously and precisely.

## Second knob
- "P_total" (optional): total power budget CAP in [1, 100] (default 40).
  Your command is smoothed: applied = 0.5*command + 0.5*current. With
  "min_power" you normally do NOT need to lower P_total; declare the
  constraint and let the optimizer find the minimal power itself.

## Escalation (a first-class declaration)
Besides the objective and the constraints, you may declare an ESCALATION:
  "escalate": {"resource": "P_total", "request": <number>, "reason": "<why>"}
An escalation asks the system for more of a resource than the current budget
allows. It is the declaration you use when the constraints you were asked to
enforce cannot be met with what you have -- that is, when the status is
INFEASIBLE. Regulation cannot resolve an INFEASIBLE status; only a changed
declaration can.
Unlike an ordinary "P_total" command, which is smoothed as a servo signal, an
escalation is a discrete request and is applied IN FULL on the next step:
"escalate" with request R sets the budget cap to exactly R. Ask for the amount
you actually believe is needed, not a cautious step towards it.

## Your task
Translate the active policy into (objective family, constraints). Prefer the
direct expression: e.g. "minimize power while keeping rate above X" ->
{"objective": {"family": "min_power"}, "constraints": [{"metric": "sum_rate", "min": X}]}.

Respond ONLY with a JSON object (no markdown fences, no extra text). Examples:
{"objective": {"family": "min_power"}, "constraints": [{"metric": "sum_rate", "min": 10}], "reasoning": "..."}
{"objective": {"family": "min_power"}, "constraints": [{"metric": "min_channel_mi", "min": 1.0}], "reasoning": "min power, every channel >= 1 bit"}
{"objective": {"family": "soft_min", "beta": 10}, "reasoning": "equalize rates"}
{"objective": {"family": "weighted_sum", "weights": [1,1,1,1,1,1,3,3]}, "reasoning": "prioritize ch 6,7"}
{"objective": {"family": "power_target", "targets": [0,0,0,0,10,10,10,10]}, "reasoning": "shut down channels 0-3, focus power on 4-7"}
{"objective": {"family": "min_power"}, "constraints": [{"metric": "sum_rate", "min": 15}], "escalate": {"resource": "P_total", "request": 80, "reason": "status INFEASIBLE: 15 bits is unreachable within a cap of 40"}}
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
and the state observed at each, plus the measured effect of your most recent
action. Check whether your previous action had the intended effect before
issuing a new one.
"""

STATUS_SEMANTIC_SECTION = """
## Optimizer status signals (computed and certified by the optimizer; read-only)
Each call you receive a status line. Meaning and how to act on it:
- status TRANSIENT: the optimizer is still moving; its numbers do not yet
  reflect your last decision. Do NOT re-steer based on them; keep your
  declaration unless the policy itself changed.
- status CONVERGED: the allocation is settled. "gap_bound" (bits) certifies
  how far it can be from the best achievable under your declared objective
  and constraints; values at or below ~0.1 bits mean "as good as measurable".
  If the policy is satisfied, the correct action is NO CHANGE: re-emit the
  same declaration.
- status INFEASIBLE: the optimizer certifies your declared constraints cannot
  be met within the current budget. Regulation cannot fix this; only you can:
  raise P_total, relax the constraint, or change the objective.
- status DISTURBED: the environment changed and the optimizer is re-adapting
  on its own. Reconsider only whether your DECLARATION still matches the
  policy; do not micro-correct the numbers.
- shadow_price mu: how much the objective would improve per bit of constraint
  relaxation. Large mu = the constraint is expensive (is it worth its cost?);
  mu near 0 = the constraint is effectively free.
You never need to compute or verify these signals yourself.
"""


def system_prompt(condition: str) -> str:
    """condition: 'semantic' (signals explained) or 'plain' (signals bare)."""
    s = system_prompt_base() + HISTORY_SECTION
    if condition == "semantic":
        s += STATUS_SEMANTIC_SECTION
    elif condition != "plain":
        raise ValueError(f"unknown condition: {condition}")
    return s


# =====================================================================
# KPIs + history (exp02-compatible, with status columns)
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


@dataclass
class HistRecord:
    call_no: int
    objective: str
    constraint: str
    kpis: dict
    status: str


class ControlHistory:
    def __init__(self, K: int):
        self.K = K
        self._recs: list[HistRecord] = []

    def add(self, call_no: int, objective: str, constraint: str,
            kpis: dict, status: str):
        if self.K == 0:
            return
        self._recs.append(HistRecord(call_no, objective, constraint, kpis, status))
        if len(self._recs) > self.K:
            self._recs = self._recs[-self.K:]

    def clear(self):
        self._recs = []

    def render(self, current_kpis: dict) -> str:
        if self.K == 0 or not self._recs:
            return ""
        lines = ["Recent control history (oldest first; state measured at each action):"]
        lines.append("  call  objective                  constraint        status      sum_rate  total_P")
        for r in self._recs:
            k = r.kpis
            lines.append(
                f"  {r.call_no:4d}  {r.objective[:24]:24s}  {r.constraint[:16]:16s}  "
                f"{r.status:10s}  {k['sum_rate_bits']:7.3f}  {k['total_power']:7.2f}"
            )
        last = self._recs[-1]
        lk, ck = last.kpis, current_kpis
        lines.append(
            f"Effect since your last action (call {last.call_no}): "
            f"sum_rate {lk['sum_rate_bits']:.3f} -> {ck['sum_rate_bits']:.3f}, "
            f"total power {lk['total_power']:.2f} -> {ck['total_power']:.2f}."
        )
        return "\n".join(lines)


# =====================================================================
# Message building / response parsing
# =====================================================================
def build_user_message(h_abs2, mi_nats, powers, p_total, spec_desc,
                       constraint_desc, status_line, policy,
                       history_block: str = "") -> str:
    lines = ["Current system state:"]
    lines.append(f"  Total MI = {mi_nats.sum() * NATS2BITS:.3f} bits")
    lines.append(f"  Total Power = {powers.sum():.2f} (budget cap P_total = {p_total:.1f})")
    lines.append(f"  Current objective: {spec_desc}")
    lines.append(f"  Declared constraint: {constraint_desc}")
    lines.append(f"  Optimizer status: {status_line}")
    lines.append("")
    lines.append("  ch  |h_i|^2    MI_i     P_i")
    lines.append("  --  -------  ------  -------")
    for i in range(N):
        lines.append(f"  {i:2d}  {h_abs2[i]:7.4f}  {mi_nats[i]:6.4f}  {powers[i]:7.3f}")
    lines.append("")
    if history_block:
        lines.append(history_block)
        lines.append("")
    lines.append(f"Active policy: {policy}" if policy
                 else "Active policy: (none — weighted_sum, equal weights)")
    return "\n".join(lines)


def parse_response(raw: str):
    """Returns (ObjectiveSpec, tau_bits, tau_ch_bits, p_total_cmd, reasoning).

    tau_bits: declared sum-rate floor (bits) or None.
    tau_ch_bits: declared per-channel MI floor (bits) or None.
    """
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

    tau_bits = None
    tau_ch_bits = None
    for c in data.get("constraints", []) or []:
        if c.get("metric") == "sum_rate" and "min" in c:
            tau_bits = float(np.clip(float(c["min"]), 0.0, 15.5))
        if c.get("metric") == "min_channel_mi" and "min" in c:
            tau_ch_bits = float(np.clip(float(c["min"]), 0.0, 1.95))

    p_total = data.get("P_total", None)
    if p_total is not None:
        p_total = float(np.clip(float(p_total), 1.0, 100.0))

    # An escalation lands on the same knob as P_total but is returned
    # separately, because it is actuated in full rather than smoothed.
    esc = data.get("escalate", None)
    escalated = None
    if isinstance(esc, dict) and esc.get("resource") == "P_total":
        req = esc.get("request", None)
        if req is not None:
            escalated = float(np.clip(float(req), 1.0, 100.0))

    return (spec, tau_bits, tau_ch_bits, p_total,
            data.get("reasoning", ""), escalated)


def ema_merge_spec(spec_new: ObjectiveSpec, spec_old: ObjectiveSpec,
                   alpha: float, p_total: float) -> ObjectiveSpec:
    if spec_new.family == spec_old.family:
        spec_new.weights = alpha * spec_new.weights + (1 - alpha) * spec_old.weights
        spec_new.targets = alpha * spec_new.targets + (1 - alpha) * spec_old.targets
        spec_new.alpha = alpha * spec_new.alpha + (1 - alpha) * spec_old.alpha
        spec_new.beta = alpha * spec_new.beta + (1 - alpha) * spec_old.beta
    return spec_new.validate(p_total)
