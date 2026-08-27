#!/usr/bin/env python3
"""
exp01-objective-steering: LLM Objective Steering Demo
=====================================================

Successor to the conference-version exp06 demo. Two changes:

1. Objective steering: the LLM no longer only tunes linear weights; it selects
   an objective FAMILY and its parameters:
     - weighted_sum(weights)   : throughput / channel-priority policies
     - alpha_fair(alpha)       : tunable fairness on MI
     - soft_min(beta)          : strict max-min fairness on MI
     - power_target(targets)   : explicit power distribution (equalize / save power)
2. Sphere projection: monotone families always use the full power budget,
   so redistribution is immediate (power_target keeps ball projection).

Requires LM Studio on localhost:1234 (OpenAI-compatible). UI runs without it.
"""

import json
import math
import textwrap
import threading

import numpy as np
import torch
# The backend is left to matplotlib's auto-selection, which tries macosx,
# qtagg, gtk4agg, gtk3agg, tkagg, wxagg in that order: the native macOS
# backend here, Qt/GTK/Tk on Linux/Windows. Forcing "macosx" would fail
# outright off macOS and also disables that fallback.
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from openai import OpenAI

from core import (
    N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
    QPSKMonteCarloEstimator, ObjectiveSpec, optimize_step,
    effective_weights, measure_mi, OBJECTIVE_FAMILIES,
)

# =====================================================================
# Configuration
# =====================================================================
N_MC_DISPLAY = 50_000
N_MC_GRAD = 10_000
STEPS_PER_FRAME = 3
HISTORY_WINDOW = 200
LLM_CALL_INTERVAL = 1.0
EMA_ALPHA_P = 0.5          # P_total smoothing
EMA_ALPHA_PARAM = 0.5      # smoothing of weights/targets/alpha/beta within a family
DEFAULT_POLICY = "Maximize total throughput while ensuring all channels get some power"

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_TIMEOUT = 30.0

device = torch.device("cpu")
NATS2BITS = 1.0 / math.log(2)


# =====================================================================
# SharedState
# =====================================================================
class SharedState:
    """Thread-safe shared state between optimizer, UI, and LLM controller."""

    def __init__(self):
        self._lock = threading.Lock()
        self.h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
        self.h_re = torch.tensor(np.sqrt(self.h_abs2), dtype=torch.float32, device=device)
        self.lam = torch.ones(N, device=device) * math.sqrt(P_TOTAL_DEFAULT / N)
        self.p_total = P_TOTAL_DEFAULT
        self.mi_history: list[float] = []
        self.iteration = 0
        self.optimizer_running = True
        self.active_policy = ""
        self._display = self._default_display()

    @staticmethod
    def _default_display():
        return {
            "mi_sub": np.zeros(N),
            "powers": np.zeros(N),
            "w_eff": np.ones(N) / N,
            "objective": "weighted_sum (equal)",
            "p_total": P_TOTAL_DEFAULT,
            "reasoning": "",
            "status": "Idle",
            "iteration": 0,
            "policy": "",
        }

    def set_policy(self, policy: str):
        with self._lock:
            self.active_policy = policy

    def get_policy(self) -> str:
        with self._lock:
            return self.active_policy

    def set_h_abs2(self, h_abs2: np.ndarray):
        with self._lock:
            self.h_abs2[:] = h_abs2
            self.h_re = torch.tensor(np.sqrt(self.h_abs2), dtype=torch.float32, device=device)

    def get_h_re(self) -> torch.Tensor:
        with self._lock:
            return self.h_re.clone()

    def get_h_abs2(self) -> np.ndarray:
        with self._lock:
            return self.h_abs2.copy()

    def get_lam(self) -> torch.Tensor:
        with self._lock:
            return self.lam.clone()

    def set_lam(self, lam: torch.Tensor):
        with self._lock:
            self.lam = lam

    def set_p_total(self, p_total: float):
        with self._lock:
            self.p_total = p_total

    def append_mi_history(self, mi_sum: float):
        with self._lock:
            self.mi_history.append(mi_sum)
            if len(self.mi_history) > HISTORY_WINDOW:
                self.mi_history = self.mi_history[-HISTORY_WINDOW:]

    def get_mi_history(self) -> list[float]:
        with self._lock:
            return list(self.mi_history)

    def increment_iteration(self, n: int = 1):
        with self._lock:
            self.iteration += n

    def get_iteration(self) -> int:
        with self._lock:
            return self.iteration

    def publish_display(self, **kw):
        with self._lock:
            self._display.update(kw)

    def get_display_snapshot(self) -> dict:
        with self._lock:
            d = dict(self._display)
            d["mi_sub"] = d["mi_sub"].copy()
            d["powers"] = d["powers"].copy()
            d["w_eff"] = d["w_eff"].copy()
            return d

    def reset(self):
        with self._lock:
            self.h_abs2[:] = H_ABS2_DEFAULT
            self.h_re = torch.tensor(np.sqrt(self.h_abs2), dtype=torch.float32, device=device)
            self.lam = torch.ones(N, device=device) * math.sqrt(P_TOTAL_DEFAULT / N)
            self.p_total = P_TOTAL_DEFAULT
            self.mi_history.clear()
            self.iteration = 0
            self.active_policy = ""
            self._display = self._default_display()


# =====================================================================
# LLMController — steers the objective family
# =====================================================================
class LLMController:
    SYSTEM_PROMPT = """\
You are an autonomous controller for a QPSK parallel communication system with N=8 independent sub-channels.
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

    def __init__(self):
        self.spec = ObjectiveSpec()
        self.p_total = P_TOTAL_DEFAULT
        self.active_policy = ""
        self.last_reasoning = ""
        self.status = "Idle"
        self.is_calling = False
        self.call_id = 0
        self._lock = threading.Lock()
        self._retrigger_args = None
        self._model_name = None
        self._user_selected_model = False
        self._periodic_stop = threading.Event()
        self._periodic_thread = None
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT)

    # ---- model selection ----
    def set_model(self, name: str):
        with self._lock:
            self._model_name = name
            self._user_selected_model = True

    def get_model(self) -> str | None:
        with self._lock:
            return self._model_name

    def _resolve_model(self):
        if self._model_name is not None:
            return self._model_name
        models = self._client.models.list()
        if models.data:
            self._model_name = models.data[0].id
            return self._model_name
        raise RuntimeError("No models loaded in LM Studio")

    # ---- policy management ----
    def set_policy(self, text: str, shared: SharedState):
        with self._lock:
            self.active_policy = text.strip()
            self.call_id += 1
        shared.set_policy(self.active_policy)
        if self.active_policy:
            self._start_periodic(shared)
        else:
            self._stop_periodic()

    def clear_policy(self, shared: SharedState):
        with self._lock:
            self.active_policy = ""
            self.spec = ObjectiveSpec()
            self.last_reasoning = ""
            self.status = "Policy cleared"
            self.call_id += 1
        shared.set_policy("")
        self._stop_periodic()

    def _start_periodic(self, shared: SharedState):
        self._stop_periodic()
        self._periodic_stop = threading.Event()
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop, args=(shared,), daemon=True)
        self._periodic_thread.start()

    def _stop_periodic(self):
        if self._periodic_thread is not None:
            self._periodic_stop.set()
            self._periodic_thread = None

    def _periodic_loop(self, shared: SharedState):
        while not self._periodic_stop.wait(timeout=LLM_CALL_INTERVAL):
            with self._lock:
                if not self.active_policy:
                    break
            snap = shared.get_display_snapshot()
            self.trigger_llm_call(shared.get_h_abs2(), snap["mi_sub"], snap["powers"])

    # ---- accessors for the optimizer loop ----
    def get_control(self):
        with self._lock:
            spec = ObjectiveSpec(
                family=self.spec.family,
                weights=self.spec.weights.copy(),
                alpha=self.spec.alpha,
                beta=self.spec.beta,
                targets=self.spec.targets.copy(),
            )
            return spec, self.p_total, self.last_reasoning, self.status

    # ---- LLM call machinery ----
    def trigger_llm_call(self, h_abs2, mi_sub, powers):
        with self._lock:
            if self.is_calling:
                self._retrigger_args = (h_abs2.copy(), mi_sub.copy(), powers.copy())
                return
            self.is_calling = True
            self._retrigger_args = None
            current_id = self.call_id
            self.status = "Calling LLM..."

        threading.Thread(
            target=self._call_llm,
            args=(h_abs2.copy(), mi_sub.copy(), powers.copy(), current_id),
            daemon=True,
        ).start()

    def _call_llm(self, h_abs2, mi_sub, powers, my_id):
        try:
            user_msg = self._build_user_message(h_abs2, mi_sub, powers)
            model_name = self._resolve_model()
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            spec_new, p_total_new, reasoning = self._parse_response(raw)

            with self._lock:
                if self.call_id == my_id:
                    self._apply_spec(spec_new)
                    if p_total_new is not None:
                        self.p_total = EMA_ALPHA_P * p_total_new + (1 - EMA_ALPHA_P) * self.p_total
                    self.last_reasoning = reasoning
                    self.status = f"OK [{self.spec.family}]"
                else:
                    self.status = "Stale (new call arrived)"
                self.is_calling = False

        except Exception as e:
            with self._lock:
                if not self._user_selected_model:
                    self._model_name = None
                err = str(e)
                self.status = f"Error: {err[:80]}" + ("..." if len(err) > 80 else "")
                self.is_calling = False

        self._maybe_retrigger()

    def _apply_spec(self, spec_new: ObjectiveSpec):
        """Adopt a new objective spec; smooth parameters within the same family."""
        a = EMA_ALPHA_PARAM
        if spec_new.family == self.spec.family:
            spec_new.weights = a * spec_new.weights + (1 - a) * self.spec.weights
            spec_new.targets = a * spec_new.targets + (1 - a) * self.spec.targets
            spec_new.alpha = a * spec_new.alpha + (1 - a) * self.spec.alpha
            spec_new.beta = a * spec_new.beta + (1 - a) * self.spec.beta
        self.spec = spec_new.validate(self.p_total)

    def _build_user_message(self, h_abs2, mi_sub, powers):
        with self._lock:
            policy = self.active_policy
            spec_desc = self.spec.describe()
            p_total = self.p_total

        lines = ["Current system state:"]
        lines.append(f"  Total MI = {mi_sub.sum():.4f} nats ({mi_sub.sum() * NATS2BITS:.3f} bits)")
        lines.append(f"  Total Power = {powers.sum():.2f} (budget P_total = {p_total:.1f})")
        lines.append(f"  Current objective: {spec_desc}")
        lines.append("")
        lines.append("  ch  |h_i|^2    MI_i     P_i")
        lines.append("  --  -------  ------  -------")
        for i in range(N):
            lines.append(f"  {i:2d}  {h_abs2[i]:7.4f}  {mi_sub[i]:6.4f}  {powers[i]:7.3f}")
        lines.append("")
        if policy:
            lines.append(f"Active policy: {policy}")
        else:
            lines.append("Active policy: (none — use weighted_sum with equal weights)")
        return "\n".join(lines)

    def _maybe_retrigger(self):
        with self._lock:
            args = self._retrigger_args
            self._retrigger_args = None
        if args is not None:
            self.trigger_llm_call(*args)

    def _parse_response(self, raw):
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

        reasoning = data.get("reasoning", "")
        return spec, p_total, reasoning

    def reset(self, shared: SharedState):
        self._stop_periodic()
        with self._lock:
            self.spec = ObjectiveSpec()
            self.p_total = P_TOTAL_DEFAULT
            self.active_policy = ""
            self.last_reasoning = ""
            self.status = "Reset"
            self.call_id += 1
            self._retrigger_args = None
        shared.set_policy("")


# =====================================================================
# Optimizer loop
# =====================================================================
def run_optimizer_loop(shared: SharedState, llm_ctrl: LLMController,
                       estimator: QPSKMonteCarloEstimator):
    while shared.optimizer_running:
        spec, cur_p_total, reasoning, status = llm_ctrl.get_control()
        shared.set_p_total(cur_p_total)

        h_re = shared.get_h_re()
        lam = shared.get_lam()

        for _ in range(STEPS_PER_FRAME):
            lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                                cur_p_total, estimator, projection="sphere")
            shared.increment_iteration()

        shared.set_lam(lam)

        mi_sub = measure_mi(lam, h_re, estimator, N_MC_DISPLAY, device)
        powers = (lam ** 2).cpu().numpy()
        w_eff = effective_weights(mi_sub, spec)

        shared.publish_display(
            mi_sub=mi_sub, powers=powers, w_eff=w_eff,
            objective=spec.describe(), p_total=cur_p_total,
            reasoning=reasoning, status=status,
            iteration=shared.get_iteration(), policy=shared.get_policy(),
        )
        shared.append_mi_history(mi_sub.sum())


# =====================================================================
# UI
# =====================================================================
def create_and_run_ui(shared: SharedState, llm_ctrl: LLMController,
                      estimator: QPSKMonteCarloEstimator):
    opt_thread = threading.Thread(
        target=run_optimizer_loop, args=(shared, llm_ctrl, estimator), daemon=True)
    opt_thread.start()

    fig = plt.figure(figsize=(14, 12))
    fig.canvas.manager.set_window_title("exp01: LLM Objective Steering Demo")

    gs = fig.add_gridspec(
        5, 3,
        height_ratios=[2.5, 2.5, 0.3, 1.2, 0.6],
        hspace=0.45, wspace=0.35,
        left=0.08, right=0.95, top=0.95, bottom=0.02,
    )

    # Top: MI history
    ax_hist = fig.add_subplot(gs[0, :])
    ax_hist.set_xlabel("Iteration")
    ax_hist.set_ylabel("Total MI [bits]")
    ax_hist.set_title("Sum Rate History: $\\Sigma_i I_i$")
    ax_hist.grid(True, alpha=0.3)
    line_hist, = ax_hist.plot([], [], "b-", linewidth=1.5)
    max_mi_bits = math.log2(4) * N
    ax_hist.axhline(max_mi_bits, color="green", linestyle="--", linewidth=1,
                    label=f"log2(4)*N = {max_mi_bits:.1f}")
    ax_hist.legend(loc="lower right", fontsize=8)
    text_mi_total = ax_hist.text(
        0.98, 0.95, "", transform=ax_hist.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    x_pos = np.arange(N)

    # Middle-left: per-channel MI
    ax_mi = fig.add_subplot(gs[1, 0])
    bars_mi = ax_mi.bar(x_pos, np.zeros(N), color="steelblue", width=0.6)
    ax_mi.axhline(2.0, color="green", linestyle="--", linewidth=1, label="log2(4)=2.0")
    ax_mi.set_xlabel("Channel")
    ax_mi.set_ylabel("MI [bits]")
    ax_mi.set_title("Per-channel MI")
    ax_mi.set_xticks(x_pos)
    ax_mi.set_ylim(0, 2.3)
    ax_mi.legend(fontsize=7)
    ax_mi.grid(True, alpha=0.3, axis="y")

    # Middle-center: power allocation
    ax_pow = fig.add_subplot(gs[1, 1])
    bars_pow = ax_pow.bar(x_pos, np.zeros(N), color="coral", width=0.6)
    ax_pow.set_xlabel("Channel")
    ax_pow.set_ylabel("Power $\\lambda_i^2$")
    ax_pow.set_title("Power Allocation")
    ax_pow.set_xticks(x_pos)
    ax_pow.set_ylim(0, P_TOTAL_DEFAULT * 0.5)
    ax_pow.grid(True, alpha=0.3, axis="y")
    text_p_total = ax_pow.text(
        0.98, 0.95, f"$P_{{total}}$ = {P_TOTAL_DEFAULT:.1f}",
        transform=ax_pow.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Middle-right: effective weights + objective label
    ax_w = fig.add_subplot(gs[1, 2])
    bars_w = ax_w.bar(x_pos, np.ones(N) / N, color="mediumseagreen", width=0.6)
    ax_w.set_xlabel("Channel")
    ax_w.set_ylabel("Effective weight $\\partial J/\\partial I_i$ (norm.)")
    ax_w.set_title("Objective: weighted_sum")
    ax_w.set_xticks(x_pos)
    ax_w.set_ylim(0, 0.5)
    ax_w.grid(True, alpha=0.3, axis="y")

    # Sliders: |h_i|
    sliders = []
    slider_y_start = 0.30
    fig.text(0.5, slider_y_start + 0.02, "Channel Gains",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
    for i in range(N):
        y_pos = slider_y_start - i * 0.021
        ax_s = fig.add_axes([0.15, y_pos, 0.65, 0.014])
        s = Slider(ax_s, f"|h{i}|", 0.0, 1.5,
                   valinit=math.sqrt(H_ABS2_DEFAULT[i]), valstep=0.01)
        s.vline.set_visible(False)
        sliders.append(s)

    # Policy control
    textbox_y = slider_y_start - N * 0.021 - 0.02
    ax_textbox = fig.add_axes([0.15, textbox_y, 0.68, 0.025])
    textbox = TextBox(ax_textbox, "Policy:", initial=DEFAULT_POLICY, textalignment="left")

    btn_x = 0.85
    ax_set = fig.add_axes([btn_x, textbox_y, 0.06, 0.025])
    btn_set = Button(ax_set, "Set")
    ax_clear = fig.add_axes([btn_x + 0.065, textbox_y, 0.06, 0.025])
    btn_clear = Button(ax_clear, "Clear")

    btn_row2_y = textbox_y - 0.03
    bw2, bg2 = 0.042, 0.005
    ax_rand = fig.add_axes([btn_x, btn_row2_y, bw2, 0.025])
    btn_rand = Button(ax_rand, "Rand G")
    ax_equal = fig.add_axes([btn_x + bw2 + bg2, btn_row2_y, bw2, 0.025])
    btn_equal = Button(ax_equal, "Eq G")
    ax_reset = fig.add_axes([btn_x + 2 * (bw2 + bg2), btn_row2_y, bw2, 0.025])
    btn_reset = Button(ax_reset, "Reset")

    # Active policy display
    ax_policy = fig.add_axes([0.15, textbox_y - 0.03, 0.68, 0.025])
    ax_policy.axis("off")
    text_policy_display = ax_policy.text(
        0.0, 0.5, "Active policy: (none)", transform=ax_policy.transAxes,
        fontsize=9, va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))

    # LLM status
    ax_status = fig.add_axes([0.08, textbox_y - 0.10, 0.87, 0.06])
    ax_status.axis("off")
    text_llm_status = ax_status.text(
        0.0, 0.95, "LLM Status: Idle", transform=ax_status.transAxes,
        fontsize=9, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    text_llm_reasoning = ax_status.text(
        0.0, 0.45, "", transform=ax_status.transAxes,
        fontsize=8, va="top", family="monospace", color="gray")

    # ---- callbacks ----
    def on_slider_change(val):
        shared.set_h_abs2(np.array([s.val ** 2 for s in sliders], dtype=np.float64))

    for s in sliders:
        s.on_changed(on_slider_change)

    def on_random(event):
        rand_h = np.random.uniform(0.5, 1.5, size=N)
        for i in range(N):
            sliders[i].set_val(rand_h[i])
        shared.set_h_abs2(np.array([v ** 2 for v in rand_h], dtype=np.float64))

    btn_rand.on_clicked(on_random)

    def on_equal(event):
        for i in range(N):
            sliders[i].set_val(1.0)
        shared.set_h_abs2(np.ones(N, dtype=np.float64))

    btn_equal.on_clicked(on_equal)

    def on_set(event):
        text = textbox.text.strip()
        if text:
            llm_ctrl.set_policy(text, shared)
            snap = shared.get_display_snapshot()
            llm_ctrl.trigger_llm_call(shared.get_h_abs2(), snap["mi_sub"], snap["powers"])

    btn_set.on_clicked(on_set)
    textbox.on_submit(lambda text: on_set(None))

    def on_clear(event):
        llm_ctrl.clear_policy(shared)
        textbox.set_val(DEFAULT_POLICY)

    btn_clear.on_clicked(on_clear)

    def on_reset(event):
        shared.reset()
        llm_ctrl.reset(shared)
        for i in range(N):
            sliders[i].set_val(math.sqrt(H_ABS2_DEFAULT[i]))
        textbox.set_val(DEFAULT_POLICY)

    btn_reset.on_clicked(on_reset)

    # ---- display update ----
    def on_timer():
        snap = shared.get_display_snapshot()
        mi_sub, powers, w_eff = snap["mi_sub"], snap["powers"], snap["w_eff"]
        cur_p_total = snap["p_total"]

        hist = shared.get_mi_history()
        iteration = shared.get_iteration()
        x_start = iteration - len(hist)
        visible_bits = [v * NATS2BITS for v in hist]
        line_hist.set_data(list(range(x_start, iteration)), visible_bits)
        ax_hist.set_xlim(x_start, max(iteration, x_start + 10))
        if visible_bits:
            ax_hist.set_ylim(0, max(max(visible_bits), max_mi_bits) * 1.05)

        mi_bits = mi_sub * NATS2BITS
        for i, bar in enumerate(bars_mi):
            bar.set_height(mi_bits[i])
        text_mi_total.set_text(f"$\\Sigma I_i$ = {mi_bits.sum():.3f} bits")

        for i, bar in enumerate(bars_pow):
            bar.set_height(powers[i])
        ax_pow.set_ylim(0, max(powers.max() * 1.3, cur_p_total * 0.5, 1.0))
        text_p_total.set_text(f"$P_{{total}}$ = {cur_p_total:.1f}")

        for i, bar in enumerate(bars_w):
            bar.set_height(w_eff[i])
        ax_w.set_ylim(0, max(w_eff.max() * 1.3, 0.2))
        obj_desc = snap["objective"]
        ax_w.set_title(f"Objective: {obj_desc}" if len(obj_desc) < 40
                       else f"Objective: {obj_desc[:37]}...", fontsize=9)

        policy = snap["policy"]
        ptext = f"Active policy: {policy}" if policy else "Active policy: (none)"
        text_policy_display.set_text(ptext[:100] + ("..." if len(ptext) > 100 else ""))

        model_name = llm_ctrl.get_model()
        model_tag = f"[{model_name}] " if model_name else ""
        text_llm_status.set_text(f"LLM Status: {model_tag}{snap['status']}")
        if snap["reasoning"]:
            text_llm_reasoning.set_text(
                textwrap.fill(f"Reasoning: {snap['reasoning']}", width=90))
        else:
            text_llm_reasoning.set_text("")

        fig.canvas.draw_idle()

    llm_ctrl.set_policy(DEFAULT_POLICY, shared)
    snap = shared.get_display_snapshot()
    llm_ctrl.trigger_llm_call(shared.get_h_abs2(), snap["mi_sub"], snap["powers"])

    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    shared.optimizer_running = False


def main():
    print(f"Device: {device}")
    print(f"N={N}, P_total={P_TOTAL_DEFAULT}, objective families: {OBJECTIVE_FAMILIES}")
    print(f"LLM endpoint: {LLM_BASE_URL}")
    print()
    shared = SharedState()
    llm_ctrl = LLMController()
    estimator = QPSKMonteCarloEstimator()
    create_and_run_ui(shared, llm_ctrl, estimator)


if __name__ == "__main__":
    main()
