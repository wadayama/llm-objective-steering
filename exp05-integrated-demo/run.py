#!/usr/bin/env python3
"""
exp05: Integrated LLM Steering Demo
===================================

Interactive demo integrating everything validated in exp01-exp04:

- exp01: LLM-selectable objective families + sphere projection;
         effective-weights panel
- exp02: bounded control history (K=8) in the LLM prompt
- exp03: constraint declaration enforced by a PHR augmented Lagrangian;
         KKT residuals -> certified status (TRANSIENT/CONVERGED/
         INFEASIBLE/DISTURBED), shadow price mu; status shown in the UI
         and reported to the LLM (semantic prompt)
- exp02c: disclosed EMA actuator on P_total (kept, per ablation result)
- exp04b: state verbalization only where grounded (status categories and
          constraint margins; no free-floating severity language)

Requires LM Studio on localhost:1234. Set DEMO_HEADLESS=1 to import
without a GUI backend (for smoke tests).
"""

import json
import math
import os
import textwrap
import threading
import urllib.request

import numpy as np
import torch
import matplotlib
if os.environ.get("DEMO_HEADLESS"):
    matplotlib.use("Agg")
# Otherwise the backend is left to matplotlib's auto-selection, which tries
# macosx, qtagg, gtk4agg, gtk3agg, tkagg, wxagg in that order: the native
# macOS backend here, Qt/GTK/Tk on Linux/Windows. Forcing "macosx" would fail
# outright off macOS and also disables that fallback (use() clears
# rcParams["backend_fallback"]).
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from openai import OpenAI

from core import (
    N, T_VAL, P_TOTAL_DEFAULT, H_ABS2_DEFAULT,
    QPSKMonteCarloEstimator, ObjectiveSpec, AugLagState,
    optimize_step, measure_mi, effective_weights,
    kkt_residuals, classify_status,
    STATUS_TRANSIENT, STATUS_CONVERGED, STATUS_INFEASIBLE, STATUS_DISTURBED,
)
from steering import (
    ControlHistory, build_user_message, compute_kpis, ema_merge_spec,
    parse_response, system_prompt, NATS2BITS,
)

# =====================================================================
# Configuration
# =====================================================================
N_MC_DISPLAY = 50_000
N_MC_GRAD = 10_000
STEPS_PER_FRAME = 5
DUAL_EVERY_STEPS = 10     # dual update cadence (multiple of STEPS_PER_FRAME)
HISTORY_WINDOW = 200
K_HIST = 8
LLM_CALL_INTERVAL = 1.0
EMA_ALPHA_P = 0.5
EMA_ALPHA_PARAM = 0.5
DEFAULT_POLICY = "Maximize total throughput while ensuring all channels get some power"

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_TIMEOUT = 30.0
# Pin the model with DEMO_LLM_MODEL when LM Studio holds more than one;
# otherwise a loaded, non-embedding model is selected automatically.
LLM_MODEL = os.environ.get("DEMO_LLM_MODEL", "").strip()
# Reasoning models spend this budget on hidden thought before they answer, and
# a reply truncated mid-thought arrives with empty content. Raise it for those.
LLM_MAX_TOKENS = int(os.environ.get("DEMO_MAX_TOKENS", "2048"))

device = torch.device("cpu")

STATUS_COLORS = {
    STATUS_TRANSIENT: "lightgray",
    STATUS_CONVERGED: "lightgreen",
    STATUS_INFEASIBLE: "lightcoral",
    STATUS_DISTURBED: "moccasin",
}


# =====================================================================
# SharedState
# =====================================================================
class SharedState:
    """Thread-safe state shared by optimizer loop, UI, and LLM controller."""

    def __init__(self):
        self._lock = threading.Lock()
        self.h_abs2 = np.array(H_ABS2_DEFAULT, dtype=np.float64)
        self.h_re = torch.tensor(np.sqrt(self.h_abs2), dtype=torch.float32, device=device)
        self.lam = torch.ones(N, device=device) * math.sqrt(P_TOTAL_DEFAULT / N)
        self.p_total = P_TOTAL_DEFAULT
        self.al = AugLagState()
        self.status = STATUS_TRANSIENT
        self.kkt = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
        self.mi_history: list[float] = []
        self.iteration = 0
        self.optimizer_running = True
        self.active_policy = ""
        self._display = self._default_display()

    @staticmethod
    def _default_display():
        return {
            "mi_sub": np.zeros(N), "powers": np.zeros(N),
            "w_eff": np.ones(N) / N, "objective": "weighted_sum (equal)",
            "constraint": "(none)", "p_total": P_TOTAL_DEFAULT,
            "status": STATUS_TRANSIENT, "gap": 1.0, "mu": 0.0,
            "reasoning": "", "llm_status": "Idle", "iteration": 0, "policy": "",
        }

    # ---- policy ----
    def set_policy(self, policy: str):
        with self._lock:
            self.active_policy = policy

    def get_policy(self) -> str:
        with self._lock:
            return self.active_policy

    # ---- physics ----
    def set_h_abs2(self, h_abs2: np.ndarray):
        with self._lock:
            self.h_abs2[:] = h_abs2
            self.h_re = torch.tensor(np.sqrt(self.h_abs2), dtype=torch.float32, device=device)

    def get_h_abs2(self) -> np.ndarray:
        with self._lock:
            return self.h_abs2.copy()

    def get_h_re(self) -> torch.Tensor:
        with self._lock:
            return self.h_re.clone()

    def get_lam(self) -> torch.Tensor:
        with self._lock:
            return self.lam.clone()

    def set_lam(self, lam: torch.Tensor):
        with self._lock:
            self.lam = lam

    def set_p_total(self, p: float):
        with self._lock:
            self.p_total = p

    # ---- augmented Lagrangian ----
    def set_constraints(self, tau_bits: float | None,
                        tau_ch_bits: float | None):
        with self._lock:
            self.al.set_constraint(tau_bits)
            self.al.set_channel_constraint(tau_ch_bits)

    def get_taus(self):
        with self._lock:
            return self.al.tau_bits, self.al.tau_ch_bits

    def dual_update_and_classify(self, sr_bits: float, mi_bits: np.ndarray,
                                 lam_now: torch.Tensor,
                                 lam_prev: torch.Tensor):
        with self._lock:
            self.al.dual_update(sr_bits, mi_bits)
            self.kkt = kkt_residuals(sr_bits, lam_now, lam_prev, self.al,
                                     mi_bits_meas=mi_bits)
            self.status = classify_status(self.kkt, self.al, self.status)
            return self.status, dict(self.kkt), self.al.mu

    def get_al_snapshot(self):
        with self._lock:
            mu_eff = self.al.mu + float(self.al.mu_ch.max())
            return (self.al.tau_bits, self.al.tau_ch_bits, mu_eff,
                    self.status, dict(self.kkt))

    # ---- history & display ----
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
            for k in ("mi_sub", "powers", "w_eff"):
                d[k] = d[k].copy()
            return d

    def reset(self):
        with self._lock:
            self.h_abs2[:] = H_ABS2_DEFAULT
            self.h_re = torch.tensor(np.sqrt(self.h_abs2), dtype=torch.float32, device=device)
            self.lam = torch.ones(N, device=device) * math.sqrt(P_TOTAL_DEFAULT / N)
            self.p_total = P_TOTAL_DEFAULT
            self.al = AugLagState()
            self.status = STATUS_TRANSIENT
            self.kkt = {"r_stat": 1.0, "r_feas": 0.0, "r_comp": 0.0, "gap_bound": 1.0}
            self.mi_history.clear()
            self.iteration = 0
            self.active_policy = ""
            self._display = self._default_display()


# =====================================================================
# LLMController (declaring System 2 with memory)
# =====================================================================
class LLMController:
    def __init__(self):
        self.spec = ObjectiveSpec()
        self.p_total = P_TOTAL_DEFAULT
        self.active_policy = ""
        self.last_reasoning = ""
        self.llm_status = "Idle"
        self.is_calling = False
        self.call_id = 0
        self.call_no = 0
        self.hist = ControlHistory(K_HIST)
        self._lock = threading.Lock()
        self._retrigger = False
        self._model_name = None
        self._periodic_stop = threading.Event()
        self._periodic_thread = None
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                              timeout=LLM_TIMEOUT)
        self.sys_prompt = system_prompt("semantic")

    def get_model(self):
        with self._lock:
            return self._model_name

    def _resolve_model(self):
        if self._model_name is not None:
            return self._model_name
        self._model_name = LLM_MODEL or self._auto_select_model()
        return self._model_name

    def _auto_select_model(self):
        """Pick a loaded, non-embedding model.

        /v1/models lists everything LM Studio knows about -- embeddings and
        unloaded models included -- so taking the first entry picks the wrong
        one as soon as more than one model is around. LM Studio's own endpoint
        reports type and load state; fall back to the OpenAI-compatible list.
        """
        for m in self._list_native_models():
            if m.get("state") == "loaded" and m.get("type") != "embeddings":
                return m["id"]
        models = self._client.models.list()
        if models.data:
            return models.data[0].id
        raise RuntimeError("No models loaded in LM Studio")

    def _list_native_models(self):
        url = LLM_BASE_URL.rsplit("/v1", 1)[0] + "/api/v0/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read()).get("data", [])
        except Exception:
            return []

    # ---- policy management ----
    def set_policy(self, text: str, shared: SharedState):
        with self._lock:
            self.active_policy = text.strip()
            self.call_id += 1
            self.hist.clear()
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
            self.llm_status = "Policy cleared"
            self.call_id += 1
            self.hist.clear()
        shared.set_policy("")
        shared.set_constraints(None, None)
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
            self.trigger_llm_call(shared)

    # ---- accessors ----
    def get_control(self):
        with self._lock:
            spec = ObjectiveSpec(
                family=self.spec.family, weights=self.spec.weights.copy(),
                alpha=self.spec.alpha, beta=self.spec.beta,
                targets=self.spec.targets.copy())
            return spec, self.p_total, self.last_reasoning, self.llm_status

    # ---- LLM call ----
    def trigger_llm_call(self, shared: SharedState):
        with self._lock:
            if self.is_calling:
                self._retrigger = True
                return
            self.is_calling = True
            self._retrigger = False
            my_id = self.call_id
            self.llm_status = "Calling LLM..."
        threading.Thread(target=self._call_llm, args=(shared, my_id),
                         daemon=True).start()

    def _call_llm(self, shared: SharedState, my_id: int):
        try:
            snap = shared.get_display_snapshot()
            tau, tau_ch, mu, status, kkt = shared.get_al_snapshot()
            mi_nats = snap["mi_sub"]
            powers = snap["powers"]
            kpis = compute_kpis(mi_nats, powers, self.p_total)
            parts = []
            if tau is not None:
                parts.append(f"sum_rate>={tau:.1f}")
            if tau_ch is not None:
                parts.append(f"each_MI>={tau_ch:.2f}")
            cdesc = "; ".join(parts) if parts else "(none)"
            sline = (f"{status} | gap_bound = {kkt['gap_bound']:.3f} bits | "
                     f"constraint {cdesc} | shadow_price mu = {mu:.3f}")
            with self._lock:
                policy = self.active_policy
                hist_block = self.hist.render(kpis)
            msg = build_user_message(
                shared.get_h_abs2(), mi_nats, powers, self.p_total,
                self.spec.describe(), cdesc, sline, policy,
                history_block=hist_block)

            model_name = self._resolve_model()
            resp = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": self.sys_prompt},
                          {"role": "user", "content": msg}],
                temperature=0.0, max_tokens=LLM_MAX_TOKENS)
            choice = resp.choices[0]
            raw = (choice.message.content or "").strip()
            if not raw:
                # A reasoning model can spend the whole budget on hidden
                # thought and answer with nothing; name that instead of
                # failing downstream with an opaque JSON error.
                raise RuntimeError(
                    f"model hit the {LLM_MAX_TOKENS}-token limit "
                    "(raise DEMO_MAX_TOKENS)"
                    if choice.finish_reason == "length"
                    else "model returned an empty reply")
            (spec_new, tau_new, tau_ch_new, p_cmd,
             reasoning, escalated) = parse_response(raw)

            # Guardrail: unconstrained min_power = "shut everything down",
            # which is never a sane translation. Reject the declaration
            # (keep the previous objective) and tell the LLM why via status.
            rejected = (spec_new.family == "min_power"
                        and tau_new is None and tau_ch_new is None)

            with self._lock:
                if self.call_id == my_id:
                    if rejected:
                        self.llm_status = ("Rejected: min_power requires a "
                                           "declared constraint")
                        self.last_reasoning = reasoning
                        self.hist.add(self.call_no,
                                      "REJECTED min_power (no constraint); kept "
                                      + self.spec.describe(),
                                      cdesc, kpis, status)
                        self.call_no += 1
                        apply_tau = False
                    else:
                        if escalated is not None:
                            # An escalation is a discrete request for more of a
                            # resource, not a servo command, so it bypasses the
                            # EMA that damps ordinary budget commands.
                            self.p_total = escalated
                        elif p_cmd is not None:
                            self.p_total = EMA_ALPHA_P * p_cmd + (1 - EMA_ALPHA_P) * self.p_total
                        self.spec = ema_merge_spec(spec_new, self.spec,
                                                   EMA_ALPHA_PARAM, self.p_total)
                        self.last_reasoning = reasoning
                        self.llm_status = (
                            f"OK [{self.spec.family}]"
                            + ("" if escalated is None
                               else f" — escalated P_total to {escalated:.0f}"))
                        self.hist.add(self.call_no, self.spec.describe(), cdesc,
                                      kpis, status)
                        self.call_no += 1
                        apply_tau = True
                else:
                    self.llm_status = "Stale (new call arrived)"
                    apply_tau = False
                self.is_calling = False
            if apply_tau:
                shared.set_constraints(tau_new, tau_ch_new)

        except Exception as e:
            with self._lock:
                err = str(e)
                self.llm_status = f"Error: {err[:70]}" + ("..." if len(err) > 70 else "")
                self.is_calling = False

        with self._lock:
            retrig = self._retrigger
            self._retrigger = False
        if retrig:
            self.trigger_llm_call(shared)

    def reset(self, shared: SharedState):
        self._stop_periodic()
        with self._lock:
            self.spec = ObjectiveSpec()
            self.p_total = P_TOTAL_DEFAULT
            self.active_policy = ""
            self.last_reasoning = ""
            self.llm_status = "Reset"
            self.call_id += 1
            self.call_no = 0
            self.hist.clear()
        shared.set_policy("")


# =====================================================================
# Optimizer loop (System 1 with AL + dual updates)
# =====================================================================
def run_optimizer_loop(shared: SharedState, llm_ctrl: LLMController,
                       estimator: QPSKMonteCarloEstimator):
    steps_since_dual = 0
    lam_interval = shared.get_lam()
    while shared.optimizer_running:
        spec, cur_p_total, reasoning, llm_status = llm_ctrl.get_control()
        shared.set_p_total(cur_p_total)

        h_re = shared.get_h_re()
        lam = shared.get_lam()

        for _ in range(STEPS_PER_FRAME):
            with shared._lock:
                al = shared.al  # optimize_step only reads al fields
            lam = optimize_step(lam, h_re, spec, T_VAL, N_MC_GRAD, device,
                                cur_p_total, estimator, al=al)
            shared.increment_iteration()
        steps_since_dual += STEPS_PER_FRAME
        shared.set_lam(lam)

        mi_sub = measure_mi(lam, h_re, estimator, N_MC_DISPLAY, device)
        sr_bits = float(mi_sub.sum() * NATS2BITS)

        if steps_since_dual >= DUAL_EVERY_STEPS:
            shared.dual_update_and_classify(sr_bits, mi_sub * NATS2BITS,
                                            lam, lam_interval)
            lam_interval = lam.clone()
            steps_since_dual = 0

        tau, tau_ch, mu, status, kkt = shared.get_al_snapshot()
        powers = (lam ** 2).cpu().numpy()
        w_eff = effective_weights(mi_sub, spec)

        shared.publish_display(
            mi_sub=mi_sub, powers=powers, w_eff=w_eff,
            objective=spec.describe(),
            constraint=(("; ".join(
                ([f"sum_rate >= {tau:.1f}"] if tau is not None else []) +
                ([f"each MI >= {tau_ch:.2f}"] if tau_ch is not None else []))
                or "(none)")),
            p_total=cur_p_total, status=status, gap=kkt["gap_bound"], mu=mu,
            reasoning=reasoning, llm_status=llm_status,
            iteration=shared.get_iteration(), policy=shared.get_policy(),
        )
        shared.append_mi_history(float(mi_sub.sum()))


# =====================================================================
# UI
# =====================================================================
def create_and_run_ui(shared: SharedState, llm_ctrl: LLMController,
                      estimator: QPSKMonteCarloEstimator):
    opt_thread = threading.Thread(
        target=run_optimizer_loop, args=(shared, llm_ctrl, estimator), daemon=True)
    opt_thread.start()

    fig = plt.figure(figsize=(14, 12))
    fig.canvas.manager.set_window_title("exp05: Integrated LLM Steering Demo")

    gs = fig.add_gridspec(
        5, 3, height_ratios=[2.5, 2.5, 0.3, 1.2, 0.6],
        hspace=0.45, wspace=0.35, left=0.08, right=0.95, top=0.95, bottom=0.02)

    # Top: MI history + constraint line
    ax_hist = fig.add_subplot(gs[0, :])
    ax_hist.set_xlabel("Iteration")
    ax_hist.set_ylabel("Total MI [bits]")
    ax_hist.set_title("Sum Rate History: $\\Sigma_i I_i$")
    ax_hist.grid(True, alpha=0.3)
    line_hist, = ax_hist.plot([], [], "b-", linewidth=1.5)
    max_mi_bits = 2.0 * N
    ax_hist.axhline(max_mi_bits, color="green", linestyle="--", linewidth=1,
                    label=f"max = {max_mi_bits:.0f} bits")
    line_tau = ax_hist.axhline(0, color="red", linestyle="--", linewidth=1.2,
                               visible=False, label="declared constraint")
    ax_hist.legend(loc="lower right", fontsize=8)
    text_mi_total = ax_hist.text(
        0.98, 0.95, "", transform=ax_hist.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    x_pos = np.arange(N)

    ax_mi = fig.add_subplot(gs[1, 0])
    bars_mi = ax_mi.bar(x_pos, np.zeros(N), color="steelblue", width=0.6)
    ax_mi.axhline(2.0, color="green", linestyle="--", linewidth=1)
    line_tau_ch = ax_mi.axhline(0, color="red", linestyle="--", linewidth=1.2,
                                visible=False)
    ax_mi.set_xlabel("Channel")
    ax_mi.set_ylabel("MI [bits]")
    ax_mi.set_title("Per-channel MI")
    ax_mi.set_xticks(x_pos)
    ax_mi.set_ylim(0, 2.3)
    ax_mi.grid(True, alpha=0.3, axis="y")

    ax_pow = fig.add_subplot(gs[1, 1])
    bars_pow = ax_pow.bar(x_pos, np.zeros(N), color="coral", width=0.6)
    ax_pow.set_xlabel("Channel")
    ax_pow.set_ylabel("Power $\\lambda_i^2$")
    ax_pow.set_title("Power Allocation")
    ax_pow.set_xticks(x_pos)
    ax_pow.set_ylim(0, P_TOTAL_DEFAULT * 0.5)
    ax_pow.grid(True, alpha=0.3, axis="y")
    text_p_total = ax_pow.text(
        0.98, 0.95, "", transform=ax_pow.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax_w = fig.add_subplot(gs[1, 2])
    bars_w = ax_w.bar(x_pos, np.ones(N) / N, color="mediumseagreen", width=0.6)
    ax_w.set_xlabel("Channel")
    ax_w.set_ylabel("Effective weight (norm.)")
    ax_w.set_title("Objective: weighted_sum")
    ax_w.set_xticks(x_pos)
    ax_w.set_ylim(0, 0.5)
    ax_w.grid(True, alpha=0.3, axis="y")

    # Sliders
    sliders = []
    slider_y_start = 0.30
    fig.text(0.5, slider_y_start + 0.02, "Channel Gains",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
    for i in range(N):
        ax_s = fig.add_axes([0.15, slider_y_start - i * 0.021, 0.65, 0.014])
        s = Slider(ax_s, f"|h{i}|", 0.0, 1.5,
                   valinit=math.sqrt(H_ABS2_DEFAULT[i]), valstep=0.01)
        s.vline.set_visible(False)
        sliders.append(s)

    # Policy controls
    textbox_y = slider_y_start - N * 0.021 - 0.02
    ax_textbox = fig.add_axes([0.15, textbox_y, 0.68, 0.025])
    textbox = TextBox(ax_textbox, "Policy:", initial=DEFAULT_POLICY,
                      textalignment="left")
    btn_x = 0.85
    btn_set = Button(fig.add_axes([btn_x, textbox_y, 0.06, 0.025]), "Set")
    btn_clear = Button(fig.add_axes([btn_x + 0.065, textbox_y, 0.06, 0.025]), "Clear")
    btn_row2_y = textbox_y - 0.03
    bw2, bg2 = 0.042, 0.005
    btn_rand = Button(fig.add_axes([btn_x, btn_row2_y, bw2, 0.025]), "Rand G")
    btn_equal = Button(fig.add_axes([btn_x + bw2 + bg2, btn_row2_y, bw2, 0.025]), "Eq G")
    btn_reset = Button(fig.add_axes([btn_x + 2 * (bw2 + bg2), btn_row2_y, bw2, 0.025]), "Reset")

    # Policy display
    ax_policy = fig.add_axes([0.15, textbox_y - 0.03, 0.68, 0.025])
    ax_policy.axis("off")
    text_policy = ax_policy.text(
        0.0, 0.5, "Active policy: (none)", transform=ax_policy.transAxes,
        fontsize=9, va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))

    # Optimizer certificate + LLM status
    ax_status = fig.add_axes([0.08, textbox_y - 0.115, 0.87, 0.075])
    ax_status.axis("off")
    text_opt_status = ax_status.text(
        0.0, 0.98, "", transform=ax_status.transAxes, fontsize=9, va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.9))
    text_llm_status = ax_status.text(
        0.0, 0.60, "LLM Status: Idle", transform=ax_status.transAxes,
        fontsize=9, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    text_llm_reasoning = ax_status.text(
        0.0, 0.24, "", transform=ax_status.transAxes,
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
            llm_ctrl.trigger_llm_call(shared)

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

        # constraint lines
        tau, tau_ch = shared.get_taus()
        if tau is not None:
            line_tau.set_ydata([tau, tau])
            line_tau.set_visible(True)
        else:
            line_tau.set_visible(False)
        if tau_ch is not None:
            line_tau_ch.set_ydata([tau_ch, tau_ch])
            line_tau_ch.set_visible(True)
        else:
            line_tau_ch.set_visible(False)

        mi_bits = mi_sub * NATS2BITS
        for i, bar in enumerate(bars_mi):
            bar.set_height(mi_bits[i])
        text_mi_total.set_text(f"$\\Sigma I_i$ = {mi_bits.sum():.3f} bits")

        for i, bar in enumerate(bars_pow):
            bar.set_height(powers[i])
        ax_pow.set_ylim(0, max(powers.max() * 1.3, cur_p_total * 0.5, 1.0))
        text_p_total.set_text(
            f"$\\Sigma P_i$ = {powers.sum():.1f} / cap {cur_p_total:.1f}")

        for i, bar in enumerate(bars_w):
            bar.set_height(w_eff[i])
        ax_w.set_ylim(0, max(w_eff.max() * 1.3, 0.2))
        obj = snap["objective"]
        ax_w.set_title(f"Objective: {obj[:37] + '...' if len(obj) > 40 else obj}",
                       fontsize=9)

        policy = snap["policy"]
        ptext = f"Active policy: {policy}" if policy else "Active policy: (none)"
        text_policy.set_text(ptext[:100] + ("..." if len(ptext) > 100 else ""))

        # optimizer certificate (grounded verbalization only)
        status = snap["status"]
        cert = (f"Optimizer: {status:10s} | constraint {snap['constraint']:22s} | "
                f"gap <= {snap['gap']:.3f} bits | shadow price mu = {snap['mu']:.3f}")
        text_opt_status.set_text(cert)
        text_opt_status.get_bbox_patch().set_facecolor(
            STATUS_COLORS.get(status, "lightgray"))

        model_name = llm_ctrl.get_model()
        tag = f"[{model_name}] " if model_name else ""
        text_llm_status.set_text(f"LLM Status: {tag}{snap['llm_status']}")
        if snap["reasoning"]:
            text_llm_reasoning.set_text(
                textwrap.fill(f"Reasoning: {snap['reasoning']}", width=90))
        else:
            text_llm_reasoning.set_text("")

        fig.canvas.draw_idle()

    llm_ctrl.set_policy(DEFAULT_POLICY, shared)
    llm_ctrl.trigger_llm_call(shared)

    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    shared.optimizer_running = False


def main():
    print(f"exp05 integrated demo — device: {device}")
    print(f"N={N}, P_total={P_TOTAL_DEFAULT}, K_hist={K_HIST}, "
          f"dual update every {DUAL_EVERY_STEPS} steps")
    print(f"LLM endpoint: {LLM_BASE_URL}")
    print(f"LLM model: {LLM_MODEL or '(auto: first loaded non-embedding model)'}"
          f", max_tokens={LLM_MAX_TOKENS}")
    shared = SharedState()
    llm_ctrl = LLMController()
    estimator = QPSKMonteCarloEstimator()
    create_and_run_ui(shared, llm_ctrl, estimator)


if __name__ == "__main__":
    main()
