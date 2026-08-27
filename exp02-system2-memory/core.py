"""Core System 1: QPSK MI estimation + objective-family gradient optimizer.

Extends the conference-version System 1 (weighted sum only, ball projection)
with:
  (1) a menu of objective families the LLM can select from
      - weighted_sum : J = sum_i w_i * MI_i
      - alpha_fair   : J = (1/N) sum_i U_alpha(MI_i)
      - soft_min     : J = -(1/beta) log sum_i exp(-beta * MI_i)
      - power_target : J = -(1/(4 P_total)) sum_i (P_i - Pi_i)^2
  (2) sphere projection for monotone families (always use the full budget),
      ball projection for power_target (targets may deliberately underuse it).
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# =====================================================================
# Constants
# =====================================================================
N = 8
SIGMA2 = 1.0
T_VAL = SIGMA2 / 2.0  # = 0.5
P_TOTAL_DEFAULT = 40.0
LAMBDA_MIN = 0.1  # lower bound on lambda_i to prevent zero-gradient trap
ALPHA_EPS = 1e-4  # regularizer for alpha_fair utility near MI = 0

QPSK_CONST = torch.tensor(
    [[+1, +1], [+1, -1], [-1, +1], [-1, -1]],
    dtype=torch.float32,
) / math.sqrt(2)

H_ABS2_DEFAULT = [0.25, 0.36, 0.49, 0.64, 0.81, 1.0, 1.44, 2.25]

OBJECTIVE_FAMILIES = ("weighted_sum", "alpha_fair", "soft_min", "power_target")

# Per-family default step sizes (gradient scales differ between families)
ETA_DEFAULTS = {
    "weighted_sum": 1.0,
    "alpha_fair": 1.0,
    "soft_min": 1.0,
    "power_target": 2.0,
}


# =====================================================================
# MI estimator (identical math to the conference version)
# =====================================================================
class QPSKMonteCarloEstimator:
    """QPSK Monte Carlo MI estimator (display: no-grad, diff: autograd-capable)."""

    @torch.no_grad()
    def mi_display(self, a: float, t_val: float, N_mc: int, dev: torch.device) -> float:
        M = 4
        const = QPSK_CONST.to(dev)
        z = torch.randn(N_mc, 2, device=dev)
        total_lse = 0.0
        for k in range(M):
            diffs = const[k].unsqueeze(0) - const
            a_diffs = a * diffs
            sq_norms = (a_diffs ** 2).sum(dim=1)
            cross = a_diffs @ z.T
            exponents = -sq_norms.unsqueeze(1) / (2 * t_val) - cross / math.sqrt(t_val)
            lse = torch.logsumexp(exponents, dim=0)
            total_lse += lse.mean().item()
        return math.log(M) - total_lse / M

    def mi_diff(self, a: torch.Tensor, t_val: float, N_mc: int, dev: torch.device) -> torch.Tensor:
        M = 4
        const = QPSK_CONST.to(dev)
        z = torch.randn(N_mc, 2, device=dev)
        total_lse = torch.tensor(0.0, device=dev)
        for k in range(M):
            diffs = const[k].unsqueeze(0) - const
            a_diffs = a * diffs
            sq_norms = (a_diffs ** 2).sum(dim=1)
            cross = a_diffs @ z.T
            exponents = -sq_norms.unsqueeze(1) / (2 * t_val) - cross / math.sqrt(t_val)
            lse = torch.logsumexp(exponents, dim=0)
            total_lse = total_lse + lse.mean()
        return math.log(M) - total_lse / M


# =====================================================================
# Objective specification
# =====================================================================
@dataclass
class ObjectiveSpec:
    """Objective family + parameters, as steered by System 2 (the LLM)."""

    family: str = "weighted_sum"
    weights: np.ndarray = field(default_factory=lambda: np.ones(N) / N)
    alpha: float = 2.0     # alpha_fair: fairness strength (1 = proportional fair)
    beta: float = 10.0     # soft_min: sharpness of the soft minimum
    targets: np.ndarray = field(default_factory=lambda: np.ones(N) * (P_TOTAL_DEFAULT / N))

    def validate(self, p_total: float) -> "ObjectiveSpec":
        """Clamp parameters to safe ranges (guardrail against LLM output)."""
        if self.family not in OBJECTIVE_FAMILIES:
            raise ValueError(f"unknown objective family: {self.family}")
        w = np.clip(np.asarray(self.weights, dtype=np.float64), 0.0, None)
        self.weights = w / w.sum() if w.sum() > 0 else np.ones(N) / N
        self.alpha = float(np.clip(self.alpha, 0.5, 20.0))
        self.beta = float(np.clip(self.beta, 1.0, 50.0))
        tg = np.clip(np.asarray(self.targets, dtype=np.float64), 0.0, None)
        if tg.sum() > p_total:  # scale down infeasible targets
            tg = tg * (p_total / tg.sum())
        self.targets = tg
        return self

    def describe(self) -> str:
        if self.family == "weighted_sum":
            return "weighted_sum(w=[" + ", ".join(f"{v:.2f}" for v in self.weights) + "])"
        if self.family == "alpha_fair":
            return f"alpha_fair(alpha={self.alpha:.1f})"
        if self.family == "soft_min":
            return f"soft_min(beta={self.beta:.1f})"
        return "power_target(P=[" + ", ".join(f"{v:.1f}" for v in self.targets) + "])"


def compute_objective(mi: torch.Tensor, lam: torch.Tensor, spec: ObjectiveSpec,
                      p_total: float, dev: torch.device) -> torch.Tensor:
    """J(mi, lam) for the selected family. All families are maximized."""
    if spec.family == "weighted_sum":
        w = torch.tensor(spec.weights, dtype=torch.float32, device=dev)
        return (w * mi).sum()
    if spec.family == "alpha_fair":
        # Normalized alpha-fair ascent direction: the raw utility gradient
        # (MI + eps)^-alpha explodes as MI -> 0, so we use detached weights
        # normalized to sum 1. Same fixed points, weighted_sum-scale gradient.
        w = (mi.detach() + ALPHA_EPS) ** (-spec.alpha)
        w = w / w.sum()
        return (w * mi).sum()
    if spec.family == "soft_min":
        b = spec.beta
        return -torch.logsumexp(-b * mi, dim=0) / b
    if spec.family == "power_target":
        tg = torch.tensor(spec.targets, dtype=torch.float32, device=dev)
        return -((lam ** 2 - tg) ** 2).sum() / (4.0 * p_total)
    raise ValueError(spec.family)


def effective_weights(mi: np.ndarray, spec: ObjectiveSpec) -> np.ndarray:
    """dJ/dMI_i normalized to sum 1 — the 'weights the objective implies now'.

    Unifies the display across families: for weighted_sum these are the w_i
    themselves; for soft_min a softmax over -beta*MI; for alpha_fair MI^-alpha
    normalized. For power_target (not a function of MI) the normalized targets.
    """
    if spec.family == "weighted_sum":
        return spec.weights.copy()
    if spec.family == "alpha_fair":
        g = (mi + ALPHA_EPS) ** (-spec.alpha)
        return g / g.sum()
    if spec.family == "soft_min":
        e = np.exp(-spec.beta * (mi - mi.min()))
        return e / e.sum()
    tg = spec.targets
    s = tg.sum()
    return tg / s if s > 0 else np.ones(N) / N


# =====================================================================
# Optimizer step: joint gradient + projection
# =====================================================================
def optimize_step(lam: torch.Tensor, h_re: torch.Tensor, spec: ObjectiveSpec,
                  t_val: float, N_mc: int, dev: torch.device, p_total: float,
                  estimator: QPSKMonteCarloEstimator, eta: float | None = None,
                  projection: str = "sphere") -> torch.Tensor:
    """One gradient-ascent step on J followed by projection and clamping.

    projection: "sphere" renormalizes to the full budget every step (used for
    monotone families); "ball" only rescales when the budget is exceeded.
    power_target always uses ball projection regardless of the argument,
    because its targets may deliberately sum to less than p_total.
    """
    if eta is None:
        eta = ETA_DEFAULTS[spec.family]

    lam_t = lam.detach().clone().requires_grad_(True)
    mi_list = [
        estimator.mi_diff(h_re[i] * lam_t[i], t_val, N_mc, dev)
        for i in range(N)
    ]
    mi = torch.stack(mi_list)
    J = compute_objective(mi, lam_t, spec, p_total, dev)
    grad = torch.autograd.grad(J, lam_t)[0]

    with torch.no_grad():
        lam = lam + eta * grad
        norm_sq = (lam ** 2).sum().item()
        use_ball = (spec.family == "power_target") or (projection == "ball")
        if use_ball:
            if norm_sq > p_total:
                lam *= math.sqrt(p_total / norm_sq)
        else:  # sphere: always use the full budget
            if norm_sq > 0:
                lam *= math.sqrt(p_total / norm_sq)
        lam.clamp_(min=LAMBDA_MIN)
    return lam


# =====================================================================
# Metrics
# =====================================================================
def jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    if s <= 0:
        return 0.0
    return float(s ** 2 / (len(x) * (x ** 2).sum()))


def measure_mi(lam: torch.Tensor, h_re: torch.Tensor,
               estimator: QPSKMonteCarloEstimator, N_mc: int,
               dev: torch.device) -> np.ndarray:
    mi = np.zeros(N)
    for i in range(N):
        a_i = float(h_re[i]) * float(lam[i])
        mi[i] = estimator.mi_display(a_i, T_VAL, N_mc, dev)
    return mi
