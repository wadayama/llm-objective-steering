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
# A declared sum-rate floor is clipped to this fraction of the 2N-bit
# ceiling: demanding the ceiling itself is never attainable at finite power.
# At N = 8 this is 15.5, the value that was hard-coded before N was swept.
SR_CEILING_FRACTION = 0.96875

QPSK_CONST = torch.tensor(
    [[+1, +1], [+1, -1], [-1, +1], [-1, -1]],
    dtype=torch.float32,
) / math.sqrt(2)

H_ABS2_DEFAULT = [0.25, 0.36, 0.49, 0.64, 0.81, 1.0, 1.44, 2.25]

OBJECTIVE_FAMILIES = ("weighted_sum", "alpha_fair", "soft_min", "power_target",
                      "min_power")
NATS2BITS_CORE = 1.0 / math.log(2)

# Per-family default step sizes (gradient scales differ between families)
ETA_DEFAULTS = {
    "weighted_sum": 1.0,
    "alpha_fair": 1.0,
    "soft_min": 1.0,
    "power_target": 2.0,
    "min_power": 0.3,  # weak objective force vs stiff AL penalty: keep small
}

# Families whose objective decreases in power: never force the full budget
BALL_FAMILIES = ("power_target", "min_power")


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
        if self.family == "min_power":
            return "min_power"
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
    if spec.family == "min_power":
        # Normalized so the gradient scale matches the other families
        return -(lam ** 2).sum() / P_TOTAL_DEFAULT
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
    if spec.family == "min_power":
        return np.ones(N) / N  # dJ/dMI = 0; uniform placeholder for display
    tg = spec.targets
    s = tg.sum()
    return tg / s if s > 0 else np.ones(N) / N


# =====================================================================
# Optimizer step: joint gradient + projection
# =====================================================================
def optimize_step(lam: torch.Tensor, h_re: torch.Tensor, spec: ObjectiveSpec,
                  t_val: float, N_mc: int, dev: torch.device, p_total: float,
                  estimator: QPSKMonteCarloEstimator, eta: float | None = None,
                  projection: str = "sphere",
                  al: "AugLagState | None" = None) -> torch.Tensor:
    """One gradient-ascent step on the (augmented) objective + projection.

    al: optional PHR augmented-Lagrangian state. When it carries an active
    sum-rate constraint, the PHR penalty term is added to the objective so
    the constraint is enforced by System 1 itself.

    projection: "sphere" renormalizes to the full budget every step (used
    for monotone families); families in BALL_FAMILIES always use the ball
    projection because their objectives may deliberately underuse the budget.
    """
    if eta is None:
        eta = ETA_DEFAULTS[spec.family]

    lam_t = lam.detach().clone().requires_grad_(True)
    mi_list = [
        estimator.mi_diff(h_re[i] * lam_t[i], t_val, N_mc, dev)
        for i in range(N)
    ]
    mi = torch.stack(mi_list)
    L = compute_objective(mi, lam_t, spec, p_total, dev)
    if al is not None and al.tau_bits is not None:
        sr_bits = mi.sum() * NATS2BITS_CORE
        g = al.tau_eff - sr_bits  # violation when positive
        pen = torch.clamp(al.mu + al.rho * g, min=0.0)
        L = L - (pen ** 2 - al.mu ** 2) / (2.0 * al.rho)
    grad = torch.autograd.grad(L, lam_t)[0]

    with torch.no_grad():
        lam = lam + eta * grad
        norm_sq = (lam ** 2).sum().item()
        use_ball = (spec.family in BALL_FAMILIES) or (projection == "ball")
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


# =====================================================================
# PHR augmented Lagrangian: dual state, dual update, KKT residuals
# =====================================================================
@dataclass
class AugLagState:
    """Dual state for the sum-rate constraint SR >= tau_bits.

    tau_eff = tau_bits + margin_bits is the internally enforced level:
    a 2-sigma chance-constraint margin against Monte-Carlo measurement
    noise (sigma ~ 0.15 bits at the 20k-sample state measurement).
    mu / rho follow the PHR (Powell-Hestenes-Rockafellar) scheme with the
    standard safeguards: mu clamped to [0, mu_max], rho increased only
    while the violation fails to decrease, capped at rho_max.
    """

    tau_bits: float | None = None
    margin_bits: float = 0.3
    mu: float = 0.0
    rho: float = 1.0
    rho_max: float = 10.0   # keep the penalty from stiffening the dynamics
    rho_growth: float = 1.5
    mu_max: float = 20.0
    g_prev: float | None = None

    @property
    def tau_eff(self) -> float | None:
        if self.tau_bits is None:
            return None
        return self.tau_bits + self.margin_bits

    def set_constraint(self, tau_bits: float | None):
        """Declare / update the constraint. Dual state is kept (warm start)."""
        if tau_bits is None:
            self.tau_bits = None
            self.mu = 0.0
            self.g_prev = None
        else:
            self.tau_bits = float(np.clip(tau_bits, 0.0,
                                          SR_CEILING_FRACTION * 2.0 * N))

    def dual_update(self, sr_bits_meas: float) -> float:
        """Outer-iteration multiplier update from a measured sum rate.

        Returns the signed constraint value g = tau_eff - SR (positive =
        violated). Uses the PHR update mu <- max(0, mu + rho*g) and grows
        rho only when the violation is not shrinking (with a noise floor
        so MC fluctuations do not trigger penalty growth).
        """
        if self.tau_bits is None:
            return 0.0
        g = self.tau_eff - sr_bits_meas
        self.mu = float(np.clip(self.mu + self.rho * g, 0.0, self.mu_max))
        if (self.g_prev is not None and g > 0.15
                and g > 0.9 * self.g_prev):
            self.rho = min(self.rho * self.rho_growth, self.rho_max)
        self.g_prev = g
        return g


STATUS_TRANSIENT = "TRANSIENT"
STATUS_CONVERGED = "CONVERGED"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_DISTURBED = "DISTURBED"

EPS_STAT = 0.05    # settled when ||lam_t - lam_{t-T}||_inf below this
EPS_FEAS = 0.15    # feasibility residual noise floor [bits]


def kkt_residuals(sr_bits_meas: float, lam_now: torch.Tensor,
                  lam_prev_interval: torch.Tensor,
                  al: AugLagState) -> dict:
    """Residual triple + a loose optimality-gap bound, per dual interval."""
    r_stat = float((lam_now - lam_prev_interval).abs().max())
    if al.tau_bits is None:
        r_feas, r_comp = 0.0, 0.0
    else:
        r_feas = max(0.0, al.tau_bits - sr_bits_meas)
        r_comp = al.mu * abs(al.tau_eff - sr_bits_meas)
    gap_bound = al.mu * r_feas + r_stat
    return {"r_stat": r_stat, "r_feas": r_feas, "r_comp": r_comp,
            "gap_bound": gap_bound}


def classify_status(res: dict, al: AugLagState, prev_status: str) -> str:
    """Hysteretic status category for System 2 consumption."""
    infeasible = (al.tau_bits is not None
                  and al.mu >= 0.9 * al.mu_max
                  and res["r_feas"] > 2 * EPS_FEAS)
    if infeasible:
        return STATUS_INFEASIBLE
    settled = res["r_stat"] < EPS_STAT and res["r_feas"] < EPS_FEAS
    if settled:
        return STATUS_CONVERGED
    if prev_status == STATUS_CONVERGED and (
            res["r_stat"] > 2 * EPS_STAT or res["r_feas"] > 2 * EPS_FEAS):
        return STATUS_DISTURBED
    if prev_status == STATUS_DISTURBED and not settled:
        return STATUS_DISTURBED
    return STATUS_TRANSIENT
