"""Benchmark v0.1 scenario specifications.

Each scenario is a dict:
  id, category, description, policy
  kind: "translation" (single LLM call against a fixed synthetic state)
        or "episode" (full closed loop)
  For translation: check(spec, tau, tau_ch) -> (ok: bool, note: str)
  For episodes: steps, seeds, oracle (declaration for the oracle baseline
        and translation scoring), env_schedule (optional gain changes),
        pass_fn(metrics) -> bool, expected_fail (diagnostic scenarios)

All scenarios use N=8 with the default gains unless stated.
"""

import numpy as np

from core import N, H_ABS2_DEFAULT

# Fixed synthetic state for translation scenarios (mid-convergence snapshot)
TRANSLATION_STATE = {
    "mi_nats": np.array([0.62, 0.71, 0.78, 0.83, 0.88, 0.92, 0.98, 1.02]),
    "powers": np.array([5.0] * N),
    "p_total": 40.0,
}


def _w(spec):
    return np.asarray(spec.weights, dtype=float)


SCENARIOS = [
    # ---------------- translation ----------------
    dict(
        id="T1-throughput",
        category="translation",
        policy="Maximize total throughput",
        kind="translation",
        check=lambda spec, tau, tau_ch: (
            spec.family == "weighted_sum"
            and float(_w(spec).std() / max(_w(spec).mean(), 1e-9)) < 0.2,
            f"family={spec.family}",
        ),
    ),
    dict(
        id="T2-fair-rates",
        category="translation",
        policy="Ensure all channels achieve similar data rates",
        kind="translation",
        check=lambda spec, tau, tau_ch: (
            spec.family in ("soft_min", "alpha_fair"),
            f"family={spec.family}",
        ),
    ),
    dict(
        id="T3-equal-power",
        category="translation",
        policy="Equalize transmit power across all channels",
        kind="translation",
        check=lambda spec, tau, tau_ch: (
            spec.family == "power_target"
            and float(np.asarray(spec.targets).std()) < 0.5,
            f"family={spec.family}",
        ),
    ),
    dict(
        id="T4-shutdown-subset",
        category="translation",
        policy="Shut down channels 0 to 3",
        kind="translation",
        check=lambda spec, tau, tau_ch: (
            (spec.family == "power_target"
             and all(np.asarray(spec.targets)[:4] < 0.5)
             and any(np.asarray(spec.targets)[4:] > 1.0))
            or (spec.family == "weighted_sum"
                and all(_w(spec)[:4] < 1e-3)),
            f"family={spec.family}",
        ),
    ),
    dict(
        id="T5-prioritize",
        category="translation",
        policy="Prioritize channels 6 and 7",
        kind="translation",
        check=lambda spec, tau, tau_ch: (
            spec.family == "weighted_sum"
            and float(_w(spec)[6:].min()) > 1.5 * float(_w(spec)[:6].max()),
            f"family={spec.family}",
        ),
    ),
    dict(
        id="T6-servo-decl",
        category="translation",
        policy=("Minimize total transmit power while keeping the total "
                "data rate above 10 bits"),
        kind="translation",
        check=lambda spec, tau, tau_ch: (
            spec.family == "min_power" and tau is not None
            and abs(tau - 10.0) < 0.51,
            f"family={spec.family} tau={tau}",
        ),
    ),
    dict(
        id="T7-perch-decl",
        category="translation",
        policy=("minimize total power while keeping MI larger than 1.0 bit "
                "for all the channels"),
        kind="translation",
        check=lambda spec, tau, tau_ch: (
            spec.family == "min_power" and tau_ch is not None
            and abs(tau_ch - 1.0) < 0.11,
            f"family={spec.family} tau_ch={tau_ch}",
        ),
    ),
    # ---------------- servo episodes ----------------
    dict(
        id="S1-sumrate-servo",
        category="servo",
        policy=("Minimize total transmit power while keeping the total "
                "data rate above 10 bits"),
        kind="episode",
        steps=250,
        seeds=[42, 43],
        oracle=dict(family="min_power", tau=10.0, tau_ch=None),
        pass_fn=lambda m: (m["violation_rate"] <= 0.05
                           and m["power_regret"] <= 3.0),
    ),
    dict(
        id="S2-perchannel-servo",
        category="servo",
        policy=("minimize total power while keeping MI larger than 1.0 bit "
                "for all the channels"),
        kind="episode",
        steps=250,
        seeds=[42],
        oracle=dict(family="min_power", tau=None, tau_ch=1.0),
        pass_fn=lambda m: (m["ch_violation_rate"] <= 0.05
                           and m["power_regret"] <= 3.0),
    ),
    # ---------------- infeasibility arbitration ----------------
    dict(
        id="I1-infeasible",
        category="infeasible",
        policy=("Minimize total transmit power while keeping the total "
                "data rate above 15 bits"),
        kind="episode",
        steps=250,
        seeds=[42],
        oracle=dict(family="min_power", tau=15.0, tau_ch=None),
        # pass = the loop certifies infeasibility AND the LLM resolves it
        # (final sum rate within 0.5 bits of the demanded 15)
        pass_fn=lambda m: (m["reached_infeasible"]
                           and m["final_sum_rate"] >= 14.5),
    ),
    # ---------------- disturbance resilience ----------------
    dict(
        id="D1-gain-shuffle",
        category="disturbance",
        policy=("Minimize total transmit power while keeping the total "
                "data rate above 10 bits"),
        kind="episode",
        steps=300,
        seeds=[42],
        env_schedule={150: list(np.array(H_ABS2_DEFAULT)[
            np.random.default_rng(7).permutation(N)])},
        oracle=dict(family="min_power", tau=10.0, tau_ch=None),
        pass_fn=lambda m: (m["violation_rate_post"] <= 0.25
                           and m["recovered_after_disturbance"]),
    ),
    # ---------------- switching stability (semantic oscillation) ----------
    dict(
        id="X1-shutdown-stability",
        category="switching",
        policy="Shut down channels 0 to 3",
        kind="episode",
        steps=200,
        seeds=[42],
        oracle=dict(family="power_target", tau=None, tau_ch=None,
                    targets=[0, 0, 0, 0, 10, 10, 10, 10]),
        pass_fn=lambda m: (m["family_switches_after_settle"] == 0
                           and m["weak_channels_off"]),
    ),
    # ---------------- expressiveness gap (diagnostic) -------------------
    dict(
        id="E1-unexpressible",
        category="expressiveness",
        policy=("Minimize total power while keeping the decoding latency of "
                "every channel below 5 milliseconds"),
        kind="translation",
        expected_fail=True,  # grammar has no latency vocabulary (v0.1 gap)
        # v0.1 diagnostic: record what the LLM does; "pass" would require a
        # coverage/escalation mechanism that does not exist yet.
        check=lambda spec, tau, tau_ch: (
            False,
            f"grammar cannot express latency; LLM chose family={spec.family} "
            f"tau={tau} tau_ch={tau_ch}",
        ),
    ),
]
