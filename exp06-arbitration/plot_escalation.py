"""Fig: what an instruction cannot do and a typed field can (4B model)."""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _need(path, produced_by):
    if not os.path.exists(path):
        sys.exit(f"{path} not found -- run {produced_by} first")
    return path


TAG = "qwen3-4b"
ARMS = [("C0", "no escalation field", "tab:gray", "-"),
        ("C1", "no field, instruction added", "tab:orange", "--"),
        ("C3", "typed escalation field", "tab:blue", "-")]
SEED = 42
TAU = 15.0

fig, ax = plt.subplots(2, 1, figsize=(3.4, 2.7), sharex=True)
for cond, label, color, ls in ARMS:
    d = json.load(open(_need(f"results/arb_{TAG}_{cond}_s{SEED}.json",
                            "run_arbitration.py")))
    steps = [c["step"] for c in d["calls"]]
    ps = [c["p_total"] for c in d["calls"]]
    lw = 1.4 if cond == "C3" else 1.0
    ax[0].plot(steps, ps, color=color, ls=ls, lw=lw, label=label)
    ax[1].plot(d["ts_steps"], d["ts_sum_rate"], color=color, ls=ls, lw=lw)

ax[0].set_ylabel("power cap $P_{\\mathrm{total}}$", fontsize=8)
ax[0].tick_params(labelsize=7)
ax[0].legend(fontsize=6.2, loc="center right", framealpha=0.95)
ax[1].axhline(TAU, color="k", ls=":", lw=0.8)
ax[1].set_ylim(top=17.2)
ax[1].text(170, TAU + 1.1, "demanded rate", fontsize=6.2, ha="center")
ax[1].set_ylabel("sum rate [bits]", fontsize=8)
ax[1].set_xlabel("optimizer step", fontsize=8)
ax[1].tick_params(labelsize=7)
for a in ax:
    a.grid(alpha=0.3, lw=0.4)
fig.tight_layout(pad=0.4)
# default next to the data; pass a path to write elsewhere
out = sys.argv[1] if len(sys.argv) > 1 else "results/escalation.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
