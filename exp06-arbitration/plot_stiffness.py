"""Fig: remaining infeasible is not a neutral holding state (no LLM involved)."""
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


d = json.load(open(_need("results/verify_stiffness.json",
                        "verify_stiffness.py")))
inf, fea = d["P40-infeasible"]["ts"], d["P70-feasible"]["ts"]

fig, ax0 = plt.subplots(figsize=(3.4, 2.1))
ax0.plot(fea["steps"], fea["sr"], color="tab:blue", lw=1.1,
         label="cap sufficient (feasible)")
ax0.plot(inf["steps"], inf["sr"], color="tab:red", lw=1.1,
         label="cap insufficient (infeasible)")
ax0.axhline(15.0, color="k", ls=":", lw=0.8)
ax0.set_ylim(top=17.4)
ax0.text(8, 15.9, "demanded rate", fontsize=6.2)
ax0.set_ylabel("sum rate [bits]", fontsize=8)
ax0.set_xlabel("optimizer step", fontsize=8)
ax0.legend(fontsize=6.2, loc="center right", framealpha=0.95)
ax0.grid(alpha=0.3, lw=0.4)
ax0.tick_params(labelsize=7)
fig.tight_layout(pad=0.4)
# default next to the data; pass a path to write elsewhere
out = sys.argv[1] if len(sys.argv) > 1 else "results/stiffness.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
