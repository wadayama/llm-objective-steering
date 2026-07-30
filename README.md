# LLM Objective Steering — Interactive Demo

An interactive demo of a two-system architecture for power allocation over
N=8 parallel QPSK-AWGN channels. The LLM **declares an objective and
constraints**; a projected-gradient optimizer enforces them and certifies the
result. Mutual information (MI) is estimated by Monte Carlo, and the power
allocation $P_i = \lambda_i^2$ is optimized online.

- **System 1 (optimizer)** — gradient ascent on MI with projection. Declared
  constraints are enforced by a PHR augmented Lagrangian, and KKT residuals
  yield a status category (TRANSIENT / CONVERGED / INFEASIBLE / DISTURBED)
  together with a shadow price μ.
- **System 2 (LLM)** — translates a natural-language policy into an objective
  family plus constraint declarations. It declares; it does not regulate
  numbers call by call.

Objective families the LLM can select:

| Family | Objective | Use for |
|---|---|---|
| `weighted_sum` | $\sum_i w_i \mathrm{MI}_i$ | throughput, channel priorities |
| `alpha_fair` | $\alpha$-fair utility | fairness on MI (tunable strength) |
| `soft_min` | soft minimum | max-min style fairness |
| `power_target` | quadratic penalty toward target powers | equalizing power, shutting channels down |
| `min_power` | $-\sum_i P_i$ | minimizing total power (requires a constraint) |

Constraints it can declare: `sum_rate >= τ` (floor on the total rate) and
`min_channel_mi >= τ_ch` (floor on every channel's MI).

Every LLM output passes through clamping, normalization, and projection, so
**feasibility holds by construction** no matter what the model emits. If the
declared constraints cannot be met, the system returns INFEASIBLE *with a
certificate* rather than failing silently.

This repository contains only what is needed to run the demo.

```
exp05-integrated-demo/
├── run.py        UI, LLM controller, optimizer loop
├── core.py       System 1 (MI estimation, objectives, projection,
│                 augmented Lagrangian, KKT residuals)
└── steering.py   System 2 (prompt construction, response parsing,
                  control history)
```

## Requirements

- **Python 3.13+** and [uv](https://docs.astral.sh/uv/)
- **No GPU needed** — PyTorch runs on CPU
- **An OpenAI-compatible local server** on `localhost:1234`. Verified with
  [LM Studio](https://lmstudio.ai/). The model used for testing was
  `gpt-oss-20b` (MXFP4-Q8, MLX runtime, ~12 GB), which needs a machine with
  enough RAM — it runs on 24 GB.
- **A matplotlib GUI backend** (see below)

## Setup

```bash
git clone https://github.com/wadayama/llm-objective-steering.git
cd llm-objective-steering
uv sync
```

### About the GUI backend

The demo leaves backend selection to matplotlib, which tries
`macosx → qtagg → gtk4agg → gtk3agg → tkagg → wxagg → agg` in that order.

- **macOS** — `macosx` is picked with no extra setup.
- **Linux / Windows** — you need Qt or Tk. With neither available, matplotlib
  falls back to the non-interactive `agg` backend, no window opens, and the
  demo exits immediately. Install a GUI toolkit:
  ```bash
  uv add pyqt6          # or install python3-tk through your OS package manager
  ```
- To force a specific backend, set the environment variable:
  ```bash
  MPLBACKEND=qtagg uv run python run.py
  ```

### Starting the LLM server (LM Studio)

You can use the GUI, but the CLI is enough:

```bash
lms server start                  # start the server on port 1234
lms load openai/gpt-oss-20b       # load a model (list them with: lms ls)
lms ps                            # check which models are in memory
```

Verify connectivity:

```bash
curl -s http://localhost:1234/v1/models
```

If a model ID comes back, you are ready. Note that **starting the server and
loading a model are two separate steps** — you need both. The demo does not
hardcode a model name; it uses the **first model** returned by `/v1/models`.

## Running the demo

```bash
cd exp05-integrated-demo
uv run python run.py
```

What you see:

- **Top left** — per-channel MI in bits. A declared `sum_rate >= τ` constraint
  appears as a red dashed line.
- **Top right** — effective weights $\partial J / \partial \mathrm{MI}_i$, i.e.
  what the current objective is actually prioritizing.
- **Middle** — per-channel power, total power against the budget cap.
- **Status badge** —
  ```
  Optimizer: CONVERGED | constraint sum_rate>=10.0 | gap <= 0.043 bits | shadow price mu = 0.118
  ```
  The background color encodes the state: gray TRANSIENT, **green CONVERGED**,
  **red INFEASIBLE**, orange DISTURBED.
- **`LLM Status`** — the outcome of the last call (e.g. `OK [min_power]`) and
  the model's reasoning.

Controls:

- Type a policy in the text box and press **Set** (or Enter) to send it to the
  LLM.
- Eight sliders adjust the channel gains — use them to inject disturbances.
- **Random** / **Equal** set all gains at once, **Clear** removes the policy,
  **Reset** returns to the initial state.
- Quit by closing the window or pressing `Ctrl-C`.

Policies worth trying:

| Policy | What should happen |
|---|---|
| `Minimize total transmit power while keeping the total data rate above 10 bits` | `min_power` plus a constraint declaration; power descends on its own and settles at the boundary as CONVERGED |
| `Keep the total data rate above 15 bits with minimal power` | infeasible within a budget of 40 → μ climbs sharply → INFEASIBLE → the LLM raises the budget and recovers |
| `minimize total power while keeping MI larger than 1.0 bit for all the channels` | declares the per-channel constraint (vector multipliers) |
| `Ensure all channels achieve similar data rates` | `soft_min` |
| `Equalize transmit power across all channels` | `power_target` with equal targets |
| `Shut down channels 0 to 3` | `power_target` with zeros for the first four |
| `Prioritize channels 6 and 7` | `weighted_sum` with skewed weights |

Move the gain sliders far enough and the status turns DISTURBED (orange); the
optimizer re-converges by itself.

## Troubleshooting

**`RuntimeError: No models loaded in LM Studio`**
The server is up but no model is loaded. Run `lms load <model>`.

**The UI shows `LLM Status: Error: Connection error...`**
LM Studio is not reachable. The demo does not crash: System 1 keeps optimizing
the most recent objective, and calls are retried once per second. **You can
start LM Studio while the demo is running and it will recover on its own** —
no restart needed.

**Resizing the window prints `AttributeError: 'ResizeEvent' object has no attribute 'inaxes'`**
A known matplotlib 3.11.0 defect: `TextBox._resize` is bound to `resize_event`
but goes through a decorator that assumes a `LocationEvent`. matplotlib catches
the exception internally, so **the demo keeps running**. If the noise bothers
you, add this right after the matplotlib import:

```python
matplotlib.widgets.TextBox._resize = lambda self, event: self.stop_typing()
```

**No window appears and the demo exits right away**
No GUI backend is available, so `agg` was selected. See "About the GUI backend"
above. To check which backend you got:

```bash
uv run python -c "import matplotlib.pyplot; import matplotlib; print(matplotlib.get_backend())"
```

**Checking things without a GUI**
`DEMO_HEADLESS=1` imports the demo under `agg`, which is useful as a smoke test:

```bash
cd exp05-integrated-demo && DEMO_HEADLESS=1 uv run python -c "import run; print('OK')"
```

## License

MIT License — see [LICENSE](LICENSE).
