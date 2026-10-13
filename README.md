# autorlhf

A local-first terminal workbench that turns a character/behaviour specification into a
fine-tuned Hugging Face causal language model, using **Jev** (TypeSafe's typed decision
model, served through OpenRouter) as the judge.

Pipeline: **generate → judge → curate → train → evaluate → inspect.**

Everything runs on your machine except the OpenRouter calls (generation + Jev
judgements) and, if you explicitly enable them, the optional W&B/Weave trackers.

## Pipeline

| Stage        | What happens                                                                                                                |
| ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **generate** | Synthesise prompts from the character spec and generate candidate responses (OpenRouter teacher, the local model, or both). |
| **judge**    | Score/compare candidates with Jev against the rubric.                                                                       |
| **curate**   | Split prompts into train/validation/eval, filter by score, and (in `-i`) collect human review decisions.                    |
| **train**    | SFT warm-up and/or DPO with LoRA / QLoRA / FFT, using a hardware-feasible preset.                                           |
| **evaluate** | Held-out evaluation of the base model vs the fine-tuned adapter on the _same_ prompts.                                      |
| **inspect**  | Browse runs, artefacts, metrics, and the full event history.                                                                |

## Install

```bash
uv sync                         # CUDA 12.4 Torch on Linux; CPU still works without a GPU
uv sync --extra tracking       # optional W&B + Weave
uv sync --extra qlora          # optional bitsandbytes for CUDA QLoRA
uv sync --all-extras           # everything
```

On Linux, `pyproject.toml` and `uv.lock` pin Torch 2.6.0 to the official CUDA 12.4
index. This avoids silently resolving a newer CUDA runtime that the installed NVIDIA
driver cannot initialize. Use the project `.venv`; do not activate a second environment.

On shared GPU hosts that export a global `LD_LIBRARY_PATH`, remove it for the project
shell if it points at another Python installation's Torch/NCCL libraries:

```bash
unset VIRTUAL_ENV LD_LIBRARY_PATH
uv sync
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
uv run autorlhf --doctor
```

Requires Python 3.11 or 3.12. CPU-only machines can still generate/judge/curate data and
run the tiny CPU LoRA preset.

## Credentials

Run the setup wizard (masked input, written `0600`):

```bash
uv run autorlhf --setup
```

| Key                                                | Required | Purpose                                    |
| -------------------------------------------------- | -------- | ------------------------------------------ |
| `OPENROUTER_API_KEY`                               | yes      | Generation and Jev judgements.             |
| `HF_TOKEN`                                         | optional | Gated/private model repositories.          |
| `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_ENTITY` | optional | Training metrics to W&B.                   |
| `WEAVE_PROJECT` / `WEAVE_ENABLED`                  | optional | Prompt/response/judgement traces to Weave. |

Secrets live in `~/.local/share/autorlhf/config.env` (or `$AUTORLHF_HOME/config.env`).
A project-local `.env` is read as a lowest-precedence fallback but never written;
`.env` is gitignored. Real environment variables always win. See `.env.example`.

When W&B is enabled in setup, enter the API key and project; setup can test the
credentials/project immediately and reports a safe success or remediation message.
To install the optional client first, run `uv sync --extra tracking`.

## Commands

```bash
uv run autorlhf                       # dashboard if runs exist, else first-run setup
uv run autorlhf -i                    # interactive: human review of every comparison
uv run autorlhf --doctor              # environment report (exit 1 on a failed check)
uv run autorlhf --doctor --live       # also verify the OpenRouter / HF credentials
uv run autorlhf --setup               # headless credential wizard
uv run autorlhf --headless --config run.json     # new run without Textual
uv run autorlhf --headless --run-id ID --resume  # resume without Textual
uv run autorlhf --list-runs
uv run autorlhf --export RUN_ID DEST [--with-checkpoints]
uv run autorlhf --version
```

Exit codes: `0` success, `1` a check or run failed, `2` the command could not start.

## Artifact layout

```
$AUTORLHF_HOME/                     (~/.local/share/autorlhf by default)
  config.env                        credentials (0600)
  autorlhf.db                       SQLite: runs, phases, reviews, cache
  runs/<run-id>-<slug>/
    config.json                     immutable RunConfig + fingerprint
    character.json  rubric.json
    dataset.jsonl  preferences.jsonl  sft.jsonl  eval_prompts.jsonl
    judgments.jsonl  reviews.jsonl  splits.json
    baseline_eval.json  final_eval.json  comparison.json
    manifest.json                   self-describing run record (no secrets)
    adapter/  checkpoints/  logs/   trained weights + logs
```

The formats are frozen in `docs/DATA_FORMATS.md`; every record carries `run_id`,
`dataset_version`, and `created_at`.

## Training, resuming, inspecting

Training picks a preset from `config.presets` that the **measured** hardware can actually
run; the judge may only _rank_ the surviving presets, never invent hyperparameters.
Runs are resumable from their last complete checkpoint — resuming reloads the saved
`config.json` so a run can never silently change its model, prompt, or data.

Inspect a run with `--list-runs` and `--export`, or open the dashboard (`autorlhf`) and browse its phases, metrics, and samples. In the live run screen, enable Verbose (or press v) to stream per-prompt evaluation progress, prompts, generated responses, judgements, and API request details into Activity. These are live events, not a persisted full event-history artifact; credentials are excluded.

Use **Abandon** (or press x) on an incomplete run in the dashboard to permanently delete its database record and run artifacts. A confirmation prompt prevents accidental deletion.

## Base vs fine-tuned evaluation

Held-out prompts are never used to generate training data, choose preferences, or tune
hyperparameters. Both `baseline_eval.json` (base model) and `final_eval.json` (adapter)
record the same `eval_prompts_hash`, so the two reports provably used identical prompts;
`comparison.json` holds the pairwise Jev verdict. A run with too few scored responses is
marked **inconclusive** rather than reported as a win.

## Hardware requirements per method

| Method    | Requirement                                                  | Notes                                    |
| --------- | ------------------------------------------------------------ | ---------------------------------------- |
| **LoRA**  | CPU-capable (tiny preset, no VRAM floor) up to >= 24 GiB GPU | Default; adapter is a few MB.            |
| **QLoRA** | CUDA/ROCm GPU with `bitsandbytes`, >= 6 GiB VRAM            | 4-bit NF4 base; accelerator-only at runtime. |
| **FFT**   | >= 48 GiB VRAM (large GPU / multi-GPU)                       | **Experimental**; rewrites every weight. |

Accelerator presets use CUDA/ROCm-reported VRAM. On partitioned GPUs such as MIG, this is the
memory exposed to the process, not the physical card's advertised capacity. Feasibility checks
compare estimated need with reported total VRAM; concurrent processes can still consume memory
and cause an out-of-memory failure. QLoRA is offered only when a compatible CUDA/ROCm device is
detected and `bitsandbytes` imports successfully. On CPU-only machines, only tiny LoRA is offered.

## Honesty about the feedback signal

Jev is an **AI judge**: this is RLAIF (reinforcement/feedback from AI), not human
preference data. Running with `-i` adds real human review decisions, which are recorded
alongside the automated ones. A Jev score probability is a **preference judgement** — it
is _not_ a probability that training will succeed.

## Privacy

- **W&B** (off by default): uploads training metrics and a credential-free config excerpt
  (model id, method, objective, run id). No prompts or responses.
- **Weave** (off by default): uploads prompts, model responses, and Jev judgements to an
  external service. It requires explicit, recorded consent; without it the tracer is
  inert and training continues locally.
- Credentials are never written to SQLite, logs, datasets, or exports.

## Not implemented (by design)

- **No hosted/cloud service and no multi-user mode.** autorlhf is a single-machine
  workbench; there is no server deployment, auth, or run scheduling.
- **No chat playground.** The TUI inspects datasets, judgements, and evaluations — it is
  not a chat front-end for the trained model.
- **No automatic model upload or publishing.** Fine-tuned adapters stay on disk; you
  upload them yourself.
- **No hyperparameter search.** Training uses a fixed, curated preset set filtered by
  measured hardware; Jev may only _rank_ feasible presets.
- **Jev never generates prose.** It is a typed decision model (score/choice/noul), not an
  LLM, and is not used for text generation.
- **FFT is experimental** and gated behind a large-VRAM requirement.

Run `uv run autorlhf --doctor` for the current picture of what your install can do.
