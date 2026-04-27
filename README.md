# LLM → Nyquist SAL: structural audio evaluation

Evaluate whether **large language models** can follow **verifiable musical instructions** when they write **Nyquist SAL** (or `.ny`): generate code, render to **WAV**, then measure **time-varying structure** (energy, spectral brightness, onset density, tempo) with **librosa**-based features and simple adherence scores.

**What this repo is:** batch generation (`run_batch.py`), post-export analysis (`analyze_wavs.py`), optional A/B comparison (`compare_task_batches.py`), and plotting. **Large run trees** (`outputs/`, WAVs, API caches) belong **out of version control** — see `.gitignore`. Re-run analysis after you add or replace `out.wav` files to refresh `analysis_*.json` and plots.

---

## Requirements

- **Python 3.10+** (3.11+ recommended)
- **Nyquist / NyquistIDE** (SAL path: open `generated.sal`, export `out.wav` beside `meta.json`)
- **LLM:** Google **Gemini** (API key) and/or **Ollama** with a coding model (e.g. `qwen2.5-coder:7b`)

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit **`.env`**: `GEMINI_API_KEY` or `GOOGLE_API_KEY`; optional `GEMINI_MODEL`, `NYQUIST_BIN`, `OLLAMA_HOST`, `OLLAMA_MODEL`, or OpenAI-compatible `OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_MODEL` (see `.env.example`).

Optional **Nyquist subprocess** check (no LLM):

```bash
python smoke_test_nyquist.py
```

---

## Render modes

| Mode | Command flag | What happens |
|------|----------------|----------------|
| **SAL (recommended)** | `--format sal` | Writes `generated.sal` + `meta.json`. You render in **NyquistIDE** and save **`out.wav`** in the same run folder, then run **`analyze_wavs.py`**. |
| **Nyquist CLI** | `--format ny` | Pipeline invokes the **`ny`** subprocess to render `out.wav`. Depends on a working `ny` on `PATH` or **`NYQUIST_BIN`**; can be fragile across installs. |

---

## Workflows

### 1) Generate a batch

```bash
python run_batch.py --task energy --per-prompt 3 --format sal
# Tasks: energy | spectral | tempo | density | all
# Backends: --llm gemini (default) | ollama-qwen | openai-compat
```

Default output: **`outputs/run_<UTC>_<task>_<tag>/`**. Override with **`--out`**.  
**`--per-prompt`:** repeats per **prompt template** (each task defines several in code). **`--prompt 1,3`:** 1-based indices, comma-separated. **`--duration-hint SEC`:** passed into the system message so the model targets a plausible length.

**Ollama (local):** install [Ollama](https://ollama.com), e.g. `ollama pull qwen2.5-coder:7b`, then:

```bash
python run_batch.py --llm ollama-qwen --task energy --per-prompt 3 --format sal
```

**OpenAI-compatible API:** set `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL` in `.env`, then `--llm openai-compat`.

### 2) SAL: export WAVs, then analyze

In NyquistIDE, run each `generated.sal` and save **`out.wav`** next to `meta.json` (exact path also in the file header / `; MANUAL_WAV_PATH:` when the model follows instructions).

```bash
python analyze_wavs.py outputs/run_<timestamp>_<task>_<tag>
```

### 3) Compare two completed batches

Same task layout on both sides (matching run structure):

```bash
python compare_task_batches.py \
  --a outputs/run_..._batch_a \
  --b outputs/run_..._batch_b \
  --label-a gemini --label-b ollama \
  --output outputs/compare_<name>
```

Writes **`combined/`**, **`batch_a/`**, **`batch_b/`** each with analysis artifacts; **`combined/`** also gets **`analysis_prompt_pairwise.{json,csv}`** and **`plots/analysis_prompt_pairwise_mean_adherence.png`**.

---

## Tasks and metrics (summary)

| Task | Features (high level) | Score idea |
|------|------------------------|------------|
| **energy** | Short-time **RMS** (smoothed) | **Spearman** correlation of envelope vs time vs prompt direction; `energy_append_failure_heuristics` |
| **spectral** | **Spectral centroid** (Hz), smoothed | Spearman for monotonic prompts; prompt index **2** uses a **V-shape** score (first half fall, second half rise) |
| **density** | **Onset** strength + `onset_detect` | Event rate in **first vs last 25%** (rise/fall); **arc** prompt compares **middle 50%** to outer quarters |
| **tempo** | Onset strength + **`beat_track`** | BPM error vs value parsed from prompt → `adherence_primary`; SAL pulse trains are brittle |

**Source of truth for prompts:** `generate_prompts` in `src/tasks/{energy,spectral,density,tempo}.py`. Abridged intent:

- **Energy (0–2):** ~15 s arc with quiet→climax+fade; 12 s crescendo; 14 s decrescendo.
- **Spectral (0–2):** bright→dark 12 s; dull→bright 10 s; **14 s V** (bright–dark–bright).
- **Density (0–2):** sparse→dense ~12 s; dense→sparse; **arc** (sparse ends, denser middle).
- **Tempo (0–2):** ~12 s **100 BPM** clicks; ~12 s **120 BPM** pattern; **90 BPM** drum-like loop **≥ 10 s**.

**Duration hints** (seconds, used for `duration_shortfall` / accountability) are task- and prompt-dependent — see **`effective_duration_hint_sec`** / `_duration_accountability` in **`src/pipeline/run.py`** and per-task modules.

---

## Evaluation vocabulary (`src/pipeline/evaluation_axes.py`)

Analysis CSV/JSON prefix **`eval__*`** fields separate **well-formedness** from **semantic** success.

| Axis | Typical field prefix | Meaning |
|------|----------------------|---------|
| **WF: LLM OK** | `eval__wf_llm_ok` | No quota / empty-generation failure. |
| **WF: code committed** | `eval__wf_code_committed` | Code artifact path exists. |
| **WF: Nyquist / export** | `eval__wf_nyquist_subprocess_ok` | Not in a hard failure set for the render path. |
| **WF: playable non-silent WAV** | `eval__wf_playable_nonsilent_wav` | `success` or **`duration_shortfall`** (audio still loaded for metrics). |
| **SEM: duration OK** | `eval__sem_duration_ok` | Not short vs the pipeline duration hint. |
| **SEM: directional adherence** | from `scores` | Task-specific **\[0, 1\]** adherence (`directional_adherence` or `adherence_primary`). |
| **SEM: “strong” (≥ 0.72)** | `eval__sem_energy_strong_direction` | True when adherence ≥ **`_SEM_ADHERENCE_STRONG`** (`0.72`) — used for **energy, spectral, tempo, and density** rows in summaries and bar charts. |
| **SEM: heuristic issue** | `eval__sem_energy_heuristic_issue` | Task-specific weak / flat / misaligned flags (`*_append_failure_heuristics` in each task). |

**Tempo caveat:** plots that label the third semantic bar may still use the shared field name **`sem_energy_strong_direction`**; for tempo runs the **underlying score is BPM adherence**, not RMS energy — read row `task_id` / `scores` when interpreting.

---

## Run statuses (`src/pipeline/status.py`)

Common values in `runs.jsonl` / `analysis_runs.jsonl`:

| Status | Meaning |
|--------|---------|
| **`success`** | WAV OK; features and scores computed. |
| **`pending_manual_render`** | SAL mode: waiting for **`out.wav`** in the run folder. |
| **`duration_shortfall`** | WAV shorter than the task duration hint (still often analyzed). |
| **`generation_failed`** | Unusable LLM output or non-quota API error. |
| **`llm_quota_exceeded`** | 429 / quota (Gemini, etc.). |
| **`nyquist_not_found`** | `ny` missing; set **`NYQUIST_BIN`**. |
| **`nyquist_timeout`** | Subprocess exceeded **`NYQUIST_TIMEOUT_SEC`** (default 300). |
| **`syntax_error`** / **`runtime_error`** | Nyquist failed (stderr heuristics). |
| **`render_failure`** | Process exited OK but expected WAV missing. |
| **`silent_audio`** / **`invalid_audio`** | Unusable or too-short waveform. |

---

## Batch artifacts

Under each **batch root** (and under `batch_a/`, `batch_b/`, `combined/` after compare):

| Artifact | Role |
|----------|------|
| `runs.jsonl`, `runs_summary.csv` | Generation log (from `run_batch.py`). |
| `analysis_runs.jsonl` | One JSON row per analyzed run (post-`analyze_wavs` / compare). |
| `analysis_evaluation_summary.json` | Aggregated rates (WF / SEM / strong / heuristics). |
| `analysis_by_prompt.json`, `.csv` | Per-prompt rollups. |
| `plots/` | Summary adherence bars, well-formed vs semantic breakdowns, etc. |
| `<task>/<run_id>/` | `generated.sal` or `generated.ny`, `meta.json`, optional `out.wav`, `features.json`, `scores.json`, `plot.png`, `sal_error.txt` on SAL failures. |

---

## CLI reference

```text
python run_batch.py [--task energy|spectral|tempo|density|all] [--per-prompt N]
                    [--prompt I[,I...]] [--format ny|sal]
                    [--llm gemini|ollama-qwen|openai-compat]
                    [--sleep-between-llm SEC] [--out DIR] [--duration-hint SEC]

python analyze_wavs.py <batch_folder>

python compare_task_batches.py --a <batch_a> --b <batch_b>
                                --label-a <name> --label-b <name> --output <dir>
```

---

## Repo layout

```
run_batch.py
analyze_wavs.py
compare_task_batches.py
smoke_test_nyquist.py
requirements.txt
.env.example
src/
  config.py
  tasks/           # base.py + energy, spectral, density, tempo
  pipeline/        # llm.py, nyquist_runner.py, run.py, status.py, evaluation_axes.py
references/        # sal_*_example.sal few-shots
outputs/           # local only — gitignored
```

**Adding a task:** mirror `src/tasks/energy.py` (register with `register_task`), then expose the id in `run_batch.py` if you want it on the CLI.

---

## Troubleshooting

- **`nyquist_not_found`:** Set **`NYQUIST_BIN`** to the real `ny` binary (macOS CMU build often lives at **`NyquistIDE.app/Contents/Java/ny`** under `Applications` or `~/Applications`). The runner may auto-try common paths; override if your install differs. Ensure **`XLISPPATH`** / runtime can find Nyquist libs (see smoke test and `.env.example`).
- **SAL / `sal_error.txt`:** Model output may not be valid SAL (Python idioms, wrong builtins, typos). Improve few-shots in **`references/`** or tighten **`src/tasks/sal_prompt_rules.py`** if present for that task.
- **Short `out.wav` / low density–spectral scores:** Quarter-based and trajectory metrics assume enough duration; **`duration_shortfall`** flags clips below the hint.
- **Gemini `429` / quota:** Smaller batches, different **`GEMINI_MODEL`**, **`--llm ollama-qwen`**, or wait / billing per [Google rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). **`--sleep-between-llm`** helps per-minute throttling, not daily caps.
- **Empty LLM output / safety block:** Check `runs.jsonl` for `prompt_feedback`; shorten prompts or switch model.

---

## License

Add a **`LICENSE`** file when you publish the repository if you need explicit terms (this tree may not ship one).
