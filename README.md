# LLM → Nyquist SAL — audio structure evaluation

This repo asks an LLM to write **Nyquist SAL** (or `.ny`), turns it into **WAV**, then scores how well the audio matches the prompt (loudness, brightness, rhythm density, tempo) using **librosa** and simple metrics.

You get: batch generation, optional analysis after you export WAVs from NyquistIDE, optional comparison of two batches, and plots. Put real runs under `outputs/` and keep that folder **out of git** (see `.gitignore`).

---

## What you need

- Python **3.10+** (3.11+ is fine)
- **NyquistIDE** (or a working `ny` CLI if you use `--format ny`)
- An LLM: **Gemini** (API key) and/or **Ollama** with a coding model

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` — at minimum **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`**. Everything else is optional (model name, `NYQUIST_BIN`, Ollama host/model, OpenAI-compatible URLs). Details are in `.env.example`.

Check Nyquist without calling an LLM:

```bash
python smoke_test_nyquist.py
```

---

## Typical flow (SAL)

1. **Generate** — writes `generated.sal` and metadata under `outputs/run_<time>_<task>_<model>/`:

   ```bash
   python run_batch.py --task energy --per-prompt 3 --format sal
   ```

   Use `--task spectral`, `tempo`, `density`, or `all`. Add `--llm ollama-qwen` for local Ollama (after `ollama pull qwen2.5-coder:7b` or similar), or `--llm openai-compat` with the vars from `.env.example`.

2. **Export** — open each script in **NyquistIDE**, play/render, and save **`out.wav`** in the **same folder** as `meta.json`.

3. **Analyze** — scores every run that has `out.wav`:

   ```bash
   python analyze_wavs.py outputs/run_<your_batch_folder>
   ```

**`--format ny`** skips the manual step: the tool calls `ny` for you. That path is handy when it works but can be picky about install paths.

Useful flags: `--out` for a custom folder, `--per-prompt N` for repeats per prompt, `--prompt 1,2` for specific prompts (1-based), `--duration-hint` to nudge length in the system message.

---

## Compare two batches

When both folders are analyzed and have the same task layout:

```bash
python compare_task_batches.py \
  --a outputs/run_batch_a \
  --b outputs/run_batch_b \
  --label-a gemini --label-b ollama \
  --output outputs/compare_my_run
```

You get `combined/`, `batch_a/`, and `batch_b/` with summaries, CSV/JSON, and plots (including pairwise prompt stats under `combined/`).

---

## Tasks (in plain language)

| Task | Roughly measures |
|------|-------------------|
| **energy** | Gets louder or softer over time (RMS trend) |
| **spectral** | Bright vs dark over time (spectral centroid); one prompt is a “V” shape |
| **density** | How busy the sound is early vs late (onset counts in time slices) |
| **tempo** | Estimated BPM vs the number in the prompt (`beat_track`; can be noisy) |

Exact prompt text lives in `src/tasks/energy.py`, `spectral.py`, `density.py`, and `tempo.py` inside `generate_prompts()`.

---

## Outputs you’ll see

- **`runs.jsonl`** — what happened during generation  
- **`analysis_runs.jsonl`** / **`analysis_evaluation_summary.json`** — after `analyze_wavs.py`  
- **`analysis_by_prompt.*`** — rolled up by prompt  
- **`plots/`** — charts  
- **`energy/…/`, `spectral/…/`, etc.** — each run folder has code, `meta.json`, WAV, scores, and maybe `sal_error.txt`  

For “did the WAV load?” vs “did it match the prompt?”, the pipeline tags rows with `eval__*` fields (see `src/pipeline/evaluation_axes.py`). **`success`** and **`duration_shortfall`** are common statuses; **`pending_manual_render`** means SAL is waiting for your `out.wav`.

---

## Project layout

```
run_batch.py          # generate
analyze_wavs.py       # score a batch
compare_task_batches.py
smoke_test_nyquist.py
src/tasks/            # prompts + metrics per task
src/pipeline/         # LLM, Nyquist, orchestration
references/           # example SAL for few-shots
```

New task: copy the pattern in `energy.py`, register it, then add the id to `run_batch.py` if you want it on the CLI.

---

## If something breaks

- **No `ny`:** set `NYQUIST_BIN` to your Nyquist binary (on macOS it’s often inside `NyquistIDE.app/Contents/Java/ny`).
- **SAL errors:** the model may have written invalid SAL; tweak `references/*.sal` or the task prompts.
- **Very short WAV:** many metrics expect a full-length clip; you’ll see `duration_shortfall`.
- **Gemini 429 / quota:** smaller batches, another model, or use Ollama locally.

---

## License

Add a `LICENSE` file when you publish if you need one.
