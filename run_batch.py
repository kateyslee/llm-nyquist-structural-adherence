#!/usr/bin/env python3
"""Batch runner: modular Nyquist adherence pipeline (energy, spectral, tempo, density)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUTS_DIR, gemini_model, ollama_openai_defaults, openai_compatible_config
from src.pipeline.llm import build_default_client
from src.pipeline.evaluation_axes import aggregate_analyzed_evaluations
from src.pipeline.run import (
    RunRecord,
    append_jsonl,
    plot_density_adherence,
    plot_energy_adherence,
    plot_failure_breakdown,
    plot_spectral_adherence,
    plot_tempo_adherence,
    run_single_generation,
    write_runs_csv,
)
from src.tasks import get_task, list_tasks

_VALID_LLMS = frozenset({"gemini", "ollama_qwen", "openai_compat"})


def _model_label_for_backend(backend_norm: str) -> str:
    if backend_norm == "gemini":
        return gemini_model()
    if backend_norm == "ollama_qwen":
        *_, label = ollama_openai_defaults()
        return label
    _b, _k, m = openai_compatible_config()
    return m or "openai_compat"


def _default_batch_folder_tag(backend_norm: str) -> str:
    """Short LLM tag for default `outputs/run_<stamp>_<task>_<tag>/` (not full model id)."""
    return {"gemini": "gemini", "ollama_qwen": "ollama", "openai_compat": "openai"}.get(
        backend_norm, backend_norm
    )


def _parse_prompt_filter(raw: str | None) -> list[int]:
    """Parse --prompt into ordered 1-based indices; empty string → []."""
    if raw is None:
        return []
    s = raw.strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError as e:
            raise ValueError(
                f"Invalid --prompt fragment {p!r}; use comma-separated integers (1-based), e.g. 2 or 1,3."
            ) from e
    return out


def _prompt_indices_zero_based(
    one_based: list[int], n_prompts: int, task_id: str
) -> list[int]:
    """Map 1-based user indices to 0-based prompt_index; [] means all prompts."""
    if not one_based:
        return list(range(n_prompts))
    result: list[int] = []
    for u in one_based:
        if u < 1 or u > n_prompts:
            raise ValueError(
                f"Task {task_id!r} has {n_prompts} prompt template(s); --prompt value {u} is out of range "
                f"(use 1..{n_prompts}, same as printed 'prompt k/n')."
            )
        result.append(u - 1)
    return result


def _normalize_llm_backend(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s not in _VALID_LLMS:
        raise ValueError(
            f"Unknown --llm/--backend {raw!r}. Use: gemini, ollama-qwen, openai-compat "
            f"(underscores optional)."
        )
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Run LLM → Nyquist → metric pipeline")
    p.add_argument(
        "--task",
        choices=["energy", "spectral", "tempo", "density", "all"],
        default="energy",
        help="Which task module to run",
    )
    p.add_argument(
        "--per-prompt",
        type=int,
        default=3,
        help="Generations per selected prompt template (e.g. --task spectral --prompt 2 --per-prompt 3)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        metavar="I[,I...]",
        help="1-based prompt index(es), comma-separated (same as log 'prompt k/n'). "
        "Example: --task spectral --prompt 2 --per-prompt 3 runs only the second template three times. "
        "Default: all templates for each task. With --task all, each task uses the same list (must fit each task).",
    )
    p.add_argument(
        "--llm",
        "--backend",
        dest="llm_backend",
        type=str,
        default="gemini",
        metavar="BACKEND",
        help="LLM: gemini (API key) | ollama-qwen (free local Ollama + Qwen) | openai-compat (OPENAI_* in .env)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run output directory (default: outputs/run_<utc>_<task>_gemini|ollama|openai; <task> from --task)",
    )
    p.add_argument("--duration-hint", type=float, default=15.0, help="Seconds hint for system prompt")
    p.add_argument(
        "--format",
        choices=["ny", "sal"],
        default="ny",
        help="ny: subprocess render + metrics. sal: write generated.sal + meta.json; "
        "export WAV in NyquistIDE then run analyze_wavs.py on this batch folder.",
    )
    p.add_argument(
        "--sleep-between-llm",
        type=float,
        default=float(os.environ.get("GEMINI_SLEEP_BETWEEN_SEC", "0") or "0"),
        metavar="SEC",
        help="Pause SEC seconds between LLM calls (reduces burst 429s). "
        "Default: 0 or GEMINI_SLEEP_BETWEEN_SEC env.",
    )
    args = p.parse_args()

    try:
        backend_norm = _normalize_llm_backend(args.llm_backend)
        prompt_filter_1based = _parse_prompt_filter(args.prompt)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_label = _model_label_for_backend(backend_norm)
    folder_tag = _default_batch_folder_tag(backend_norm)
    run_root = (args.out or (OUTPUTS_DIR / f"run_{stamp}_{args.task}_{folder_tag}")).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    known = list_tasks()
    task_ids = known if args.task == "all" else [args.task]
    if any(t not in known for t in task_ids):
        print(f"Unknown task in {task_ids}; known: {known}", file=sys.stderr)
        return 2

    llm = build_default_client(prefer=backend_norm)

    rows: list[RunRecord] = []
    jsonl_path = run_root / "runs.jsonl"
    llm_call_idx = 0

    for task_id in task_ids:
        n_prompts = len(get_task(task_id).generate_prompts())
        try:
            prompt_indices = _prompt_indices_zero_based(
                prompt_filter_1based, n_prompts, task_id
            )
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        for pi in prompt_indices:
            for _ in range(args.per_prompt):
                if llm_call_idx > 0 and args.sleep_between_llm > 0:
                    time.sleep(args.sleep_between_llm)
                llm_call_idx += 1
                print(f"[{task_id}] prompt {pi + 1}/{n_prompts} …", flush=True)
                rec = run_single_generation(
                    task_id=task_id,
                    prompt_index=pi,
                    llm=llm,
                    model_label=model_label,
                    llm_backend=backend_norm,
                    run_root=run_root,
                    duration_hint_sec=args.duration_hint,
                    code_format=args.format,
                )
                rows.append(rec)
                append_jsonl(jsonl_path, rec.to_json_dict())
                print(f"  → {rec.status.value}", flush=True)

    csv_path = run_root / "runs_summary.csv"
    write_runs_csv(csv_path, rows)
    plot_failure_breakdown(rows, run_root / "plots" / "failure_breakdown.png")
    plot_energy_adherence(rows, run_root / "plots" / "energy_adherence.png")
    plot_spectral_adherence(rows, run_root / "plots" / "spectral_adherence.png")
    plot_tempo_adherence(rows, run_root / "plots" / "tempo_adherence.png")
    plot_density_adherence(rows, run_root / "plots" / "density_adherence.png")

    ok = sum(1 for r in rows if r.status.value == "success")
    pending = sum(1 for r in rows if r.status.value == "pending_manual_render")
    dur_sf = sum(1 for r in rows if r.status.value == "duration_shortfall")
    print(f"\nDone. {ok}/{len(rows)} full success (audio + metrics). Artifacts: {run_root}")

    ev_agg = aggregate_analyzed_evaluations(rows)
    if ev_agg.get("n_analyzed"):
        unk = int(ev_agg.get("count_wf_playable_unknown") or 0)
        if unk == int(ev_agg["n_analyzed"]):
            print(
                "Well-formedness vs semantic: all runs still pending WAV — "
                "run `python analyze_wavs.py <this batch>` after NyquistIDE export for WF/SEM rates."
            )
        else:
            wf_d = (ev_agg.get("count_wf_playable_nonsilent_wav") or 0) + (
                ev_agg.get("count_wf_playable_false") or 0
            )
            if wf_d:
                print(
                    f"Well-formedness (playable non-silent WAV): "
                    f"{ev_agg.get('count_wf_playable_nonsilent_wav', 0)}/{wf_d} "
                    f"({100 * float(ev_agg.get('rate_wf_playable_nonsilent_wav') or 0):.0f}%) "
                    f"(excludes {unk} unknown/pending)."
                )
        ny_f = sum(
            1
            for r in rows
            if r.code_format == "ny"
            and r.status.value
            in ("syntax_error", "runtime_error", "nyquist_timeout", "nyquist_not_found")
        )
        ny_n = sum(1 for r in rows if r.code_format == "ny")
        if ny_n:
            print(
                f"Nyquist subprocess hard-fail count (.ny only): {ny_f}/{ny_n} "
                "(syntax/runtime/timeout/not_found — see runs.jsonl)."
            )
    if dur_sf:
        print(
            f"{dur_sf} run(s) flagged duration_shortfall (rendered WAV shorter than --duration-hint); "
            "see scores / score__llm_duration_shortfall in CSV."
        )
    if pending:
        print(
            f"{pending} run(s) pending manual WAV export (SAL). "
            f"Then: python analyze_wavs.py {run_root}"
        )
    n_quota = sum(1 for r in rows if r.status.value == "llm_quota_exceeded")
    if n_quota and backend_norm == "gemini":
        print(
            "\nGemini returned 429 (quota / rate limit) on "
            f"{n_quota} run(s). Free tier often caps **requests per day per model** (e.g. 20 for "
            "gemini-2.5-flash) — spacing calls does not reset the daily cap.\n"
            "Options: wait until the quota resets, try another GEMINI_MODEL, enable billing, or run "
            "locally with no API quota:\n"
            "  python run_batch.py --llm ollama-qwen ...\n",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
