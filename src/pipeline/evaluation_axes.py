"""
Separate **well-formedness** (syntactic / toolchain / playable audio) from **semantic
correctness** (prompt-aligned behavior), matching common feedback on LLM→audio work:

- **Well-formedness**: Did the LLM return usable code? Did Nyquist run without
  syntax/runtime/timeout failures (automated `.ny`)? For SAL, did the user obtain a
  non-silent, loadable WAV (proxy for “NyquistIDE executed the script far enough to
  export audio”)? We do not statically prove SAL correctness without a separate checker.
- **Semantic correctness**: Duration vs spec, directional adherence (energy RMS / spectral centroid /
  tempo BPM / density quarter onset trajectory), heuristic flags — already in `scores` / `RunStatus` for some cases.

Downstream: `RunRecord.evaluation` is flattened as `eval__*` in JSON/CSV and summarized
after `analyze_wavs.py` in `analysis_evaluation_summary.json` + `plots/analysis_wellformed_vs_semantic.png`.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .status import RunStatus

# Optional headline threshold for “strong” monotonic energy shape (not normative).
_SEM_ADHERENCE_STRONG = 0.72

_NYQUIST_SUBPROCESS_FAILURE = frozenset(
    {
        RunStatus.SYNTAX_ERROR,
        RunStatus.RUNTIME_ERROR,
        RunStatus.NYQUIST_TIMEOUT,
        RunStatus.NYQUIST_NOT_FOUND,
    }
)


def build_evaluation_axes(
    *,
    status: RunStatus,
    code_format: str,
    task_id: str,
    scores: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Return booleans / floats for reporting. None values mean “not applicable”
    (e.g. SAL not yet exported).
    """
    scores = dict(scores or {})
    fmt = (code_format or "ny").lower()

    wf_llm_ok = status not in (
        RunStatus.LLM_QUOTA_EXCEEDED,
        RunStatus.GENERATION_FAILED,
    )
    wf_code_committed = wf_llm_ok and status not in (RunStatus.GENERATION_FAILED,)

    wf_nyquist_subprocess_ok: bool | None
    if fmt == "ny":
        wf_nyquist_subprocess_ok = status not in _NYQUIST_SUBPROCESS_FAILURE
        if status == RunStatus.RENDER_FAILURE:
            wf_nyquist_subprocess_ok = False
    elif fmt == "sal":
        if status == RunStatus.PENDING_MANUAL_RENDER:
            wf_nyquist_subprocess_ok = None
        else:
            # Any post-batch status implies the human render path was attempted for `.ny`
            # batches we cannot observe; for SAL, non-pending means IDE export happened.
            wf_nyquist_subprocess_ok = status not in _NYQUIST_SUBPROCESS_FAILURE
    else:
        wf_nyquist_subprocess_ok = None

    if status == RunStatus.PENDING_MANUAL_RENDER:
        wf_playable_nonsilent_wav: bool | None = None
    elif status in (RunStatus.SUCCESS, RunStatus.DURATION_SHORTFALL):
        wf_playable_nonsilent_wav = True
    else:
        wf_playable_nonsilent_wav = False

    if status == RunStatus.PENDING_MANUAL_RENDER:
        sem_metrics_computed: bool | None = None
        sem_duration_ok: bool | None = None
    else:
        sem_metrics_computed = bool(scores) and wf_playable_nonsilent_wav is True
        if wf_playable_nonsilent_wav is not True:
            sem_duration_ok = False
        else:
            sem_duration_ok = status != RunStatus.DURATION_SHORTFALL and not bool(
                scores.get("llm_duration_shortfall")
            )

    adherence = scores.get("directional_adherence")
    try:
        adherence_f = float(adherence) if adherence is not None else None
    except (TypeError, ValueError):
        adherence_f = None

    sem_energy_strong_direction: bool | None
    if status == RunStatus.PENDING_MANUAL_RENDER:
        sem_energy_strong_direction = None
    elif task_id in ("energy", "spectral", "tempo", "density") and adherence_f is not None:
        sem_energy_strong_direction = adherence_f >= _SEM_ADHERENCE_STRONG
    else:
        sem_energy_strong_direction = None

    flags = (
        bool(scores.get("flag_flat_energy_trajectory")),
        bool(scores.get("flag_weak_prompt_alignment")),
        bool(scores.get("flag_likely_wrong_energy_direction")),
    )
    if status == RunStatus.PENDING_MANUAL_RENDER:
        sem_energy_heuristic_issue: bool | None = None
    else:
        sem_energy_heuristic_issue = (
            any(flags) if task_id in ("energy", "spectral", "tempo", "density") and scores else False
        )

    return {
        "wf_llm_ok": wf_llm_ok,
        "wf_code_committed": wf_code_committed,
        "wf_nyquist_subprocess_ok": wf_nyquist_subprocess_ok,
        "wf_playable_nonsilent_wav": wf_playable_nonsilent_wav,
        "sem_metrics_computed": sem_metrics_computed,
        "sem_duration_ok": sem_duration_ok,
        "sem_energy_directional_adherence": adherence_f,
        "sem_energy_strong_direction": sem_energy_strong_direction,
        "sem_energy_heuristic_issue": sem_energy_heuristic_issue,
    }


def load_runs_jsonl_status_counts(path: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    if not path.is_file():
        return c
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        st = row.get("status")
        if isinstance(st, str):
            c[st] += 1
    return c


def nyquist_subprocess_stats_from_batch_jsonl(batch_root: Path) -> dict[str, Any]:
    """
    From `runs.jsonl` (post-`run_batch.py`), estimate automated Nyquist outcomes for `.ny`.
    SAL rows are mostly `pending_manual_render` until the user exports WAVs.
    """
    path = batch_root / "runs.jsonl"
    c = load_runs_jsonl_status_counts(path)
    total = sum(c.values())
    if total == 0:
        return {"runs_jsonl_present": False, "total": 0}

    syntax = c.get(RunStatus.SYNTAX_ERROR.value, 0)
    runtime = c.get(RunStatus.RUNTIME_ERROR.value, 0)
    timeout = c.get(RunStatus.NYQUIST_TIMEOUT.value, 0)
    not_found = c.get(RunStatus.NYQUIST_NOT_FOUND.value, 0)
    render_fail = c.get(RunStatus.RENDER_FAILURE.value, 0)
    ny_fail = syntax + runtime + timeout + not_found + render_fail
    success = c.get(RunStatus.SUCCESS.value, 0)
    pending = c.get(RunStatus.PENDING_MANUAL_RENDER.value, 0)

    denom_exec = total - pending
    rate_no_crash = None if denom_exec <= 0 else (denom_exec - ny_fail) / float(denom_exec)

    return {
        "runs_jsonl_present": True,
        "total": total,
        "pending_manual_render": pending,
        "automated_nyquist_denominator": denom_exec,
        "nyquist_syntax_error": syntax,
        "nyquist_runtime_error": runtime,
        "nyquist_timeout": timeout,
        "nyquist_not_found": not_found,
        "render_failure": render_fail,
        "nyquist_subprocess_failure_total": ny_fail,
        "full_pipeline_success": success,
        "rate_nyquist_subprocess_no_hard_fail": rate_no_crash,
        "note": "For --format sal, `runs.jsonl` usually stays `pending_manual_render` until you export WAVs; "
        "then use `wf_playable_nonsilent_wav` from `analyze_wavs.py` for well-formedness. "
        "Automated subprocess syntax/runtime counts apply to `--format ny` batches.",
    }


def aggregate_analyzed_evaluations(rows: list[Any]) -> dict[str, Any]:
    """Summarize `evaluation` dict on each RunRecord (post-batch or post-analyze)."""
    n = len(rows)
    if n == 0:
        return {"n_analyzed": 0}

    def mean_true(key: str) -> float | None:
        vals = [getattr(r, "evaluation", {}).get(key) for r in rows]
        use = [v for v in vals if isinstance(v, bool)]
        if not use:
            return None
        return sum(1 for v in use if v) / float(len(use))

    def count_true(key: str) -> int:
        return sum(1 for r in rows if getattr(r, "evaluation", {}).get(key) is True)

    def count_bool(key: str) -> tuple[int, int]:
        true_n = sum(1 for r in rows if getattr(r, "evaluation", {}).get(key) is True)
        false_n = sum(1 for r in rows if getattr(r, "evaluation", {}).get(key) is False)
        return true_n, false_n

    wf_play_t, wf_play_f = count_bool("wf_playable_nonsilent_wav")
    wf_play_na = sum(1 for r in rows if getattr(r, "evaluation", {}).get("wf_playable_nonsilent_wav") is None)
    sem_dur_t, sem_dur_f = count_bool("sem_duration_ok")
    sem_dur_na = sum(1 for r in rows if getattr(r, "evaluation", {}).get("sem_duration_ok") is None)
    strong_vals = [
        getattr(r, "evaluation", {}).get("sem_energy_strong_direction")
        for r in rows
        if r.task_id in ("energy", "spectral", "tempo", "density")
    ]
    strong_n = sum(1 for v in strong_vals if v is True)
    strong_defined = sum(1 for v in strong_vals if isinstance(v, bool))

    wf_denom = wf_play_t + wf_play_f
    sem_denom = sem_dur_t + sem_dur_f

    return {
        "n_analyzed": n,
        "rate_wf_playable_nonsilent_wav": wf_play_t / wf_denom if wf_denom else None,
        "count_wf_playable_nonsilent_wav": wf_play_t,
        "count_wf_playable_false": wf_play_f,
        "count_wf_playable_unknown": wf_play_na,
        "rate_sem_duration_ok": sem_dur_t / sem_denom if sem_denom else None,
        "count_sem_duration_ok": sem_dur_t,
        "count_sem_duration_false": sem_dur_f,
        "count_sem_duration_unknown": sem_dur_na,
        "rate_sem_energy_strong_direction_among_energy": (
            strong_n / float(strong_defined) if strong_defined else None
        ),
        "count_sem_energy_strong_direction": strong_n,
        "n_energy_rows_for_strong_metric": strong_defined,
        "mean_rate_wf_llm_ok": mean_true("wf_llm_ok"),
        "mean_rate_wf_code_committed": mean_true("wf_code_committed"),
        "count_sem_energy_heuristic_issue": count_true("sem_energy_heuristic_issue"),
    }


_OK_ADHERENCE_STATUS = frozenset({RunStatus.SUCCESS, RunStatus.DURATION_SHORTFALL})


def _directional_adherence_value(r: Any) -> float | None:
    if getattr(r, "status", None) not in _OK_ADHERENCE_STATUS:
        return None
    scores = getattr(r, "scores", None) or {}
    raw = scores.get("directional_adherence", scores.get("adherence_primary"))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def group_runs_by_task_prompt(rows: list[Any]) -> dict[tuple[str, int], list[Any]]:
    g: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for r in rows:
        g[(r.task_id, int(r.prompt_index))].append(r)
    return dict(g)


def summarize_prompt_run_group(rows_same_prompt: list[Any]) -> dict[str, Any]:
    """One (task_id, prompt_index) bucket: aggregate metrics across replicate runs."""
    rows_same_prompt = list(rows_same_prompt)
    if not rows_same_prompt:
        return {}
    first = rows_same_prompt[0]
    adh = [_directional_adherence_value(r) for r in rows_same_prompt]
    adh_f = [x for x in adh if x is not None]
    st_counts: Counter[str] = Counter(r.status.value for r in rows_same_prompt)
    ev = aggregate_analyzed_evaluations(rows_same_prompt)
    return {
        "task_id": first.task_id,
        "prompt_index": int(first.prompt_index),
        "prompt": first.prompt,
        "n_runs": len(rows_same_prompt),
        "run_ids": sorted(r.run_id for r in rows_same_prompt),
        "status_counts": dict(sorted(st_counts.items())),
        "mean_directional_adherence": (sum(adh_f) / len(adh_f)) if adh_f else None,
        "min_directional_adherence": min(adh_f) if adh_f else None,
        "max_directional_adherence": max(adh_f) if adh_f else None,
        "evaluation_summary": ev,
    }


def per_prompt_evaluation_document(rows: list[Any]) -> dict[str, Any]:
    """Table keyed by (task_id, prompt_index) for a single batch or combined rows."""
    grouped = group_runs_by_task_prompt(rows)
    by_prompt = [summarize_prompt_run_group(grouped[k]) for k in sorted(grouped)]
    return {"n_prompt_keys": len(by_prompt), "by_prompt": by_prompt}


def per_prompt_flat_csv_rows(by_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten `by_prompt` entries for spreadsheet export."""
    out: list[dict[str, Any]] = []
    for p in by_prompt:
        ev = p.get("evaluation_summary") or {}
        prompt = p.get("prompt") or ""
        out.append(
            {
                "task_id": p.get("task_id"),
                "prompt_index": p.get("prompt_index"),
                "prompt_preview": (prompt[:200] + "…") if len(prompt) > 200 else prompt,
                "n_runs": p.get("n_runs"),
                "run_ids": ";".join(p.get("run_ids") or []),
                "mean_directional_adherence": p.get("mean_directional_adherence"),
                "min_directional_adherence": p.get("min_directional_adherence"),
                "max_directional_adherence": p.get("max_directional_adherence"),
                "rate_wf_playable_nonsilent_wav": ev.get("rate_wf_playable_nonsilent_wav"),
                "rate_sem_duration_ok": ev.get("rate_sem_duration_ok"),
                "rate_sem_energy_strong_direction_among_energy": ev.get(
                    "rate_sem_energy_strong_direction_among_energy"
                ),
                "n_analyzed_eval": ev.get("n_analyzed"),
            }
        )
    return out


def write_per_prompt_csv(path: Path, rows_flat: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_flat:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows_flat for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows_flat:
            w.writerow({k: row.get(k, "") for k in keys})


def pairwise_prompt_comparison_document(
    rows_a: list[Any],
    rows_b: list[Any],
    *,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """
    Align batches on (task_id, prompt_index). Each side may have multiple runs per prompt;
    summaries use means within that bucket.
    """
    ga = group_runs_by_task_prompt(rows_a)
    gb = group_runs_by_task_prompt(rows_b)
    keys_a = set(ga)
    keys_b = set(gb)
    common = sorted(keys_a & keys_b)
    pairs: list[dict[str, Any]] = []
    for k in common:
        sa = summarize_prompt_run_group(ga[k])
        sb = summarize_prompt_run_group(gb[k])
        ma = sa.get("mean_directional_adherence")
        mb = sb.get("mean_directional_adherence")
        delta = None if ma is None or mb is None else float(ma) - float(mb)
        winner: str | None
        if ma is None or mb is None:
            winner = None
        elif float(ma) > float(mb) + 1e-12:
            winner = label_a
        elif float(mb) > float(ma) + 1e-12:
            winner = label_b
        else:
            winner = "tie"
        pairs.append(
            {
                "task_id": sa["task_id"],
                "prompt_index": sa["prompt_index"],
                "prompt": sa["prompt"],
                label_a: sa,
                label_b: sb,
                "mean_directional_adherence_delta_first_minus_second": delta,
                "winner_by_mean_directional_adherence": winner,
            }
        )

    def key_row(tid: str, pi: int) -> dict[str, str | int]:
        return {"task_id": tid, "prompt_index": pi}

    only_a = [key_row(tid, pi) for tid, pi in sorted(keys_a - keys_b)]
    only_b = [key_row(tid, pi) for tid, pi in sorted(keys_b - keys_a)]
    return {
        "label_first": label_a,
        "label_second": label_b,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "keys_only_in_first_batch": only_a,
        "keys_only_in_second_batch": only_b,
    }


def pairwise_prompt_flat_csv_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten pairwise `pairs` for CSV (dynamic column prefix from labels)."""
    la = str(doc.get("label_first") or "batch_a")
    lb = str(doc.get("label_second") or "batch_b")
    out: list[dict[str, Any]] = []
    for row in doc.get("pairs") or []:
        a = row.get(la) or {}
        b = row.get(lb) or {}
        eva = a.get("evaluation_summary") or {}
        evb = b.get("evaluation_summary") or {}
        prompt = row.get("prompt") or ""
        out.append(
            {
                "task_id": row.get("task_id"),
                "prompt_index": row.get("prompt_index"),
                "prompt_preview": (prompt[:200] + "…") if len(str(prompt)) > 200 else prompt,
                f"n_runs__{la}": a.get("n_runs"),
                f"n_runs__{lb}": b.get("n_runs"),
                f"mean_directional_adherence__{la}": a.get("mean_directional_adherence"),
                f"mean_directional_adherence__{lb}": b.get("mean_directional_adherence"),
                "mean_adherence_delta_first_minus_second": row.get(
                    "mean_directional_adherence_delta_first_minus_second"
                ),
                "winner_by_mean_directional_adherence": row.get(
                    "winner_by_mean_directional_adherence"
                ),
                f"rate_wf_playable__{la}": eva.get("rate_wf_playable_nonsilent_wav"),
                f"rate_wf_playable__{lb}": evb.get("rate_wf_playable_nonsilent_wav"),
                f"rate_sem_strong_direction__{la}": eva.get(
                    "rate_sem_energy_strong_direction_among_energy"
                ),
                f"rate_sem_strong_direction__{lb}": evb.get(
                    "rate_sem_energy_strong_direction_among_energy"
                ),
            }
        )
    return out
