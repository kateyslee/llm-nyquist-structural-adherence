#!/usr/bin/env python3
"""
After manual NyquistIDE export: find `out.wav` next to each `meta.json` under a batch
directory (same layout as `run_batch.py --format sal`), run librosa metrics, write
`features.json`, `scores.json`, `plot.png`, and update `meta.json` with `analyzed_utc`.

Usage:
  python analyze_wavs.py outputs/run_20260403T200643Z
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.evaluation_axes import (
    aggregate_analyzed_evaluations,
    nyquist_subprocess_stats_from_batch_jsonl,
    per_prompt_evaluation_document,
    per_prompt_flat_csv_rows,
    write_per_prompt_csv,
)
from src.pipeline.run import (
    RunRecord,
    analyze_run_dir,
    plot_density_adherence,
    plot_energy_adherence,
    plot_failure_breakdown,
    plot_spectral_adherence,
    plot_tempo_adherence,
    plot_wellformed_vs_semantic,
    write_runs_csv,
)
from src.pipeline.status import RunStatus


def iter_batch_run_dirs(batch_root: Path) -> list[Path]:
    """All `task_id/run_id/` dirs under a batch root that contain `meta.json`."""
    batch_root = batch_root.resolve()
    out: list[Path] = []
    for sub in sorted(batch_root.iterdir()):
        if not sub.is_dir() or sub.name.startswith(".") or sub.name == "plots":
            continue
        if sub.name in ("energy", "tempo", "spectral", "density"):
            for run_dir in sorted(sub.iterdir()):
                if run_dir.is_dir() and (run_dir / "meta.json").is_file():
                    out.append(run_dir)
    return out


def collect_analyzed_run_records(batch_root: Path) -> tuple[list[RunRecord], int]:
    """Run `analyze_run_dir` for every run with `out.wav`; return (records, skipped_no_wav)."""
    rows: list[RunRecord] = []
    skipped = 0
    for rd in iter_batch_run_dirs(batch_root):
        if not (rd / "out.wav").is_file():
            skipped += 1
            continue
        rec = analyze_run_dir(rd)
        if rec is not None:
            rows.append(rec)
    return rows, skipped


def write_batch_analysis_artifacts(
    batch_root: Path,
    rows: list[RunRecord],
    *,
    jsonl_row_extra: Callable[[RunRecord], dict[str, Any]] | None = None,
    nyquist_stats_override: dict[str, Any] | None = None,
    eval_summary_extra: dict[str, Any] | None = None,
) -> None:
    """Write analysis_runs.jsonl, CSV, plots, and analysis_evaluation_summary.json under batch_root."""
    batch_root = batch_root.resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = batch_root / "analysis_runs.jsonl"
    lines: list[str] = []
    for r in rows:
        d = r.to_json_dict()
        if jsonl_row_extra:
            d.update(jsonl_row_extra(r))
        lines.append(json.dumps(d, ensure_ascii=False))
    out_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    write_runs_csv(batch_root / "analysis_summary.csv", rows)
    plot_dir = batch_root / "plots"
    plot_failure_breakdown(rows, plot_dir / "analysis_failure_breakdown.png", title="Analysis outcomes")
    plot_energy_adherence(rows, plot_dir / "analysis_energy_adherence.png")
    plot_spectral_adherence(rows, plot_dir / "analysis_spectral_adherence.png")
    plot_tempo_adherence(rows, plot_dir / "analysis_tempo_adherence.png")
    plot_density_adherence(rows, plot_dir / "analysis_density_adherence.png")
    plot_wellformed_vs_semantic(rows, plot_dir / "analysis_wellformed_vs_semantic.png")

    ppt = per_prompt_evaluation_document(rows)
    (batch_root / "analysis_by_prompt.json").write_text(
        json.dumps(ppt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_per_prompt_csv(batch_root / "analysis_by_prompt.csv", per_prompt_flat_csv_rows(ppt["by_prompt"]))

    nstat = (
        nyquist_stats_override
        if nyquist_stats_override is not None
        else nyquist_subprocess_stats_from_batch_jsonl(batch_root)
    )
    eval_summary: dict[str, Any] = {
        "batch_root": str(batch_root),
        "from_runs_jsonl_nyquistexec": nstat,
        "from_analyzed_wavs": aggregate_analyzed_evaluations(rows),
    }
    if eval_summary_extra:
        eval_summary.update(eval_summary_extra)
    (batch_root / "analysis_evaluation_summary.json").write_text(
        json.dumps(eval_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run feature extraction + scores on manual out.wav exports (SAL workflow)"
    )
    p.add_argument(
        "batch_root",
        type=Path,
        help="Batch folder from run_batch.py (contains energy/, tempo/, spectral/, density/, …)",
    )
    args = p.parse_args()
    root = args.batch_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    run_dirs = iter_batch_run_dirs(root)
    if not run_dirs:
        print(f"No meta.json under {root}/energy|tempo|spectral|density/* — nothing to analyze.", file=sys.stderr)
        return 1

    rows: list[RunRecord] = []
    skipped = 0
    for rd in run_dirs:
        if not (rd / "out.wav").is_file():
            skipped += 1
            continue
        rec = analyze_run_dir(rd)
        if rec is not None:
            rows.append(rec)
            extra = ""
            if rec.status == RunStatus.DURATION_SHORTFALL:
                extra = " (LLM duration shortfall — WAV shorter than duration_hint_sec)"
            print(f"{rec.task_id}/{rd.name} → {rec.status.value}{extra}")

    write_batch_analysis_artifacts(root, rows)

    ev = aggregate_analyzed_evaluations(rows)
    rj = nyquist_subprocess_stats_from_batch_jsonl(root)

    ok = sum(1 for r in rows if r.status.value == "success")
    dur_sf = sum(1 for r in rows if r.status.value == "duration_shortfall")
    print(f"\nAnalyzed {len(rows)} run(s), {ok} success. Skipped {skipped} (no out.wav).")
    if dur_sf:
        print(
            f"{dur_sf} run(s) status=duration_shortfall (WAV shorter than batch duration hint); "
            "see duration_shortfall.txt and scores.json in those run dirs."
        )
    if ev.get("n_analyzed"):
        print(
            "\nWell-formedness vs semantic (see analysis_evaluation_summary.json, "
            "plots/analysis_wellformed_vs_semantic.png):"
        )
        n_ev = int(ev["n_analyzed"])
        print(
            f"  WF playable non-silent WAV: {ev.get('count_wf_playable_nonsilent_wav', 0)}/{n_ev} "
            f"({100 * float(ev.get('rate_wf_playable_nonsilent_wav') or 0):.0f}%)"
        )
        print(
            f"  SEM duration OK: {ev.get('count_sem_duration_ok', 0)}/{n_ev} "
            f"({100 * float(ev.get('rate_sem_duration_ok') or 0):.0f}%)"
        )
        ne = int(ev.get("n_energy_rows_for_strong_metric") or 0)
        if ne:
            print(
                f"  SEM primary adherence ≥0.72 (energy, spectral, tempo, density): "
                f"{ev.get('count_sem_energy_strong_direction', 0)}/{ne} "
                f"({100 * float(ev.get('rate_sem_energy_strong_direction_among_energy') or 0):.0f}%)"
            )
    if rj.get("runs_jsonl_present") and rj.get("total", 0) > 0:
        denom = int(rj.get("automated_nyquist_denominator") or 0)
        if denom <= 0:
            print(
                "\nNyquist subprocess (from runs.jsonl): no automated `.ny` runs in this snapshot "
                f"({rj.get('pending_manual_render', 0)}/{rj.get('total', 0)} still "
                "`pending_manual_render` — typical for SAL before export). "
                "Use WF rates above after WAV analysis."
            )
        else:
            rate = rj.get("rate_nyquist_subprocess_no_hard_fail")
            rate_s = f"{100 * float(rate):.0f}%" if rate is not None else "n/a"
            print(
                "\nNyquist subprocess (from runs.jsonl at batch generation): "
                f"failures syntax/runtime/timeout/not_found/render={rj.get('nyquist_subprocess_failure_total', 0)} "
                f"of {denom} non-pending; rate no hard fail={rate_s}. "
                f"{rj.get('note', '')}"
            )
    print(f"Wrote {root / 'analysis_runs.jsonl'} and analysis_summary.csv under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
