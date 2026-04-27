#!/usr/bin/env python3
"""
Compare two completed SAL batches (same task layout): writes

  <output>/
    combined/   — 18 rows if each input has 9 analyzed WAVs; same plots/JSON as analyze_wavs.py
    batch_a/    — copy of batch A analysis (recomputed from disk)
    batch_b/    — copy of batch B analysis

Each subfolder gets `analysis_runs.jsonl`, `analysis_summary.csv`, `plots/*.png`,
`analysis_by_prompt.json`, `analysis_by_prompt.csv`, and `analysis_evaluation_summary.json`.

`combined/` additionally gets `analysis_prompt_pairwise.json`, `analysis_prompt_pairwise.csv`,
and `plots/analysis_prompt_pairwise_mean_adherence.png` (same prompt key in both batches).

`combined/analysis_runs.jsonl` adds `comparison_model` and `comparison_batch_root`
per row.

Usage:
  python compare_task_batches.py \\
    --a outputs/run_20260421T201216Z_gemini_spectral1 \\
    --b outputs/run_20260421T201430Z_spectral_ollama \\
    --label-a gemini --label-b ollama \\
    --output outputs/compare_spectral_gemini_vs_ollama
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.evaluation_axes import (
    aggregate_analyzed_evaluations,
    nyquist_subprocess_stats_from_batch_jsonl,
    pairwise_prompt_comparison_document,
    pairwise_prompt_flat_csv_rows,
    write_per_prompt_csv,
)
from src.pipeline.run import plot_pairwise_prompt_mean_adherence

from analyze_wavs import collect_analyzed_run_records, write_batch_analysis_artifacts


def _write_labeled_subset(
    out_dir: Path,
    rows: list,
    *,
    label: str,
    source_root: Path,
    nyquist_root: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def extra(_r) -> dict:
        return {"comparison_model": label, "comparison_batch_root": str(source_root.resolve())}

    nstat = nyquist_subprocess_stats_from_batch_jsonl(nyquist_root)
    write_batch_analysis_artifacts(
        out_dir,
        rows,
        jsonl_row_extra=extra,
        nyquist_stats_override=nstat,
        eval_summary_extra={
            "comparison_source_batch": str(source_root.resolve()),
            "comparison_model": label,
            "from_analyzed_wavs_this_batch_only": aggregate_analyzed_evaluations(rows),
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Pairwise batch analysis: per-batch + combined folder")
    p.add_argument("--a", type=Path, required=True, metavar="BATCH_A", help="First batch root")
    p.add_argument("--b", type=Path, required=True, metavar="BATCH_B", help="Second batch root")
    p.add_argument("--output", type=Path, required=True, metavar="DIR", help="Output directory (created)")
    p.add_argument("--label-a", type=str, default="batch_a", help="Tag for model A in combined JSONL")
    p.add_argument("--label-b", type=str, default="batch_b", help="Tag for model B in combined JSONL")
    args = p.parse_args()

    a = args.a.resolve()
    b = args.b.resolve()
    out = args.output.resolve()
    if not a.is_dir() or not b.is_dir():
        print("Both --a and --b must be existing directories.", file=sys.stderr)
        return 2

    rows_a, sk_a = collect_analyzed_run_records(a)
    rows_b, sk_b = collect_analyzed_run_records(b)
    if not rows_a:
        print(f"No analyzed runs with out.wav under {a}", file=sys.stderr)
        return 1
    if not rows_b:
        print(f"No analyzed runs with out.wav under {b}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    combined = rows_a + rows_b
    ids_a = {r.run_id for r in rows_a}

    def extra_combined(r) -> dict:
        if r.run_id in ids_a:
            return {"comparison_model": args.label_a, "comparison_batch_root": str(a)}
        return {"comparison_model": args.label_b, "comparison_batch_root": str(b)}

    nstat = {
        "batch_a": str(a),
        "batch_b": str(b),
        "from_runs_jsonl_nyquistexec_a": nyquist_subprocess_stats_from_batch_jsonl(a),
        "from_runs_jsonl_nyquistexec_b": nyquist_subprocess_stats_from_batch_jsonl(b),
    }
    write_batch_analysis_artifacts(
        out / "combined",
        combined,
        jsonl_row_extra=extra_combined,
        nyquist_stats_override=nstat,
        eval_summary_extra={
            "comparison_batch_a": str(a),
            "comparison_batch_b": str(b),
            "comparison_label_a": args.label_a,
            "comparison_label_b": args.label_b,
            "from_analyzed_wavs_combined": aggregate_analyzed_evaluations(combined),
            "from_analyzed_wavs_batch_a_only": aggregate_analyzed_evaluations(rows_a),
            "from_analyzed_wavs_batch_b_only": aggregate_analyzed_evaluations(rows_b),
        },
    )

    pair_doc = pairwise_prompt_comparison_document(
        rows_a, rows_b, label_a=args.label_a, label_b=args.label_b
    )
    comb = out / "combined"
    (comb / "analysis_prompt_pairwise.json").write_text(
        json.dumps(pair_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_per_prompt_csv(
        comb / "analysis_prompt_pairwise.csv",
        pairwise_prompt_flat_csv_rows(pair_doc),
    )
    plot_pairwise_prompt_mean_adherence(
        pair_doc, comb / "plots" / "analysis_prompt_pairwise_mean_adherence.png"
    )

    _write_labeled_subset(out / "batch_a", rows_a, label=args.label_a, source_root=a, nyquist_root=a)
    _write_labeled_subset(out / "batch_b", rows_b, label=args.label_b, source_root=b, nyquist_root=b)

    print(
        f"Wrote {out / 'combined'} ({len(combined)} runs), "
        f"{out / 'batch_a'} ({len(rows_a)} runs, skipped {sk_a} no wav), "
        f"{out / 'batch_b'} ({len(rows_b)} runs, skipped {sk_b} no wav)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
