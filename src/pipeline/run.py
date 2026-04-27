"""Orchestration: prompt → LLM → Nyquist → features → scores → logs."""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from google.api_core import exceptions as google_api_exceptions

from ..config import OUTPUTS_DIR
from ..tasks.base import GenerationContext, default_nyquist_rules, default_sal_rules, get_task
from ..tasks.energy import (
    energy_append_failure_heuristics,
    energy_directional_score,
    energy_prompt_direction,
)
from ..tasks.density import density_append_failure_heuristics
from ..tasks.tempo import parse_target_bpm_from_prompt, tempo_append_failure_heuristics
from .llm import LLMClient
from .evaluation_axes import build_evaluation_axes
from .nyquist_runner import run_nyquist_code, validate_wav
from .status import RunStatus

# Rendered WAV shorter than pipeline duration hint → flag as LLM/synthesis fault (not infra).
_DURATION_OK_MIN_RATIO = 0.92
_DURATION_OK_FLOOR_SEC = 0.25


def _duration_accountability(
    *,
    features: dict[str, Any],
    scores: dict[str, Any],
    duration_hint_sec: float | None,
) -> tuple[RunStatus | None, str | None]:
    """
    If duration_hint_sec is set, record target vs actual in scores. Returns
    (DURATION_SHORTFALL, message) when audio is materially shorter than the hint.
    """
    if duration_hint_sec is None or duration_hint_sec <= 0:
        return None, None
    target = float(duration_hint_sec)
    actual = float(features.get("duration_sec", 0.0))
    min_ok = max(_DURATION_OK_FLOOR_SEC, target * _DURATION_OK_MIN_RATIO)
    scores["target_duration_sec"] = target
    scores["actual_duration_sec"] = actual
    scores["duration_min_acceptable_sec"] = min_ok
    scores["duration_deficit_sec"] = max(0.0, target - actual)
    short = actual < min_ok
    scores["llm_duration_shortfall"] = short
    if not short:
        return None, None
    msg = (
        f"LLM/synthesis duration fault: WAV is {actual:.2f}s but pipeline target was {target:.2f}s "
        f"(min acceptable {min_ok:.2f}s — likely wrong SAL/Nyquist duration or export range)."
    )
    return RunStatus.DURATION_SHORTFALL, msg


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    prompt_index: int
    prompt: str
    model_label: str
    status: RunStatus
    llm_backend: str
    code_format: str = "ny"
    code_path: str | None = None
    wav_path: str | None = None
    features_path: str | None = None
    plot_path: str | None = None
    stderr_snippet: str | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    features_summary: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        ev = d.pop("evaluation", None) or {}
        for k, v in ev.items():
            d[f"eval__{k}"] = v
        return d


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def effective_duration_hint_sec(task_id: str, prompt_index: int, cli_default: float) -> float:
    """
    Duration used in system prompts, meta.json, and shortfall checks. Matches written prompts
    in energy/tempo/spectral/density tasks so we do not flag e.g. a 12 s clip as short vs a global 15 s CLI default.
    """
    if task_id == "energy":
        return {0: 15.0, 1: 12.0, 2: 14.0}.get(prompt_index, float(cli_default))
    if task_id == "tempo":
        return {0: 12.0, 1: 10.0}.get(prompt_index, float(cli_default))
    if task_id == "spectral":
        return {0: 12.0, 1: 10.0, 2: 14.0}.get(prompt_index, float(cli_default))
    if task_id == "density":
        return {0: 12.0, 1: 11.0, 2: 14.0}.get(prompt_index, float(cli_default))
    return float(cli_default)


def compute_metrics_for_wav(
    *,
    task_id: str,
    prompt_index: int,
    prompt: str,
    wav_path: Path,
    run_dir: Path,
    run_id: str,
    model_label: str,
    llm_backend: str,
    code_path: str | None,
    code_format: str,
    created_utc: str,
    duration_hint_sec: float | None = None,
) -> RunRecord:
    """Librosa features + scores + plot; wav must exist on disk."""
    def _eval_axes(st: RunStatus, sc: dict[str, Any]) -> dict[str, Any]:
        return build_evaluation_axes(
            status=st,
            code_format=code_format,
            task_id=task_id,
            scores=sc,
        )

    wav_path = wav_path.resolve()
    audio_status = validate_wav(wav_path)
    record = RunRecord(
        run_id=run_id,
        task_id=task_id,
        prompt_index=prompt_index,
        prompt=prompt,
        model_label=model_label,
        status=audio_status,
        llm_backend=llm_backend,
        code_format=code_format,
        code_path=code_path,
        wav_path=str(wav_path) if wav_path.is_file() else None,
        created_utc=created_utc,
    )
    if audio_status != RunStatus.SUCCESS:
        record.evaluation = _eval_axes(audio_status, {})
        return record

    task = get_task(task_id)
    features = task.extract_features(wav_path)
    if task_id == "tempo":
        features = {**features, "target_bpm": parse_target_bpm_from_prompt(prompt)}
    if task_id == "spectral":
        features = {**features, "prompt_index": prompt_index}
    if task_id == "density":
        features = {**features, "prompt_index": prompt_index}
    feat_path = run_dir / "features.json"
    serializable = {k: _jsonify(v) for k, v in features.items()}
    feat_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    record.features_path = str(feat_path)

    scores = task.score(features)
    if task_id == "energy":
        direction = energy_prompt_direction(prompt_index)
        scores["direction"] = direction
        scores["directional_adherence"] = energy_directional_score(scores, direction)
        scores["adherence_primary"] = scores["directional_adherence"]
        energy_append_failure_heuristics(scores, direction)
    elif task_id == "tempo":
        ap = float(scores.get("adherence_primary", 0.0))
        scores["direction"] = "tempo_bpm"
        scores["directional_adherence"] = ap
        tempo_append_failure_heuristics(scores, features)
    elif task_id == "density":
        ap = float(scores.get("adherence_primary", 0.0))
        scores["directional_adherence"] = ap
        density_append_failure_heuristics(scores, features)

    dur_status, dur_msg = _duration_accountability(
        features=features, scores=scores, duration_hint_sec=duration_hint_sec
    )
    if dur_status is not None:
        record.status = dur_status
        record.stderr_snippet = dur_msg

    record.scores = {k: _jsonify(v) for k, v in scores.items()}
    _feat_summary_keys = (
        "duration_sec",
        "estimated_bpm",
        "target_bpm",
        "dynamic_range_db",
        "spectral_span_hz",
        "spectral_centroid_mean_hz",
        "spectral_centroid_std_hz",
        "self_similarity_mean_offdiag",
        "self_similarity_max_offdiag",
        "chroma_recurrence_frames",
        "onset_contrast_ratio",
        "onset_density_hz",
        "onset_density_first_quarter_hz",
        "onset_density_last_quarter_hz",
        "onset_density_delta_hz",
        "onset_density_middle_half_hz",
    )
    record.features_summary = {
        k: _jsonify(v) for k, v in features.items() if k in _feat_summary_keys
    }

    plot_path = run_dir / "plot.png"
    title = f"{task.spec.name} | {model_label} | run {run_id}"
    task.plot(features, plot_path, title=title)
    record.plot_path = str(plot_path)

    (run_dir / "scores.json").write_text(json.dumps(record.scores, indent=2), encoding="utf-8")
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["analyzed_utc"] = _utc_now()
        if duration_hint_sec is not None:
            meta["duration_hint_sec"] = duration_hint_sec
        if "llm_duration_shortfall" in scores:
            meta["llm_duration_shortfall"] = bool(scores["llm_duration_shortfall"])
        if scores.get("target_duration_sec") is not None:
            meta["target_duration_sec"] = scores["target_duration_sec"]
        if scores.get("actual_duration_sec") is not None:
            meta["actual_duration_sec"] = scores["actual_duration_sec"]
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    dsf = run_dir / "duration_shortfall.txt"
    if record.status == RunStatus.DURATION_SHORTFALL and dur_msg:
        dsf.write_text(dur_msg + "\n", encoding="utf-8")
    elif dsf.is_file():
        dsf.unlink()

    record.evaluation = _eval_axes(record.status, record.scores)
    return record


def analyze_run_dir(run_dir: Path) -> RunRecord | None:
    """
    If `meta.json` + `out.wav` exist, compute features/scores (manual SAL → WAV workflow).
    Returns None if prerequisites missing.
    """
    run_dir = run_dir.resolve()
    meta_path = run_dir / "meta.json"
    wav_path = run_dir / "out.wav"
    if not meta_path.is_file() or not wav_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    code_file = meta.get("code_filename", "generated.sal")
    cp = run_dir / code_file
    code_path_str = str(cp) if cp.is_file() else None
    raw_hint = meta.get("duration_hint_sec")
    duration_hint_sec = float(raw_hint) if raw_hint is not None else None
    return compute_metrics_for_wav(
        task_id=meta["task_id"],
        prompt_index=int(meta["prompt_index"]),
        prompt=meta["prompt"],
        wav_path=wav_path,
        run_dir=run_dir,
        run_id=meta["run_id"],
        model_label=meta.get("model_label", ""),
        llm_backend=meta.get("llm_backend", ""),
        code_path=code_path_str,
        code_format=meta.get("code_format", "sal"),
        created_utc=meta.get("created_utc", ""),
        duration_hint_sec=duration_hint_sec,
    )


def run_single_generation(
    *,
    task_id: str,
    prompt_index: int,
    llm: LLMClient,
    model_label: str,
    llm_backend: str,
    run_root: Path | None = None,
    duration_hint_sec: float = 15.0,
    code_format: str = "ny",
) -> RunRecord:
    run_id = uuid.uuid4().hex[:12]
    root = (run_root or OUTPUTS_DIR).resolve()
    run_dir = root / task_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(task_id)
    prompts = task.generate_prompts()
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(f"prompt_index {prompt_index} out of range for task {task_id}")
    prompt = prompts[prompt_index]

    wav_path = run_dir / "out.wav"
    dur_hint = effective_duration_hint_sec(task_id, prompt_index, duration_hint_sec)
    ctx = GenerationContext(output_wav_abs=wav_path.as_posix(), duration_hint_sec=dur_hint)
    code_format = (code_format or "ny").lower().strip()
    if code_format not in ("ny", "sal"):
        code_format = "ny"
    if code_format == "sal":
        system = default_sal_rules(ctx, task_id=task_id) + "\n\n" + task.system_prompt_addon_sal()
    else:
        system = default_nyquist_rules(ctx) + "\n\n" + task.system_prompt_addon()

    record = RunRecord(
        run_id=run_id,
        task_id=task_id,
        prompt_index=prompt_index,
        prompt=prompt,
        model_label=model_label,
        status=RunStatus.GENERATION_FAILED,
        llm_backend=llm_backend,
        code_format=code_format,
        created_utc=_utc_now(),
    )

    def _finalize() -> RunRecord:
        record.evaluation = build_evaluation_axes(
            status=record.status,
            code_format=record.code_format,
            task_id=record.task_id,
            scores=record.scores,
        )
        return record

    try:
        code = llm.generate(system, prompt)
    except google_api_exceptions.ResourceExhausted as exc:
        record.status = RunStatus.LLM_QUOTA_EXCEEDED
        record.stderr_snippet = f"LLM quota / rate limit (429): {exc}"
        (run_dir / "error.txt").write_text(record.stderr_snippet, encoding="utf-8")
        return _finalize()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        low = msg.lower()
        if "429" in msg or "quota exceeded" in low or "resource exhausted" in low:
            record.status = RunStatus.LLM_QUOTA_EXCEEDED
            record.stderr_snippet = f"LLM quota / rate limit: {exc}"
        else:
            record.stderr_snippet = f"LLM error: {exc}"
        (run_dir / "error.txt").write_text(record.stderr_snippet, encoding="utf-8")
        return _finalize()

    if not code.strip():
        record.stderr_snippet = "Empty LLM output (no text returned)."
        (run_dir / "error.txt").write_text(record.stderr_snippet, encoding="utf-8")
        return _finalize()

    if code_format == "sal":
        code_filename = "generated.sal"
        code_path = run_dir / code_filename
        header = (
            "; --- Pipeline (SAL): open in NyquistIDE ---\n"
            f"; Save exported audio to (exact path):\n"
            f";   {wav_path.as_posix()}\n"
            "; Then from the project folder:\n"
            ";   python analyze_wavs.py <batch_root>\n"
            "; where <batch_root> is the outputs/run_<timestamp>_<task>_... directory that contains "
            "energy/, tempo/, spectral/, or density/.\n"
            "; ------------------------------------------\n\n"
        )
        code_path.write_text(header + code, encoding="utf-8")
        record.code_path = str(code_path)
        meta = {
            "run_id": run_id,
            "task_id": task_id,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "model_label": model_label,
            "llm_backend": llm_backend,
            "code_format": "sal",
            "code_filename": code_filename,
            "output_wav_filename": "out.wav",
            "created_utc": record.created_utc,
            "duration_hint_sec": dur_hint,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        record.status = RunStatus.PENDING_MANUAL_RENDER
        record.stderr_snippet = (
            "SAL saved; open in NyquistIDE, render, export to out.wav here, "
            "then: python analyze_wavs.py <batch_root>"
        )
        return _finalize()

    code_path = run_dir / "generated.ny"
    code_path.write_text(code, encoding="utf-8")
    record.code_path = str(code_path)

    nyq = run_nyquist_code(code, wav_path)
    record.status = nyq.status
    record.stderr_snippet = nyq.stderr_snippet
    if nyq.wav_path:
        record.wav_path = str(nyq.wav_path.resolve())

    if record.status != RunStatus.SUCCESS:
        return _finalize()

    return compute_metrics_for_wav(
        task_id=task_id,
        prompt_index=prompt_index,
        prompt=prompt,
        wav_path=wav_path,
        run_dir=run_dir,
        run_id=run_id,
        model_label=model_label,
        llm_backend=llm_backend,
        code_path=str(code_path),
        code_format="ny",
        created_utc=record.created_utc,
        duration_hint_sec=dur_hint,
    )


def _jsonify(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    return str(v)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_runs_csv(path: Path, rows: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    flat: list[dict[str, Any]] = []
    for r in rows:
        d = r.to_json_dict()
        scores = d.pop("scores", {}) or {}
        for k, v in scores.items():
            d[f"score__{k}"] = v
        fs = d.pop("features_summary", {}) or {}
        for k, v in fs.items():
            d[f"feat__{k}"] = v
        flat.append(d)
    keys = sorted({k for row in flat for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in flat:
            w.writerow({k: row.get(k, "") for k in keys})


def plot_failure_breakdown(rows: list[RunRecord], out_path: Path, title: str | None = None) -> None:
    from collections import Counter

    c = Counter(r.status.value for r in rows)
    labels = list(c.keys())
    vals = [c[k] for k in labels]
    plt.figure(figsize=(7, 3.5))
    plt.bar(labels, vals, color="#4C72B0")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title(title or "Failure / outcome breakdown")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_energy_adherence(rows: list[RunRecord], out_path: Path) -> None:
    _ok = frozenset({RunStatus.SUCCESS, RunStatus.DURATION_SHORTFALL})
    sub = [r for r in rows if r.task_id == "energy" and r.status in _ok]
    if not sub:
        return
    xs = range(len(sub))
    ys = [float(r.scores.get("directional_adherence", 0.0)) for r in sub]
    plt.figure(figsize=(8, 3))
    plt.bar(list(xs), ys, color="#55A868")
    plt.xlabel("Sample index (batch order)")
    plt.ylabel("Directional adherence [0,1]")
    plt.title("Energy task: directional RMS adherence")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tempo_adherence(rows: list[RunRecord], out_path: Path) -> None:
    _ok = frozenset({RunStatus.SUCCESS, RunStatus.DURATION_SHORTFALL})
    sub = [r for r in rows if r.task_id == "tempo" and r.status in _ok]
    if not sub:
        return
    xs = range(len(sub))
    ys = [float(r.scores.get("directional_adherence", r.scores.get("adherence_primary", 0.0))) for r in sub]
    plt.figure(figsize=(8, 3))
    plt.bar(list(xs), ys, color="#E2975D")
    plt.xlabel("Sample index (batch order)")
    plt.ylabel("BPM adherence [0,1]")
    plt.title("Tempo task: beat-track adherence vs target BPM")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectral_adherence(rows: list[RunRecord], out_path: Path) -> None:
    _ok = frozenset({RunStatus.SUCCESS, RunStatus.DURATION_SHORTFALL})
    sub = [r for r in rows if r.task_id == "spectral" and r.status in _ok]
    if not sub:
        return
    xs = range(len(sub))
    ys = [float(r.scores.get("directional_adherence", 0.0)) for r in sub]
    plt.figure(figsize=(8, 3))
    plt.bar(list(xs), ys, color="#8172B2")
    plt.xlabel("Sample index (batch order)")
    plt.ylabel("Directional adherence [0,1]")
    plt.title("Spectral task: centroid trajectory adherence")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_density_adherence(rows: list[RunRecord], out_path: Path) -> None:
    _ok = frozenset({RunStatus.SUCCESS, RunStatus.DURATION_SHORTFALL})
    sub = [r for r in rows if r.task_id == "density" and r.status in _ok]
    if not sub:
        return
    xs = range(len(sub))
    ys = [float(r.scores.get("directional_adherence", r.scores.get("adherence_primary", 0.0))) for r in sub]
    plt.figure(figsize=(8, 3))
    plt.bar(list(xs), ys, color="#64B5CD")
    plt.xlabel("Sample index (batch order)")
    plt.ylabel("Density adherence [0,1]")
    plt.title("Density task: first vs last quarter onset rate (directional)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_wellformed_vs_semantic(rows: list[RunRecord], out_path: Path, title: str | None = None) -> None:
    """Batch-level rates: playable WAV (WF) vs duration / adherence (SEM)."""
    from .evaluation_axes import aggregate_analyzed_evaluations

    agg = aggregate_analyzed_evaluations(rows)
    n = int(agg.get("n_analyzed") or 0)
    if n == 0:
        return

    labels = [
        "WF: playable\nnon-silent WAV",
        "SEM: duration\nacceptable",
        "SEM: primary\nadherence ≥0.72",
    ]
    y1 = float(agg.get("rate_wf_playable_nonsilent_wav") or 0.0)
    y2 = float(agg.get("rate_sem_duration_ok") or 0.0)
    y3_raw = agg.get("rate_sem_energy_strong_direction_among_energy")
    y3 = float(y3_raw) if y3_raw is not None else 0.0

    plt.figure(figsize=(7.5, 3.8))
    colors = ("#8da0cb", "#66c2a5", "#fc8d62")
    bars = plt.bar(labels, [y1, y2, y3], color=colors)
    plt.ylabel("Fraction of analyzed runs")
    plt.ylim(0, 1.05)
    plt.axhline(1.0, color="#333", linewidth=0.5, linestyle="--", alpha=0.4)
    for b, v in zip(bars, [y1, y2, y3], strict=True):
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            min(v + 0.03, 1.02),
            f"{v:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title(title or f"Well-formedness vs semantic (n={n} analyzed)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pairwise_prompt_mean_adherence(pairwise_doc: dict[str, Any], out_path: Path) -> None:
    """Grouped bars: mean directional adherence per (task_id, prompt_index) for two batches."""
    la = str(pairwise_doc.get("label_first") or "batch_a")
    lb = str(pairwise_doc.get("label_second") or "batch_b")
    pair_rows: list[dict[str, Any]] = list(pairwise_doc.get("pairs") or [])
    xs_labels: list[str] = []
    ys_a: list[float] = []
    ys_b: list[float] = []
    for row in pair_rows:
        a = row.get(la) or {}
        b = row.get(lb) or {}
        ma = a.get("mean_directional_adherence")
        mb = b.get("mean_directional_adherence")
        if ma is None or mb is None:
            continue
        tid = str(row.get("task_id", ""))
        pi = int(row.get("prompt_index", 0))
        xs_labels.append(f"{tid[:4]}… p{pi}")
        ys_a.append(float(ma) if ma is not None else 0.0)
        ys_b.append(float(mb) if mb is not None else 0.0)
    if not xs_labels:
        return
    n = len(xs_labels)
    x = list(range(n))
    width = 0.36
    plt.figure(figsize=(max(6.0, n * 1.15), 3.8))
    plt.bar([i - width / 2.0 for i in x], ys_a, width, label=la, color="#8da0cb")
    plt.bar([i + width / 2.0 for i in x], ys_b, width, label=lb, color="#fc8d62")
    plt.xticks(x, xs_labels, rotation=35, ha="right")
    plt.ylabel("Mean directional adherence [0,1]")
    plt.ylim(0, 1.05)
    plt.axhline(1.0, color="#333", linewidth=0.5, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", fontsize=9)
    plt.title("Per-prompt comparison (mean over replicates per batch)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
