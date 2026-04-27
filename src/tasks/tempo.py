"""Tempo task: steady BPM adherence via librosa onset strength + beat_track."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np

from .base import Task, TaskSpec, register_task

_HOP = 512
_MAX_ONSET_PLOT_POINTS = 400


def parse_target_bpm_from_prompt(prompt: str) -> float:
    import re

    m = re.search(r"(\d+)\s*BPM", prompt, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 120.0


def tempo_append_failure_heuristics(scores: dict[str, Any], features: dict[str, Any]) -> None:
    """
    Reuses energy-task flag *names* for CSV compatibility; semantics are tempo-specific.
    """
    adherence = float(scores.get("adherence_primary", 0.0))
    err = float(scores.get("bpm_error") or 0.0)
    cvr = float(features.get("onset_contrast_ratio", 0.0))
    scores["flag_flat_energy_trajectory"] = cvr < 1.35
    scores["flag_weak_prompt_alignment"] = adherence < 0.45
    scores["flag_likely_wrong_energy_direction"] = err > 22.0


class TempoTask(Task):
    spec = TaskSpec(
        id="tempo",
        name="Tempo adherence",
        description="Estimated tempo should be near target BPM (librosa beat_track).",
    )

    def generate_prompts(self) -> list[str]:
        return [
            (
                "Create a steady click track at exactly 100 BPM for about 12 seconds. "
                "Use short, sharp pulses on every beat (one pulse per beat). "
                "Avoid continuous tones; each beat should be clearly separated and percussive."
            ),
            (
                "Create a steady rhythmic pattern at exactly 120 BPM for about 12 seconds. "
                "Use clear percussive or pulsed elements on each beat so the tempo is obvious."
            ),
            (
                "Generate a simple drum-like loop at 90 BPM, stable tempo, at least 10 seconds."
            ),
        ]

    def system_prompt_addon(self) -> str:
        return (
            "Honor the requested BPM literally. Space beats evenly in time; "
            "Nyquist uses seconds, so beat period = 60/BPM seconds."
        )

    def system_prompt_addon_sal(self) -> str:
        return (
            "Honor the requested BPM. In SAL, use **period = 60.0 / BPM** seconds between beats; "
            "shape pulses with **`pwl`** on a **`hzosc`** carrier or `sim` of short clicks; repeat the "
            "pattern to fill the clip duration. **`noise(dur)`** must be a **function call**, never bare "
            "`noise`. `return` the sound from `define function` … `end`, then `play your-entry()` on the last line."
        )

    def extract_features(self, audio_path: Path, sr: int | None = None) -> dict[str, Any]:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=_HOP)
        oenv = np.asarray(onset_env, dtype=float).ravel()
        mean_o = float(np.mean(oenv) + 1e-12)
        max_o = float(np.max(oenv) + 1e-12)
        contrast = max_o / mean_o

        tempo_arr, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=_HOP)
        if np.isscalar(tempo_arr):
            tempo_val = float(tempo_arr)
        else:
            tempo_val = float(np.asarray(tempo_arr).ravel()[0])

        n = len(times)
        if n <= _MAX_ONSET_PLOT_POINTS:
            idx = np.arange(n)
        else:
            idx = np.unique(
                np.linspace(0, n - 1, num=_MAX_ONSET_PLOT_POINTS, dtype=float).astype(int)
            )

        return {
            "sr": int(sr),
            "hop_length": _HOP,
            "duration_sec": float(len(y) / sr),
            "estimated_bpm": tempo_val,
            "onset_contrast_ratio": contrast,
            "onset_strength_times": times[idx].astype(float).tolist(),
            "onset_strength": oenv[idx].astype(float).tolist(),
            "beat_frames": [int(x) for x in np.asarray(beats).ravel()[:256]],
        }

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        target = float(features.get("target_bpm", 120))
        est = float(features.get("estimated_bpm", 0.0))
        if target <= 0:
            return {
                "target_bpm": target,
                "estimated_bpm": est,
                "bpm_error": None,
                "adherence_primary": 0.0,
            }
        err = abs(est - target)
        adherence = max(0.0, 1.0 - max(0.0, err - 5.0) / 35.0)
        return {
            "target_bpm": target,
            "estimated_bpm": est,
            "bpm_error": err,
            "adherence_primary": adherence,
        }

    def plot(self, features: dict[str, Any], out_path: Path, title: str | None = None) -> None:
        tx = features.get("onset_strength_times") or []
        oy = features.get("onset_strength") or []
        tgt = features.get("target_bpm")
        est = features.get("estimated_bpm")

        fig, axes = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [2.2, 1]})
        if len(tx) == len(oy) and len(tx) > 1:
            axes[0].plot(np.asarray(tx, dtype=float), np.asarray(oy, dtype=float), color="#4C72B0", lw=1.0)
            axes[0].set_ylabel("Onset strength")
            axes[0].set_title(title or "Tempo (onset envelope)")
        else:
            axes[0].text(0.5, 0.5, "Onset strength N/A", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_xlabel("Time (s)")

        if tgt is not None and est is not None:
            axes[1].axis("off")
            msg = (
                f"Target BPM: {float(tgt):.1f}\n"
                f"Estimated BPM: {float(est):.1f}\n"
                f"(librosa.beat.beat_track — may halve/double vs true pulse rate)"
            )
            axes[1].text(0.02, 0.95, msg, transform=axes[1].transAxes, va="top", fontsize=11, family="monospace")
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()


register_task(TempoTask())
