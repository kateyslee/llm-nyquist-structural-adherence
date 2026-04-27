"""Spectral trajectory task: brightness (centroid Hz) vs time vs prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d

from .base import Task, TaskSpec, register_task
from .energy import _chroma_self_similarity_scalars, _spectral_centroid_track


def _smooth_centroid_series(cent: np.ndarray) -> np.ndarray:
    if cent.size < 4:
        return cent.astype(float)
    win = max(3, cent.size // 20 | 1)
    if win % 2 == 0:
        win += 1
    return uniform_filter1d(cent.astype(float), size=win, mode="nearest")


def _spearman_rise_fall(y: np.ndarray) -> tuple[float, float, float]:
    """Spearman of normalized time vs y and vs -y; monotonic_strength = max(|rho|)."""
    if y.size < 4:
        return 0.0, 0.0, 0.0
    x = np.linspace(0.0, 1.0, num=y.size)
    rho_up, _ = stats.spearmanr(x, y)
    rho_down, _ = stats.spearmanr(x, -y)
    if not np.isfinite(rho_up):
        rho_up = 0.0
    if not np.isfinite(rho_down):
        rho_down = 0.0
    best = max(abs(float(rho_up)), abs(float(rho_down)))
    return float(rho_up), float(rho_down), float(best)


def _directional_from_rhos(rho_rise: float, rho_fall: float, direction: str) -> float:
    if direction == "fall":
        rho = rho_fall
    else:
        rho = rho_rise
    return float((rho + 1.0) / 2.0)


def _v_shape_adherence(times: np.ndarray, smooth: np.ndarray) -> dict[str, float]:
    """Midpoint split: first half should fall in centroid, second half rise."""
    if smooth.size < 8 or times.size != smooth.size:
        return {
            "directional_adherence": 0.0,
            "v_first_half_fall_adherence": 0.0,
            "v_second_half_rise_adherence": 0.0,
        }
    mid_t = 0.5 * (float(times[0]) + float(times[-1]))
    split = int(np.searchsorted(times, mid_t, side="right"))
    split = max(4, min(split, smooth.size - 4))
    y1, y2 = smooth[:split], smooth[split:]
    x1 = np.linspace(0.0, 1.0, num=y1.size)
    x2 = np.linspace(0.0, 1.0, num=y2.size)
    ru1, rf1, _ = _spearman_rise_fall(y1)
    ru2, rf2, _ = _spearman_rise_fall(y2)
    rho_fall_1, _ = stats.spearmanr(x1, -y1)
    rho_rise_2, _ = stats.spearmanr(x2, y2)
    if not np.isfinite(rho_fall_1):
        rho_fall_1 = 0.0
    if not np.isfinite(rho_rise_2):
        rho_rise_2 = 0.0
    a1 = float((float(rho_fall_1) + 1.0) / 2.0)
    a2 = float((float(rho_rise_2) + 1.0) / 2.0)
    joint = min(a1, a2)
    return {
        "directional_adherence": joint,
        "v_first_half_fall_adherence": a1,
        "v_second_half_rise_adherence": a2,
        "spearman_vs_rise_first_half": float(ru1),
        "spearman_vs_fall_first_half": float(rf1),
        "spearman_vs_rise_second_half": float(ru2),
        "spearman_vs_fall_second_half": float(rf2),
    }


def spectral_post_score_flags(scores: dict[str, Any]) -> None:
    """Attach failure heuristics after `score()` (mirrors energy pipeline in run.py)."""
    direction = str(scores.get("direction", "rise"))
    spectral_append_failure_heuristics(scores, direction)


def spectral_append_failure_heuristics(scores: dict[str, Any], direction: str) -> None:
    if scores.get("reason") == "too_few_frames":
        return
    mono = float(scores.get("monotonic_strength", 0.0))
    span = float(scores.get("spectral_span_hz", 0.0))
    da = float(scores.get("directional_adherence", 0.0))
    sr = float(scores.get("spearman_vs_rise", 0.0))
    sf = float(scores.get("spearman_vs_fall", 0.0))
    scores["flag_flat_energy_trajectory"] = mono < 0.12 and span < 120.0
    scores["flag_weak_prompt_alignment"] = da < 0.45
    if direction == "fall":
        scores["flag_likely_wrong_energy_direction"] = sf < 0.05
    elif direction == "rise":
        scores["flag_likely_wrong_energy_direction"] = sr < 0.05
    else:
        v1 = float(scores.get("v_first_half_fall_adherence", 0.0))
        v2 = float(scores.get("v_second_half_rise_adherence", 0.0))
        scores["flag_likely_wrong_energy_direction"] = v1 < 0.45 or v2 < 0.45


class SpectralTrajectoryTask(Task):
    spec = TaskSpec(
        id="spectral",
        name="Spectral trajectory",
        description="Spectral centroid should rise, fall, or form a V per prompt.",
    )

    def generate_prompts(self) -> list[str]:
        return [
            (
                "Begin with a very bright, high-frequency timbre and gradually transition to a dark, "
                "low-frequency sound over 12 seconds. Keep the change smooth and continuous."
            ),
            (
                "Start with a dull, low-frequency tone and steadily increase brightness into a sharp, "
                "high-frequency texture over 10 seconds, without abrupt changes."
            ),
            (
                "Create a 14-second piece that begins bright, becomes progressively darker until the "
                "midpoint, then returns to a bright timbre by the end."
            ),
        ]

    def system_prompt_addon(self) -> str:
        return (
            "Match the spectral / timbre trajectory in the user message (brightness vs time). "
            "Use filters, oscillators, or noise shaping so centroid motion is audible and smooth."
        )

    def system_prompt_addon_sal(self) -> str:
        return (
            "Match the spectral trajectory in the user message (brightness vs time). Prefer **`hzosc` "
            "layers + `pwl` mix weights** (see few-shot). For airy texture use **`noise(dur)`** or "
            "`noise()` — never bare **`noise`** (unbound variable). Filters like `hp`/`lp` take a "
            "**sound** as the first argument, e.g. `hp(noise(dur), 5000.0)`. "
            "Automated scoring uses **spectral centroid of the entire mix**; a loud **broadband** "
            "`noise` layer can move centroid independently of your carrier `pwl`, so trajectory scores "
            "may no longer match what you hear—keep additive noise **quiet**, **band-limited**, or omit "
            "if you want centroid to track the main brightness arc. "
            "`return` the sound from `define function` … `end`, then `play your-entry()` on the last line."
        )

    def extract_features(self, audio_path: Path, sr: int | None = None) -> dict[str, Any]:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        hop = 512
        if len(y) < hop * 2:
            cent = np.array([], dtype=float)
            times = np.array([], dtype=float)
        else:
            cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
            times = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop)
        centroid_meta = _spectral_centroid_track(y, sr, hop_length=hop)
        self_sim = _chroma_self_similarity_scalars(y, sr, hop_length=hop)
        smooth = _smooth_centroid_series(cent) if cent.size else cent.astype(float)
        span = float(np.max(smooth) - np.min(smooth)) if smooth.size else 0.0
        base: dict[str, Any] = {
            "sr": sr,
            "times": times,
            "centroid_hz": cent,
            "centroid_smooth": smooth,
            "hop_length": hop,
            "duration_sec": float(len(y) / sr),
            "spectral_span_hz": span,
            **centroid_meta,
            **self_sim,
        }
        return base

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        smooth = np.asarray(features.get("centroid_smooth", features.get("centroid_hz", [])))
        times = np.asarray(features.get("times", []))
        if smooth.size < 4 or times.size != smooth.size:
            return {
                "adherence_primary": 0.0,
                "directional_adherence": 0.0,
                "direction": "rise",
                "trajectory_kind": "rise",
                "reason": "too_few_frames",
                "spearman_vs_rise": None,
                "spearman_vs_fall": None,
                "monotonic_strength": 0.0,
                "spectral_span_hz": float(features.get("spectral_span_hz", 0.0)),
                "flag_flat_energy_trajectory": True,
                "flag_weak_prompt_alignment": True,
                "flag_likely_wrong_energy_direction": True,
            }

        prompt_index = int(features.get("prompt_index", 0))
        if prompt_index == 2:
            v = _v_shape_adherence(times, smooth)
            rho_r, rho_f, mono = _spearman_rise_fall(smooth)
            out = {
                "trajectory_kind": "v_shape",
                "direction": "v_shape",
                "spearman_vs_rise": rho_r,
                "spearman_vs_fall": rho_f,
                "monotonic_strength": mono,
                "spectral_span_hz": float(np.max(smooth) - np.min(smooth)),
                **v,
            }
            out["adherence_primary"] = float(v["directional_adherence"])
            spectral_post_score_flags(out)
            return out

        direction = "fall" if prompt_index == 0 else "rise"
        rho_r, rho_f, mono = _spearman_rise_fall(smooth)
        da = _directional_from_rhos(rho_r, rho_f, direction)
        out = {
            "trajectory_kind": direction,
            "direction": direction,
            "spearman_vs_rise": rho_r,
            "spearman_vs_fall": rho_f,
            "monotonic_strength": mono,
            "adherence_primary": da,
            "directional_adherence": da,
            "spectral_span_hz": float(np.max(smooth) - np.min(smooth)),
        }
        spectral_post_score_flags(out)
        return out

    def plot(self, features: dict[str, Any], out_path: Path, title: str | None = None) -> None:
        times = np.asarray(features.get("times", []))
        cent = np.asarray(features.get("centroid_hz", []))
        smooth = np.asarray(features.get("centroid_smooth", cent))
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
        if times.size == cent.size and times.size > 1:
            ax.plot(times, cent, alpha=0.35, label="Spectral centroid")
            ax.plot(times, smooth, linewidth=2, color="#C44E52", label="Smoothed centroid")
        else:
            ax.text(0.5, 0.5, "Centroid N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Centroid (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(title or "Spectral trajectory (centroid)")
        ax.legend(loc="best")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()


register_task(SpectralTrajectoryTask())
