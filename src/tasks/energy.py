"""Energy trajectory task: crescendo / decrescendo adherence via RMS trend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d

from .base import Task, TaskSpec, register_task

_MAX_CHROMA_FRAMES_RECURRENCE = 200
_MAX_CENTROID_PLOT_POINTS = 160


def _rms_envelope(y: np.ndarray, sr: int, hop_length: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms


def _spectral_centroid_track(y: np.ndarray, sr: int, hop_length: int) -> dict[str, Any]:
    """Mean/std + downsampled (time, Hz) for JSON and plotting."""
    if len(y) < hop_length * 2:
        return {
            "spectral_centroid_mean_hz": None,
            "spectral_centroid_std_hz": None,
            "spectral_centroid_times": [],
            "spectral_centroid_hz": [],
        }
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    ct = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop_length)
    mean_hz = float(np.mean(cent))
    std_hz = float(np.std(cent))
    n = len(cent)
    if n <= _MAX_CENTROID_PLOT_POINTS:
        idx = np.arange(n)
    else:
        idx = np.unique(
            np.linspace(0, n - 1, num=_MAX_CENTROID_PLOT_POINTS, dtype=float).astype(int)
        )
    return {
        "spectral_centroid_mean_hz": mean_hz,
        "spectral_centroid_std_hz": std_hz,
        "spectral_centroid_times": ct[idx].astype(float).tolist(),
        "spectral_centroid_hz": cent[idx].astype(float).tolist(),
    }


def _chroma_self_similarity_scalars(y: np.ndarray, sr: int, hop_length: int) -> dict[str, Any]:
    """
    Compact chroma recurrence stats (not the full matrix) for structure / repetition cues.
    """
    if len(y) < hop_length * 4:
        return {
            "self_similarity_mean_offdiag": None,
            "self_similarity_std_offdiag": None,
            "self_similarity_max_offdiag": None,
            "chroma_recurrence_frames": 0,
        }
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=2048)
    n_frames = chroma.shape[1]
    if n_frames < 4:
        return {
            "self_similarity_mean_offdiag": None,
            "self_similarity_std_offdiag": None,
            "self_similarity_max_offdiag": None,
            "chroma_recurrence_frames": int(n_frames),
        }
    if n_frames > _MAX_CHROMA_FRAMES_RECURRENCE:
        pick = np.unique(
            np.linspace(0, n_frames - 1, num=_MAX_CHROMA_FRAMES_RECURRENCE, dtype=float).astype(
                int
            )
        )
        chroma = chroma[:, pick]
        n_frames = chroma.shape[1]
    r = librosa.segment.recurrence_matrix(
        chroma, mode="affinity", sym=True, width=3, sparse=False
    )
    r = np.asarray(r, dtype=float)
    np.fill_diagonal(r, 0.0)
    iu = np.triu_indices(r.shape[0], k=1)
    off = r[iu]
    if off.size == 0:
        return {
            "self_similarity_mean_offdiag": 0.0,
            "self_similarity_std_offdiag": 0.0,
            "self_similarity_max_offdiag": 0.0,
            "chroma_recurrence_frames": int(n_frames),
        }
    return {
        "self_similarity_mean_offdiag": float(np.mean(off)),
        "self_similarity_std_offdiag": float(np.std(off)),
        "self_similarity_max_offdiag": float(np.max(off)),
        "chroma_recurrence_frames": int(n_frames),
    }


class EnergyTask(Task):
    spec = TaskSpec(
        id="energy",
        name="Energy trajectory",
        description="RMS should increase (crescendo) or decrease per prompt.",
    )

    def generate_prompts(self) -> list[str]:
        return [
            (
                "Begin very quietly and gradually increase to a loud climax over 10 seconds, "
                "then fade out in the final 5 seconds. Total length about 15 seconds."
            ),
            (
                "Create a 12-second crescendo: start near silence and end at maximum loudness "
                "without clipping, smooth exponential-style growth."
            ),
            (
                "Slow decrescendo: start loud and become very quiet over 14 seconds, "
                "mostly linear in perceived loudness."
            ),
        ]

    def system_prompt_addon(self) -> str:
        return (
            "Match the energy contour described in the user message. "
            "Use amplitude envelopes (e.g. pwl, env) so loudness changes are audible."
        )

    def system_prompt_addon_sal(self) -> str:
        return (
            "Match the energy contour in the user message. In SAL, shape loudness with `pwl` (or "
            "similar envelopes); use `carrier * envelope` or `sim` / `mult` as appropriate. "
            "`return` the sound from `define function` … `end`, then `play your-entry()` on the "
            "last line."
        )

    def extract_features(self, audio_path: Path, sr: int | None = None) -> dict[str, Any]:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        hop = 512
        times, rms = _rms_envelope(y, sr, hop_length=hop)
        centroid = _spectral_centroid_track(y, sr, hop_length=hop)
        self_sim = _chroma_self_similarity_scalars(y, sr, hop_length=hop)
        base: dict[str, Any] = {
            "sr": sr,
            "times": times,
            "rms": rms,
            "hop_length": hop,
            "duration_sec": float(len(y) / sr),
            **centroid,
            **self_sim,
        }
        if len(rms) < 4:
            base["rms_smooth"] = rms
            return base
        win = max(3, len(rms) // 20 | 1)
        if win % 2 == 0:
            win += 1
        smooth = uniform_filter1d(rms.astype(float), size=win, mode="nearest")
        base["rms_smooth"] = smooth
        base["rms_max"] = float(np.max(rms) + 1e-12)
        base["rms_mean"] = float(np.mean(rms) + 1e-12)
        return base

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        smooth = np.asarray(features.get("rms_smooth", features.get("rms", [])))
        if smooth.size < 4:
            return {
                "adherence_primary": 0.0,
                "reason": "too_few_frames",
                "spearman_vs_rise": None,
                "spearman_vs_fall": None,
                "monotonic_strength": 0.0,
                "dynamic_range_db": 0.0,
                "flag_flat_energy_trajectory": True,
                "flag_weak_prompt_alignment": True,
                "flag_likely_wrong_energy_direction": True,
            }
        x = np.linspace(0, 1, num=smooth.size)
        rho_up, _ = stats.spearmanr(x, smooth)
        rho_down, _ = stats.spearmanr(x, -smooth)
        if not np.isfinite(rho_up):
            rho_up = 0.0
        if not np.isfinite(rho_down):
            rho_down = 0.0
        # Symmetric score: how strongly monotonic in either direction (prompt-specific scoring
        # would need prompt tags; we report both and let the batch script pick by prompt index).
        best = max(float(rho_up), float(rho_down))
        dr_db = float(20 * np.log10((np.max(smooth) + 1e-9) / (np.min(smooth) + 1e-9)))
        return {
            "spearman_vs_rise": float(rho_up),
            "spearman_vs_fall": float(rho_down),
            "monotonic_strength": best,
            "adherence_primary": (best + 1) / 2,
            "dynamic_range_db": dr_db,
        }

    def plot(self, features: dict[str, Any], out_path: Path, title: str | None = None) -> None:
        times = np.asarray(features["times"])
        rms = np.asarray(features["rms"])
        smooth = np.asarray(features.get("rms_smooth", rms))
        ct = features.get("spectral_centroid_times") or []
        chz = features.get("spectral_centroid_hz") or []
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        axes[0].plot(times, rms, alpha=0.35, label="RMS")
        axes[0].plot(times, smooth, linewidth=2, label="Smoothed RMS")
        axes[0].set_ylabel("RMS")
        axes[0].legend(loc="best")
        axes[0].set_title(title or "Energy (RMS)")
        if len(ct) == len(chz) and len(ct) > 1:
            axes[1].plot(np.asarray(ct, dtype=float), np.asarray(chz, dtype=float), color="#C44E52", lw=1.5)
            axes[1].set_ylabel("Spectral centroid (Hz)")
        else:
            axes[1].text(0.5, 0.5, "Spectral centroid N/A", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_ylabel("Spectral centroid (Hz)")
        axes[1].set_xlabel("Time (s)")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()


def energy_prompt_direction(prompt_index: int) -> str:
    """Heuristic: which Spearman to treat as target for bundled prompts."""
    if prompt_index == 2:
        return "fall"
    return "rise"


def energy_directional_score(score_row: dict[str, Any], direction: str) -> float:
    """Map Spearman to [0,1] for the intended direction."""
    if direction == "fall":
        rho = score_row.get("spearman_vs_fall")
    else:
        rho = score_row.get("spearman_vs_rise")
    if rho is None:
        return 0.0
    return float((rho + 1) / 2)


def energy_append_failure_heuristics(scores: dict[str, Any], direction: str) -> None:
    """
    Derived flags for interim reporting (not mutually exclusive).
    Uses thresholds tuned for normalized RMS contours; adjust if needed.
    """
    if scores.get("reason") == "too_few_frames":
        return
    mono = float(scores.get("monotonic_strength", 0.0))
    dr = float(scores.get("dynamic_range_db", 0.0))
    da = float(scores.get("directional_adherence", 0.0))
    sr = float(scores.get("spearman_vs_rise", 0.0))
    sf = float(scores.get("spearman_vs_fall", 0.0))
    scores["flag_flat_energy_trajectory"] = mono < 0.12 and dr < 8.0
    scores["flag_weak_prompt_alignment"] = da < 0.45
    if direction == "fall":
        scores["flag_likely_wrong_energy_direction"] = sf < 0.05
    else:
        scores["flag_likely_wrong_energy_direction"] = sr < 0.05


register_task(EnergyTask())
