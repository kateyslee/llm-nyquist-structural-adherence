"""Event-density task: onset rate in first vs last quarter of the clip (directional trajectory)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import librosa
import matplotlib.pyplot as plt
import numpy as np

from .base import Task, TaskSpec, register_task

_HOP = 512
_MAX_ONSET_PLOT_POINTS = 400
_MERGE_ONSET_SEC = 0.035
_QUARTER = 0.25
# Map (last_quarter_rate - first_quarter_rate) in Hz to [0,1] when sign matches prompt.
_SLACK_HZ = 0.22
_SCALE_HZ = 2.85


def density_prompt_direction(prompt_index: int) -> Literal["rise", "fall", "arc"]:
    """0 = sparse→dense (rise); 1 = dense→sparse (fall); 2 = arc (dense middle vs sparse ends)."""
    if int(prompt_index) == 1:
        return "fall"
    if int(prompt_index) == 2:
        return "arc"
    return "rise"


def _merge_onset_times(times: np.ndarray, min_sep: float) -> np.ndarray:
    if times.size == 0:
        return np.asarray([], dtype=float)
    t = np.sort(np.asarray(times, dtype=float).ravel())
    out: list[float] = [float(t[0])]
    for x in t[1:]:
        if x - out[-1] >= min_sep:
            out.append(float(x))
    return np.asarray(out, dtype=float)


def _quarter_onset_rates(merged: np.ndarray, duration_sec: float) -> tuple[float, float, float]:
    """Events per second in [0, 0.25T) and [0.75T, T]; same-length windows."""
    if duration_sec <= 1e-9:
        return 0.0, 0.0, 0.0
    q = _QUARTER * duration_sec
    t0, t1 = 0.0, q
    t2, t3 = (1.0 - _QUARTER) * duration_sec, duration_sec
    n_first = int(np.sum((merged >= t0) & (merged < t1)))
    n_last = int(np.sum((merged >= t2) & (merged <= t3)))
    d_first = float(n_first / max(q, 1e-9))
    d_last = float(n_last / max(q, 1e-9))
    return d_first, d_last, float(d_last - d_first)


def _middle_half_onset_rate(merged: np.ndarray, duration_sec: float) -> float:
    """Events per second in [0.25T, 0.75T) — middle 50% (for arc prompt)."""
    if duration_sec <= 1e-9:
        return 0.0
    lo = _QUARTER * duration_sec
    hi = (1.0 - _QUARTER) * duration_sec
    win = hi - lo
    n_mid = int(np.sum((merged >= lo) & (merged < hi)))
    return float(n_mid / max(win, 1e-9))


def _directional_adherence_from_delta(
    raw_delta: float, direction: Literal["rise", "fall"]
) -> tuple[float, float]:
    """signed_good > 0 means motion matches prompt. Returns (adherence, signed_good)."""
    signed = raw_delta if direction == "rise" else -raw_delta
    if signed <= 0:
        return 0.0, signed
    adj = max(0.0, signed - _SLACK_HZ)
    adh = float(min(1.0, adj / _SCALE_HZ))
    return adh, signed


_ARC_SLACK_HZ = 0.18
_ARC_SCALE_HZ = 1.6


def _arc_adherence(d_first: float, d_mid: float, d_last: float) -> tuple[float, float]:
    """Middle half denser than average of first+last quarters → positive gain."""
    outer = 0.5 * (d_first + d_last)
    gain = d_mid - outer
    if gain <= 0:
        return 0.0, gain
    adj = max(0.0, gain - _ARC_SLACK_HZ)
    adh = float(min(1.0, adj / _ARC_SCALE_HZ))
    return adh, gain


class DensityTask(Task):
    spec = TaskSpec(
        id="density",
        name="Event density",
        description="Onset rate in first 25% vs last 25% of duration; score matches rise or fall prompt.",
    )

    def generate_prompts(self) -> list[str]:
        return [
            (
                "Create a sound that begins sparse with isolated tones and gradually becomes dense and layered over about 12 seconds."
            ),
            (
                "Create a sound that begins dense and busy, then gradually becomes sparse with only a few isolated tones by the end."
            ),
            (
                "Create a sound with a sparse opening, a denser middle section, and a sparse ending, with the middle clearly more crowded than the outer sections."
            ),
        ]

    def system_prompt_addon(self) -> str:
        return (
            "Automated scoring uses **librosa onset-detect** in time windows: **first 25% vs last 25%** "
            "for build-up or thinning prompts; for the **arc** prompt it compares the **middle 50%** "
            "to the **average of those two outer quarters**. Put clear transient energy in the windows "
            "that matter—smooth sustained chords may yield few detected onsets."
        )

    def system_prompt_addon_sal(self) -> str:
        return (
            "Scoring uses **onset peaks** in the **first 25% vs last 25%** of duration (rise = more in the "
            "last quarter; fall = more in the first). For the **arc** prompt, the **middle 50%** should be "
            "busier than the **outer quarters**—use distinct attacks or grains in the center. **`pwl`** "
            "spikes, **`sim`**, or **`sound @ time`** layers work better than one unchanging drone. "
            "**`noise(dur)`** must be a call, never bare `noise`. `return` the sound from `define function` "
            "… `end`, then `play your-entry()`."
        )

    def extract_features(self, audio_path: Path, sr: int | None = None) -> dict[str, Any]:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        duration_sec = float(len(y) / sr)
        empty_tail = {
            "sr": int(sr),
            "hop_length": _HOP,
            "duration_sec": duration_sec,
            "onset_density_hz": 0.0,
            "onset_density_first_quarter_hz": 0.0,
            "onset_density_last_quarter_hz": 0.0,
            "onset_density_delta_hz": 0.0,
            "onset_density_middle_half_hz": 0.0,
            "onset_contrast_ratio": 0.0,
            "onset_detect_times": [],
            "onset_strength_times": [],
            "onset_strength": [],
        }
        if len(y) < _HOP * 4:
            return empty_tail

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=_HOP)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=_HOP,
            units="frames",
            backtrack=True,
        )
        raw_t = librosa.frames_to_time(onset_frames, sr=sr, hop_length=_HOP)
        merged = _merge_onset_times(raw_t, _MERGE_ONSET_SEC)
        global_hz = float(merged.size / max(duration_sec, 1e-6))
        d_first, d_last, delta = _quarter_onset_rates(merged, duration_sec)
        d_mid = _middle_half_onset_rate(merged, duration_sec)

        mean_o = float(np.mean(onset_env) + 1e-12)
        max_o = float(np.max(onset_env) + 1e-12)
        contrast = max_o / mean_o

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
            "duration_sec": duration_sec,
            "onset_density_hz": global_hz,
            "onset_density_first_quarter_hz": d_first,
            "onset_density_last_quarter_hz": d_last,
            "onset_density_delta_hz": delta,
            "onset_density_middle_half_hz": d_mid,
            "onset_contrast_ratio": contrast,
            "onset_detect_times": merged.astype(float).tolist(),
            "onset_strength_times": times[idx].astype(float).tolist(),
            "onset_strength": np.asarray(onset_env[idx], dtype=float).tolist(),
        }

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        prompt_index = int(features.get("prompt_index", 0))
        direction = density_prompt_direction(prompt_index)
        d0 = float(features.get("onset_density_first_quarter_hz", 0.0))
        d1 = float(features.get("onset_density_last_quarter_hz", 0.0))
        raw_delta = float(features.get("onset_density_delta_hz", d1 - d0))
        d_mid = float(features.get("onset_density_middle_half_hz", 0.0))

        if features.get("duration_sec", 0.0) < 0.5:
            return {
                "direction": direction,
                "onset_density_first_quarter_hz": d0,
                "onset_density_last_quarter_hz": d1,
                "onset_density_middle_half_hz": d_mid,
                "onset_density_delta_hz": raw_delta,
                "density_signed_alignment_hz": 0.0,
                "adherence_primary": 0.0,
                "directional_adherence": 0.0,
                "reason": "too_short",
            }

        if direction == "arc":
            adh, gain = _arc_adherence(d0, d_mid, d1)
            return {
                "direction": direction,
                "onset_density_first_quarter_hz": d0,
                "onset_density_last_quarter_hz": d1,
                "onset_density_middle_half_hz": d_mid,
                "onset_density_delta_hz": raw_delta,
                "density_arc_gain_hz": gain,
                "density_signed_alignment_hz": gain,
                "adherence_primary": adh,
                "directional_adherence": adh,
            }

        adh, signed = _directional_adherence_from_delta(raw_delta, direction)
        return {
            "direction": direction,
            "onset_density_first_quarter_hz": d0,
            "onset_density_last_quarter_hz": d1,
            "onset_density_middle_half_hz": d_mid,
            "onset_density_delta_hz": raw_delta,
            "density_signed_alignment_hz": signed,
            "adherence_primary": adh,
            "directional_adherence": adh,
        }

    def plot(self, features: dict[str, Any], out_path: Path, title: str | None = None) -> None:
        tx = features.get("onset_strength_times") or []
        oy = features.get("onset_strength") or []
        marks = np.asarray(features.get("onset_detect_times") or [], dtype=float)
        dur = float(features.get("duration_sec") or 0.0)
        d0 = features.get("onset_density_first_quarter_hz")
        d1 = features.get("onset_density_last_quarter_hz")
        delta = features.get("onset_density_delta_hz")

        fig, axes = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [2.0, 1]})
        if len(tx) == len(oy) and len(tx) > 1 and dur > 0:
            txa = np.asarray(tx, dtype=float)
            oya = np.asarray(oy, dtype=float)
            axes[0].plot(txa, oya, color="#4C72B0", lw=1.0)
            q = 0.25 * dur
            ymax = float(np.max(oya)) if len(oya) else 1.0
            axes[0].axvspan(0.0, q, color="#55A868", alpha=0.12, label="First 25%")
            axes[0].axvspan(dur - q, dur, color="#C44E52", alpha=0.12, label="Last 25%")
            if marks.size:
                axes[0].vlines(marks, 0.0, ymax, color="#333333", alpha=0.25, linewidth=0.7)
            axes[0].set_ylabel("Onset strength")
            axes[0].set_title(title or "Density (onset envelope + quarters)")
            axes[0].legend(loc="upper right", fontsize=8)
        else:
            axes[0].text(0.5, 0.5, "Onset strength N/A", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_xlabel("Time (s)")

        axes[1].axis("off")
        if d0 is not None and d1 is not None and delta is not None:
            msg = (
                f"Onset rate first 25%: {float(d0):.2f} events/s\n"
                f"Onset rate last 25%:  {float(d1):.2f} events/s\n"
                f"Δ (last − first):     {float(delta):.2f} Hz\n"
                f"(merged onsets ≥ {_MERGE_ONSET_SEC * 1000:.0f} ms apart)"
            )
            axes[1].text(0.02, 0.95, msg, transform=axes[1].transAxes, va="top", fontsize=11, family="monospace")
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()


def density_append_failure_heuristics(scores: dict[str, Any], features: dict[str, Any]) -> None:
    """Reuse CSV flag names; semantics are density-specific."""
    adherence = float(scores.get("adherence_primary", 0.0))
    raw_delta = float(features.get("onset_density_delta_hz", 0.0))
    direction = str(scores.get("direction", "rise"))
    cvr = float(features.get("onset_contrast_ratio", 0.0))
    d0 = float(features.get("onset_density_first_quarter_hz", 0.0))
    d1 = float(features.get("onset_density_last_quarter_hz", 0.0))
    d_mid = float(features.get("onset_density_middle_half_hz", 0.0))

    if direction == "arc":
        gain = float(scores.get("density_arc_gain_hz", d_mid - 0.5 * (d0 + d1)))
        signed = gain
        scores["flag_flat_energy_trajectory"] = (
            max(d0, d1, d_mid) < 0.55 and abs(raw_delta) < 0.35 and cvr < 1.5
        )
    else:
        signed = raw_delta if direction == "rise" else -raw_delta
        scores["flag_flat_energy_trajectory"] = abs(raw_delta) < 0.35 and max(d0, d1) < 1.1 and cvr < 1.4

    scores["flag_weak_prompt_alignment"] = adherence < 0.45
    scores["flag_likely_wrong_energy_direction"] = signed <= 0.0


register_task(DensityTask())
