"""Modular task interface: prompt templates, features, scoring, plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sal_prompt_rules import default_sal_rules

_TASKS: dict[str, "Task"] = {}


def register_task(task: "Task") -> None:
    _TASKS[task.spec.id] = task


def get_task(task_id: str) -> "Task":
    if task_id not in _TASKS:
        raise KeyError(f"Unknown task {task_id!r}; known: {sorted(_TASKS)}")
    return _TASKS[task_id]


def list_tasks() -> list[str]:
    return sorted(_TASKS.keys())


@dataclass
class TaskSpec:
    id: str
    name: str
    description: str = ""


class Task(ABC):
    spec: TaskSpec

    @abstractmethod
    def generate_prompts(self) -> list[str]:
        """Return one or more user prompts for this task."""

    @abstractmethod
    def system_prompt_addon(self) -> str:
        """Extra system instructions appended to the global Nyquist authoring rules."""

    def system_prompt_addon_sal(self) -> str:
        """SAL-specific hints (defaults to XLISP addon; override for SAL idioms)."""
        return self.system_prompt_addon()

    @abstractmethod
    def extract_features(self, audio_path: Path, sr: int | None = None) -> dict[str, Any]:
        """Compute task-specific features from rendered audio."""

    @abstractmethod
    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        """Return numeric metrics and human-readable summary keys."""

    def plot(self, features: dict[str, Any], out_path: Path, title: str | None = None) -> None:
        """Optional visualization for a single clip."""
        del features, out_path, title


@dataclass
class GenerationContext:
    """Per-run strings passed into prompt templates if needed."""

    output_wav_abs: str
    duration_hint_sec: float = 15.0


def default_nyquist_rules(ctx: GenerationContext) -> str:
    return f"""You write valid Nyquist (XLISP) code for audio synthesis.
Requirements:
- The code must be self-contained and runnable by the Nyquist interpreter.
- Save the final sound to this exact file path using s-save (use forward slashes):
  (s-save <your-sound-expression> {ctx.output_wav_abs!r} ny:all)
- Do not use play() as the only output; the WAV file is what we evaluate.
- Prefer a total duration around {ctx.duration_hint_sec:.0f} seconds unless the user prompt specifies otherwise.
- Avoid external file dependencies; generate sound with oscillators, noise, envelopes, etc.
Return ONLY the Nyquist source code, no markdown fences or explanation."""
