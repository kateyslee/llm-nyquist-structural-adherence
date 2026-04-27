"""Tight SAL rules + task-specific SAL few-shot for LLM system prompts.

Default: `references/sal_energy_example.sal`. **Spectral** uses `references/sal_spectral_example.sal`;
**tempo** uses `references/sal_tempo_example.sal` (rhythmic pulses / BPM period).
**Density** uses `references/sal_density_example.sal` (more attacks toward the end of the clip).

Imported from `base` as `default_sal_rules` so all backends use one prompt path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

_ROOT = Path(__file__).resolve().parents[2]
_SAL_FEW_SHOT_ENERGY = _ROOT / "references" / "sal_energy_example.sal"
_SAL_FEW_SHOT_SPECTRAL = _ROOT / "references" / "sal_spectral_example.sal"
_SAL_FEW_SHOT_TEMPO = _ROOT / "references" / "sal_tempo_example.sal"
_SAL_FEW_SHOT_DENSITY = _ROOT / "references" / "sal_density_example.sal"


class _SalGenerationContext(Protocol):
    output_wav_abs: str
    duration_hint_sec: float


def default_sal_rules(ctx: _SalGenerationContext, task_id: str = "energy") -> str:
    if task_id == "spectral":
        path = _SAL_FEW_SHOT_SPECTRAL
    elif task_id == "tempo":
        path = _SAL_FEW_SHOT_TEMPO
    elif task_id == "density":
        path = _SAL_FEW_SHOT_DENSITY
    else:
        path = _SAL_FEW_SHOT_ENERGY
    try:
        example = path.read_text(encoding="utf-8")
    except OSError:
        example = (
            f"; (missing {path.name} — use set/begin with, play name() at end)\n"
        )

    core = f"""You output a single Nyquist SAL program for Nyquist IDE (not Lisp .ny).

Hard constraints (violations break the run):
- Inside `begin` … `end`, bind variables ONLY with `set name = expr` or a `begin with name = expr` … `end` block. NEVER use `local`, `let`, or a bare `name = expr` line.
- Top-level sound generators use **`define function name(args)`** … `end` (course style) or `define name` … `end` if your SAL accepts it. The body must `return` a sound expression. NEVER `return play(...)`. Callback-style helpers may use **`function name(...)`** … `end` when the API requires it.
- The LAST non-comment line of the file MUST be **`play your-entry()`** (e.g. `play main()`). Do not end with a bare `my-name()` without `play`.
- Use only real SAL/Nyquist constructs you are sure exist: e.g. `pwl`, `hzosc`, **`*`** for carrier × envelope, **`sim`** for sum/mix, `sampler` / `s-read` / `load` only when the prompt needs samples or libs. Do not invent primitives.
- **White noise is a function, not a variable:** write **`noise()`** or **`noise(dur)`** (seconds). Never write bare `noise` or `noise ~ dur` — that raises **unbound variable NOISE**. Same idea for other generators: always call them with `()` when they are functions.

Dialect reminders:
- Piecewise linear envelopes: `pwl(t0, v0, t1, v1, …)` with time/value pairs.
- Duration: honor the requested duration hint in seconds in your `pwl` and logic.

Complete example (structure and syntax to emulate; adapt frequencies/envelopes to the prompt):

```sal
{example}
```
"""

    return f"""{core}

Hard requirements (this run):
- Target total duration ~ {ctx.duration_hint_sec:.0f} seconds unless the user prompt says otherwise.
- Include this exact comment near the top: `; MANUAL_WAV_PATH: {ctx.output_wav_abs}`

Return ONLY the SAL source, no markdown fences or explanation."""


def system_prompt_addon_sal() -> str:
    """Optional one-line addon if tasks already inject duration; keeps repetition minimal."""
    return (
        "SAL: only `set`/`begin with` for bindings inside begin; never local/let/bare `=`; "
        "`return` a sound, not `return play(...)`; final line must be `play your-fn()`."
    )
