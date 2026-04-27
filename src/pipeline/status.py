"""Run outcome taxonomy.

Individual `RunStatus` values mix toolchain and quality; **well-formedness vs semantic**
layers are derived in `evaluation_axes.build_evaluation_axes` for reporting.
"""

from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    GENERATION_FAILED = "generation_failed"
    LLM_QUOTA_EXCEEDED = "llm_quota_exceeded"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    NYQUIST_NOT_FOUND = "nyquist_not_found"
    NYQUIST_TIMEOUT = "nyquist_timeout"
    RENDER_FAILURE = "render_failure"
    SILENT_AUDIO = "silent_audio"
    INVALID_AUDIO = "invalid_audio"
    PENDING_MANUAL_RENDER = "pending_manual_render"
    DURATION_SHORTFALL = "duration_shortfall"
    SUCCESS = "success"
