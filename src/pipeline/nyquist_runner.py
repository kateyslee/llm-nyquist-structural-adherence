"""Execute Nyquist in a subprocess and classify outcomes."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import nyquist_subprocess_env, nyquist_timeout_sec, resolve_nyquist_executable
from .status import RunStatus


@dataclass
class NyquistResult:
    status: RunStatus
    wav_path: Path | None
    stderr_snippet: str
    returncode: int | None


def _truncate(s: str, n: int = 4000) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "\n... (truncated)"


def classify_stderr(stderr: str, returncode: int) -> RunStatus:
    low = stderr.lower()
    if returncode != 0:
        if "syntax error" in low or ("unexpected" in low and "error" in low):
            return RunStatus.SYNTAX_ERROR
        return RunStatus.RUNTIME_ERROR
    if "error" in low and returncode == 0:
        # Some builds still print to stderr without failing
        if "syntax" in low:
            return RunStatus.SYNTAX_ERROR
    return RunStatus.SUCCESS


def validate_wav(path: Path) -> RunStatus:
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception:
        return RunStatus.RENDER_FAILURE
    if data is None or np.size(data) == 0:
        return RunStatus.INVALID_AUDIO
    y = np.asarray(data, dtype=float).ravel()
    dur = float(len(y) / float(sr))
    if dur < 0.2:
        return RunStatus.INVALID_AUDIO
    peak = float(np.max(np.abs(y)))
    if peak < 1e-5:
        return RunStatus.SILENT_AUDIO
    return RunStatus.SUCCESS


def run_nyquist_code(code: str, out_wav: Path, timeout_sec: int | None = None) -> NyquistResult:
    if timeout_sec is None:
        timeout_sec = nyquist_timeout_sec()

    out_wav = out_wav.resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists():
        out_wav.unlink()

    ny = resolve_nyquist_executable()
    if not ny:
        return NyquistResult(
            status=RunStatus.NYQUIST_NOT_FOUND,
            wav_path=None,
            stderr_snippet=(
                "Nyquist executable not found. On macOS (CMU installer): set\n"
                "  NYQUIST_BIN=\"$HOME/Applications/NyquistIDE.app/Contents/Java/ny\"\n"
                "or /Applications/... if you put NyquistIDE there. "
                "Keep the `nyquist` folder in your home directory for runtime/lib; "
                "optional: NYQUIST_HOME if it lives elsewhere."
            ),
            returncode=None,
        )

    with tempfile.TemporaryDirectory(prefix="nyq_run_") as tmp:
        tmp_path = Path(tmp)
        user_path = tmp_path / "user_gen.ny"
        driver_path = tmp_path / "driver.ny"
        user_path.write_text(code, encoding="utf-8")
        # Load user code, then exit — otherwise many `ny` builds sit in the REPL (timeouts).
        driver_path.write_text(
            f'(load "{user_path.as_posix()}")\n(exit)\n',
            encoding="utf-8",
        )

        try:
            proc = subprocess.run(
                [ny, str(driver_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(tmp_path),
                env=nyquist_subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            proc = getattr(exc, "process", None)
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            return NyquistResult(
                status=RunStatus.NYQUIST_TIMEOUT,
                wav_path=None,
                stderr_snippet=_truncate(
                    f"Nyquist subprocess exceeded {timeout_sec}s (no WAV produced). "
                    f"Rendering can be slow; set NYQUIST_TIMEOUT_SEC higher, or simplify generated code. "
                    f"Partial stderr: {getattr(exc, 'stderr', None) or ''}"
                ),
                returncode=None,
            )

        stderr = proc.stderr or ""
        stdout = proc.stdout or ""
        combined = stderr + "\n" + stdout

        if proc.returncode != 0:
            st = classify_stderr(combined, proc.returncode)
            return NyquistResult(status=st, wav_path=None, stderr_snippet=_truncate(combined), returncode=proc.returncode)

        if not out_wav.is_file():
            return NyquistResult(
                status=RunStatus.RENDER_FAILURE,
                wav_path=None,
                stderr_snippet=_truncate(
                    combined or "Nyquist exited 0 but output WAV missing; check s-save path in code."
                ),
                returncode=proc.returncode,
            )

        audio_status = validate_wav(out_wav)
        if audio_status != RunStatus.SUCCESS:
            return NyquistResult(
                status=audio_status,
                wav_path=out_wav,
                stderr_snippet=_truncate(combined),
                returncode=proc.returncode,
            )

        return NyquistResult(
            status=RunStatus.SUCCESS,
            wav_path=out_wav,
            stderr_snippet=_truncate(combined),
            returncode=proc.returncode,
        )
