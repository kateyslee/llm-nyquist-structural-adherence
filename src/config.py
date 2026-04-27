"""Environment-driven configuration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = Path(os.environ.get("PIPELINE_OUTPUTS", PROJECT_ROOT / "outputs"))


def gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def gemini_model() -> str:
    # Default avoids gemini-2.0-flash when some API projects show free-tier limit: 0 for that model.
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def openai_compatible_config() -> tuple[str | None, str | None, str | None]:
    """Returns (base_url, api_key, model) for OpenAI-compatible APIs (Ollama, vLLM, etc.)."""
    base = os.environ.get("OPENAI_API_BASE")
    key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    return base, key, model


def ollama_openai_defaults() -> tuple[str, str, str]:
    """
    Preset for local Ollama (free) with a Qwen coder model via OpenAI-compatible API.

    Install: https://ollama.com — then e.g. `ollama pull qwen2.5-coder:7b`
    Override with OLLAMA_HOST (base only), OLLAMA_MODEL, OLLAMA_API_KEY (dummy ok).
    """
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    base = f"{host}/v1"
    key = os.environ.get("OLLAMA_API_KEY", "ollama")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    return base, key, model


def nyquist_bin() -> str:
    """Raw preference: full path or executable name searched on PATH."""
    return os.environ.get("NYQUIST_BIN", "ny")


def nyquist_home() -> Path | None:
    """Directory containing `runtime/` and `lib/` (CMU install: ~/nyquist)."""
    explicit = os.environ.get("NYQUIST_HOME", "").strip()
    if explicit:
        p = Path(os.path.expanduser(explicit))
        return p if p.is_dir() else None
    default = Path.home() / "nyquist"
    return default if default.is_dir() else None


def resolve_nyquist_executable() -> str | None:
    """
    Executable for batch Nyquist runs.

    Order: NYQUIST_BIN if it exists, then `ny` on PATH, then common macOS CMU bundle paths
    (see nyquist/doc/readme-mac.txt: NyquistIDE.app/Contents/Java/ny).
    """
    raw = os.environ.get("NYQUIST_BIN", "").strip()
    if raw:
        p = Path(os.path.expanduser(raw))
        if p.is_file():
            return str(p.resolve())
    found = shutil.which("ny")
    if found:
        return found
    home = Path.home()
    for candidate in (
        home / "Applications/NyquistIDE.app/Contents/Java/ny",
        Path("/Applications/NyquistIDE.app/Contents/Java/ny"),
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def nyquist_timeout_sec() -> int:
    """Subprocess limit for batch `ny` runs (s-save can be slow on some builds)."""
    raw = os.environ.get("NYQUIST_TIMEOUT_SEC", "").strip()
    if raw.isdigit():
        return max(10, int(raw))
    return 300


def nyquist_subprocess_env() -> dict[str, str]:
    """Child process env; sets XLISPPATH for CMU CLI per readme-mac.txt."""
    env = os.environ.copy()
    base = nyquist_home()
    if base is not None:
        runtime = base / "runtime"
        lib = base / "lib"
        if runtime.is_dir() and lib.is_dir():
            env["XLISPPATH"] = f"{runtime.resolve().as_posix()}:{lib.resolve().as_posix()}"
    return env
