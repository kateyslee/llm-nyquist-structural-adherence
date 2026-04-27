#!/usr/bin/env python3
"""
Smoke test: run a tiny fixed Nyquist program (no LLM) through the same
subprocess path as the batch pipeline. Use this to verify NYQUIST_BIN /
XLISPPATH / timeouts before running expensive generations.

  python smoke_test_nyquist.py
  python smoke_test_nyquist.py --out /tmp/nyq_smoke.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf

from src.pipeline.nyquist_runner import run_nyquist_code, validate_wav
from src.pipeline.status import RunStatus


def main() -> int:
    p = argparse.ArgumentParser(description="2s sine smoke test for automated Nyquist")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "smoke_test" / "out.wav",
        help="Output WAV path (default: outputs/smoke_test/out.wav)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Subprocess timeout seconds (default: 60; smoke should finish in seconds)",
    )
    args = p.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Same pattern as LLM runs: s-save to an absolute path; 2 s of low-level sine.
    code = (
        "; pipeline smoke test — fixed 2s tone, not model-generated\n"
        f'(s-save (extract 0 2.0 (mult 0.1 (osc c4))) {out.as_posix()!r} ny:all)\n'
    )

    print("Running Nyquist smoke test (2.0 s sine at low amplitude)…")
    print(f"  executable: (from resolve_nyquist_executable / NYQUIST_BIN)")
    print(f"  output:     {out}")

    res = run_nyquist_code(code, out, timeout_sec=args.timeout)

    if res.status != RunStatus.SUCCESS:
        print(f"\nFAIL: status={res.status.value}")
        print(res.stderr_snippet or "(no stderr)")
        return 1

    v = validate_wav(out)
    if v != RunStatus.SUCCESS:
        print(f"\nFAIL: WAV validation {v.value}")
        return 1

    data, sr = sf.read(out, always_2d=False)
    dur = len(data) / float(sr)
    print(f"\nOK: Nyquist automation path works.")
    print(f"  sample_rate={sr} Hz, duration≈{dur:.3f} s, path={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
