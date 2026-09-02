#!/usr/bin/env python3
"""Step 2 L1 puncture: verify triton-ascend Python backend discovery."""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    print("=== Step 2 L1: entry_points backend discovery ===")

    # 1. Host triton
    import triton

    print(f"host triton: {getattr(triton, '__version__', '?')} @ {triton.__file__}")

    import triton._C.libtriton as libtriton

    print(f"  extend_with: {hasattr(libtriton.passes.plugin, 'extend_with')}")
    print(f"  libtriton.ascend: {hasattr(libtriton, 'ascend')}")

    # 2. Backend discovery
    from triton.backends import backends

    print(f"discovered backends: {sorted(backends.keys())}")

    if "ascend" not in backends:
        print("FAIL: ascend not in backends (entry_points not registered?)")
        return 1

    print("PASS: ascend backend registered via entry_points")

    # 3. Import ascend backend modules (may fail at L3 without native bindings)
    try:
        compiler = importlib.import_module("triton.backends.ascend.compiler")
        print(f"PASS: imported compiler ({compiler.__name__})")
    except Exception as exc:
        print(f"EXPECTED at L1/L2 boundary: compiler import failed: {exc!r}")
        print("  (needs libtriton.ascend or out-of-tree native .so — Step 3+)")

    try:
        import triton.language.extra.cann  # noqa: F401

        print("PASS: triton.language.extra.cann importable")
    except Exception as exc:
        print(f"SKIP L2b: cann extension not in L1 wheel: {exc!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
