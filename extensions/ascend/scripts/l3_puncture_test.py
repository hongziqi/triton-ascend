#!/usr/bin/env python3
"""Step 3 L3 puncture: load LoopSplit probe via extend_with."""

from __future__ import annotations

import sys


def main() -> int:
    print("=== Step 3 L3: extend_with + LoopSplit probe ===")
    import triton._C.libtriton as libtriton

    if not hasattr(libtriton.passes.plugin, "extend_with"):
        print("FAIL: host triton has no passes.plugin.extend_with")
        print("  Run: bash extensions/ascend/scripts/build_triton_ext_host.sh")
        return 1

    print("PASS: extend_with available")

    try:
        import triton_loop_split_probe  # noqa: F401
    except Exception as exc:
        print(f"FAIL: could not import/build probe wheel: {exc!r}")
        print("  Build probe: pip wheel extensions/ascend/puncture/loop_split_probe")
        return 1

    if not hasattr(libtriton.passes.plugin, "add_loop_split_probe"):
        # extend_with wraps pass name as add_<pass>
        candidates = [n for n in dir(libtriton.passes.plugin) if "loop" in n.lower()]
        print(f"plugin pass fns: {candidates[:10]}")
        print("FAIL: add_loop_split_probe not registered")
        return 1

    print("PASS: LoopSplit probe pass registered via extend_with")
    return 0


if __name__ == "__main__":
    sys.exit(main())
