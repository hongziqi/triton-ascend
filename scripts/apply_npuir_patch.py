#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Apply AscendNPU-IR LLVM-23 adaptations (idempotent per clean checkout).

Keeps the AscendNPU-IR submodule tree clean at HEAD and re-applies
``patch/npuir_adapter_to_llvm_23.patch`` before every plugin build.
Called from ``setup.py`` and standalone CMake.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATCH = _PKG_ROOT / "patch" / "npuir_adapter_to_llvm_23.patch"
_DEFAULT_NPUIR = _PKG_ROOT / "AscendNPU-IR"


def _get_patch_files(patch_path: Path) -> list[str]:
    files: list[str] = []
    with open(patch_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("diff --git a/"):
                target = line.split(" b/", 1)[-1].rstrip("\n")
                if target != "/dev/null":
                    files.append(target)
    return files


def _normalize_crlf(files: list[str], *, cwd: Path) -> None:
    """CRLF working trees reject LF patches; CMake/tblgen accept LF."""
    for rel in files:
        path = cwd / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\r\n" in data:
            path.write_bytes(data.replace(b"\r\n", b"\n"))


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def apply_npuir_patch(
    *,
    npuir_dir: Path = _DEFAULT_NPUIR,
    patch_path: Path = _DEFAULT_PATCH,
) -> None:
    if not patch_path.is_file():
        raise RuntimeError(f"patch not found: {patch_path}")
    if not npuir_dir.is_dir():
        raise RuntimeError(
            f"AscendNPU-IR not found at {npuir_dir}. "
            "Init the submodule: git submodule update --init AscendNPU-IR"
        )
    if not (npuir_dir / ".git").exists():
        raise RuntimeError(
            f"{npuir_dir} is not a git checkout; AscendNPU-IR must be a submodule "
            "so the adapter patch can be applied cleanly at build time."
        )

    patch_files = _get_patch_files(patch_path)
    if not patch_files:
        raise RuntimeError(f"patch has no file sections: {patch_path}")

    tracked = subprocess.run(
        ["git", "ls-files", "--"] + patch_files,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(npuir_dir),
        text=True,
    ).stdout.splitlines()
    if tracked:
        _run(["git", "checkout", "-f", "--"] + tracked, cwd=npuir_dir)

    patch_abs = str(patch_path.resolve())
    # Already applied? (reverse --check succeeds)
    already = subprocess.run(
        ["git", "apply", "--reverse", "--check", patch_abs],
        cwd=str(npuir_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if already.returncode == 0:
        print(f"[triton-ascend] {patch_path.name} already applied → {npuir_dir}")
        return

    apply = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", patch_abs],
        cwd=str(npuir_dir),
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        # Some checkouts are CRLF and reject the LF patch until normalized;
        # others are the opposite. Retry once after forcing LF.
        if tracked:
            _normalize_crlf(tracked, cwd=npuir_dir)
        apply = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", patch_abs],
            cwd=str(npuir_dir),
            capture_output=True,
            text=True,
        )
    if apply.returncode != 0:
        raise RuntimeError(
            f"git apply failed for {patch_path.name}: {apply.stderr or apply.stdout}"
        )
    print(f"[triton-ascend] applied {patch_path.name} → {npuir_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npuir-dir", type=Path, default=_DEFAULT_NPUIR)
    p.add_argument("--patch", type=Path, default=_DEFAULT_PATCH)
    args = p.parse_args(argv)
    try:
        apply_npuir_patch(npuir_dir=args.npuir_dir, patch_path=args.patch)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"[triton-ascend] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
