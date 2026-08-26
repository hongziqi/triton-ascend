#!/usr/bin/env python3
"""Minimal Ascend out-of-tree build entry (plugin puncture path).

Compared to setup_ascend.py, this only adds what community setup.py cannot do:

  1. Apply triton-ascend-3.7.0.patch (+ optional dev patch)
  2. Apply npuir_adapter_to_llvm_23.patch
  3. Pass -DASCENDNPU_IR_TAG / -DLLVM_MAJOR_VERSION_23_COMPATIBLE via cmake
  4. Export TRITON_PLUGIN_DIRS=<repo>/ascend so setup.py registers the backend

Everything else (backends list, cmake, packaging, nvidia/amd, …) is community
setup.py. Prefer this for pluginization puncture; keep setup_ascend.py for
release/wheel/coverage/distributed paths.

Usage (plugin env)::

    conda activate plugin
    export LLVM_SYSPATH=${LLVM_INSTALL_PREFIX_3_7_PATCH}   # required: Ascend LLVM
    DEBUG=1 TRITON_BUILD_PROTON=OFF \\
      python3 setup_plugin.py install
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ASCEND_DIR = _THIS_DIR / "ascend"
_TRITON_SETUP = _THIS_DIR / "setup.py"


def _is_dev_mode() -> bool:
    if os.getenv("IS_MANYLINUX", "FALSE").upper() not in ["ON", "1", "YES", "TRUE", "Y"]:
        return True
    if os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""):
        return True
    version_file = _THIS_DIR / "version.txt"
    if version_file.exists() and "dev" in version_file.read_text():
        return True
    return False


def _run_git(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd or _THIS_DIR),
    )


def _apply_patch(patch_path: Path, *, directory: str | None = None) -> None:
    """Apply patch from repo root. ``directory`` must be repo-relative (not absolute).

    Absolute ``--directory`` makes git invent invalid paths under the submodule
    (seen as: error: invalid path '/.../ascend/AscendNPU-IR/CMakeLists.txt').
    """
    if not patch_path.is_file():
        raise RuntimeError(f"patch({patch_path}) not found.")

    rel_patch = os.path.relpath(patch_path, _THIS_DIR)
    cmd = ["apply"]
    if directory:
        # Always repo-relative, matching setup_ascend.py.
        rel_dir = directory if not os.path.isabs(directory) else os.path.relpath(directory, _THIS_DIR)
        cmd.extend(["--directory", rel_dir])
    cmd.append(rel_patch)

    # Already applied → skip (common on re-install without resetting the tree).
    reverse = _run_git([*cmd[:1], "-R", "--check", *cmd[1:]], check=False)
    if reverse.returncode == 0:
        print(f"[setup_plugin] skip already-applied patch: {rel_patch}"
              + (f" (dir={cmd[cmd.index('--directory')+1]})" if "--directory" in cmd else ""))
        return

    try:
        _run_git(cmd, check=True)
        print(f"[setup_plugin] applied patch: {rel_patch}"
              + (f" (dir={cmd[cmd.index('--directory')+1]})" if "--directory" in cmd else ""))
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"patch({rel_patch}) failed\n{detail}") from None


def _checkout_file(files: list[str], *, cwd: Path | None = None) -> None:
    try:
        _run_git(["checkout", "--", *files], cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"init code failed, list:{files}\n{detail}") from None


def _triton_patch_files() -> tuple[list[str], list[str]]:
    patch_files = [
        "CMakeLists.txt",
        "include/triton/Dialect/Triton/IR/TritonAttrDefs.td",
        "lib/Dialect/Triton/IR/Traits.cpp",
        "python/src/ir.cc",
        "python/triton/_utils.py",
        "python/triton/compiler/code_generator.py",
        "python/triton/compiler/compiler.py",
        "python/triton/compiler/errors.py",
        "python/triton/language/math.py",
        "python/triton/language/semantic.py",
        "python/triton/language/standard.py",
        "python/triton/runtime/interpreter.py",
        "python/triton/runtime/jit.py",
        "bin/RegisterTritonDialects.h",
        "bin/triton-opt.cpp",
        "bin/CMakeLists.txt",
    ]
    dev_patch_files = ["python/triton/runtime/autotuner.py"]
    return patch_files, dev_patch_files


def _npuir_patch_files() -> list[str]:
    return [
        "CMakeLists.txt",
        "bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td",
        "bishengir/include/bishengir/Dialect/HIVM/IR/CMakeLists.txt",
        "bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td",
        "bishengir/include/bishengir/Dialect/HFusion/IR/CMakeLists.txt",
        "bishengir/include/bishengir/Dialect/Scope/IR/ScopeOps.td",
        "bishengir/include/bishengir/Dialect/Scope/IR/CMakeLists.txt",
        "bishengir/lib/Dialect/Scope/IR/ScopeOps.cpp",
        "bishengir/triton/lib/Dialect/TritonGPU/IR/Ops.cpp",
        "bishengir/lib/Dialect/HIVM/IR/HIVMImpl.cpp",
        "bishengir/lib/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.cpp",
    ]


def apply_ascend_patches() -> None:
    """(1)(2) Triton Ascend core patch + NPU-IR LLVM23 adapter.

    Set SKIP_TA_PATCH=1 to leave Triton core untouched (pure TRITON_PLUGIN_DIRS
    experiment). Ascend NPUIR LLVM23 adapter can still be applied unless
    SKIP_NPUIR_PATCH=1.
    """
    skip_ta = os.getenv("SKIP_TA_PATCH", "").upper() in ("1", "ON", "TRUE", "YES", "Y")
    skip_npuir = os.getenv("SKIP_NPUIR_PATCH", "").upper() in ("1", "ON", "TRUE", "YES", "Y")

    patch_dir = _ASCEND_DIR / "patch"
    triton_patch = patch_dir / "triton-ascend-3.7.0.patch"
    dev_patch = patch_dir / "triton-ascend-dev-3.7.0.patch"
    npuir_patch = patch_dir / "npuir_adapter_to_llvm_23.patch"
    npuir_dir_rel = "ascend/AscendNPU-IR"
    npuir_dir = _THIS_DIR / npuir_dir_rel

    if skip_ta:
        print("[setup_plugin] SKIP_TA_PATCH=1 — not applying triton-ascend-*.patch")
    else:
        patch_files, dev_patch_files = _triton_patch_files()
        if _is_dev_mode() and dev_patch.is_file():
            _checkout_file(dev_patch_files)
            _apply_patch(dev_patch)
        _checkout_file(patch_files)
        _apply_patch(triton_patch)

    if skip_npuir:
        print("[setup_plugin] SKIP_NPUIR_PATCH=1 — not applying npuir_adapter_to_llvm_23.patch")
        return

    if not npuir_dir.is_dir():
        raise RuntimeError(f"AscendNPU-IR not found at {npuir_dir}")
    _checkout_file(_npuir_patch_files(), cwd=npuir_dir)
    _apply_patch(npuir_patch, directory=npuir_dir_rel)


def configure_plugin_env() -> None:
    """Wire Layer A/B plugin dir + Ascend-only cmake flags into community setup.py."""
    if not (_ASCEND_DIR / "backend" / "name.conf").is_file():
        raise RuntimeError(f"Ascend plugin root missing name.conf under {_ASCEND_DIR}")

    # Community setup.py: BackendInstaller.copy_externals() reads this.
    existing = os.environ.get("TRITON_PLUGIN_DIRS", "").strip()
    ascend = str(_ASCEND_DIR.resolve())
    if existing:
        dirs = [d for d in existing.split(";") if d]
        if ascend not in dirs:
            dirs.append(ascend)
        os.environ["TRITON_PLUGIN_DIRS"] = ";".join(dirs)
    else:
        os.environ["TRITON_PLUGIN_DIRS"] = ascend

    # (3) NPU-IR tag (optional) + LLVM23 compat (required for Ascend 3.7 / LLVM 23).
    extras = ["-DLLVM_MAJOR_VERSION_23_COMPATIBLE=ON"]
    tag = os.getenv("ASCENDNPU_IR_TAG")
    if tag:
        extras.append(f"-DASCENDNPU_IR_TAG={tag}")

    append = os.environ.get("TRITON_APPEND_CMAKE_ARGS", "").strip()
    for flag in extras:
        if flag not in append:
            append = f"{append} {flag}".strip()
    os.environ["TRITON_APPEND_CMAKE_ARGS"] = append

    # Sensible defaults for Ascend plugin puncture; override via env if needed.
    os.environ.setdefault("TRITON_BUILD_PROTON", "OFF")
    os.environ.setdefault("TRITON_BUILD_TD", "OFF")

    print(f"[setup_plugin] TRITON_PLUGIN_DIRS={os.environ['TRITON_PLUGIN_DIRS']}")
    print(f"[setup_plugin] TRITON_APPEND_CMAKE_ARGS={os.environ['TRITON_APPEND_CMAKE_ARGS']}")
    if not os.environ.get("LLVM_SYSPATH"):
        print(
            "[setup_plugin] WARNING: LLVM_SYSPATH unset; community setup.py may "
            "download vanilla LLVM (Ascend usually needs the patched LLVM tree).",
            file=sys.stderr,
        )


def main() -> None:
    configure_plugin_env()
    apply_ascend_patches()
    # Hand off to community setup.py with the same argv (install / develop / …).
    sys.argv[0] = str(_TRITON_SETUP)
    runpy.run_path(str(_TRITON_SETUP), run_name="__main__")


if __name__ == "__main__":
    main()
