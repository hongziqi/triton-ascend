#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``triton-ascend`` wheel — scheme 1 community plugin.

The native artifact is ``libtriton_ascend.so``. It exports
``tritonGetPluginInfo``, links the installed ``libtriton`` (and shared
``libnanobind`` when the host is plugin API v3), and embeds missing MLIR
dialect objects produced by ``standalone/link_ext_dialects.py``.
Requires: ``LLVM_SYSPATH``, a host ``triton`` built with
``TRITON_EXT_ENABLED=1``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

_PKG_ROOT = Path(__file__).resolve().parent
_STANDALONE = _PKG_ROOT / "standalone"
_BUILD_DIR = Path(
    os.environ.get("TRITON_ASCEND_BUILD_DIR", _PKG_ROOT / "build" / "plugin")
)


def _install_language_into_triton() -> None:
    """Optionally place Ascend DSL under triton.language.extra.cann.

    Nightly ``triton_key()`` does ``pkgutil.walk_packages`` on ``language.extra``
    and imports every subpackage. Dropping the full cann tree there crashes
    ``import triton`` / compile until the owned IR builder exists. Default is
    therefore off; ``triton_ascend._ensure_cann_language`` injects the module
    via ``sys.modules`` instead.
    """
    if os.environ.get("TRITON_ASCEND_INSTALL_CANN_INTO_EXTRA", "0") not in (
        "1", "true", "TRUE", "yes",
    ):
        return
    import triton

    dest = Path(triton.__file__).resolve().parent / "language" / "extra" / "cann"
    src = _PKG_ROOT / "language" / "cann"
    if dest.is_symlink() or dest.exists():
        if dest.resolve() == src.resolve():
            return
        if dest.is_symlink():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(src, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dest)
    print(f"[triton-ascend] language -> {dest}")


class CMakeExtension(Extension):
    def __init__(self, name: str):
        super().__init__(name, sources=[])


def _find_cmake() -> str:
    """Prefer a real cmake binary over the pip ``cmake`` console script.

    ``pip install`` with build isolation keeps ``$CONDA_PREFIX/bin`` on PATH, so
    ``cmake`` may resolve to a Python wrapper that does ``from cmake import cmake``.
    The isolated env usually lacks that module → ModuleNotFoundError.
    """
    for cand in (
        os.environ.get("CMAKE"),
        "/usr/bin/cmake",
        "/bin/cmake",
        shutil.which("cmake"),
    ):
        if not cand:
            continue
        path = Path(cand)
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")[:120]
        except OSError:
            text = ""
        # Skip pip entry-point wrappers that import the cmake package.
        if "from cmake import" in text:
            continue
        return str(path)
    raise RuntimeError(
        "cmake not found (need system /usr/bin/cmake). "
        "Or: pip install . --no-build-isolation with a working cmake on PATH."
    )


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        llvm = os.environ.get("LLVM_SYSPATH")
        if not llvm:
            raise RuntimeError(
                "LLVM_SYSPATH must be set (same LLVM used to build host triton)"
            )

        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
        out_c = _PKG_ROOT / "_C"
        out_c.mkdir(parents=True, exist_ok=True)
        cmake = _find_cmake()

        # AscendNPU-IR is a submodule: keep HEAD clean and re-apply the LLVM-24
        # adapter patch on every build (same contract as monorepo setup_ascend).
        subprocess.check_call(
            [sys.executable, str(_PKG_ROOT / "scripts" / "apply_npuir_patch.py")],
            cwd=str(_PKG_ROOT),
        )

        cmake_args = [
            f"-DLLVM_SYSPATH={llvm}",
            f"-DASCEND_ROOT={_PKG_ROOT}",
            f"-DSTANDALONE_OUTPUT_DIR={out_c}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        print(f"[triton-ascend] plugin cmake src={_STANDALONE}")
        print(f"[triton-ascend] plugin build={_BUILD_DIR}")
        print(f"[triton-ascend] LLVM_SYSPATH={llvm}")
        print(f"[triton-ascend] cmake={cmake}")
        print("[triton-ascend] scheme 1: host libtriton + missing dialect .o")

        # pip's default build isolation injects a temporary PYTHONPATH that only
        # has setuptools. CMake then runs Python_EXECUTABLE to `import triton`
        # and fails with ModuleNotFoundError even though the host env has it.
        # Prefer: pip install -e . --no-build-isolation
        cmake_env = os.environ.copy()
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "_PYPROJECT_HOOKS_BACKEND_PATH",
            "PEP517_BUILD_BACKEND",
        ):
            cmake_env.pop(key, None)

        subprocess.check_call(
            [cmake, "-S", str(_STANDALONE), "-B", str(_BUILD_DIR), *cmake_args],
            env=cmake_env,
        )
        jobs = os.environ.get("MAX_JOBS", str(os.cpu_count() or 4))
        subprocess.check_call(
            [cmake, "--build", str(_BUILD_DIR), "-j", jobs, "--target", "libtriton_ascend"],
            env=cmake_env,
        )

        staged = out_c / "libtriton_ascend.so"
        if not staged.is_file():
            matches = list(out_c.glob("libtriton_ascend*.so"))
            if not matches:
                raise RuntimeError(f"libtriton_ascend.so missing under {out_c}")
            staged = matches[0]

        ext_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_path.parent
        extdir.mkdir(parents=True, exist_ok=True)
        if staged.resolve() != ext_path:
            shutil.copy2(staged, ext_path)
        # Keep untagged name in sync so package_data and plugin_library()
        # cannot pick up a stale smaller copy.
        if staged.resolve() != (out_c / "libtriton_ascend.so").resolve():
            shutil.copy2(staged, out_c / "libtriton_ascend.so")
        if ext_path.name != "libtriton_ascend.so":
            shutil.copy2(ext_path, extdir / "libtriton_ascend.so")
        # Drop stale tagged copies under _C (old EXT builds were ~2x larger and
        # used to win a size-based picker, loading the wrong .so at runtime).
        for stale in out_c.glob("libtriton_ascend.cpython-*.so"):
            if stale.resolve() != staged.resolve():
                try:
                    stale.unlink()
                except OSError:
                    pass
        print(f"[triton-ascend] native -> {ext_path}")


class _PostInstallMixin:
    def run(self):
        super().run()
        try:
            _install_language_into_triton()
        except Exception as e:
            print(f"[triton-ascend] WARNING: language install failed: {e}", file=sys.stderr)


class Install(_PostInstallMixin, install):
    pass


class Develop(_PostInstallMixin, develop):
    pass


# Keep egg-info under build/ so `pip show` from the repo root does not report
# Location=<source tree> and confuse uninstall/reinstall.
_EGG_BASE = _PKG_ROOT / "build" / "egg-info"
_EGG_BASE.mkdir(parents=True, exist_ok=True)

setup(
    name="triton-ascend",
    version=os.environ.get("TRITON_ASCEND_VERSION", "3.8.0"),
    description="Ascend NPU backend plugin for community Triton",
    # Runtime-only wheel: backend + language + _C.so. No community trees,
    # no AscendNPU-IR / lib / include / standalone sources.
    packages=[
        "triton_ascend",
        "triton_ascend.backend",
        "triton_ascend.backend.runtime",
        "triton_ascend.backend.runtime.cv_autotune",
        "triton_ascend.backend.runtime.cv_autotune.normalization",
        "triton_ascend.backend.runtime.dsl_analysis",
        "triton_ascend._C",
    ],
    package_dir={
        "triton_ascend": ".",
        "triton_ascend.backend": "backend",
        "triton_ascend.backend.runtime": "backend/runtime",
        "triton_ascend.backend.runtime.cv_autotune": "backend/runtime/cv_autotune",
        "triton_ascend.backend.runtime.cv_autotune.normalization": "backend/runtime/cv_autotune/normalization",
        "triton_ascend.backend.runtime.dsl_analysis": "backend/runtime/dsl_analysis",
        "triton_ascend._C": "_C",
    },
    include_package_data=False,
    package_data={
        "triton_ascend": [
            "language/cann/**/*.py",
            "language/kernels/**/*.py",
        ],
        "triton_ascend.backend": ["npu_utils.cpp"],
        # Only the untagged .so — tagged copies are build leftovers.
        "triton_ascend._C": ["*.py", "libtriton_ascend.so"],
    },
    # Ensure native plugin .so is packaged even when not discovered as an ext.
    # CMakeBuild still builds CMakeExtension; this covers staged libtriton_ascend.so.
    ext_modules=[CMakeExtension("triton_ascend._C.libtriton_ascend")],
    cmdclass={"build_ext": CMakeBuild, "install": Install, "develop": Develop},
    python_requires=">=3.9",
    install_requires=["triton"],
    entry_points={
        "triton.backends": [
            "ascend = triton_ascend.backend",
        ],
    },
    zip_safe=False,
    options={"egg_info": {"egg_base": str(_EGG_BASE)}},
)
