#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``triton-ascend`` wheel — scheme 1 community plugin for Triton 3.7.1.

The native artifact is ``libtriton_ascend.so``. It exports
``tritonGetPluginInfo``, links the installed ``libtriton``, and embeds
missing MLIR dialect objects produced by ``standalone/link_ext_dialects.py``.

Triton 3.7 loads plugins from ``TRITON_PLUGIN_PATHS`` at ``libtriton`` import
time (no ``extend_with``). Headers come from the ``third_party/triton``
submodule because the 3.7.1 wheel does not ship them.

Requires: ``LLVM_SYSPATH``, host ``triton==3.7.1`` built with
``TRITON_EXT_ENABLED=1``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
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
_CMAKE_PCT_RE = re.compile(r"\[\s*(\d+)%\]")


def _progress(msg: str) -> None:
    """Print build progress where the user can see it.

    ``pip install`` (non-verbose) captures the build backend's stdout/stderr and
    only dumps it on failure, so a long cmake run looks "stuck". Writing to
    ``/dev/tty`` bypasses that pipe while still logging to stdout for ``pip -v``.
    """
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except OSError:
        pass
    try:
        with open("/dev/tty", "w", encoding="utf-8", errors="replace") as tty:
            tty.write(line)
            tty.flush()
    except OSError:
        pass


def _run_logged(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    label: str,
    cwd: str | None = None,
    progress_throttle_s: float = 1.0,
) -> None:
    """Run a command, tee full output to ``log_path``, and surface progress live."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(f"[triton-ascend] {label}")
    _progress(f"[triton-ascend]   cmd: {' '.join(cmd)}")
    _progress(f"[triton-ascend]   log: {log_path}  (tail -f to follow)")

    last_pct = -1
    last_emit = 0.0
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        logf.write(f"# {label}\n# {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            logf.write(raw)
            logf.flush()
            line = raw.rstrip("\n")
            tail.append(line)
            if len(tail) > 80:
                del tail[: len(tail) - 80]
            m = _CMAKE_PCT_RE.search(line)
            now = time.monotonic()
            if m:
                pct = int(m.group(1))
                if pct != last_pct and (now - last_emit) >= progress_throttle_s:
                    # Keep one short line so -j192 spam stays readable.
                    snippet = line.strip()
                    if len(snippet) > 120:
                        snippet = snippet[:117] + "..."
                    _progress(f"[triton-ascend] [{pct:3d}%] {snippet}")
                    last_pct = pct
                    last_emit = now
            elif any(
                key in line
                for key in (
                    "Configuring done",
                    "Generating done",
                    "Build files have been written",
                    "Linking CXX",
                    "Built target libtriton_ascend",
                    "error:",
                    "fatal error:",
                )
            ):
                _progress(f"[triton-ascend] {line.strip()}")
                last_emit = now
        rc = proc.wait()
    if rc != 0:
        _progress(f"[triton-ascend] FAILED: {label} (exit {rc})")
        _progress(f"[triton-ascend] last log lines ({log_path}):")
        for line in tail[-40:]:
            _progress(f"  {line}")
        raise subprocess.CalledProcessError(rc, cmd)
    _progress(f"[triton-ascend] done: {label}")


def _invalidate_stale_cmake_cache(build_dir: Path, llvm: str) -> None:
    """Drop CMakeCache when LLVM_DIR no longer matches ``LLVM_SYSPATH``.

    Re-running cmake -B reuses a cached LLVM_DIR/MLIR_DIR; a leftover 3.9
    cache with a 3.7.1 ``LLVM_SYSPATH`` silently builds against the wrong tree.
    """
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return
    llvm_resolved = str(Path(llvm).resolve())
    cached_llvm_dir = ""
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("LLVM_DIR:"):
            # LLVM_DIR:PATH=/path/to/.../lib/cmake/llvm
            cached_llvm_dir = line.split("=", 1)[-1].strip()
            break
    if not cached_llvm_dir:
        return
    if cached_llvm_dir.startswith(llvm_resolved + "/") or cached_llvm_dir.startswith(
        llvm.rstrip("/") + "/"
    ):
        return
    _progress(
        f"[triton-ascend] stale CMake cache LLVM_DIR={cached_llvm_dir} "
        f"(want under {llvm}); removing {build_dir}"
    )
    shutil.rmtree(build_dir, ignore_errors=True)


def _install_plugin_pth() -> None:
    """Install a site ``.pth`` that sets ``TRITON_PLUGIN_PATHS`` at startup."""
    import site

    # After install the package lives in site-packages; during the same
    # process ``import triton_ascend`` may still resolve to the source tree.
    line = "import triton_ascend._autoload_plugin\n"
    candidates: list[Path] = []
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        if not site_dir:
            continue
        if os.environ.get("CONDA_PREFIX") and not str(site_dir).startswith(
            os.environ["CONDA_PREFIX"]
        ):
            continue
        root = Path(site_dir)
        if not root.is_dir():
            continue
        autoload = root / "triton_ascend" / "_autoload_plugin.py"
        if autoload.is_file() or (_PKG_ROOT / "_autoload_plugin.py").is_file():
            candidates.append(root / "triton_ascend_plugin.pth")
    if not candidates:
        print("[triton-ascend] WARNING: no site-packages for .pth")
        return
    for pth in candidates:
        try:
            pth.write_text(line)
            print(f"[triton-ascend] wrote {pth}")
            return
        except OSError as e:
            print(f"[triton-ascend] WARNING: could not write {pth}: {e}")
    print("[triton-ascend] WARNING: no writable site-packages for .pth")


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
    def run(self) -> None:
        super().run()
        # PEP 660 ``pip install -e .`` builds an editable wheel and does NOT
        # invoke setuptools ``develop`` / ``install`` cmdclass hooks. The Ascend
        # ``.pth`` that sets TRITON_PLUGIN_PATHS before ``import triton`` must
        # therefore be written from ``build_ext``, which still runs.
        try:
            _install_plugin_pth()
        except Exception as e:
            print(
                f"[triton-ascend] WARNING: plugin .pth install failed: {e}",
                file=sys.stderr,
            )
        try:
            _install_language_into_triton()
        except Exception as e:
            print(
                f"[triton-ascend] WARNING: language install failed: {e}",
                file=sys.stderr,
            )

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        llvm = os.environ.get("LLVM_SYSPATH")
        if not llvm:
            raise RuntimeError(
                "LLVM_SYSPATH must be set (same LLVM used to build host triton)"
            )

        _invalidate_stale_cmake_cache(_BUILD_DIR, llvm)
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
        out_c = _PKG_ROOT / "_C"
        out_c.mkdir(parents=True, exist_ok=True)
        cmake = _find_cmake()
        log_dir = _BUILD_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # AscendNPU-IR is a submodule: keep HEAD clean and re-apply the LLVM-23
        # adapter patch on every build (same contract as monorepo setup_ascend).
        _progress("[triton-ascend] applying AscendNPU-IR llvm23 adapter patch")
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

        _progress(f"[triton-ascend] plugin cmake src={_STANDALONE}")
        _progress(f"[triton-ascend] plugin build={_BUILD_DIR}")
        _progress(f"[triton-ascend] LLVM_SYSPATH={llvm}")
        _progress(f"[triton-ascend] cmake={cmake}")
        _progress("[triton-ascend] scheme 1: host libtriton + missing dialect .o")
        _progress(
            "[triton-ascend] tip: pip hides cmake by default; progress goes to "
            f"/dev/tty + {log_dir}/ (or use: pip install -v -e . --no-build-isolation)"
        )

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

        _run_logged(
            [cmake, "-S", str(_STANDALONE), "-B", str(_BUILD_DIR), *cmake_args],
            env=cmake_env,
            log_path=log_dir / "cmake-configure.log",
            label="cmake configure",
        )
        jobs = os.environ.get("MAX_JOBS", str(os.cpu_count() or 4))
        _progress(f"[triton-ascend] cmake build -j{jobs} target=libtriton_ascend")
        _run_logged(
            [
                cmake,
                "--build",
                str(_BUILD_DIR),
                "-j",
                jobs,
                "--target",
                "libtriton_ascend",
            ],
            env=cmake_env,
            log_path=log_dir / "cmake-build.log",
            label="cmake build libtriton_ascend",
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
        _progress(f"[triton-ascend] native -> {ext_path}")
        # PEP 660 ``pip install -e .`` builds the editable wheel via build_ext but
        # does *not* run setuptools ``develop``/``install`` cmdclass hooks. Write
        # the site ``.pth`` here so TRITON_PLUGIN_PATHS is set before libtriton
        # loads on the next interpreter start.
        try:
            _install_plugin_pth()
        except Exception as e:
            _progress(f"[triton-ascend] WARNING: plugin .pth install failed: {e}")
        try:
            _install_language_into_triton()
        except Exception as e:
            _progress(f"[triton-ascend] WARNING: language install failed: {e}")


class _PostInstallMixin:
    def run(self):
        super().run()
        # Keep for classic ``python setup.py develop`` / non-editable install.
        try:
            _install_plugin_pth()
        except Exception as e:
            print(f"[triton-ascend] WARNING: plugin .pth install failed: {e}", file=sys.stderr)
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
    version=os.environ.get("TRITON_ASCEND_VERSION", "3.7.1"),
    description="Ascend NPU backend plugin for community Triton 3.7.1",
    # Runtime-only wheel: backend + language + _C.so. No community trees,
    # no AscendNPU-IR / lib / include / standalone sources.
    #
    # ``_C/`` is a build output (gitignore), not a Python package. Declaring it
    # in ``packages`` / ``package_dir`` makes ``egg_info`` fail on a fresh clone
    # before ``build_ext`` can create the directory. Ship the .so via the
    # extension + ``package_data`` instead; runtime uses ``plugin_library()``.
    packages=[
        "triton_ascend",
        "triton_ascend.backend",
        "triton_ascend.backend.reuse",
        "triton_ascend.backend.reuse.dsl",
        "triton_ascend.backend.runtime",
        "triton_ascend.backend.runtime.cv_autotune",
        "triton_ascend.backend.runtime.cv_autotune.normalization",
        "triton_ascend.backend.runtime.dsl_analysis",
    ],
    package_dir={
        "triton_ascend": ".",
        "triton_ascend.backend": "backend",
        "triton_ascend.backend.reuse": "backend/reuse",
        "triton_ascend.backend.reuse.dsl": "backend/reuse/dsl",
        "triton_ascend.backend.runtime": "backend/runtime",
        "triton_ascend.backend.runtime.cv_autotune": "backend/runtime/cv_autotune",
        "triton_ascend.backend.runtime.cv_autotune.normalization": "backend/runtime/cv_autotune/normalization",
        "triton_ascend.backend.runtime.dsl_analysis": "backend/runtime/dsl_analysis",
    },
    include_package_data=False,
    package_data={
        "triton_ascend": [
            "language/cann/**/*.py",
            "language/kernels/**/*.py",
            "language/tla/**/*.py",
            "_autoload_plugin.py",
            # Untagged build product under package root; tagged .so are leftovers.
            "_C/libtriton_ascend.so",
        ],
        "triton_ascend.backend": ["npu_utils.cpp"],
    },
    # Extension name places the .so at triton_ascend/_C/… without needing a
    # Python package named triton_ascend._C. CMakeBuild stages the same path.
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
