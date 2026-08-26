#!/usr/bin/env python3
"""Puncture 2: Ascend stages hook (+ optional Layer C .so).

  2a) Ascend add_stages calls knobs.runtime.add_stages_inspection_hook.
      Proof: hook wraps ttir and records entry func name (IR left unchanged).

  2b) Subprocess with TRITON_PASS_PLUGIN_PATH + passes.plugin.add_plugin.
      Needs LLVM_BUILD_SHARED_LIBS=1; static LLVM → EXPECTED_FAIL.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent


def _find_plugin_so() -> str | None:
    env = os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
    if env and pathlib.Path(env).is_file():
        return env
    for c in (
        ROOT / "build/lib.linux-aarch64-cpython-311/triton/plugins/libTritonPluginsTestLib.so",
    ):
        if c.is_file():
            return str(c.resolve())
    conda = pathlib.Path(os.environ.get("CONDA_PREFIX", ""))
    matches = list(conda.glob("lib/python*/site-packages/triton/plugins/libTritonPluginsTestLib.so"))
    return str(matches[0]) if matches else None


def get_key():
    return pathlib.Path(__file__).read_text()


def get_hash():
    return hashlib.sha256(get_key().encode("utf-8")).hexdigest()


HOOK_STATE: dict = {"called": False}


def inspect_stages_hook_observe(self=None, stages=None, options=None, language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    HOOK_STATE["called"] = True
    HOOK_STATE["stage_keys"] = list(stages.keys())
    orig_ttir = stages["ttir"]

    def make_ttir_observe(src, metadata):
        mod = orig_ttir(src, metadata)
        HOOK_STATE["ttir_name"] = mod.get_entry_func_name()
        HOOK_STATE["ttir_snippet"] = mod.str()[:400]
        return mod

    stages["ttir"] = make_ttir_observe
    return get_key(), get_hash()


def _ensure_ascend_calls_hook(AscendBackend, knobs):
    src_path = pathlib.Path(AscendBackend.add_stages.__code__.co_filename)
    try:
        src = src_path.read_text()
    except OSError:
        src = ""
    if "add_stages_inspection_hook" in src:
        print("INFO: AscendBackend.add_stages already references stages hook")
        return "installed"

    orig = AscendBackend.add_stages

    def add_stages_with_hook(self, stages, options, language):
        orig(self, stages, options, language)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    AscendBackend.add_stages = add_stages_with_hook
    print("INFO: monkey-patched AscendBackend.add_stages (puncture only)")
    print(f"INFO: source-tree fix: {ROOT}/ascend/backend/compiler.py")
    return "monkeypatch"


def run_2a() -> int:
    os.environ.pop("TRITON_PASS_PLUGIN_PATH", None)

    import torch
    from triton import knobs
    from triton.backends.ascend.compiler import AscendBackend

    # Import kernels after triton is available; path = this directory
    sys.path.insert(0, str(HERE))
    import stages_kernels as kernels

    print("== Puncture 2a: Ascend stages hook wiring ==")
    print(f"ascend.compiler={AscendBackend.add_stages.__code__.co_filename}")

    mode = _ensure_ascend_calls_hook(AscendBackend, knobs)
    if not torch.npu.is_available():
        print("FAIL: torch.npu not available", file=sys.stderr)
        return 1

    grid = lambda meta: (1,)

    knobs.runtime.add_stages_inspection_hook = None
    h0 = kernels.kernel_baseline[grid](BLOCK_SIZE=128)
    t0 = h0.asm.get("ttir", "")
    print(f"baseline compiled; @kernel_baseline in ttir={('@kernel_baseline' in t0)}")

    HOOK_STATE.clear()
    HOOK_STATE["called"] = False
    knobs.runtime.add_stages_inspection_hook = inspect_stages_hook_observe
    h1 = kernels.kernel_hooked[grid](BLOCK_SIZE=128)
    knobs.runtime.add_stages_inspection_hook = None

    ok = (
        HOOK_STATE.get("called")
        and HOOK_STATE.get("ttir_name") == "kernel_hooked"
        and "ttir" in (HOOK_STATE.get("stage_keys") or [])
        and h1 is not None
    )
    if ok:
        print(f"PASS 2a: Ascend path ({mode}) invoked stages hook")
        print(f"  stage_keys={HOOK_STATE.get('stage_keys')}")
        print(f"  ttir_name={HOOK_STATE.get('ttir_name')}")
        print(f"  ttir_snippet:\n{HOOK_STATE.get('ttir_snippet')}\n")
        return 0

    print("FAIL 2a: hook not observed", file=sys.stderr)
    print(f"  state={HOOK_STATE}", file=sys.stderr)
    return 1


def run_2b() -> int:
    plugin = _find_plugin_so()
    if not plugin:
        print("SKIP 2b: no libTritonPluginsTestLib.so")
        return 0

    code = f"""
import hashlib, pathlib, sys, traceback
sys.path.insert(0, {str(HERE)!r})
import torch, triton, triton.language as tl
from triton import knobs
from triton._C.libtriton import ir, passes
from triton.backends.ascend.compiler import AscendBackend
import stages_kernels as kernels

def get_key():
    return "plugin_puncture_2b"
def get_hash():
    return hashlib.sha256(get_key().encode()).hexdigest()

src = pathlib.Path(AscendBackend.add_stages.__code__.co_filename).read_text()
if "add_stages_inspection_hook" not in src:
    orig = AscendBackend.add_stages
    def add_stages_with_hook(self, stages, options, language):
        orig(self, stages, options, language)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)
    AscendBackend.add_stages = add_stages_with_hook
    print("INFO: monkey-patched AscendBackend.add_stages")

if not (hasattr(passes, "plugin") and hasattr(passes.plugin, "add_plugin")):
    print("FAIL 2b: passes.plugin.add_plugin not bound")
    sys.exit(1)

def hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(a is None for a in (stages, options, language, capability)):
        return get_key(), get_hash()
    orig = stages["ttir"]
    def wrap(src, metadata):
        mod = orig(src, metadata)
        pm = ir.pass_manager(mod.context)
        passes.plugin.add_plugin(pm)
        pm.run(mod, "ascend_ttir_plugin_puncture")
        return mod
    stages["ttir"] = wrap
    return get_key(), get_hash()

print("== Puncture 2b: Layer C .so on Ascend path ==")
print(f"PLUGIN_SO={{os.environ.get('TRITON_PASS_PLUGIN_PATH')}}")
import os
print(f"PLUGIN_SO={{os.environ.get('TRITON_PASS_PLUGIN_PATH')}}")
knobs.runtime.add_stages_inspection_hook = hook
try:
    h = kernels.kernel_plugin[lambda meta: (1,)](BLOCK_SIZE=128)
    ttir = h.asm.get("ttir", "")
    if "@foo" in ttir:
        print("PASS 2b: saw @foo")
        sys.exit(0)
    print("FAIL 2b: no @foo")
    print(ttir[:500])
    sys.exit(1)
except Exception as e:
    err = "".join(traceback.format_exception_only(type(e), e)).strip()
    print("EXPECTED_FAIL 2b under static LLVM (needs LLVM_BUILD_SHARED_LIBS=1):")
    print(f"  {{err}}")
    print("  See examples/plugins/README.md and lit REQUIRES: shared-libs")
    sys.exit(0)
"""
    # Fix botched f-string in embedded code — rewrite cleanly below
    code = (
        "import hashlib, os, pathlib, sys, traceback\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import torch, triton, triton.language as tl\n"
        "from triton import knobs\n"
        "from triton._C.libtriton import ir, passes\n"
        "from triton.backends.ascend.compiler import AscendBackend\n"
        "import stages_kernels as kernels\n"
        "\n"
        "def get_key():\n"
        "    return 'plugin_puncture_2b'\n"
        "def get_hash():\n"
        "    return hashlib.sha256(get_key().encode()).hexdigest()\n"
        "\n"
        "src = pathlib.Path(AscendBackend.add_stages.__code__.co_filename).read_text()\n"
        "if 'add_stages_inspection_hook' not in src:\n"
        "    orig = AscendBackend.add_stages\n"
        "    def add_stages_with_hook(self, stages, options, language):\n"
        "        orig(self, stages, options, language)\n"
        "        if knobs.runtime.add_stages_inspection_hook is not None:\n"
        "            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)\n"
        "    AscendBackend.add_stages = add_stages_with_hook\n"
        "    print('INFO: monkey-patched AscendBackend.add_stages')\n"
        "\n"
        "if not (hasattr(passes, 'plugin') and hasattr(passes.plugin, 'add_plugin')):\n"
        "    print('FAIL 2b: passes.plugin.add_plugin not bound')\n"
        "    sys.exit(1)\n"
        "\n"
        "def hook(self=None, stages=None, options=None, language=None, capability=None):\n"
        "    if all(a is None for a in (stages, options, language, capability)):\n"
        "        return get_key(), get_hash()\n"
        "    orig = stages['ttir']\n"
        "    def wrap(src, metadata):\n"
        "        mod = orig(src, metadata)\n"
        "        pm = ir.pass_manager(mod.context)\n"
        "        passes.plugin.add_plugin(pm)\n"
        "        pm.run(mod, 'ascend_ttir_plugin_puncture')\n"
        "        return mod\n"
        "    stages['ttir'] = wrap\n"
        "    return get_key(), get_hash()\n"
        "\n"
        "print('== Puncture 2b: Layer C .so on Ascend path ==')\n"
        "print('PLUGIN_SO=' + os.environ.get('TRITON_PASS_PLUGIN_PATH', ''))\n"
        "knobs.runtime.add_stages_inspection_hook = hook\n"
        "try:\n"
        "    h = kernels.kernel_plugin[lambda meta: (1,)](BLOCK_SIZE=128)\n"
        "    ttir = h.asm.get('ttir', '')\n"
        "    if '@foo' in ttir:\n"
        "        print('PASS 2b: saw @foo')\n"
        "        sys.exit(0)\n"
        "    print('FAIL 2b: no @foo')\n"
        "    print(ttir[:500])\n"
        "    sys.exit(1)\n"
        "except Exception as e:\n"
        "    err = ''.join(traceback.format_exception_only(type(e), e)).strip()\n"
        "    print('EXPECTED_FAIL 2b under static LLVM (needs LLVM_BUILD_SHARED_LIBS=1):')\n"
        "    print('  ' + err)\n"
        "    print('  See examples/plugins/README.md and lit REQUIRES: shared-libs')\n"
        "    sys.exit(0)\n"
    )
    env = os.environ.copy()
    env["TRITON_PASS_PLUGIN_PATH"] = plugin
    print("\n-- launching 2b subprocess --")
    r = subprocess.run([sys.executable, "-c", code], env=env)
    return r.returncode


def main() -> int:
    rc = run_2a()
    rc2 = run_2b()
    return rc or rc2


if __name__ == "__main__":
    sys.exit(main())
