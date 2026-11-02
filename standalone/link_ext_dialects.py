#!/usr/bin/env python3
"""Partial-link four MLIR dialect archives for libtriton_ascend.so.

The archives contribute Linalg / Tensor / Complex / Bufferization. Any global
symbol that libtriton.so already exports is rewritten to undefined so the
plugin binds that symbol to the host copy. Symbols that exist only in these
archives stay defined in the plugin, so each MLIR symbol has one definition.

Also supports --bind-so: after the final link, Ascend/bishengir .o files may
still define Unique copies of host TypeIDs (notably OpTrait::IsTerminator).
Those must be rewritten to SHN_UNDEF in the finished .so so CSE/verify from
libtriton see the same TypeID that Yield registration recorded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile

# Elf64_Sym
_SYM64 = struct.Struct("<IBBHQQ")
SHN_UNDEF = 0
KEEP_TYPES = {"STT_OBJECT", "STT_FUNC", "STT_NOTYPE", "STT_TLS", "STT_GNU_IFUNC"}


def _defined_dynamic(lib: Path) -> set[str]:
    out = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(lib)], text=True, errors="replace"
    )
    names: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and len(parts[1]) == 1:
            names.add(parts[2])
    return names


def _find_ld() -> str:
    """Prefer LLVM ld.lld; GNU ld -r --whole-archive can yield an empty .o here."""
    llvm_sys = os.environ.get("LLVM_SYSPATH", "")
    candidates = [
        os.environ.get("LD"),
        shutil.which("ld.lld"),
        str(Path(llvm_sys) / "bin" / "ld.lld") if llvm_sys else None,
        "/usr/bin/ld.lld",
        shutil.which("ld"),
        "ld",
    ]
    for cand in candidates:
        if not cand:
            continue
        if Path(cand).is_file() or shutil.which(cand):
            return str(cand)
    return "ld"


def _partial_link(archives: list[Path], combined: Path) -> None:
    ld = _find_ld()
    cmd = [ld, "-r", "-o", str(combined), "--whole-archive", *[str(a) for a in archives], "--no-whole-archive"]
    subprocess.check_call(cmd)
    if not combined.is_file() or combined.stat().st_size == 0:
        raise SystemExit(
            f"partial link produced empty {combined} via {ld}; "
            "set LD=.../ld.lld or LLVM_SYSPATH"
        )


def _rewrite_sym_section(
    data: bytearray,
    elf: ELFFile,
    section_name: str,
    host_defined: set[str],
    name_substr: str | None = None,
) -> tuple[int, int, int]:
    """Rewrite defined host-overlapping symbols in one symtab to SHN_UNDEF."""
    symtab = elf.get_section_by_name(section_name)
    if symtab is None:
        return 0, 0, 0
    off = symtab["sh_offset"]
    entsize = symtab["sh_entsize"] or _SYM64.size
    if entsize != _SYM64.size:
        raise SystemExit(f"unexpected symbol size {entsize} in {section_name}")
    n = symtab.num_symbols()
    kept = stripped = skipped = 0
    for i, sym in enumerate(symtab.iter_symbols()):
        shndx = sym["st_shndx"]
        if shndx == "SHN_UNDEF" or isinstance(shndx, str):
            continue
        st_type_name = sym["st_info"]["type"]
        name = sym.name
        if not name or st_type_name not in KEEP_TYPES:
            skipped += 1
            continue
        if name_substr is not None and name_substr not in name:
            kept += 1
            continue
        if name not in host_defined:
            kept += 1
            continue
        base = off + i * entsize
        st_name, st_info, st_other, _shndx, _val, st_size = _SYM64.unpack_from(
            data, base
        )
        # Keep original bind/visibility; only clear the local definition so the
        # dynamic linker resolves the symbol via libtriton.
        _SYM64.pack_into(
            data, base, st_name, st_info, st_other, SHN_UNDEF, 0, st_size
        )
        stripped += 1
    if n and (off + n * entsize) > len(data):
        raise SystemExit(f"symbol table {section_name} extends past file")
    return kept, stripped, skipped


def _rewrite(
    obj: Path,
    host_defined: set[str],
    shared: bool = False,
    name_substr: str | None = None,
) -> tuple[int, int, int]:
    data = bytearray(obj.read_bytes())
    with obj.open("rb") as f:
        elf = ELFFile(f)
        sections = [".dynsym", ".symtab"] if shared else [".symtab"]
        kept = stripped = skipped = 0
        for sec in sections:
            k, s, sk = _rewrite_sym_section(
                data, elf, sec, host_defined, name_substr=name_substr
            )
            kept += k
            stripped += s
            skipped += sk
        if shared and elf.get_section_by_name(".dynsym") is None:
            raise SystemExit(f"{obj} has no .dynsym (not a shared object?)")
        if not shared and elf.get_section_by_name(".symtab") is None:
            raise SystemExit(f"{obj} has no .symtab")
    obj.write_bytes(data)
    return kept, stripped, skipped


def _bind_so(libtriton: Path, shared_lib: Path) -> int:
    """Rebind Unique TypeIDResolver copies in the finished .so to host.

    Only touches TypeIDResolver symbols. Broad host-overlap rewrites can turn
    host-hidden helpers into hard undefineds and break the link.
    """
    if not libtriton.is_file():
        print(f"missing {libtriton}", file=sys.stderr)
        return 1
    if not shared_lib.is_file():
        print(f"missing {shared_lib}", file=sys.stderr)
        return 1
    host = _defined_dynamic(libtriton)
    staging = shared_lib.with_suffix(shared_lib.suffix + ".bind_staging")
    if staging.exists():
        staging.unlink()
    shutil.copyfile(shared_lib, staging)
    kept, stripped, skipped = _rewrite(
        staging, host, shared=True, name_substr="TypeIDResolver"
    )
    staging.replace(shared_lib)
    print(
        f"bind-so: host_exports={len(host)} kept={kept} "
        f"bound_to_host={stripped} skipped={skipped} -> {shared_lib}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--libtriton", type=Path, required=True)
    p.add_argument(
        "--bind-so",
        type=Path,
        help="Rewrite host-overlapping dynsym/symtab entries in a finished .so",
    )
    p.add_argument("--archive", type=Path, action="append")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    if args.bind_so is not None:
        return _bind_so(args.libtriton, args.bind_so)

    if not args.archive or args.output is None:
        p.error("--archive/--output required unless --bind-so is set")
    for a in args.archive:
        if not a.is_file():
            print(f"missing archive {a}", file=sys.stderr)
            return 1
    if not args.libtriton.is_file():
        print(f"missing {args.libtriton}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Work on a temp path then atomically replace the cmake OUTPUT so a failed
    # rewrite never leaves a truncated .o for the next gmake/elftools pass.
    combined = args.output.with_suffix(".combined.o")
    staging = args.output.with_suffix(".staging.o")
    if staging.exists():
        staging.unlink()
    _partial_link(args.archive, combined)
    host = _defined_dynamic(args.libtriton)
    shutil.copyfile(combined, staging)
    kept, stripped, skipped = _rewrite(staging, host, shared=False)
    staging.replace(args.output)
    print(
        f"ext dialects: host_exports={len(host)} kept={kept} "
        f"bound_to_host={stripped} skipped={skipped} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
