#!/usr/bin/env python3
"""Partial-link four MLIR dialect archives for libtriton_ascend.so.

The archives contribute Linalg / Tensor / Complex / Bufferization. Any global
symbol that libtriton.so already exports is rewritten to undefined so the
plugin binds that symbol to the host copy. Symbols that exist only in these
archives stay defined in the plugin, so each MLIR symbol has one definition.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile

# Elf64_Sym
_ST_SHNDX = 6
_ST_VALUE = 8
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


def _partial_link(archives: list[Path], combined: Path) -> None:
    cmd = ["ld", "-r", "-o", str(combined), "--whole-archive", *[str(a) for a in archives], "--no-whole-archive"]
    subprocess.check_call(cmd)


def _rewrite(obj: Path, host_defined: set[str]) -> tuple[int, int, int]:
    data = bytearray(obj.read_bytes())
    with obj.open("rb") as f:
        elf = ELFFile(f)
        symtab = elf.get_section_by_name(".symtab")
        if symtab is None:
            raise SystemExit(f"{obj} has no .symtab")
        off = symtab["sh_offset"]
        entsize = symtab["sh_entsize"] or _SYM64.size
        if entsize != _SYM64.size:
            raise SystemExit(f"unexpected symbol size {entsize}")
        n = symtab.num_symbols()
        kept = stripped = skipped = 0
        for i, sym in enumerate(symtab.iter_symbols()):
            shndx = sym["st_shndx"]
            if shndx == "SHN_UNDEF" or isinstance(shndx, str):
                continue
            st_type = sym["st_info"]["type"]
            name = sym.name
            if not name or st_type not in KEEP_TYPES:
                skipped += 1
                continue
            if name not in host_defined:
                kept += 1
                continue
            base = off + i * entsize
            st_name, st_info, st_other, _shndx, _val, st_size = _SYM64.unpack_from(data, base)
            _SYM64.pack_into(data, base, st_name, st_info, st_other, SHN_UNDEF, 0, st_size)
            stripped += 1
        if n and (off + n * entsize) > len(data):
            raise SystemExit("symbol table extends past file")
    obj.write_bytes(data)
    return kept, stripped, skipped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--libtriton", type=Path, required=True)
    p.add_argument("--archive", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for a in args.archive:
        if not a.is_file():
            print(f"missing archive {a}", file=sys.stderr)
            return 1
    if not args.libtriton.is_file():
        print(f"missing {args.libtriton}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = args.output.with_suffix(".combined.o")
    _partial_link(args.archive, combined)
    host = _defined_dynamic(args.libtriton)
    shutil.copyfile(combined, args.output)
    kept, stripped, skipped = _rewrite(args.output, host)
    print(
        f"ext dialects: host_exports={len(host)} kept={kept} "
        f"bound_to_host={stripped} skipped={skipped} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
