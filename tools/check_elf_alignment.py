#!/usr/bin/env python3
"""Report PT_LOAD alignment of ELF shared libraries (Android 16 KB page check).

Android 15+ requires every PT_LOAD segment in a bundled .so to be aligned to
16 KB (0x4000). Older NDK builds used 4 KB, which triggers the "not 16 KB
compatible" warning and runs the app in compatibility mode.

  python tools/check_elf_alignment.py path/to/lib.so [...]
"""

from __future__ import annotations

import struct
import sys

PT_LOAD = 1


def check(path: str) -> bool:
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != b"\x7fELF":
        print(f"{path}: not an ELF file")
        return False
    is64 = data[4] == 2
    little = data[5] == 1
    endian = "<" if little else ">"

    if is64:
        e_phoff = struct.unpack_from(endian + "Q", data, 0x20)[0]
        e_phentsize = struct.unpack_from(endian + "H", data, 0x36)[0]
        e_phnum = struct.unpack_from(endian + "H", data, 0x38)[0]
        fmt = endian + "IIQQQQQQ"
    else:
        e_phoff = struct.unpack_from(endian + "I", data, 0x1C)[0]
        e_phentsize = struct.unpack_from(endian + "H", data, 0x2A)[0]
        e_phnum = struct.unpack_from(endian + "H", data, 0x2C)[0]
        fmt = endian + "IIIIIIII"

    ok = True
    aligns = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        fields = struct.unpack_from(fmt, data, off)
        p_type = fields[0]
        if p_type != PT_LOAD:
            continue
        if is64:
            p_offset, p_vaddr, p_align = fields[2], fields[3], fields[7]
        else:
            p_offset, p_vaddr, p_align = fields[1], fields[2], fields[6]
        aligns.append(p_align)
        congruent = (p_offset % 0x4000) == (p_vaddr % 0x4000)
        if p_align < 0x4000 or not congruent:
            ok = False

    label = "OK  " if ok else "FAIL"
    pretty = ", ".join(f"0x{a:x}" for a in aligns)
    print(f"{label} {path}: LOAD p_align = [{pretty}]")
    return ok


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: check_elf_alignment.py lib.so [...]")
    sys.exit(0 if all(check(p) for p in paths) else 1)
