# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Set ``TRITON_PLUGIN_PATHS`` before host ``libtriton`` is imported.

Triton 3.7.x loads plugins only from the env var at ``libtriton`` init time
(no ``extend_with``). A site ``.pth`` imports this module at interpreter start.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_plugin_path() -> Path | None:
    """Prepend ``libtriton_ascend.so`` to ``TRITON_PLUGIN_PATHS`` if present."""
    env = os.environ.get("TRITON_ASCEND_EXTENSION") or os.environ.get(
        "TRITON_ASCEND_PASS_PLUGIN"
    )
    if env:
        so = Path(env)
    else:
        here = Path(__file__).resolve().parent
        candidates = (
            here / "_C" / "libtriton_ascend.so",
            # Installed wheel layout: triton_ascend/_C/...
            here / "triton_ascend" / "_C" / "libtriton_ascend.so",
        )
        so = next((p for p in candidates if p.is_file()), candidates[0])
    if not so.is_file():
        return None

    path = str(so.resolve())
    existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    parts = [p for p in existing.split(":") if p]
    if path not in parts:
        os.environ["TRITON_PLUGIN_PATHS"] = (
            path if not parts else path + ":" + ":".join(parts)
        )
    return so


ensure_plugin_path()
