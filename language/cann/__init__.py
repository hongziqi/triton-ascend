# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys

from triton.language import math

libdevice = None  # type: ignore
extension = None  # type: ignore


def _drop_failed_submodule(name: str) -> None:
    """Remove a failed-import ``None`` sentinel so a later retry can load."""
    key = f"{__name__}.{name}"
    if sys.modules.get(key) is None:
        sys.modules.pop(key, None)


# libdevice (tanh etc.) must not depend on HostExtend / affine_expr. Keep it
# independent so community OpInfo hosts still get cann.libdevice.
try:
    from . import libdevice as _libdevice

    libdevice = _libdevice
    try:
        math.tanh = libdevice.tanh
    except Exception:
        pass
except Exception:
    _drop_failed_submodule("libdevice")
    libdevice = None  # type: ignore

# extension needs richer IR (affine_*). Soft-fail on OpInfo-only hosts; do not
# undo a successful libdevice import.
try:
    from . import extension as _extension

    extension = _extension
    try:
        extension.parallel = extension.aux_ops.parallel
    except Exception:
        pass
except Exception:
    _drop_failed_submodule("extension")
    extension = None  # type: ignore

__all__ = ["libdevice", "extension"]
