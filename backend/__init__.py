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
__version__ = '3.6.0'

import logging
import sys

# Entry point package is triton_ascend.backend; legacy imports use
# triton.backends.ascend (utils, driver, compiler).
sys.modules.setdefault("triton.backends.ascend", sys.modules[__name__])

# Triton discovers this package via ``triton.backends`` entry points. Set
# ``TRITON_PLUGIN_PATHS`` and warn on wrong host version *here*, before
# ``backend.compiler`` imports ``libtriton`` (no site ``.pth``).
from .plugin_bootstrap import bootstrap_ascend_plugin

bootstrap_ascend_plugin()

from .testing import do_bench_npu


def _apply_ascend_patch():
    """Hot-patch community Triton. Does not edit host sources."""
    try:
        from .triton_hotpatch import install as _install_hotpatch

        _install_hotpatch()
    except Exception as e:
        logging.warning(f"[Ascend Patch] hotpatch skipped: {e}")


from .knobs import install_ascend_knobs as _install_ascend_knobs

_install_ascend_knobs()

__all__ = ["do_bench_npu"]
