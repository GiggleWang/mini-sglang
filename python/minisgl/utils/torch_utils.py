from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from minisgl.core import Context


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def cuda_time(name: str):
    """Wraps a forward() with CUDA event timing via the global DecodeTimer.

    Only records timing during decode iterations (skips prefill).
    No-op when MINISGL_DECODE_TIMING is not set.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            from minisgl.engine.decode_timer import get_decode_timer

            timer = get_decode_timer()
            if timer is None:
                return fn(self, *args, **kwargs)
            # Import here to avoid circular imports at module load time
            from minisgl.core import get_global_ctx

            ctx = get_global_ctx()
            if ctx.batch is None or ctx.batch.is_prefill:
                return fn(self, *args, **kwargs)
            start = timer.start_module(name)
            result = fn(self, *args, **kwargs)
            timer.end_module(name, start)
            return result

        return wrapper

    return decorator


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
