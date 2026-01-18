from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    BaseKVCache,
    BaseKVPress,
    KVCacheLayout,
    SizeInfo,
)
from .press import (
    # Basic methods
    L2NormPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    # Norm-based methods
    KnormPress,
    # Attention-based methods
    ExpectedAttentionPress,
    TOVAPress,
    ObservedAttentionPress,
    NonCausalAttnPress,
    # SVD/Projection-based methods
    QFilterPress,
    CURPress,
    # Structural methods
    PyramidKVPress,
    LagKVPress,
    KeyDiffPress,
    # Leverage score methods
    LeverageScorePress,
    CompactorPress,
    # Factory function
    create_kv_press,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BaseCacheManager: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache(
    model_config: ModelConfig,
    num_pages: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_layout: KVCacheLayout = KVCacheLayout.LayerFirst,
) -> BaseKVCache:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        kv_layout=cache_layout,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache_manager(device: torch.device):
    from .naive_manager import NaiveCacheManager

    return NaiveCacheManager(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache_manager(device: torch.device):
    from .radix_manager import RadixCacheManager

    return RadixCacheManager(device=device)


def create_cache_manager(device: torch.device, type: str) -> BaseCacheManager:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache",
    "create_cache_manager",
    "create_kv_press",
    "BaseKVCache",
    "BaseKVPress",
    "KVCacheLayout",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "SUPPORTED_CACHE_MANAGER",
    # Press implementations - Basic
    "RandomPress",
    "StreamingLLMPress",
    "L2NormPress",
    "SnapKVPress",
    # Press implementations - Norm-based
    "KnormPress",
    # Press implementations - Attention-based
    "ExpectedAttentionPress",
    "TOVAPress",
    "ObservedAttentionPress",
    "NonCausalAttnPress",
    # Press implementations - SVD/Projection-based
    "QFilterPress",
    "CURPress",
    # Press implementations - Structural
    "PyramidKVPress",
    "LagKVPress",
    "KeyDiffPress",
    # Press implementations - Leverage score
    "LeverageScorePress",
    "CompactorPress",
]
