from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


class BaseKVPress(ABC):
    """
    Base class for KV cache compression (similar to kvpress).
    Implementations should define how to score and select KV pairs to keep.
    """

    def __init__(self, compression_ratio: float = 0.5):
        """
        Args:
            compression_ratio: Ratio of KV pairs to keep after compression (0.0-1.0).
                             e.g., 0.5 means keeping 50% of the KV pairs.
        """
        assert 0.0 < compression_ratio <= 1.0, "compression_ratio must be in (0, 1]"
        self.compression_ratio = compression_ratio

    @abstractmethod
    def score(
        self,
        kv_cache: "BaseKVCache",
        page_indices: torch.Tensor,
        batch: "Batch",
    ) -> torch.Tensor:
        """
        Compute importance scores for each position in the KV cache.

        Args:
            kv_cache: The KV cache containing K and V tensors.
            page_indices: Indices of pages in the KV cache for this request. Shape: (seq_len,)
            batch: The current batch being processed.

        Returns:
            scores: Importance scores for each position. Shape: (seq_len,)
                   Higher scores indicate more important positions.
        """
        ...

    def select(
        self,
        scores: torch.Tensor,
        seq_len: int,
        num_sink_tokens: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select which positions to keep based on importance scores.

        Args:
            scores: Importance scores for each position. Shape: (seq_len,)
            seq_len: Total sequence length.
            num_sink_tokens: Number of initial tokens to always keep (attention sinks).

        Returns:
            keep_indices: Indices of positions to keep. Shape: (keep_len,)
            evict_indices: Indices of positions to evict. Shape: (evict_len,)
        """
        target_len = max(num_sink_tokens, int(seq_len * self.compression_ratio))

        if target_len >= seq_len:
            # No compression needed
            return torch.arange(seq_len, device=scores.device), torch.tensor(
                [], dtype=torch.long, device=scores.device
            )

        # Always keep sink tokens (first few tokens are important for attention)
        sink_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        sink_mask[:num_sink_tokens] = True

        # Select top-k from remaining positions based on scores
        remaining_scores = scores.clone()
        remaining_scores[:num_sink_tokens] = float("-inf")  # Exclude sinks from selection

        num_to_keep = target_len - num_sink_tokens
        if num_to_keep > 0:
            _, top_indices = remaining_scores.topk(num_to_keep)
            keep_mask = sink_mask.clone()
            keep_mask[top_indices] = True
        else:
            keep_mask = sink_mask

        keep_indices = torch.where(keep_mask)[0]
        evict_indices = torch.where(~keep_mask)[0]

        return keep_indices, evict_indices


class BaseKVCache(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


class KVCacheLayout(enum.Enum):
    LayerFirst = enum.auto()
    PageFirst = enum.auto()


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class BaseCacheManager(ABC):
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[BaseCacheHandle, torch.Tensor]:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            handle (BaseCacheHandle): The handle to the matched prefix.
            indices (torch.Tensor): The indices of the longest-matched prefix in the cache.
        """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            int: The length of prefix that is already in the cache. This part is not
                 inserted, so the caller should free these indices.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
