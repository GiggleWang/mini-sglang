from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.kvcache import BaseCacheHandle, BaseKVCache, BaseKVPress, create_cache_manager, create_kv_press

if TYPE_CHECKING:
    from minisgl.core import Batch, Req

    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )

    def compress_kv(
        self,
        req: "Req",
        kv_cache: BaseKVCache,
        page_table: torch.Tensor,
        batch: "Batch",
    ) -> Tuple[int, int]:
        """
        Compress KV cache for a request after prefill.

        Args:
            req: The request to compress.
            kv_cache: The KV cache.
            page_table: The page table mapping (table_idx, seq_pos) -> page_index.
            batch: The current batch.

        Returns:
            Tuple of (kept_len, evicted_len) indicating compression results.
        """
        # Create KVPress from serializable params
        method = req.sampling_params.kv_press_method
        if method is None:
            return (req.device_len, 0)

        kv_press: BaseKVPress = create_kv_press(
            method, compression_ratio=req.sampling_params.kv_press_ratio
        )

        seq_len = req.cached_len
        if seq_len <= 1:
            return (seq_len, 0)

        # Get current page indices for this request
        page_indices = page_table[req.table_idx, :seq_len].clone()

        # Compute importance scores
        scores = kv_press.score(kv_cache, page_indices, batch)

        # Select which positions to keep/evict
        keep_indices, evict_indices = kv_press.select(scores, seq_len)

        if len(evict_indices) == 0:
            return (seq_len, 0)

        # Get the page indices to free
        evicted_pages = page_indices[evict_indices]

        # Update page table: compact the kept pages to the front
        kept_pages = page_indices[keep_indices]
        new_seq_len = len(kept_pages)
        page_table[req.table_idx, :new_seq_len] = kept_pages

        # Free the evicted pages
        self._free(evicted_pages)

        # Update request state
        # Note: We need to update cached_len and device_len to reflect compression
        # The input_ids on CPU are not modified - only the KV cache is compressed
        req.cached_len = min(req.cached_len, new_seq_len)
        req.device_len = new_seq_len + 1

        return (new_seq_len, len(evict_indices))
