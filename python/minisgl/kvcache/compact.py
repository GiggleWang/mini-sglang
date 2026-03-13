"""KV cache compaction after prefill.

After prefill, scores every token using a pluggable BaseKVCompressor,
then physically moves the kept tokens' KV vectors in-place using a staging
buffer (gather → clone → scatter), following the vLLM v1 approach.

No new pages are allocated: kept data is written to the first keep_count
slots of the original allocation, and the trailing pages are freed.
page_table needs no update since dst_slots == old_slots[:keep_count].
"""
from __future__ import annotations

import math

import torch

from .compression import BaseKVCompressor


class KVCompressionManager:
    """Orchestrates post-prefill KV cache compression for a single request.

    Usage (called by scheduler inside lazy_free_region, BEFORE cache_req):
        manager.compress_req(req, cache_manager, kv_cache, page_table, prefill_q)
    """

    def __init__(self, compressor: BaseKVCompressor, keep_ratio: float):
        self.compressor = compressor
        self.keep_ratio = keep_ratio

    def compress_req(
        self,
        req,
        cache_manager,
        kv_cache,
        page_table: torch.Tensor,
        prefill_q: torch.Tensor | None = None,
    ) -> None:
        """Compress KV cache for one request after prefill completes.

        Steps:
          1. Score tokens → select keep_count indices (GPU)
          2. Gather kept KV data into staging buffer (GPU, with clone)
          3. Scatter staging buffer back to the first keep_count slots (GPU)
          4. Free trailing pages (slots beyond the first keep_pages)
          5. Update req metadata (cached_len, device_len, input_ids)

        page_table is NOT modified: the first keep_count entries already
        point to the correct (now-overwritten) physical slots.

        Args:
            req: Req object (cached_len = input_len at call time)
            cache_manager: CacheManager (provides _free / page_size)
            kv_cache: MHAKVCache
            page_table: GPU tensor shape (max_reqs, max_seq_len)
            prefill_q: optional Q from last attention layer (for ThinkPress)
        """
        # Skip if this request reused a shared prefix — those pages cannot be remapped
        if req.cache_handle.cached_len > 0:
            return

        seq_len = req.cached_len  # = input_len after complete_one()
        keep_count = max(
            math.ceil(self.keep_ratio * seq_len),
            cache_manager.page_size,  # keep at least one full page
        )
        if keep_count >= seq_len:
            return  # nothing to compress

        # ── 1. Score and select tokens ────────────────────────────────────────
        keep_indices = self.compressor.select_tokens(
            kv_cache,
            page_table,
            req.table_idx,
            seq_len,
            keep_count,
            prefill_q,
        )  # (keep_count,) sorted ascending, on GPU

        # ── 2 & 3. In-place KV compaction (staging buffer, vLLM approach) ────
        # src_slots: physical slots of the kept tokens
        # dst_slots: first keep_count slots of the original allocation
        # Since all slot values in old_slots are unique, and keep_indices[i] >= i,
        # dst and src never alias across iterations — but we clone for safety.
        old_slots = page_table[req.table_idx, :seq_len]  # view, no copy
        src_slots = old_slots[keep_indices]               # (keep_count,) GPU
        dst_slots = old_slots[:keep_count]                # (keep_count,) GPU view

        for lid in range(kv_cache.num_layers):
            kc = kv_cache.k_cache(lid)  # (num_pages, page_size, H, D)
            vc = kv_cache.v_cache(lid)
            k_flat = kc.view(-1, kc.shape[2], kc.shape[3])  # (total_slots, H, D)
            v_flat = vc.view(-1, vc.shape[2], vc.shape[3])
            # Gather to staging buffer (clone prevents aliasing), then scatter
            k_buf = k_flat[src_slots].clone()
            v_buf = v_flat[src_slots].clone()
            k_flat[dst_slots] = k_buf
            v_flat[dst_slots] = v_buf

        # ── 4. Free trailing pages ────────────────────────────────────────────
        # Pages 0..keep_pages-1 are retained; pages keep_pages..end are freed.
        # We pass old_slots[keep_pages*page_size:] so _free's [::page_size]
        # extracts page-start indices correctly.
        keep_pages = math.ceil(keep_count / cache_manager.page_size)
        free_start = keep_pages * cache_manager.page_size
        if free_start < seq_len:
            # clone because old_slots is a view into page_table (will be stale after
            # page_table updates by other requests in the same lazy_free_region)
            cache_manager._free(old_slots[free_start:].clone())

        # ── 5. Update request metadata ────────────────────────────────────────
        req.cached_len = keep_count
        req.device_len = keep_count + 1
        req.max_device_len = keep_count + req.output_len
        # input_ids after append_host: [tok_0, ..., tok_{input_len-1}, first_generated]
        # Move keep_indices to CPU once for CPU-side input_ids indexing.
        keep_indices_cpu = keep_indices.cpu()
        req.input_ids = torch.cat([req.input_ids[keep_indices_cpu], req.input_ids[-1:]])
