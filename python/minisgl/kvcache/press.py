from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import BaseKVCache, BaseKVPress

if TYPE_CHECKING:
    from minisgl.core import Batch


class RandomPress(BaseKVPress):
    """
    Random KV cache compression for testing/baseline.
    Randomly selects which KV pairs to keep.
    """

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        return torch.rand(seq_len, device=page_indices.device)


class StreamingLLMPress(BaseKVPress):
    """
    StreamingLLM-style compression: keep sink tokens + recent tokens.
    Reference: https://arxiv.org/abs/2309.17453

    This is a simple but effective strategy that keeps:
    1. First few tokens (attention sinks)
    2. Most recent tokens (local context)
    """

    def __init__(self, compression_ratio: float = 0.5, num_sink_tokens: int = 4):
        super().__init__(compression_ratio)
        self.num_sink_tokens = num_sink_tokens

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        # Sink tokens get highest priority
        scores[: self.num_sink_tokens] = float("inf")

        # Recent tokens get linearly increasing scores
        if seq_len > self.num_sink_tokens:
            scores[self.num_sink_tokens :] = torch.arange(
                seq_len - self.num_sink_tokens,
                device=page_indices.device,
                dtype=scores.dtype,
            )

        return scores


class L2NormPress(BaseKVPress):
    """
    L2 norm-based compression: keep KV pairs with larger L2 norms.

    This is based on the observation that KV pairs with larger norms
    tend to have more influence on the attention output.
    """

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        # Compute average L2 norm across all layers
        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)  # (num_pages, 1, num_heads, head_dim)
            v_cache = kv_cache.v_cache(layer_id)

            # Get the KV pairs for this request
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)
            v_selected = v_cache[page_indices]

            # Compute L2 norm for each position
            k_norm = k_selected.float().norm(dim=-1).mean(dim=(1, 2))  # (seq_len,)
            v_norm = v_selected.float().norm(dim=-1).mean(dim=(1, 2))

            scores += k_norm + v_norm

        return scores


class SnapKVPress(BaseKVPress):
    """
    SnapKV-style compression: use attention patterns to identify important KV pairs.
    Reference: https://arxiv.org/abs/2404.14469

    This implementation uses a simplified version that looks at the attention
    pattern from the last few query positions to identify important keys.

    Note: This requires attention weights to be computed, which may not always
    be available. Falls back to L2 norm if attention weights are not available.
    """

    def __init__(
        self,
        compression_ratio: float = 0.5,
        observation_window: int = 16,
        num_sink_tokens: int = 4,
    ):
        super().__init__(compression_ratio)
        self.observation_window = observation_window
        self.num_sink_tokens = num_sink_tokens
        self._fallback = L2NormPress(compression_ratio)

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        # For now, use L2 norm as fallback since we don't have attention weights
        # In a full implementation, you would capture attention weights during forward
        # and use them here to compute importance scores
        return self._fallback.score(kv_cache, page_indices, batch)


# Registry of available press methods
SUPPORTED_PRESS = {
    "random": RandomPress,
    "streaming_llm": StreamingLLMPress,
    "l2_norm": L2NormPress,
    "snapkv": SnapKVPress,
}


def create_kv_press(method: str, compression_ratio: float = 0.5, **kwargs) -> BaseKVPress:
    """
    Create a KV press instance by method name.

    Args:
        method: Name of the compression method.
        compression_ratio: Ratio of KV pairs to keep.
        **kwargs: Additional arguments for the specific press method.

    Returns:
        A BaseKVPress instance.
    """
    if method not in SUPPORTED_PRESS:
        raise ValueError(
            f"Unknown press method: {method}. "
            f"Available methods: {list(SUPPORTED_PRESS.keys())}"
        )
    return SUPPORTED_PRESS[method](compression_ratio=compression_ratio, **kwargs)
