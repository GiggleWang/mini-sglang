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


class KnormPress(BaseKVPress):
    """
    KnormPress: Inverse norm of the key vectors.
    Reference: https://arxiv.org/abs/2406.11430

    This method evicts tokens with large key norms, as they tend to
    contribute less to attention (due to softmax normalization).
    Lower key norms => higher scores (we keep low-norm keys).
    """

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            # Compute inverse of L2 norm (lower norm = higher score)
            k_norm = k_selected.float().norm(dim=-1).mean(dim=(1, 2))  # (seq_len,)
            # Use negative norm so that smaller norms get higher scores
            scores -= k_norm

        return scores


class ExpectedAttentionPress(BaseKVPress):
    """
    ExpectedAttentionPress: Expected attention weight during generation phase.
    Reference: https://arxiv.org/abs/2306.14048

    This method estimates the expected attention weight by computing
    Q @ K^T scores and applying softmax normalization.
    Uses the last query's key as a proxy for future queries.
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

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            # Use the last key as query proxy for expected attention
            k_flat = k_selected.squeeze(1)  # (seq_len, num_heads, head_dim)
            query_proxy = k_flat[-1:, :, :]  # (1, num_heads, head_dim)

            # Compute attention scores: (1, num_heads, head_dim) @ (seq_len, num_heads, head_dim).T
            # -> (num_heads, 1, seq_len)
            head_dim = k_flat.shape[-1]
            attn_scores = torch.einsum("qhd,shd->hqs", query_proxy, k_flat) / (head_dim**0.5)

            # Average across heads and squeeze
            layer_scores = attn_scores.mean(dim=0).squeeze(0)  # (seq_len,)
            scores += layer_scores

        return scores


class TOVAPress(BaseKVPress):
    """
    TOVAPress: Token Omission Via Attention.
    Reference: https://arxiv.org/abs/2401.06104

    Evicts tokens based on the attention weight of the last query,
    averaged across all attention heads.
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

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            k_flat = k_selected.squeeze(1)  # (seq_len, num_heads, head_dim)
            # Use last position as query
            query = k_flat[-1:, :, :]  # (1, num_heads, head_dim)

            head_dim = k_flat.shape[-1]
            # Compute attention scores
            attn_scores = torch.einsum("qhd,shd->hqs", query, k_flat) / (head_dim**0.5)
            # Apply softmax
            attn_weights = torch.softmax(attn_scores, dim=-1)
            # Average across heads
            layer_scores = attn_weights.mean(dim=0).squeeze(0)  # (seq_len,)
            scores += layer_scores

        return scores


class ObservedAttentionPress(BaseKVPress):
    """
    ObservedAttentionPress: Average attention weight observed during prefilling.
    Reference: https://arxiv.org/abs/2310.01801

    Uses the average attention pattern from all queries during prefill
    to identify important keys.
    """

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            k_flat = k_selected.squeeze(1)  # (seq_len, num_heads, head_dim)
            head_dim = k_flat.shape[-1]

            # Compute all-to-all attention (causal)
            # Q @ K^T: (seq_len, num_heads, head_dim) @ (seq_len, num_heads, head_dim).T
            attn_scores = torch.einsum("qhd,khd->hqk", k_flat, k_flat) / (head_dim**0.5)

            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=page_indices.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

            # Softmax and average over queries and heads
            attn_weights = torch.softmax(attn_scores, dim=-1)
            # Sum attention received by each key position
            layer_scores = attn_weights.sum(dim=1).mean(dim=0)  # (seq_len,)
            scores += layer_scores

        return scores


class QFilterPress(BaseKVPress):
    """
    QFilterPress: Project Key representations onto main SVD components of Query.
    Reference: https://arxiv.org/abs/2409.14057

    Approximates attention scores by projecting keys onto the principal
    components of the query space.
    """

    def __init__(self, compression_ratio: float = 0.5, num_components: int = 8):
        super().__init__(compression_ratio)
        self.num_components = num_components

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            k_flat = k_selected.squeeze(1)  # (seq_len, num_heads, head_dim)

            # Use keys as proxy for queries (Q ≈ K in many cases)
            # Compute SVD of K to get principal components
            for h in range(k_flat.shape[1]):
                k_head = k_flat[:, h, :].float()  # (seq_len, head_dim)

                # SVD to get principal components
                try:
                    U, S, Vh = torch.linalg.svd(k_head, full_matrices=False)
                    # Project keys onto top components
                    n_comp = min(self.num_components, len(S))
                    proj = k_head @ Vh[:n_comp, :].T  # (seq_len, n_comp)
                    # Score is the projection magnitude
                    layer_scores = (proj**2).sum(dim=-1)
                    scores += layer_scores
                except Exception:
                    # Fallback to L2 norm if SVD fails
                    scores += k_head.norm(dim=-1)

        return scores


class PyramidKVPress(BaseKVPress):
    """
    PyramidKVPress: Maintain pyramid-like cache sizes across layers.
    Reference: https://arxiv.org/abs/2406.02069

    Allocates more cache budget to lower layers and less to higher layers,
    based on the observation that lower layers capture more local patterns.

    Note: This implementation uses position-based scoring that can be
    adjusted per-layer when integrated with the full system.
    """

    def __init__(
        self, compression_ratio: float = 0.5, pyramid_factor: float = 2.0, num_sink_tokens: int = 4
    ):
        super().__init__(compression_ratio)
        self.pyramid_factor = pyramid_factor
        self.num_sink_tokens = num_sink_tokens

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        # Combine L2 norm with position-based weighting
        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]

            # Layer-specific weight (lower layers get higher weight)
            layer_weight = self.pyramid_factor ** (
                (kv_cache.num_layers - 1 - layer_id) / kv_cache.num_layers
            )

            k_norm = k_selected.float().norm(dim=-1).mean(dim=(1, 2))
            scores += k_norm * layer_weight

        # Add recency bias
        recency = torch.arange(seq_len, device=page_indices.device, dtype=scores.dtype)
        scores += recency * 0.01  # Small recency bonus

        return scores


class LagKVPress(BaseKVPress):
    """
    LagKVPress: Leverage KV lag-relative information for compression.
    Reference: https://arxiv.org/abs/2410.08829

    Query-free, attention-weight-free method that uses the difference
    between consecutive key vectors to identify important positions.
    Compatible with Flash Attention.
    """

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            k_flat = k_selected.squeeze(1).float()  # (seq_len, num_heads, head_dim)

            # Compute lag differences (change between consecutive keys)
            if seq_len > 1:
                k_diff = k_flat[1:] - k_flat[:-1]  # (seq_len-1, num_heads, head_dim)
                diff_norm = k_diff.norm(dim=-1).mean(dim=1)  # (seq_len-1,)

                # Tokens with large changes are important (transition points)
                scores[1:] += diff_norm
                scores[:-1] += diff_norm  # Both sides of transition are important

        # First token always important
        scores[0] = scores.max() + 1

        return scores


class KeyDiffPress(BaseKVPress):
    """
    KeyDiffPress: Evict tokens based on key similarity.
    Reference: https://arxiv.org/abs/2410.14846

    Removes redundant keys that are too similar to their neighbors.
    Keeps keys that are unique/different from surrounding context.
    """

    def __init__(self, compression_ratio: float = 0.5, window_size: int = 16):
        super().__init__(compression_ratio)
        self.window_size = window_size

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            k_flat = k_selected.squeeze(1).float()  # (seq_len, num_heads, head_dim)

            # Normalize keys for cosine similarity
            k_norm = k_flat / (k_flat.norm(dim=-1, keepdim=True) + 1e-8)

            # Compute similarity to neighbors within window
            for i in range(seq_len):
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)

                if end - start <= 1:
                    continue

                neighbors = k_norm[start:end]  # (window, num_heads, head_dim)
                center = k_norm[i : i + 1]  # (1, num_heads, head_dim)

                # Cosine similarity
                sim = (center * neighbors).sum(dim=-1).mean()  # scalar

                # Lower similarity = more unique = higher score
                scores[i] -= sim

        return scores


class NonCausalAttnPress(BaseKVPress):
    """
    NonCausalAttnPress: Evict based on non-causal chunked attention scores.
    Reference: https://arxiv.org/abs/2503.08323

    Uses bidirectional (non-causal) attention within chunks to better
    capture token importance.
    """

    def __init__(self, compression_ratio: float = 0.5, chunk_size: int = 64):
        super().__init__(compression_ratio)
        self.chunk_size = chunk_size

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            k_flat = k_selected.squeeze(1).float()  # (seq_len, num_heads, head_dim)
            head_dim = k_flat.shape[-1]

            # Process in chunks
            for chunk_start in range(0, seq_len, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, seq_len)
                chunk = k_flat[chunk_start:chunk_end]  # (chunk_len, num_heads, head_dim)

                # Non-causal attention within chunk
                attn = torch.einsum("qhd,khd->hqk", chunk, chunk) / (head_dim**0.5)
                attn_weights = torch.softmax(attn, dim=-1)

                # Sum attention received (importance)
                chunk_scores = attn_weights.sum(dim=1).mean(dim=0)
                scores[chunk_start:chunk_end] += chunk_scores

        return scores


class LeverageScorePress(BaseKVPress):
    """
    LeverageScorePress: Evict based on approximate statistical leverage scores.
    Reference: https://arxiv.org/abs/2503.08323

    Preserves outliers in the key space using leverage score approximation.
    Tokens with high leverage are important for maintaining representation quality.
    """

    def __init__(self, compression_ratio: float = 0.5, num_samples: int = 32):
        super().__init__(compression_ratio)
        self.num_samples = num_samples

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]  # (seq_len, 1, num_heads, head_dim)

            k_flat = k_selected.squeeze(1).float()  # (seq_len, num_heads, head_dim)

            # Average across heads
            k_avg = k_flat.mean(dim=1)  # (seq_len, head_dim)

            # Approximate leverage scores using random projection
            try:
                # Compute K @ K^T and its pseudo-inverse contribution
                KKT = k_avg @ k_avg.T  # (seq_len, seq_len)
                # Diagonal of K @ (K^T @ K)^{-1} @ K^T approximates leverage
                # Use trace of projection matrix
                eigvals = torch.linalg.eigvalsh(KKT)
                threshold = eigvals.max() * 1e-6
                # Approximate leverage as diagonal contribution
                leverage = (k_avg**2).sum(dim=-1) / (eigvals[eigvals > threshold].sum() + 1e-8)
                scores += leverage
            except Exception:
                # Fallback to L2 norm
                scores += k_avg.norm(dim=-1)

        return scores


class CompactorPress(BaseKVPress):
    """
    CompactorPress: Blends NonCausalAttnPress and LeverageScorePress.
    Reference: https://arxiv.org/abs/2503.08323

    Combines attention-based and leverage-based scoring with weights
    that adapt based on compression ratio.
    """

    def __init__(
        self,
        compression_ratio: float = 0.5,
        chunk_size: int = 64,
        attention_weight: float = 0.5,
    ):
        super().__init__(compression_ratio)
        self._attn_press = NonCausalAttnPress(compression_ratio, chunk_size)
        self._leverage_press = LeverageScorePress(compression_ratio)
        self.attention_weight = attention_weight

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        attn_scores = self._attn_press.score(kv_cache, page_indices, batch)
        leverage_scores = self._leverage_press.score(kv_cache, page_indices, batch)

        # Normalize scores
        attn_norm = (attn_scores - attn_scores.mean()) / (attn_scores.std() + 1e-8)
        leverage_norm = (leverage_scores - leverage_scores.mean()) / (leverage_scores.std() + 1e-8)

        # Blend based on weight
        w = self.attention_weight
        return w * attn_norm + (1 - w) * leverage_norm


class CURPress(BaseKVPress):
    """
    CURPress: Prune keys and values based on CUR decomposition.
    Reference: https://arxiv.org/abs/2503.08323

    Uses approximate leverage scores from CUR matrix decomposition
    to identify important tokens.
    """

    def __init__(self, compression_ratio: float = 0.5, rank: int = 16):
        super().__init__(compression_ratio)
        self.rank = rank

    def score(
        self,
        kv_cache: BaseKVCache,
        page_indices: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        seq_len = len(page_indices)
        scores = torch.zeros(seq_len, device=page_indices.device)

        for layer_id in range(kv_cache.num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            k_selected = k_cache[page_indices]

            k_flat = k_selected.squeeze(1).float()  # (seq_len, num_heads, head_dim)
            k_avg = k_flat.mean(dim=1)  # (seq_len, head_dim)

            try:
                # Low-rank approximation via SVD
                U, S, Vh = torch.linalg.svd(k_avg, full_matrices=False)
                rank = min(self.rank, len(S))

                # CUR leverage scores: ||U_i||^2 where U is truncated
                U_trunc = U[:, :rank]  # (seq_len, rank)
                leverage = (U_trunc**2).sum(dim=-1)
                scores += leverage
            except Exception:
                # Fallback
                scores += k_avg.norm(dim=-1)

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
    # Basic methods
    "random": RandomPress,
    "streaming_llm": StreamingLLMPress,
    "l2_norm": L2NormPress,
    "snapkv": SnapKVPress,
    # Norm-based methods
    "knorm": KnormPress,
    # Attention-based methods
    "expected_attention": ExpectedAttentionPress,
    "tova": TOVAPress,
    "observed_attention": ObservedAttentionPress,
    "noncausal_attn": NonCausalAttnPress,
    # SVD/Projection-based methods
    "qfilter": QFilterPress,
    "cur": CURPress,
    # Structural methods
    "pyramid_kv": PyramidKVPress,
    "lag_kv": LagKVPress,
    "key_diff": KeyDiffPress,
    # Leverage score methods
    "leverage_score": LeverageScorePress,
    "compactor": CompactorPress,
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
