"""KV cache token scoring strategies for post-prefill compression.

Each compressor implements select_tokens() which returns sorted indices of
tokens to KEEP after compression. Supports 6 methods:
  - streaming_llm : position-based sinks + local window
  - knorm         : keep highest-norm key vectors
  - keydiff       : keep most-unique keys (farthest from mean)
  - lagkv         : partition-based relative variance scoring
  - expected_attention : attention-like scoring using mean_K as Q proxy
  - think         : channel-weighted key scoring (needs prefill Q)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseKVCompressor(ABC):
    @abstractmethod
    def select_tokens(
        self,
        kv_cache,
        page_table: torch.Tensor,
        table_idx: int,
        seq_len: int,
        keep_count: int,
        prefill_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return keep_count token indices (sorted ascending) to retain.

        Args:
            kv_cache: MHAKVCache instance
            page_table: shape (max_reqs, max_seq_len), maps position → slot
            table_idx: row index for this request in page_table
            seq_len: number of valid prefill tokens
            keep_count: number of tokens to keep
            prefill_q: optional Q tensor from last attention layer,
                       shape (tokens_in_last_chunk, num_qo_heads, head_dim)
        Returns:
            LongTensor of shape (keep_count,), sorted ascending
        """


# ── Utilities ──────────────────────────────────────────────────────────────────

def _get_slots(page_table: torch.Tensor, table_idx: int, seq_len: int) -> torch.Tensor:
    return page_table[table_idx, :seq_len]  # (seq_len,)


def _iter_k(kv_cache, slots: torch.Tensor):
    """Yield per-token K tensor (seq_len, H, D) for each layer, one at a time."""
    for lid in range(kv_cache.num_layers):
        kc = kv_cache.k_cache(lid)
        k_flat = kc.view(-1, kc.shape[2], kc.shape[3])  # (total_slots, H, D)
        yield k_flat[slots]


def _iter_kv(kv_cache, slots: torch.Tensor):
    """Yield (k, v) tensors (seq_len, H, D) for each layer, one at a time."""
    for lid in range(kv_cache.num_layers):
        kc = kv_cache.k_cache(lid)
        vc = kv_cache.v_cache(lid)
        k_flat = kc.view(-1, kc.shape[2], kc.shape[3])
        v_flat = vc.view(-1, vc.shape[2], vc.shape[3])
        yield k_flat[slots], v_flat[slots]


def _topk_indices(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Return keep_count indices with the highest scores, sorted ascending."""
    _, idx = scores.topk(keep_count, largest=True, sorted=False)
    return idx.sort().values


# ── StreamingLLM ───────────────────────────────────────────────────────────────

class StreamingLLMCompressor(BaseKVCompressor):
    """Keep first n_sink tokens (attention sinks) + last n_local tokens.

    Tokens outside these windows are evicted. No KV reads needed.
    Reference: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
    """

    def __init__(self, n_sink: int = 4, n_local: int = 1024):
        self.n_sink = n_sink
        self.n_local = n_local

    def select_tokens(
        self, kv_cache, page_table, table_idx, seq_len, keep_count, prefill_q=None
    ) -> torch.Tensor:
        scores = torch.full((seq_len,), -1.0)
        scores[: self.n_sink] = 1.0  # attention sinks: always keep
        scores[max(self.n_sink, seq_len - self.n_local) :] = 1.0  # recent window
        return _topk_indices(scores, keep_count)


# ── KNorm ──────────────────────────────────────────────────────────────────────

class KNormCompressor(BaseKVCompressor):
    """Score = mean L2 norm of K across all layers and KV heads.

    Tokens with large-magnitude key vectors carry more information.
    Reference: Devoto et al., "A Simple and Effective L2 Norm-Based Strategy
    for KV Cache Compression"
    """

    def select_tokens(
        self, kv_cache, page_table, table_idx, seq_len, keep_count, prefill_q=None
    ) -> torch.Tensor:
        slots = _get_slots(page_table, table_idx, seq_len)
        scores = torch.zeros(seq_len, device=slots.device)
        for k in _iter_k(kv_cache, slots):
            # k: (S, H, D) → norm over D → (S, H) → mean over H → (S,)
            scores += k.norm(dim=-1).mean(dim=-1)
        scores /= kv_cache.num_layers
        return _topk_indices(scores, keep_count)


# ── KeyDiff ────────────────────────────────────────────────────────────────────

class KeyDiffCompressor(BaseKVCompressor):
    """Score = -cosine_similarity(K, anchor) where anchor = mean(normalize(K)).

    Keeps tokens whose keys are most DIFFERENT from the average pattern,
    maximising diversity of the retained KV cache.
    Reference: Cai et al., "KeyDiff: Enhancing KV Cache Compression via
    Key Difference"
    """

    def select_tokens(
        self, kv_cache, page_table, table_idx, seq_len, keep_count, prefill_q=None
    ) -> torch.Tensor:
        slots = _get_slots(page_table, table_idx, seq_len)
        scores = torch.zeros(seq_len, device=slots.device)
        for k in _iter_k(kv_cache, slots):
            # k: (S, H, D)
            k_norm = F.normalize(k, p=2, dim=-1)              # (S, H, D)
            anchor = k_norm.mean(dim=0, keepdim=True)          # (1, H, D)
            sim = (k_norm * anchor).sum(dim=-1)                # (S, H)
            scores += -sim.mean(dim=-1)                        # (S,) more negative = more unique
        scores /= kv_cache.num_layers
        return _topk_indices(scores, keep_count)


# ── LagKV ──────────────────────────────────────────────────────────────────────

class LagKVCompressor(BaseKVCompressor):
    """Partition-based scoring: std(normalize(K, next_partition_range)).

    Tokens whose keys deviate from the next partition's value range are
    considered more informative. The last partition and sinks are always kept.
    Reference: Fu et al., "LazyKV: Lazy KV Cache Eviction for Efficient LLM"
    """

    def __init__(self, lag_size: int = 128, n_sink: int = 4):
        self.lag_size = lag_size
        self.n_sink = n_sink

    def _partition_score(self, k_seq: torch.Tensor) -> torch.Tensor:
        """Compute per-token importance scores for one layer.

        Args:
            k_seq: (S, H, D)
        Returns:
            scores: (S,)
        """
        S = k_seq.shape[0]
        lag = self.lag_size
        scores = torch.zeros(S, device=k_seq.device)
        scores[: self.n_sink] = float("inf")  # always keep sinks

        rest_start = self.n_sink
        rest_len = S - rest_start
        n_full = rest_len // lag
        if n_full < 2:
            # Not enough partitions to compare — keep everything after sinks
            scores[rest_start:] = float("inf")
            return scores

        for i in range(n_full - 1):
            s = rest_start + i * lag
            e = s + lag
            ns, ne = e, e + lag
            v_part = k_seq[s:e]        # (lag, H, D) current partition
            v_ref = k_seq[ns:ne]       # (lag, H, D) next partition as reference
            min_r = v_ref.amin(dim=0)  # (H, D)
            max_r = v_ref.amax(dim=0)  # (H, D)
            normalized = (v_part - min_r) / (max_r - min_r + 1e-8)  # (lag, H, D)
            # std over channels, mean over heads
            part_score = normalized.std(dim=-1).mean(dim=-1)          # (lag,)
            scores[s:e] = part_score

        # Last full partition and any remainder: keep with high score
        scores[rest_start + (n_full - 1) * lag :] = float("inf")
        return scores

    def select_tokens(
        self, kv_cache, page_table, table_idx, seq_len, keep_count, prefill_q=None
    ) -> torch.Tensor:
        slots = _get_slots(page_table, table_idx, seq_len)
        scores = torch.zeros(seq_len, device=slots.device)
        for k in _iter_k(kv_cache, slots):
            scores += self._partition_score(k)
        scores /= kv_cache.num_layers
        return _topk_indices(scores, keep_count)


# ── ExpectedAttention ──────────────────────────────────────────────────────────

class ExpectedAttentionCompressor(BaseKVCompressor):
    """Approximate expected attention score using mean_K as a Q proxy.

    score_j = softmax_over_tokens(K_j · mean_K^T / √d) * ||V_j||₂

    Using the mean key as a proxy for the expected future query captures
    tokens that would receive high attention from a "typical" future query.
    Averaged across all layers and KV heads.
    Reference: Duquennoy et al., "ExpectedAttention: Estimating the
    expected attention for KV cache compression" (adapted, Q-free version)
    """

    def select_tokens(
        self, kv_cache, page_table, table_idx, seq_len, keep_count, prefill_q=None
    ) -> torch.Tensor:
        slots = _get_slots(page_table, table_idx, seq_len)
        scores = torch.zeros(seq_len, device=slots.device)
        for k, v in _iter_kv(kv_cache, slots):
            # k, v: (S, H, D)
            D = k.shape[-1]
            mean_k = k.mean(dim=0, keepdim=True)                       # (1, H, D)
            attn_logit = (k * mean_k).sum(dim=-1) / math.sqrt(D)      # (S, H)
            attn_score = attn_logit.softmax(dim=0)                     # (S, H)
            vnorm = v.norm(dim=-1)                                     # (S, H)
            scores += (attn_score * vnorm).mean(dim=-1)                # (S,)
        scores /= kv_cache.num_layers
        return _topk_indices(scores, keep_count)


# ── ThinkPress (token-level adaptation) ────────────────────────────────────────

class ThinkPressCompressor(BaseKVCompressor):
    """Channel-weighted key scoring: (mean_Q² ⊙ K²).sum(channel).

    Rewards tokens whose key channels align with the most "active" query
    channels in the last prefill window. Averaged across all layers and heads.
    Falls back to KNorm when prefill_q is not available.
    Reference: Adapted from ThinKPress (Tang et al., "ThinK: Thinner Key
    Cache by Query-Driven Pruning") to token-level selection.
    """

    def select_tokens(
        self, kv_cache, page_table, table_idx, seq_len, keep_count, prefill_q=None
    ) -> torch.Tensor:
        slots = _get_slots(page_table, table_idx, seq_len)
        scores = torch.zeros(seq_len, device=slots.device)

        if prefill_q is None:
            # Fallback: KNorm (no Q available)
            for k in _iter_k(kv_cache, slots):
                scores += k.norm(dim=-1).mean(dim=-1)
            scores /= kv_cache.num_layers
            return _topk_indices(scores, keep_count)

        # prefill_q: (tokens_in_last_chunk, Hq, D)
        # Read Hk and D directly from KV cache shape (no data copy)
        kc0 = kv_cache.k_cache(0)
        Hk, D = kc0.shape[2], kc0.shape[3]

        Hq = prefill_q.shape[1]
        q_per_kv = max(1, Hq // Hk)
        mean_q = prefill_q.mean(dim=0)  # (Hq, D)
        if q_per_kv > 1:
            mean_q = mean_q.view(Hk, q_per_kv, D).mean(dim=1)  # (Hk, D)
        q_importance = mean_q.pow(2)  # (Hk, D) — channel importance weights

        for k in _iter_k(kv_cache, slots):
            # k: (S, Hk, D)
            # score_j = mean_heads( sum_channels(q_importance * K_j²) )
            token_score = (q_importance.unsqueeze(0) * k.pow(2)).sum(dim=-1).mean(dim=-1)
            scores += token_score  # (S,)
        scores /= kv_cache.num_layers
        return _topk_indices(scores, keep_count)


# ── Factory ────────────────────────────────────────────────────────────────────

_VALID_METHODS = frozenset(
    ["none", "streaming_llm", "knorm", "keydiff", "lagkv", "expected_attention", "think"]
)


def create_compressor(method: str, **kwargs) -> BaseKVCompressor | None:
    """Create a KV cache compressor by name.

    Args:
        method: one of "none", "streaming_llm", "knorm", "keydiff",
                "lagkv", "expected_attention", "think"
        **kwargs: method-specific parameters:
            n_sink (int): streaming_llm/lagkv — number of sink tokens (default 4)
            n_local (int): streaming_llm — local window size (default 1024)
            lag_size (int): lagkv — partition size (default 128)
    Returns:
        A BaseKVCompressor instance, or None if method == "none"
    """
    if method == "none":
        return None
    if method == "streaming_llm":
        return StreamingLLMCompressor(
            n_sink=kwargs.get("n_sink", 4),
            n_local=kwargs.get("n_local", 1024),
        )
    if method == "knorm":
        return KNormCompressor()
    if method == "keydiff":
        return KeyDiffCompressor()
    if method == "lagkv":
        return LagKVCompressor(
            lag_size=kwargs.get("lag_size", 128),
            n_sink=kwargs.get("n_sink", 4),
        )
    if method == "expected_attention":
        return ExpectedAttentionCompressor()
    if method == "think":
        return ThinkPressCompressor()
    raise ValueError(
        f"Unknown kv_compression_method: {method!r}. "
        f"Valid choices: {sorted(_VALID_METHODS)}"
    )
