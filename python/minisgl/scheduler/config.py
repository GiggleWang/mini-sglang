from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False

    # KV cache compression applied after prefill, before decode
    kv_compression_method: str = "none"      # "none"|"streaming_llm"|"knorm"|"keydiff"|
                                             # "lagkv"|"expected_attention"|"think"
    kv_compression_keep_ratio: float = 0.5  # fraction of tokens to retain
    kv_compression_n_sink: int = 4          # streaming_llm/lagkv: number of sink tokens
    kv_compression_n_local: int = 1024      # streaming_llm: local window size
    kv_compression_lag_size: int = 128      # lagkv: partition size

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
