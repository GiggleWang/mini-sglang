from __future__ import annotations

import csv
import time
from pathlib import Path

from minisgl.utils import init_logger

logger = init_logger(__name__)


class IterTimeline:
    """Records wall-clock start/end timestamps for every forward_batch call.

    Each row captures one iteration (prefill or decode) with:
    - iter:         global iteration index (0-based, covers both prefill and decode)
    - type:         "prefill" or "decode"
    - batch_size:   number of requests in the batch
    - num_tokens:   actual token count (= total input tokens for prefill, = batch_size for decode)
    - max_seq_len:  longest KV sequence length in the batch
    - t_start_ms:   wall-clock time since first iteration (ms)
    - t_end_ms:     wall-clock time since first iteration (ms)
    - duration_ms:  t_end_ms - t_start_ms

    Gap between consecutive iterations (= scheduling + Python overhead between steps) can be
    computed as: gap[i] = t_start[i+1] - t_end[i]
    """

    def __init__(self, csv_path: str):
        self._path = Path(csv_path)
        self._t0: float | None = None
        self._iter = 0
        self._file = self._path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["iter", "type", "batch_size", "num_tokens", "max_seq_len",
             "t_start_ms", "t_end_ms", "duration_ms"]
        )
        logger.info(f"IterTimeline enabled: {csv_path}")

    def record(self, batch, t_start: float, t_end: float) -> None:
        if self._t0 is None:
            self._t0 = t_start
        t0 = self._t0
        num_tokens = len(batch.input_ids)
        max_seq_len = max((req.device_len for req in batch.reqs), default=0)
        self._writer.writerow([
            self._iter,
            "prefill" if batch.is_prefill else "decode",
            batch.size,
            num_tokens,
            max_seq_len,
            round((t_start - t0) * 1000, 3),
            round((t_end - t0) * 1000, 3),
            round((t_end - t_start) * 1000, 3),
        ])
        self._file.flush()
        self._iter += 1

    def close(self) -> None:
        self._file.close()


_GLOBAL_ITER_TIMELINE: IterTimeline | None = None


def get_iter_timeline() -> IterTimeline | None:
    return _GLOBAL_ITER_TIMELINE


def set_iter_timeline(timeline: IterTimeline) -> None:
    global _GLOBAL_ITER_TIMELINE
    _GLOBAL_ITER_TIMELINE = timeline
