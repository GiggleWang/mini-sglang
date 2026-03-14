from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from minisgl.utils import init_logger

logger = init_logger(__name__)


@dataclass
class _TimingStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = float("-inf")

    def update(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        if elapsed_ms < self.min_ms:
            self.min_ms = elapsed_ms
        if elapsed_ms > self.max_ms:
            self.max_ms = elapsed_ms

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


class DecodeTimer:
    def __init__(self, timing_csv: str, metadata_csv: str, interval: int):
        self.timing_csv = timing_csv
        self.metadata_csv = metadata_csv
        self.interval = interval

        self._stats: Dict[str, _TimingStats] = {}
        self._pending_events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_metadata: List[dict] = []
        self._iter_count = 0
        self._timing_header_written = os.path.exists(timing_csv)
        self._metadata_header_written = os.path.exists(metadata_csv)

        logger.info(
            f"DecodeTimer enabled: timing={timing_csv}, metadata={metadata_csv},"
            f" interval={interval}"
        )

    def start_module(self, name: str) -> torch.cuda.Event:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def end_module(self, name: str, start: torch.cuda.Event) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending_events.append((name, start, end))

    def record_batch_metadata(self, batch) -> None:
        seq_lens = [req.device_len for req in batch.reqs]
        self._pending_metadata.append(
            {
                "iter": self._iter_count,
                "is_decode": not batch.is_prefill,
                "batch_size": batch.size,
                "num_tokens": len(batch.input_ids),
                "max_seq_len": max(seq_lens) if seq_lens else 0,
            }
        )

    def on_iteration_end(self) -> None:
        if self._pending_events:
            torch.cuda.synchronize()
            for name, start, end in self._pending_events:
                elapsed = start.elapsed_time(end)
                if name not in self._stats:
                    self._stats[name] = _TimingStats()
                self._stats[name].update(elapsed)
            self._pending_events.clear()

        self._iter_count += 1

        if self._iter_count % self.interval == 0:
            self._flush()

    def _flush(self) -> None:
        iter_end = self._iter_count

        # Write timing CSV
        timing_rows = [
            {
                "iter_end": iter_end,
                "module": name,
                "count": s.count,
                "total_ms": f"{s.total_ms:.4f}",
                "mean_ms": f"{s.mean_ms:.4f}",
                "min_ms": f"{s.min_ms:.4f}",
                "max_ms": f"{s.max_ms:.4f}",
            }
            for name, s in sorted(self._stats.items())
        ]
        if timing_rows:
            _append_csv(
                self.timing_csv,
                timing_rows,
                fieldnames=["iter_end", "module", "count", "total_ms", "mean_ms", "min_ms", "max_ms"],
                write_header=not self._timing_header_written,
            )
            self._timing_header_written = True

        # Write metadata CSV
        if self._pending_metadata:
            _append_csv(
                self.metadata_csv,
                self._pending_metadata,
                fieldnames=["iter", "is_decode", "batch_size", "num_tokens", "max_seq_len"],
                write_header=not self._metadata_header_written,
            )
            self._metadata_header_written = True

        logger.info(f"DecodeTimer flushed at iter {iter_end}")
        self._stats.clear()
        self._pending_metadata.clear()


def _append_csv(path: str, rows: list, fieldnames: list, write_header: bool) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


_GLOBAL_DECODE_TIMER: DecodeTimer | None = None


def get_decode_timer() -> DecodeTimer | None:
    return _GLOBAL_DECODE_TIMER


def set_decode_timer(timer: DecodeTimer) -> None:
    global _GLOBAL_DECODE_TIMER
    _GLOBAL_DECODE_TIMER = timer
