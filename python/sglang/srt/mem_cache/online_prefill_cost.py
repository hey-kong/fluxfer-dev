"""Online batch-level cost summaries for hybrid HiCache loading."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

LENGTH_BUCKET_TOKENS = 128
REUSE_BUCKET_WIDTH = 0.05
SAMPLES_PER_BUCKET = 8
ONLINE_PREFILL_COST_ARCHITECTURES = frozenset(
    {
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "Mistral3ForConditionalGeneration",
    }
)


def supports_online_prefill_cost(architecture: str) -> bool:
    return architecture in ONLINE_PREFILL_COST_ARCHITECTURES


class _SampleRing:
    def __init__(self) -> None:
        self.q = [0] * SAMPLES_PER_BUCKET
        self.t = [0.0] * SAMPLES_PER_BUCKET
        self.size = 0
        self.next = 0
        self.sum_q = 0
        self.sum_t = 0.0

    def add(self, q: int, average_layer_seconds: float) -> None:
        if (
            q <= 0
            or not math.isfinite(average_layer_seconds)
            or average_layer_seconds <= 0
        ):
            return
        if self.size == SAMPLES_PER_BUCKET:
            self.sum_q -= self.q[self.next]
            self.sum_t -= self.t[self.next]
        else:
            self.size += 1
        self.q[self.next] = q
        self.t[self.next] = average_layer_seconds
        self.sum_q += q
        self.sum_t += average_layer_seconds
        self.next = (self.next + 1) % SAMPLES_PER_BUCKET

    def estimate(self, q: int) -> Optional[float]:
        if self.sum_q == 0:
            return None
        return self.sum_t / self.sum_q * q


class OnlinePrefillCostSynopsis:
    """Two-dimensional (new-token, reuse-ratio) rolling synopsis."""

    def __init__(self, max_batch_tokens: int) -> None:
        self.max_length_bucket = max(0, (max_batch_tokens - 1) // LENGTH_BUCKET_TOKENS)
        self._buckets: dict[tuple[int, int], _SampleRing] = {}

    def bucket_key(self, q: int, h: int) -> tuple[int, int]:
        if q <= 0 or h < 0:
            raise ValueError("Q must be positive and H must be non-negative")
        length_bucket = min((q - 1) // LENGTH_BUCKET_TOKENS, self.max_length_bucket)
        reuse = h / (h + q)
        reuse_bucket = (
            0 if reuse == 0 else min(math.ceil(reuse / REUSE_BUCKET_WIDTH) - 1, 19)
        )
        return length_bucket, reuse_bucket

    def add(self, q: int, h: int, average_layer_seconds: float) -> None:
        key = self.bucket_key(q, h)
        self._buckets.setdefault(key, _SampleRing()).add(q, average_layer_seconds)

    def estimate(self, q: int, h: int) -> Optional[float]:
        bucket = self._buckets.get(self.bucket_key(q, h))
        return None if bucket is None else bucket.estimate(q)


@dataclass(frozen=True)
class TransferSample:
    sequence: int
    byte_count: int
    elapsed_seconds: float

    @property
    def bandwidth_bytes_per_second(self) -> float:
        return self.byte_count / self.elapsed_seconds

    @property
    def bandwidth_gbps(self) -> float:
        return self.bandwidth_bytes_per_second / 1e9


class LatestTransferSamples:
    def __init__(self) -> None:
        self.full_block: Optional[TransferSample] = None
        self.layer_wise: Optional[TransferSample] = None

    def update(self, path: str, sample: TransferSample) -> bool:
        if (
            sample.byte_count <= 0
            or sample.elapsed_seconds <= 0
            or not math.isfinite(sample.elapsed_seconds)
        ):
            return False
        current = getattr(self, path)
        if current is not None and sample.sequence <= current.sequence:
            return False
        setattr(self, path, sample)
        return True


def select_fixed_ratio_split(
    host_pages: int, page_size: int, compute_tokens: int, ratio: float = 4.0
) -> tuple[int, int]:
    layer_pages = min(host_pages, math.floor(ratio * compute_tokens / page_size))
    return host_pages - layer_pages, layer_pages


def select_batch_split(
    host_pages: int,
    bytes_per_page_per_layer: int,
    compute_seconds: Optional[float],
    layer_wise_sample: Optional[TransferSample],
) -> tuple[int, int, Optional[str]]:
    """Return (full-block pages, layer-wise pages, cold-start reason)."""
    if host_pages <= 0:
        return 0, 0, "no-host-pages"
    if compute_seconds is None:
        return 0, host_pages, "empty-compute-bucket"
    if layer_wise_sample is None:
        return 0, host_pages, "missing-layer-wise-sample"
    covered = math.floor(
        layer_wise_sample.bandwidth_bytes_per_second
        * compute_seconds
        / bytes_per_page_per_layer
    )
    layer_pages = min(host_pages, max(covered, 0))
    return host_pages - layer_pages, layer_pages, None


@dataclass
class _PendingCompute:
    sequence: int
    q: int
    h: int
    layer_events: list[tuple[object, object]]
    wait_events: list[tuple[object, object]]
    update_synopsis: bool = True


class PrefillComputeEventRecorder:
    """Records pure layer time and consumes completed CUDA events asynchronously."""

    def __init__(
        self, synopsis: OnlinePrefillCostSynopsis, event_factory, finish_callback=None
    ) -> None:
        self.synopsis = synopsis
        self.event_factory = event_factory
        self.finish_callback = finish_callback
        self.active: Optional[_PendingCompute] = None
        self.prepared = deque()
        self.pending: list[_PendingCompute] = []

    def prepare(self, sequence: int, q: int, h: int) -> None:
        self.collect()
        if q > 0:
            self.prepared.append(_PendingCompute(sequence, q, h, [], []))

    def activate_next(self) -> None:
        if self.active is None and self.prepared:
            self.active = self.prepared.popleft()

    def begin_layer(self) -> None:
        if self.active is None:
            return
        event = self.event_factory(enable_timing=True)
        event.record()
        self.active.layer_events.append((event, None))

    def end_layer(self) -> None:
        if self.active is None or not self.active.layer_events:
            return
        event = self.event_factory(enable_timing=True)
        event.record()
        start, _ = self.active.layer_events[-1]
        self.active.layer_events[-1] = (start, event)

    def begin_wait(self) -> None:
        if self.active is None:
            return
        event = self.event_factory(enable_timing=True)
        event.record()
        self.active.wait_events.append((event, None))

    def end_wait(self) -> None:
        if self.active is None or not self.active.wait_events:
            return
        event = self.event_factory(enable_timing=True)
        event.record()
        start, _ = self.active.wait_events[-1]
        self.active.wait_events[-1] = (start, event)

    def finish(self, update_synopsis: bool = True) -> None:
        if self.active is not None and self.active.layer_events:
            self.active.update_synopsis = update_synopsis
            self.pending.append(self.active)
            if self.finish_callback is not None:
                self.finish_callback(
                    self.active.sequence, self.active.layer_events[-1][1]
                )
        self.active = None

    def discard(self) -> None:
        self.active = None

    def collect(self) -> None:
        remaining = []
        for item in self.pending:
            ends = [end for _, end in item.layer_events + item.wait_events]
            if any(end is None or not end.query() for end in ends):
                remaining.append(item)
                continue
            layer_seconds = (
                sum(start.elapsed_time(end) for start, end in item.layer_events)
                / 1000.0
            )
            wait_seconds = (
                sum(start.elapsed_time(end) for start, end in item.wait_events) / 1000.0
            )
            pure_seconds = max(layer_seconds - wait_seconds, 0.0)
            if pure_seconds > 0 and item.update_synopsis:
                self.synopsis.add(item.q, item.h, pure_seconds / len(item.layer_events))
        self.pending = remaining


def allocate_preload_pages(
    operation_page_counts: list[int], full_pages: int
) -> list[int]:
    """Allocate one batch budget in operation/root order without over-allocation."""
    if full_pages < 0 or full_pages > sum(operation_page_counts):
        raise ValueError("full-block budget is outside the batch page range")
    result = []
    remaining = full_pages
    for page_count in operation_page_counts:
        assigned = min(max(page_count, 0), remaining)
        result.append(assigned)
        remaining -= assigned
    assert remaining == 0
    return result
