import math

from sglang.srt.mem_cache.online_prefill_cost import (
    LatestTransferSamples,
    allocate_preload_pages,
    OnlinePrefillCostSynopsis,
    TransferSample,
    select_batch_split,
)


def test_bucket_boundaries_and_weighted_estimate():
    synopsis = OnlinePrefillCostSynopsis(512)
    assert synopsis.bucket_key(128, 0) == (0, 0)
    assert synopsis.bucket_key(129, 0) == (1, 0)
    assert synopsis.bucket_key(95, 5) == (0, 0)  # exactly 5%
    assert synopsis.bucket_key(94, 6) == (0, 1)
    synopsis.add(64, 0, 0.064)
    synopsis.add(128, 0, 0.256)
    assert math.isclose(synopsis.estimate(96, 0), 0.16)


def test_ring_keeps_eight_recent_samples():
    synopsis = OnlinePrefillCostSynopsis(128)
    for i in range(1, 10):
        synopsis.add(128, 0, i * 0.128)
    assert math.isclose(synopsis.estimate(128, 0), sum(range(2, 10)) / 8 * 0.128)


def test_transfer_samples_are_path_local_and_ordered():
    samples = LatestTransferSamples()
    assert samples.update("full_block", TransferSample(2, 20, 2))
    assert samples.update("layer_wise", TransferSample(1, 30, 3))
    assert not samples.update("full_block", TransferSample(1, 100, 1))
    assert samples.full_block.bandwidth_bytes_per_second == 10
    assert samples.layer_wise.bandwidth_bytes_per_second == 10
    assert TransferSample(3, 2_000_000_000, 0.5).bandwidth_gbps == 4.0


def test_split_cold_start_and_all_modes():
    sample = TransferSample(1, 1000, 1)
    assert select_batch_split(0, 100, 1, sample) == (0, 0, "no-host-pages")
    assert select_batch_split(10, 100, None, sample) == (0, 10, "empty-compute-bucket")
    assert select_batch_split(10, 100, 0.5, None) == (
        0,
        10,
        "missing-layer-wise-sample",
    )
    assert select_batch_split(10, 100, 0, sample)[:2] == (10, 0)
    assert select_batch_split(10, 100, 0.5, sample)[:2] == (5, 5)
    assert select_batch_split(10, 100, 2, sample)[:2] == (0, 10)


class _Event:
    clock = 0.0

    def __init__(self, enable_timing=False):
        self.time = None

    def record(self):
        self.time = self.clock

    def query(self):
        return True

    def elapsed_time(self, other):
        return other.time - self.time


def test_compute_recorder_subtracts_only_observed_waits():
    from sglang.srt.mem_cache.online_prefill_cost import PrefillComputeEventRecorder

    synopsis = OnlinePrefillCostSynopsis(128)
    recorder = PrefillComputeEventRecorder(synopsis, _Event)
    recorder.prepare(100, 0)
    recorder.begin_layer()
    _Event.clock = 2.0
    recorder.begin_wait()
    _Event.clock = 5.0
    recorder.end_wait()
    _Event.clock = 13.0
    recorder.end_layer()
    recorder.finish()
    recorder.collect()
    # 13 ms layer span - 3 ms of actual stream waiting = 10 ms pure compute.
    assert math.isclose(synopsis.estimate(100, 0), 0.01)


def test_batch_budget_is_allocated_once_in_root_order():
    assert allocate_preload_pages([2, 3, 4], 6) == [2, 3, 1]
    assert allocate_preload_pages([2, 3], 0) == [0, 0]
