"""Unit tests for HiCache hybrid Balanced Batch Formation helpers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import (  # noqa: E402
    HICACHE_HYBRID_BBF_LOADING_BOUND_RATIO,
    HybridBalancedPrefillState,
    Scheduler,
)

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _Node:
    def __init__(self, node_id, parent=None, host_len=0, on_device=True):
        self.id = node_id
        self.parent = parent
        self.host_value = torch.arange(host_len, dtype=torch.int64) if host_len else None
        self.value = (
            torch.arange(host_len, dtype=torch.int64)
            if on_device and host_len
            else None
        )


def _req(**kwargs):
    defaults = dict(
        rid="req",
        host_hit_length=0,
        extend_input_len=0,
        best_match_node=None,
        last_node=None,
        time_stats=SimpleNamespace(wait_queue_entry_time=0.0),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestHybridBalancedPrefillHelpers(CustomTestCase):
    def _scheduler(self, root):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = True
        scheduler.tree_cache = SimpleNamespace(
            root_node=root,
            cache_controller=SimpleNamespace(io_backend="hybrid"),
        )
        scheduler.server_args = SimpleNamespace(
            enable_hybrid_balanced_batch=True,
            enable_hybrid_bubble_filling=True,
        )
        return scheduler

    def test_enabled_only_for_hybrid_hicache_with_flag(self):
        scheduler = self._scheduler(_Node(0))
        self.assertTrue(scheduler._is_hybrid_balanced_prefill_enabled())

        scheduler.server_args.enable_hybrid_balanced_batch = False
        self.assertFalse(scheduler._is_hybrid_balanced_prefill_enabled())

        scheduler.server_args.enable_hybrid_balanced_batch = True
        scheduler.tree_cache.cache_controller.io_backend = "direct"
        self.assertFalse(scheduler._is_hybrid_balanced_prefill_enabled())


    def test_bubble_filling_requires_flag_and_unfinished_preload(self):
        root = _Node(0)
        scheduler = self._scheduler(root)

        class _FinishEvent:
            def __init__(self, ready):
                self.ready = ready

            def query(self):
                return self.ready

        def _batch(ready):
            scheduler.tree_cache.cache_controller.layer_done_counter = SimpleNamespace(
                events=[SimpleNamespace(finish_event=_FinishEvent(ready))]
            )
            return SimpleNamespace(
                hicache_consumer_index=0,
                hybrid_bubble_needs_preload=True,
            )

        self.assertTrue(scheduler._should_hybrid_bubble_fill_prefill(_batch(False)))
        self.assertFalse(scheduler._should_hybrid_bubble_fill_prefill(_batch(True)))

        scheduler.server_args.enable_hybrid_bubble_filling = False
        self.assertFalse(scheduler._should_hybrid_bubble_fill_prefill(_batch(False)))

    def test_bubble_filling_detects_preload_pages_in_load_queue(self):
        scheduler = self._scheduler(_Node(0))
        scheduler.tree_cache.cache_controller.load_queue = [
            SimpleNamespace(h2d_preload_pages=0),
            SimpleNamespace(h2d_preload_pages=2),
        ]

        self.assertTrue(scheduler._hybrid_prefill_load_queue_has_preload_pages())

        scheduler.tree_cache.cache_controller.load_queue = [
            SimpleNamespace(h2d_preload_pages=0)
        ]
        self.assertFalse(scheduler._hybrid_prefill_load_queue_has_preload_pages())

    def test_load_segments_use_host_node_ids_for_dedup(self):
        root = _Node(0)
        device = _Node(1, parent=root, host_len=4, on_device=True)
        host_a = _Node(2, parent=device, host_len=5, on_device=False)
        host_b = _Node(3, parent=host_a, host_len=7, on_device=False)
        scheduler = self._scheduler(root)
        req = _req(
            rid="shared",
            host_hit_length=12,
            extend_input_len=15,
            best_match_node=host_b,
            last_node=device,
        )

        segments = scheduler._hybrid_bbf_load_segments(req)

        self.assertEqual(segments, [(("_Node", 2), 5), (("_Node", 3), 7)])

    def test_ratio_estimate_counts_only_extra_load_segments(self):
        root = _Node(0)
        device = _Node(1, parent=root, host_len=4, on_device=True)
        host_a = _Node(2, parent=device, host_len=5, on_device=False)
        host_b = _Node(3, parent=host_a, host_len=7, on_device=False)
        scheduler = self._scheduler(root)
        req = _req(
            rid="shared",
            host_hit_length=12,
            extend_input_len=15,
            best_match_node=host_b,
            last_node=device,
        )
        state = HybridBalancedPrefillState(load_tokens=5, compute_tokens=1)
        state.loaded_segment_ids.add(("_Node", 2))
        adder = SimpleNamespace(rem_chunk_tokens=None, can_run_list=[object()])

        estimate = scheduler._hybrid_bbf_estimate_req(req, adder, state)

        self.assertEqual(estimate.extra_load_tokens, 7)
        self.assertEqual(estimate.compute_tokens, 3)
        self.assertLessEqual(HICACHE_HYBRID_BBF_LOADING_BOUND_RATIO, 8.0)

    def test_ratio_guard_rejects_loading_heavy_non_empty_batch(self):
        scheduler = self._scheduler(_Node(0))
        state = HybridBalancedPrefillState(load_tokens=80, compute_tokens=10)
        adder = SimpleNamespace(rem_chunk_tokens=None, can_run_list=[object()])
        estimate = SimpleNamespace(extra_load_tokens=1, compute_tokens=0)
        req = _req()

        self.assertFalse(
            scheduler._hybrid_bbf_should_admit(req, adder, state, estimate)
        )

        compute_heavy = SimpleNamespace(extra_load_tokens=1, compute_tokens=10)
        self.assertTrue(
            scheduler._hybrid_bbf_should_admit(req, adder, state, compute_heavy)
        )


if __name__ == "__main__":
    unittest.main()
