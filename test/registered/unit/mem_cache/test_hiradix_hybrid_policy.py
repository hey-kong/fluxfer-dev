import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


class TestHiRadixHybridWriteThroughPolicy(unittest.TestCase):
    def _node(self, pages: int, page_size: int) -> TreeNode:
        node = TreeNode()
        node.key = RadixKey(list(range(pages * page_size)))
        node.value = torch.arange(pages * page_size, dtype=torch.int64)
        return node

    def _selected_pages(self, page_counts: list[int]) -> list[int]:
        page_size = 4
        cache = object.__new__(HiRadixCache)
        cache.page_size = page_size
        nodes = [self._node(pages, page_size) for pages in page_counts]
        selected = cache._select_hybrid_load_back_tail_nodes(nodes)
        return [len(node.value) // page_size for node in selected]

    def test_tail_demotion_keeps_crossing_node_for_two_plus_four_pages(self):
        self.assertEqual(self._selected_pages([2, 4]), [])

    def test_tail_demotion_exact_half_for_two_plus_one_plus_three_pages(self):
        self.assertEqual(self._selected_pages([2, 1, 3]), [3])

    def test_tail_demotion_best_effort_for_two_plus_two_plus_two_pages(self):
        self.assertEqual(self._selected_pages([2, 2, 2]), [2])

    def test_generated_node_final_state_is_dram_only_after_pending_drain(self):
        page_size = 4
        cache = object.__new__(HiRadixCache)
        cache.page_size = page_size
        cache.cache_controller = SimpleNamespace(
            io_backend="hybrid", write_policy="write_through"
        )
        cache.hybrid_pending_device_demotions = set()
        cache.root_node = TreeNode()

        node = self._node(1, page_size)
        node.parent = cache.root_node
        node.host_value = node.value.clone()

        def fake_evict_backuped(evict_node):
            evict_node.value = None
            return len(evict_node.host_value)

        cache._evict_backuped = fake_evict_backuped
        cache._mark_hybrid_generated_node_for_demotion(node)
        cache._drain_hybrid_pending_demotions()

        self.assertIsNone(node.value)
        self.assertIsNotNone(node.host_value)


if __name__ == "__main__":
    unittest.main()
