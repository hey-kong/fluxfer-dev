import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode


class TestHiRadixHybridWriteThroughPolicy(unittest.TestCase):
    def _node(self, pages: int, page_size: int) -> TreeNode:
        node = TreeNode()
        node.key = RadixKey(list(range(pages * page_size)))
        node.value = torch.arange(pages * page_size, dtype=torch.int64)
        return node

    def test_prefix_frequency_counts_hbm_and_dram_once_per_request(self):
        cache = object.__new__(HiRadixCache)
        cache.root_node = TreeNode()
        hbm_node = self._node(1, 4)
        dram_node = self._node(1, 4)
        hbm_node.parent = cache.root_node
        dram_node.parent = hbm_node
        dram_node.host_value = dram_node.value
        dram_node.value = None
        req = SimpleNamespace()

        cache._record_request_prefix_hits(req, dram_node)
        cache._record_request_prefix_hits(req, dram_node)

        self.assertEqual(hbm_node.prefix_hit_count, 1)
        self.assertEqual(dram_node.prefix_hit_count, 1)
        self.assertEqual(req.hicache_admission_threshold, 1)

    def test_admission_uses_frequency_and_full_block(self):
        page_size = 4
        cache = object.__new__(HiRadixCache)
        cache.page_size = page_size
        cache.hybrid_admission_ops_by_reqid = {}
        nodes = [self._node(1, page_size) for _ in range(3)]
        for frequency, node in zip((2, 3, 2), nodes):
            node.prefix_hit_count = frequency
        operation = CacheOperation(
            torch.arange(12),
            torch.arange(12),
            nodes[-1].id,
            h2d_preload_pages=1,
            admission_nodes=nodes,
        )
        cache.hybrid_admission_ops_by_reqid["request"] = [(operation, 2)]
        demoted = []
        cache._try_hybrid_demote_device_node = demoted.append

        cache._demote_hybrid_load_back_tail_for_req(SimpleNamespace(rid="request"))

        self.assertEqual(demoted, [nodes[2]])

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

    def test_quick_demotion_skips_internal_node_with_device_child(self):
        page_size = 4
        cache = object.__new__(HiRadixCache)
        cache.cache_controller = SimpleNamespace(
            io_backend="hybrid", write_policy="write_through"
        )
        cache.hybrid_pending_device_demotions = set()
        cache.root_node = TreeNode()
        cache.root_node.children = {}

        parent = self._node(1, page_size)
        parent.key = RadixKey([1])
        parent.children = {}
        parent.host_value = parent.value.clone()
        child = self._node(1, page_size)
        child.key = RadixKey([2])
        child.children = {}
        child.host_value = child.value.clone()
        self._attach_child(cache.root_node, parent, page_size)
        self._attach_child(parent, child, page_size)

        evicted = []
        cache._evict_backuped = lambda node: evicted.append(node)

        self.assertFalse(cache._try_hybrid_demote_device_node(parent))
        self.assertEqual(evicted, [])
        self.assertIsNotNone(parent.value)
        self.assertNotIn(parent, cache.hybrid_pending_device_demotions)

    def test_unified_quick_demotion_skips_internal_node_with_device_child(self):
        tree_components = (ComponentType.FULL,)
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(
            io_backend="hybrid", write_policy="write_through"
        )
        cache.hybrid_pending_device_demotions = set()
        cache.root_node = UnifiedTreeNode(tree_components)
        cache.root_node.children = {}

        parent = UnifiedTreeNode(tree_components)
        parent.parent = cache.root_node
        parent.key = RadixKey([1])
        parent.children = {}
        parent.component_data[ComponentType.FULL].value = torch.tensor([1])
        parent.component_data[ComponentType.FULL].host_value = torch.tensor([1])
        child = UnifiedTreeNode(tree_components)
        child.parent = parent
        child.key = RadixKey([2])
        child.children = {}
        child.component_data[ComponentType.FULL].value = torch.tensor([2])
        child.component_data[ComponentType.FULL].host_value = torch.tensor([2])
        cache.root_node.children[1] = parent
        parent.children[2] = child

        evicted = []
        cache._evict_to_host = lambda node: evicted.append(node)

        self.assertFalse(cache._try_hybrid_demote_device_node(parent))
        self.assertEqual(evicted, [])
        self.assertIsNotNone(parent.component_data[ComponentType.FULL].value)
        self.assertNotIn(parent, cache.hybrid_pending_device_demotions)

    def _attach_child(self, parent: TreeNode, child: TreeNode, page_size: int) -> None:
        child.parent = parent
        parent.children[child.key.child_key(page_size)] = child

    def _eviction_order(self, io_backend: str) -> list[str]:
        page_size = 1
        cache = object.__new__(HiRadixCache)
        cache.page_size = page_size
        cache.cache_controller = SimpleNamespace(
            io_backend=io_backend,
            write_policy="write_through",
            evict_device=lambda value: len(value),
        )
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = []
        cache.root_node.lock_ref = 1
        cache.root_node.children = {}
        cache.evictable_size_ = 3
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()

        parent = self._node(1, page_size)
        child = self._node(1, page_size)
        sibling = self._node(1, page_size)
        parent.key = RadixKey([1])
        child.key = RadixKey([2])
        sibling.key = RadixKey([3])
        parent.priority = -100
        child.priority = 0
        sibling.priority = 10
        parent.children = {}
        child.children = {}
        sibling.children = {}
        for node in (parent, child, sibling):
            node.host_value = node.value.clone()

        self._attach_child(cache.root_node, parent, page_size)
        self._attach_child(parent, child, page_size)
        self._attach_child(cache.root_node, sibling, page_size)
        cache.evictable_leaves.update([child, sibling])

        class Strategy:
            def get_priority(self, node):
                return node.priority

        cache.eviction_strategy = Strategy()
        node_names = {parent.id: "parent", child.id: "child", sibling.id: "sibling"}
        eviction_order = []
        cache._record_remove_event = lambda node, medium=None: eviction_order.append(
            node_names[node.id]
        )
        cache.update_eviction_metrics = lambda num_evicted, start_time: None

        result = cache.evict(EvictParams(num_tokens=3))

        self.assertEqual(result.num_tokens_evicted, 3)
        return eviction_order

    def test_hybrid_write_through_recollects_new_hbm_leaves_after_heap_drains(self):
        eviction_order = self._eviction_order(io_backend="hybrid")

        # The parent gets the best priority after its child is demoted, but hybrid
        # write-through keeps processing the initial HBM leaf snapshot first.
        self.assertEqual(eviction_order, ["child", "sibling", "parent"])

    def test_non_hybrid_write_through_keeps_immediate_parent_heap_push(self):
        eviction_order = self._eviction_order(io_backend="direct")

        # Non-hybrid HBM eviction retains the previous behavior: the parent is
        # immediately pushed and can preempt the remaining initial leaves.
        self.assertEqual(eviction_order, ["child", "parent", "sibling"])


if __name__ == "__main__":
    unittest.main()
