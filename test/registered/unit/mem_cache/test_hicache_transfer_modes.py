import unittest
import pytest

torch = pytest.importorskip("torch")

from sglang.srt.managers.cache_controller import HiCacheController  # noqa: E402
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (  # noqa: E402
    HybridCacheController,
)


class TestHiCacheTransferModes(unittest.TestCase):
    def test_hybrid_preload_tokens_are_page_aligned(self):
        host_indices = torch.arange(10)

        controller = object.__new__(HiCacheController)
        controller.page_size = 4
        self.assertEqual(controller._get_hybrid_preload_tokens(host_indices, 0), 0)
        self.assertEqual(controller._get_hybrid_preload_tokens(host_indices, 1), 4)
        self.assertEqual(controller._get_hybrid_preload_tokens(host_indices, 3), 8)

        hybrid_controller = object.__new__(HybridCacheController)
        hybrid_controller.page_size = 4
        self.assertEqual(
            hybrid_controller._get_hybrid_preload_tokens(host_indices, 3), 8
        )



if __name__ == "__main__":
    unittest.main()
