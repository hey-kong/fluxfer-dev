import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost


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

    def test_block_h2d_keeps_partial_tail_on_direct_path(self):
        host = object.__new__(MHATokenToKVPoolHost)
        host.layout = "page_first_direct"
        host.page_size = 4
        host.k_buffer = "host_k"
        host.v_buffer = "host_v"

        host_indices = torch.arange(10)
        device_indices = torch.arange(100, 110)
        device_pool = SimpleNamespace(
            k_buffer=[["dev_k_l0"], ["dev_k_l1"]],
            v_buffer=[["dev_v_l0"], ["dev_v_l1"]],
        )
        block_calls = []
        direct_calls = []

        def fake_block(_device_pool, block_host_indices, block_device_indices):
            block_calls.append(
                (block_host_indices.clone(), block_device_indices.clone())
            )

        def fake_direct(**kwargs):
            direct_calls.append(
                (
                    kwargs["layer_id"],
                    kwargs["src_indices"].clone(),
                    kwargs["dst_indices"].clone(),
                )
            )

        host._load_to_device_block_h2d = fake_block
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_per_layer_direct_pf_lf",
            side_effect=fake_direct,
        ):
            host.load_to_device_per_layer(
                device_pool, host_indices, device_indices, 0, "block"
            )
            host.load_to_device_per_layer(
                device_pool, host_indices, device_indices, 1, "block"
            )

        self.assertEqual(len(block_calls), 1)
        torch.testing.assert_close(block_calls[0][0], host_indices[:8])
        torch.testing.assert_close(block_calls[0][1], device_indices[:8])
        self.assertEqual([call[0] for call in direct_calls], [0, 1])
        torch.testing.assert_close(direct_calls[0][1], host_indices[8:])
        torch.testing.assert_close(direct_calls[0][2], device_indices[8:])
        torch.testing.assert_close(direct_calls[1][1], host_indices[8:])
        torch.testing.assert_close(direct_calls[1][2], device_indices[8:])


if __name__ == "__main__":
    unittest.main()
