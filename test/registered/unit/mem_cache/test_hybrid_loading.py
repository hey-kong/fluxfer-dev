from sglang.srt.mem_cache.hybrid_loading import HybridLoadingProfile


def test_dynamic_split_tracks_request_compute():
    profile = HybridLoadingProfile(24e9, 12e9, 0.5e-6, 32, 1500)
    assert profile.overlap_tokens_per_compute_token == 4.0
    assert profile.select_preload_pages(160, 16, 0) == 10
    assert profile.select_preload_pages(160, 16, 40) == 0
    middle = profile.select_preload_pages(160, 16, 20)
    assert 0 < middle < 10


def test_full_bandwidth_can_make_full_preload_optimal():
    profile = HybridLoadingProfile(100e9, 12e9, 0.5e-6, 32, 1500)
    assert profile.select_preload_pages(160, 16, 20) == 10
