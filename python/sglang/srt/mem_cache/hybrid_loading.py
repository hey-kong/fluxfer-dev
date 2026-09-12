"""Latency model for transfer-aware two-phase KV loading."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridLoadingProfile:
    """Fixed model/hardware measurements used by the request-level model."""

    full_bandwidth_bytes_per_s: float
    layer_bandwidth_bytes_per_s: float
    compute_s_per_token_layer: float
    layer_num: int
    kv_bytes_per_token_layer: float

    def __post_init__(self):
        if (
            min(
                self.full_bandwidth_bytes_per_s,
                self.layer_bandwidth_bytes_per_s,
                self.compute_s_per_token_layer,
                self.layer_num,
                self.kv_bytes_per_token_layer,
            )
            <= 0
        ):
            raise ValueError("hybrid loading profile values must be positive")

    @property
    def overlap_tokens_per_compute_token(self) -> float:
        """Host tokens whose one-layer load fits one compute-token window."""
        return (
            self.layer_bandwidth_bytes_per_s
            * self.compute_s_per_token_layer
            / self.kv_bytes_per_token_layer
        )

    def latency(
        self, host_pages: int, preload_pages: int, page_size: int, compute_tokens: int
    ) -> float:
        q = self.kv_bytes_per_token_layer
        residual_pages = host_pages - preload_pages
        preload = (
            preload_pages
            * page_size
            * self.layer_num
            * q
            / self.full_bandwidth_bytes_per_s
        )
        layer_load = residual_pages * page_size * q / self.layer_bandwidth_bytes_per_s
        layer_compute = compute_tokens * self.compute_s_per_token_layer
        return (
            preload
            + layer_load
            + (self.layer_num - 1) * max(layer_load, layer_compute)
            + layer_compute
        )

    def select_preload_pages(
        self, host_tokens: int, page_size: int, compute_tokens: int
    ) -> int:
        """Evaluate every page-aligned split point and return argmin T(m)."""
        if host_tokens <= 0:
            return 0
        pages = (host_tokens + page_size - 1) // page_size
        return min(
            range(pages + 1),
            key=lambda m: self.latency(pages, m, page_size, compute_tokens),
        )
