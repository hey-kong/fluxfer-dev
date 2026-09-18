"""Device-side per-layer timing for pure prefill batches."""

from sglang.srt.utils import get_device_module

device_module = get_device_module()


class LayerPrefillProfiler:
    """Collect exact Transformer-layer timings when HiCache is disabled."""

    def __init__(self):
        self._start = None
        self._layers = None

    def start_prefill_profile(self):
        self._start = device_module.Event(enable_timing=True)
        self._layers = []
        self._start.record()

    def record_prefill_layer_start(self, layer_index: int):
        if self._layers is None:
            return
        start = device_module.Event(enable_timing=True)
        start.record()
        self._layers.append([layer_index, start, None])

    def record_prefill_layer_end(self, layer_index: int):
        if self._layers is None:
            return
        for layer in reversed(self._layers):
            if layer[0] == layer_index and layer[2] is None:
                end = device_module.Event(enable_timing=True)
                end.record()
                layer[2] = end
                return

    def record_prefill_profile_end(self):
        if self._start is None:
            return None
        end = device_module.Event(enable_timing=True)
        end.record()
        return end

    def finish_prefill_profile(self, end=None):
        if self._start is None:
            return None
        if end is None:
            end = self.record_prefill_profile_end()
        end.synchronize()
        total_ms = self._start.elapsed_time(end)
        layer_compute_ms = [
            (layer_index, start.elapsed_time(stop), 0.0)
            for layer_index, start, stop in self._layers
            if stop is not None
        ]
        self._start = None
        self._layers = None
        return total_ms, layer_compute_ms
