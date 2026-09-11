# Fluxfer Artifact

Fluxfer is a transfer-aware and memory-efficient hierarchical prefix-caching
system for LLM serving. This repository contains the
modified SGLang engine, GPU kernels, workload traces, and the trace-replay
client used to evaluate Fluxfer and the baselines reported in the paper.

This README describes how to:

1. build and launch Fluxfer;
2. replay a workload trace and collect TTFT/TPOT metrics; and
3. reproduce the corresponding vLLM and SGLang baselines.

## Repository layout

The commands below assume the following repository layout:

```text
.
├── python/             # Modified SGLang Python package
├── sgl-kernel/         # Modified SGLang GPU kernels
├── trace-replayer/     # Rust trace-replay client
└── scaled_traces/      # Scaled traces used by the experiments
```

All commands are executed from the repository root unless a command explicitly
changes directory.

## Requirements

### Software

- Linux (x86-64)
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Rust and Cargo
- A CUDA toolkit compatible with the PyTorch version installed by the artifact
- A C/C++ compiler, CMake, Ninja, and standard Python build tools

Install `uv` if necessary:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Rust if necessary:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "${CARGO_HOME:-$HOME/.cargo}/env"
```

### Hardware and model

- One CUDA-capable NVIDIA GPU with enough memory to serve
  `Llama-3.1-8B-Instruct`
- At least 64 GiB of available host memory for the configured host KV cache,
  plus additional memory for the model, runtime, replay client, and operating
  system
- Local access to the model weights and tokenizer files

The examples use `Llama-3.1-8B-Instruct`. Access to the model is governed by
the model provider's license and is not included with this artifact.

Before running an experiment, define the following paths:

```bash
export FLUXFER_ROOT="$(pwd)"
export MODEL_PATH="/path/to/Llama-3.1-8B-Instruct"
export TRACE_PATH="$FLUXFER_ROOT/scaled_traces/qwen_traceA_blksz_16.jsonl"
export RESULTS_DIR="$FLUXFER_ROOT/results"
mkdir -p "$RESULTS_DIR"
```

The model directory must contain at least:

```text
config.json
tokenizer.json
tokenizer_config.json
```

## Quick start: Fluxfer

### 1. Create an isolated environment

```bash
cd "$FLUXFER_ROOT"
uv venv --python 3.12 .venv-fluxfer
source .venv-fluxfer/bin/activate
```

### 2. Install the modified SGLang engine

```bash
uv pip install -e "python"
```

Build the modified kernels:

```bash
cd "$FLUXFER_ROOT/sgl-kernel"
make -f Makefile.uv build \
  MAX_JOBS=32 \
  CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
cd "$FLUXFER_ROOT"
```

### 3. Start Fluxfer

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-size 64 \
  --page-size 16 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend hybrid \
  --enable-hybrid-balanced-batch \
  --enable-hybrid-bubble-filling
```

The Fluxfer-specific settings are:

| Argument | Value | Purpose |
| --- | ---: | --- |
| `--hicache-size` | `64` | Allocates a 64 GiB host-memory KV-cache pool. |
| `--page-size` | `16` | Uses 16-token KV-cache pages. |
| `--hicache-mem-layout` | `page_first_direct` | Enables the page-first host-memory layout used by Fluxfer. |
| `--hicache-io-backend` | `hybrid` | Enables Fluxfer's hybrid page/direct transfer path. |
| `--enable-hybrid-balanced-batch` | enabled | Enables transfer-aware balanced batch formation. |
| `--enable-hybrid-bubble-filling` | enabled | Enables bubble-filling scheduling. |

Wait until the server reports that it is ready. From another terminal, check
the OpenAI-compatible endpoint:

```bash
curl --fail --silent http://localhost:30000/v1/models
```

If this command fails, inspect the server log before starting the replay.

## Build the trace replayer

The replay client is written in Rust. Build the optimized binary once:

```bash
cd "$FLUXFER_ROOT/trace-replayer"
cargo build \
  -p request-sim \
  --bin client \
  --release \
  -j32
cd "$FLUXFER_ROOT"
```

The executable is created at:

```text
trace-replayer/target/release/client
```

## Replay a trace

With Fluxfer listening on `localhost:30000`, run:

```bash
"$FLUXFER_ROOT/trace-replayer/target/release/client" \
  --tokenizer "$MODEL_PATH/tokenizer.json" \
  --tokenizer-config "$MODEL_PATH/tokenizer_config.json" \
  --endpoint http://localhost:30000/v1/chat/completions \
  --api openai \
  --dataset bailian \
  --dataset-path "$TRACE_PATH" \
  --output-path "$RESULTS_DIR/fluxfer_traceA" \
  --scale-factor 1.0 \
  --time-in-secs 7500 \
  --model-name "$MODEL_PATH" \
  --stream
```

After the replay completes, aggregate metrics, including time to first token
(TTFT) and time per output token (TPOT), are written to:

```text
results/fluxfer_traceA.summary.json
```

## Baselines

### vLLM + native KV offload

Create an isolated environment and install the version used by the artifact:

```bash
cd "$FLUXFER_ROOT"
uv venv --python 3.12 .venv-vllm-native
source .venv-vllm-native/bin/activate
uv pip install "vllm==0.23.0"
```

Start the server:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
  --port 8080 \
  --enable-prefix-caching \
  --block-size 16 \
  --kv-offloading-backend native \
  --kv-offloading-size 64 \
  --disable-hybrid-kv-cache-manager
```

Replay the same trace:

```bash
"$FLUXFER_ROOT/trace-replayer/target/release/client" \
  --tokenizer "$MODEL_PATH/tokenizer.json" \
  --tokenizer-config "$MODEL_PATH/tokenizer_config.json" \
  --endpoint http://localhost:8080/v1/chat/completions \
  --api openai \
  --dataset bailian \
  --dataset-path "$TRACE_PATH" \
  --output-path "$RESULTS_DIR/vllm_native_traceA" \
  --scale-factor 1.0 \
  --time-in-secs 7500 \
  --model-name "$MODEL_PATH" \
  --stream
```

### vLLM + LMCache

Create a separate environment:

```bash
cd "$FLUXFER_ROOT"
uv venv --python 3.12 .venv-vllm-lmcache
source .venv-vllm-lmcache/bin/activate
uv pip install "vllm==0.23.0" "lmcache==0.4.5"
```

Create the LMCache configuration reproducibly:

```bash
cat > "$FLUXFER_ROOT/lmcache_config.yaml" <<'YAML'
chunk_size: 256
local_cpu: true
max_local_cpu_size: 64
use_layerwise: true
save_unfull_chunk: false
YAML
```

Start the server:

```bash
cd "$FLUXFER_ROOT"
CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE="$FLUXFER_ROOT/lmcache_config.yaml" \
vllm serve "$MODEL_PATH" \
  --port 8080 \
  --block-size 16 \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Replay the same trace:

```bash
"$FLUXFER_ROOT/trace-replayer/target/release/client" \
  --tokenizer "$MODEL_PATH/tokenizer.json" \
  --tokenizer-config "$MODEL_PATH/tokenizer_config.json" \
  --endpoint http://localhost:8080/v1/chat/completions \
  --api openai \
  --dataset bailian \
  --dataset-path "$TRACE_PATH" \
  --output-path "$RESULTS_DIR/vllm_lmcache_traceA" \
  --scale-factor 1.0 \
  --time-in-secs 7500 \
  --model-name "$MODEL_PATH" \
  --stream
```

### SGLang + Direct I/O

Create a separate environment:

```bash
cd "$FLUXFER_ROOT"
uv venv --python 3.12 .venv-sglang-direct
source .venv-sglang-direct/bin/activate
uv pip install --prerelease=allow "sglang==0.5.12" "kernels<0.15"
```

Start upstream SGLang with its hierarchical cache and direct I/O backend:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-size 64 \
  --page-size 16 \
  --hicache-io-backend direct
```

Replay the same trace:

```bash
"$FLUXFER_ROOT/trace-replayer/target/release/client" \
  --tokenizer "$MODEL_PATH/tokenizer.json" \
  --tokenizer-config "$MODEL_PATH/tokenizer_config.json" \
  --endpoint http://localhost:30000/v1/chat/completions \
  --api openai \
  --dataset bailian \
  --dataset-path "$TRACE_PATH" \
  --output-path "$RESULTS_DIR/sglang_direct_traceA" \
  --scale-factor 1.0 \
  --time-in-secs 7500 \
  --model-name "$MODEL_PATH" \
  --stream
```

### Strata

Strata has been implemented in upstream SGLang. In this artifact, we evaluate
Strata using SGLang's hierarchical cache with the kernel I/O backend.

Create a separate environment:

```bash
cd "$FLUXFER_ROOT"
uv venv --python 3.12 .venv-strata
source .venv-strata/bin/activate
uv pip install --prerelease=allow "sglang==0.5.12" "kernels<0.15"
```

Start the server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-size 64 \
  --page-size 16 \
  --hicache-io-backend kernel
```

Replay the same trace:

```bash
"$FLUXFER_ROOT/trace-replayer/target/release/client" \
  --tokenizer "$MODEL_PATH/tokenizer.json" \
  --tokenizer-config "$MODEL_PATH/tokenizer_config.json" \
  --endpoint http://localhost:30000/v1/chat/completions \
  --api openai \
  --dataset bailian \
  --dataset-path "$TRACE_PATH" \
  --output-path "$RESULTS_DIR/strata_traceA" \
  --scale-factor 1.0 \
  --time-in-secs 7500 \
  --model-name "$MODEL_PATH" \
  --stream
```
