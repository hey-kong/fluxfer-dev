## Traces

This directory contains scaled versions of production KV cache traces derived from the [Qwen-Bailian Anonymous Usage Dataset](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon), released by Alibaba Cloud alongside the paper **[KVCache in the Wild: Characterizing and Optimizing KVCache at a Large Cloud Provider](https://www.usenix.org/system/files/atc25-wang-jiahao.pdf)**.

We scale the original traces to match the processing capacity of our testbed while preserving their temporal patterns.

The scaled traces cover four usage scenarios:

| **Scenario**       | **Description**                   | **Trace File**                                                   |
|--------------------|-----------------------------------|------------------------------------------------------------------|
| **To-C Trace**     | Interactive chat services         | [`qwen_traceA_blksz_16.jsonl`](./qwen_traceA_blksz_16.jsonl)     |
| **To-B Trace**     | API-based task automation         | [`qwen_traceB_blksz_16.jsonl`](./qwen_traceB_blksz_16.jsonl)     |
| **Thinking Trace** | Reasoning-intensive conversations | [`qwen_thinking_blksz_16.jsonl`](./qwen_thinking_blksz_16.jsonl) |
| **Coder Trace**    | Code-generation workloads         | [`qwen_coder_blksz_16.jsonl`](./qwen_coder_blksz_16.jsonl)       |