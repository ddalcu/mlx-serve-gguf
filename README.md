# mlx-serve-gguf - **EXPERIMENTAL**

This is not the main repo for MLX-Serve, its an experiment.

GGUF engine for [mlx-serve](https://mlxserve.com). Serves GGUF files as is (no conversion) on MLX with custom Metal kernels, so mlx-serve can drop llama.cpp dependency and focus on speed.

It's a separate repo on purpose, it will grow as models get added, and mlx-serve only needs a thin bridge to it.

Also its a separate repo because I want to encourage the community to open many PR's here. 

So we can add support for *MODERN* models, please dont add support for llama 1,2,etc.. nobody uses that anymore. If a model does not support tool calling, we do not add support for it here.

## Status

Works end to end in mlx-serve, text only, greedy output at parity with llama.cpp (checked with `scripts/parity.py`, see below) for:

- Qwen3.5 / Qwen3.6 dense (GGUF arch `qwen35`)
- Qwen3.5 / Qwen3.6 MoE (`qwen35moe`, the 35B-A3B)
- Gemma 4 (`gemma4`): E2B & E4B (per-layer embeddings, shared KV layers) and the dense 12B (K = V on the full attention layers)
- Gemma 3 (`gemma3`): 1B / 4B / 12B / 27B text. The SentencePiece vocab has no merges in the GGUF, they are derived from the scores the way HF's converter does it (identical to google's tokenizer.json, checked).
- LFM2 (`lfm2`) and LFM2-MoE (`lfm2moe`)

- gpt-oss (`gpt-oss`, experts in MXFP4). Parity holds on raw `/v1/completions` prompts; through `/v1/chat/completions` the two engines render the harmony system prompt differently (date, reasoning level), which is mlx-serve's template layer, not this engine.

- Nemotron-H / Nemotron 3 Nano (`nemotron_h`, `nemotron_h_moe`)
- Qwen3-Next (`qwen3next`): unlike `qwen35` the converter keeps HF head order, fuses b / a (split back at load) and stores the norms as w + 1 (what mlx-lm's checkpoints carry too)

Not done: vision (mmproj), MTP heads (skipped at load).

What is where:

- GGUF reader (mmap, zero copy): `src/gguf.zig`
- CPU reference dequant, checked against ggml itself: `src/quants.zig`
- config.json / tokenizer.json / tokenizer_config.json / generation_config.json rebuilt from the GGUF metadata, in memory: `src/meta.zig`
- The arch table (`Arch`: read hparams, config.json, tensor map): `src/arch.zig`, one file per family in `src/arch/`. A new arch is a new file + one line in each of the 3 switches.
- llama.cpp layout fixes for qwen (tiled value heads, `ssm_a`), expert banks, kernel warmup at load: `src/weights.zig`
- Metal kernels: `src/kernels.zig`, one decoder file per family in `src/metal/blocks/`, the IQ codebooks in `src/metal/tables/`

Tensor types: IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, MXFP4 (+ F32/F16/BF16). So every K_M / K_S / legacy mix llama.cpp writes. Anything else (other archs, other types, unknown tensors) is declined up front and mlx-serve hands the file to llama.cpp like before.

## Checking parity and speed

    scripts/parity.py /path/model.gguf --engines mlx,llama,llamacpp --llama-server ~/llama.cpp/build/bin/llama-server

Serves the same file on this engine, on the libllama mlx-serve embeds, and on a llama.cpp `llama-server`, sends the same chat requests at temperature 0 and compares the texts. Where two engines fork, the mlx logprobs at the fork decide: a top-2 margin under 1 nat is a rounding tie (numerics, expected), more is a real bug. Timings come from each server's own `timings`.

    scripts/llmprobe_ab.sh /path/model.gguf

The same file through llmprobe (capability card + reasoning eval, greedy) on both engines, for quality at low bit rates.

## Numbers

Charts of these tables: `charts/` (`charts/make_charts.py` renders them).

M4 Max 128 GB, `scripts/parity.py`, 64 tokens greedy, decode tok/s and time to first token for a 1740 token prompt. "llama.cpp" is a llama-server built from master on 2026-09-28; the libllama mlx-serve embeds (b10809) is slower than that on every file here.

| file | this, tok/s | llama.cpp, tok/s | this, prefill | llama.cpp, prefill | parity |
|---|---|---|---|---|---|
| Qwen3.5-0.8B IQ4_NL | 340 | 275 | 207 ms | 228 ms | ok |
| Qwen3.5-0.8B Q4_0 | 456 | 279 | 205 ms | 226 ms | ok |
| Qwen3.5-0.8B IQ1_M | 373 | 258 | 206 ms | 260 ms | ok |
| Qwen3.5-4B IQ4_NL | 113 | 104 | 1102 ms | 1178 ms | ok |
| Gemma 4 E2B IQ4_NL | 171 | 152 | 600 ms | 654 ms | ok |
| Gemma 3 1B Q4_0 | 301 | 278 | 278 ms | 243 ms | ok |
| Gemma 3 4B IQ4_NL | 128 | 132 | 896 ms | 966 ms | ok |
| LFM2-1.2B Q4_0 | 382 | 400 | 307 ms | 310 ms | ok |
| LFM2-8B-A1B Q4_0 (MoE) | 255 | 258 | | | ok |
| Qwen3.6-35B-A3B UD-IQ1_M (MoE) | 101 | 87 | 1214 ms | 1357 ms | ok |
| gpt-oss-20b Q4_K_M (MXFP4 experts) | 102 | 100 | 1303 ms | 1390 ms | ok, raw prompts |
| Nemotron-3-Nano-30B-A3B IQ4_NL (Mamba2 + MoE) | 115 | 100 | 4299 ms | 1443 ms | ok |
| Qwen3-Next-80B-A3B UD-IQ2_XXS (GDN + MoE) | 91 | 66 | 2018 ms | 1738 ms | ok |

Parity "ok" = every prompt matches byte for byte or forks on a tie under 0.2 nats. The Nemotron prefill time was taken before the materialized-bank path below (the other MoE rows are after it). On a base M4 16 GB (the earlier numbers) the same kernels were bandwidth bound and ahead everywhere; the M4 Max has 4x the bandwidth and turned the small matrices latency bound, which is what the last two items below are about.

Quality on the same low-bit files, llmprobe (`scripts/llmprobe_ab.sh`, greedy, thinking off, 2048 token cap): reasoning accuracy this engine / llama.cpp: gpt-oss-20b Q4_K_M 61% / 45% (llama.cpp ran out of tokens on 43 questions, this engine on 13), Qwen3-Next IQ2_XXS 47% / 48%, Gemma 3 1B Q4_0 10% / 9%, Qwen3.5-0.8B IQ1_M 1% / 0%. Capability cards identical on every file.

What made the difference, in case you touch the kernels:

- MLX hashes a custom kernel's whole source string on EVERY dispatch (`std::hash` in `CustomKernel::eval_gpu`). With all the IQ codebooks and decoders in one header that was 90 KB and ~6 us per matvec, a quarter of a token on the E2B. Every kernel now carries only the decoder and tables of its own type (`headerFor`): 8.7 to 4.4 us per op, E2B decode 133 to 171 tok/s, the 0.8B 250 to 456.
- Small weights (`attn_kv`, 1536 -> 256) are latency bound: fewer rows per simdgroup puts more of them in flight (`MIN_SIMDGROUPS`).
- The 32 threads of a simdgroup run in LOCKSTEP. A loop costs every thread the longest trip count of any of them. Striping each weight row over the 32 threads on its own wastes lanes whenever 32 doesn't divide the units of a row (K = 1536 IQ4_NL is 48 units = 75%, Q6_K at K = 2560 is 20 units = 62%). Rotating the stripes per row does nothing (measured). The fix: a simdgroup stripes its 8 rows x units as ONE run. That alone took the Q6_K 248k row lm_head from 6.6 to 5.0 ms and the 4B from 31 to 33 tok/s on the base M4.
- On the base M4 every type sits at its bits per weight vs MLX's own 4-bit kernel (Q4_K 1.0x, IQ4_NL 1.03 to 1.1x, Q5_K 1.27x, Q6_K 1.45x). On the M4 Max the big shapes are at 1.1x (Q4_K) to 1.55x (Q6_K) and the small ones at 1.2 to 1.8x: `zig build bench` prints the table, and `per-op floor` on its first line is the fixed cost of one dispatch against a native MLX op.
- The low bit types are compute bound, not bandwidth bound: IQ1_S (1.56 bpw) is barely faster than IQ4_NL (4.5 bpw). The codebook lookups are the cost.
- Several activation rows (PLD / MTP verify, decode batches): one kernel instance decodes each weight once and dots it against up to 5 rows, about 0.65x the cost per row of separate reads. Past 5 rows per instance the accumulators spill and it's as slow as separate reads, so 6 rows run as 2 instances of 3, 8 as 2 of 4.
- Apple GPUs spill thread-local state past ~16 to 32 floats out of registers and it costs 4 to 10x. So the decoders never build a scratch array, they hand 4 weights at a time to an emitter that multiplies and accumulates right away, and a thread never keeps more than 16 running sums.
- Loop bounds have to be compile-time constants for the same reason (K, N, M are template constants, the short last prefill tile is a macro instantiation).
- `simdgroup_matrix` ops are executed by the whole simdgroup together, not per thread. The prefill kernel is one simdgroup per 32x32 output tile, threads decode one weight row each into threadgroup memory, then multiply 8x8 blocks together.
- MoE prefill, three paths by (token, expert) pairs per expert: under 8, a matvec per pair. From 8, a gathered version of the tile kernel (`gather_tile`) runs once per (expert, token tile) over just the 8-token groups that hold the expert's rows, one decode per expert per tile instead of one per pair, but few threadgroups. From 24 (and a bf16 bank under 1 GB), the bank is dequantized once and MLX's sorted `gather_mm` does the GEMMs: the fixed pass over the bank (~3.5 ms for 270M weights on the M4 Max) beats the tile kernel from there on (`zig build bench -Dm=1024 -Dshapes=moe`: 3.9 ms vs ~6, gpt-oss shapes at 256 tokens: 5.4 ms vs 14.2). The 35B's prefill went from 1778 to 1214 ms with it.
- A benchmark that reuses one hot weight lies. `zig build bench` rotates through ~400 MB of weights per shape, like a real token does.

Gemma gotcha when you compare against llama.cpp: it adds `<bos>` to a raw `/v1/completions` prompt by itself, mlx-serve doesn't, and Gemma without BOS is garbage (the 12B prints `_ _ _`). Put `<bos>` in the prompt on the mlx side.

## How it plugs in

`src/kernels.zig` needs MLX bindings, and it takes them from whoever hosts it through one import, `mlx_host`, whose root must expose `pub const mlx` (mlx-serve's `src/mlx.zig`).

- In mlx-serve: root a module at `src/root.zig` and pass mlx-serve's own main module as `mlx_host`. Same bindings, same error latch, nothing duplicated.
- Standalone: `build.zig` here builds `mlx_host` from an mlx-serve checkout (`-Dmlx-serve=<path>`, default `../mlx-serve`, use `../..` when this repo is the `lib/` submodule). It needs `lib/mlx` staged there.

The Metal source is plain text in `src/metal/` (`blocks.h` has one decoder per block format), so it can be reused outside MLX too.

## Test

    zig build test-core   # reader + reference dequant, no MLX, runs anywhere
    zig build test        # everything, kernels run on the GPU
    zig build test -Dmodel=/path/to/model.gguf   # also parses and loads a real file

CI (`.github/workflows/ci.yml`) runs the core tests on Linux and the GPU tests plus a real 0.8B file on GitHub's Apple Silicon macOS runners, against a fresh mlx-serve checkout.

Use the Zig pinned by mlx-serve (`../mlx-serve/.zig-toolchain/zig`).

    zig build bench                              # decode matvecs vs MLX's 4-bit matvec
    zig build bench -Dm=64 -Dtypes=iq4_nl,q6_k    # prefill width, some types
    zig build bench -Dm=6 -Dshapes=e2b            # a 6 row verify step, Gemma E2B shapes only

`scripts/gen_fixtures.sh` regenerates `src/fixtures/` from ggml's own dequant (needs a libllama dylib, defaults to the one staged in mlx-serve). `scripts/gen_tables.py` regenerates the IQ codebooks (`src/iq_tables.zig`, `src/metal/tables/`) from ggml-common.h.
