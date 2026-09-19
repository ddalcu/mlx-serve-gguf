# mlx-serve-gguf - **EXPERIMENTAL**

This is not the main repo for MLX-Serve, its an experiment.

GGUF engine for [mlx-serve](https://mlxserve.com). Serves GGUF files as is (no conversion) on MLX with custom Metal kernels, so mlx-serve can drop llama.cpp dependency and focus on speed.

It's a separate repo on purpose, it will grow as models get added, and mlx-serve only needs a thin bridge to it.

Also its a separate repo because I want to encourage the community to open many PR's here. 

So we can add support for *MODERN* models, please dont add support for llama 1,2,etc.. nobody uses that anymore. If a model does not support tool calling, we do not add support for it here.

## Status

Works end to end in mlx-serve, text only, for:

- Qwen3.5 / Qwen3.6 dense (GGUF arch `qwen35`)
- Qwen3.5 / Qwen3.6 MoE (`qwen35moe`, the 35B-A3B)
- Gemma 4 (`gemma4`): E2B & E4B (per-layer embeddings, shared KV layers) and the dense 12B (K = V on the full attention layers)

What is where:

- GGUF reader (mmap, zero copy): `src/gguf.zig`
- CPU reference dequant, checked against ggml itself: `src/quants.zig`
- config.json / tokenizer.json / tokenizer_config.json / generation_config.json rebuilt from the GGUF metadata, in memory: `src/meta.zig`
- The arch table (`Arch`: read hparams, config.json, tensor map): `src/arch.zig`, one file per family in `src/arch/`. A new arch is a new file + one line in each of the 3 switches.
- llama.cpp layout fixes for qwen (tiled value heads, `ssm_a`), expert banks, kernel warmup at load: `src/weights.zig`
- Metal kernels: `src/kernels.zig`, `src/metal/blocks.h`

Tensor types: IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (+ F32/F16/BF16). The IQ ones are the focus, the others are there because real "IQ" files mix them in. Anything else (other archs, other types, unknown tensors) is declined up front and mlx-serve hands the file to llama.cpp like before.

Heads up on IQ1: the decoders match ggml on the fixtures, but I have no real file that uses them. unsloth's `UD-IQ1_M` of the 35B-A3B has zero IQ1 tensors in it (experts are IQ2_XXS / IQ2_S), and there is no IQ1 quant of the 27B.

Not done yet: vision (mmproj), MTP heads, a mat-mat kernel for MoE prefill (see below), more archs.

## Numbers

Base M4, 16 GB, same GGUF file, greedy, thinking off. llama.cpp = the libllama mlx-serve embeds (b10809), run with `--engine llama`.

Decode, tok/s. "edit task" = rename a variable in ~25 lines of code, the kind of request prompt lookup decoding (PLD, on by default in mlx-serve, lossless) drafts well on. The output there is byte identical with PLD on, with `--no-pld`, and (qwen models, Gemma E2B) on llama.cpp.

| model | llama.cpp | this, raw | this, edit task with PLD |
|---|---|---|---|
| Gemma 4 E2B IQ4_NL | 52.5 to 57 | 58 to 59 | 64.5 |
| Qwen3.5-4B IQ4_NL | 27 to 29 | 33.0 | 48.4 |
| Gemma 4 12B IQ4_NL | 12.5 | 12.9 | 18.2 |
| Qwen3.6-35B-A3B UD-IQ1_M (MoE) | 29 to 32 | 40 | 52.1 |
| Qwen3.6-27B UD-IQ2_XXS (round 1 kernels) | 7.2 | 7.6 | |

Time to first token, ms (wall clock, unique prompts, kernels warm):

| prompt tokens | 4B llama.cpp | 4B this | E2B llama.cpp | E2B this | 35B-A3B llama.cpp | 35B-A3B this |
|---|---|---|---|---|---|---|
| ~34 | 222 | 208 | 90 | 69 | 422 | 303 |
| ~74 | 286 | 347 | 148 | 176 | 420 | 589 |
| ~214 | 605 | 735 | 307 | 389 | 713 | 1517 |
| ~514 | 1382 | 1404 | 673 | 784 | 1699 | 3227 |
| ~914 | 2391 | 2247 | 1433 | 1240 | 2445 | 5588 |

Gemma 12B this: 466 / 857 / 1982 / 3825 / 6565 ms (llama.cpp runs out of GPU memory on that test on 16 GB, so no column).

So decode is ahead everywhere, by a lot when PLD has something to look up. Dense prefill is ahead for short and long prompts and 15 to 25% behind in the 75 to 500 range. MoE prefill is the weak spot, 2x behind past ~75 tokens: every (token, expert) pair is a matvec there, it needs a gathered version of the tile kernel. The first request after a load no longer pays the kernel compile, `weights.warmKernels` runs every (type, shape) once during load.

What made the difference, in case you touch the kernels:

- The 32 threads of a simdgroup run in LOCKSTEP. A loop costs every thread the longest trip count of any of them. Striping each weight row over the 32 threads on its own wastes lanes whenever 32 doesn't divide the units of a row (K = 1536 IQ4_NL is 48 units = 75%, Q6_K at K = 2560 is 20 units = 62%). Rotating the stripes per row does nothing (measured). The fix: a simdgroup stripes its 8 rows x units as ONE run. That alone took the Q6_K 248k row lm_head from 6.6 to 5.0 ms and the 4B from 31 to 33 tok/s.
- After that every type sits at its bits per weight vs MLX's own 4-bit kernel (Q4_K 1.0x, IQ4_NL 1.03 to 1.1x, Q5_K 1.27x, Q6_K 1.45x).
- On the base M4 the low bit types are compute bound, not bandwidth bound: IQ1_S (1.56 bpw) is barely faster than IQ4_NL (4.5 bpw). The codebook lookups are the cost.
- Several activation rows (PLD / MTP verify, decode batches): one kernel instance decodes each weight once and dots it against up to 5 rows, about 0.65x the cost per row of separate reads. Past 5 rows per instance the accumulators spill and it's as slow as separate reads, so 6 rows run as 2 instances of 3, 8 as 2 of 4. MLX's plain nibble kernel gets extra rows almost for free (it has ~3x the compute slack), that's the gap left to chase.
- Apple GPUs spill thread-local state past ~16 to 32 floats out of registers and it costs 4 to 10x. So the decoders never build a scratch array, they hand 4 weights at a time to an emitter that multiplies and accumulates right away, and a thread never keeps more than 16 running sums.
- Loop bounds have to be compile-time constants for the same reason (K, N, M are template constants, the short last prefill tile is a macro instantiation).
- `simdgroup_matrix` ops are executed by the whole simdgroup together, not per thread. The prefill kernel is one simdgroup per 32x32 output tile, threads decode one weight row each into threadgroup memory, then multiply 8x8 blocks together.
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
    zig build test -Dmodel=/path/to/model.gguf   # also parses a real file

Use the Zig pinned by mlx-serve (`../mlx-serve/.zig-toolchain/zig`).

    zig build bench                              # decode matvecs vs MLX's 4-bit matvec
    zig build bench -Dm=64 -Dtypes=iq4_nl,q6_k    # prefill width, some types
    zig build bench -Dm=6 -Dshapes=e2b            # a 6 row verify step, Gemma E2B shapes only

`scripts/gen_fixtures.sh` regenerates `src/fixtures/` from ggml's own dequant (needs a libllama dylib, defaults to the one staged in mlx-serve). `scripts/gen_tables.py` regenerates the IQ codebooks (Zig + Metal) from ggml-common.h.
