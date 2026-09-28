//! Decode-shape matvec timings per ggml type, next to MLX's own 4-bit affine
//! matvec on the same shape (the reference to chase). `zig build bench`.
//! Every call uses a different weight copy (~400 MB per shape in rotation): a
//! decode step sweeps the whole model once, and a single hot weight flatters
//! kernels that don't pull memory bandwidth well.
const std = @import("std");
const gguf = @import("mlx_serve_gguf");
const mlx = @import("mlx_host").mlx;
const kernels = gguf.kernels;
const GgmlType = gguf.quants.GgmlType;

const ITERS_TOKENS = 192;
/// Bytes of distinct weights to rotate through per shape.
const WORKING_SET = 400 << 20;
const MAX_COPIES = 32;

/// `experts` > 0: an expert bank, timed as MoE dispatch (top-8 sorted pairs
/// per token) against MLX's gather_qmm.
const Shape = struct { name: []const u8, in: c_int, out: c_int, experts: c_int = 0 };
const TOP_K = 8;
// Qwen3.5-4B layer shapes, the lm_head last.
const shapes = [_]Shape{
    .{ .name = "ffn_up   2560->9216", .in = 2560, .out = 9216 },
    .{ .name = "ffn_down 9216->2560", .in = 9216, .out = 2560 },
    .{ .name = "attn_qkv 2560->8192", .in = 2560, .out = 8192 },
    .{ .name = "ssm_out  4096->2560", .in = 4096, .out = 2560 },
    .{ .name = "attn_kv  2560->1024", .in = 2560, .out = 1024 },
    .{ .name = "ssm_a/b  2560->32", .in = 2560, .out = 32 },
    .{ .name = "lm_head  2560->248320", .in = 2560, .out = 248320 },
    // Gemma 4 E2B: small matrices, where per-dispatch cost shows.
    .{ .name = "e2b ffn_up   1536->6144", .in = 1536, .out = 6144 },
    .{ .name = "e2b ffn_down 6144->1536", .in = 6144, .out = 1536 },
    .{ .name = "e2b attn_q   1536->2048", .in = 1536, .out = 2048 },
    .{ .name = "e2b attn_kv  1536->256", .in = 1536, .out = 256 },
    .{ .name = "e2b lm_head  1536->262144", .in = 1536, .out = 262144 },
    // Qwen3.6-35B-A3B experts: 256 of them, 8 per token.
    .{ .name = "moe gate/up 2048->512 x256", .in = 2048, .out = 512, .experts = 256 },
    .{ .name = "moe down    512->2048 x256", .in = 512, .out = 2048, .experts = 256 },
    // gpt-oss-20b experts: 32 of them, 4 per token (the bench routes 8).
    .{ .name = "gptoss experts 2880->2880 x32", .in = 2880, .out = 2880, .experts = 32 },
};
const types = kernels.all_types;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Milliseconds per call, the whole batch evaluated as one graph like a decode
/// step. `args` = { x, weight copies, stream }, call i gets copy i % len.
fn time(comptime f: anytype, args: anytype, s: mlx.mlx_stream) !f64 {
    var best: f64 = std.math.floatMax(f64);
    for (0..3) |_| {
        const t0 = nowNs();
        var outs_buf: [ITERS_TOKENS]mlx.mlx_array = undefined;
        const outs = outs_buf[0..iters];
        for (outs, 0..) |*o, i| o.* = try @call(.auto, f, .{ args[0], args[1], i, args[2] });
        const vec = mlx.mlx_vector_array_new_data(outs.ptr, outs.len);
        try mlx.check(mlx.mlx_eval(vec));
        _ = mlx.mlx_synchronize(s);
        _ = mlx.mlx_vector_array_free(vec);
        for (outs) |o| _ = mlx.mlx_array_free(o);
        best = @min(best, @as(f64, @floatFromInt(nowNs() - t0)) / 1e6 / @as(f64, @floatFromInt(iters)));
    }
    return best;
}

fn ggufMatvec(tyx: struct { GgmlType, mlx.mlx_array }, ws: []const mlx.mlx_array, i: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    return kernels.linear(.{ .ty = tyx[0] }, tyx[1], ws[i % ws.len], s);
}

fn ggufDequantMatmul(tyx: struct { GgmlType, mlx.mlx_array }, ws: []const mlx.mlx_array, i: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    return kernels.dequantMatmul(tyx[0], tyx[1], ws[i % ws.len], s);
}

fn ggufGather(tyx: struct { GgmlType, mlx.mlx_array, mlx.mlx_array }, ws: []const mlx.mlx_array, i: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    return kernels.gatherLinear(.{ .ty = tyx[0] }, tyx[1], ws[i % ws.len], .{ .ctx = null }, tyx[2], s);
}

fn mlxGatherQmm(xi: struct { mlx.mlx_array, mlx.mlx_array }, qs: []const [3]mlx.mlx_array, i: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const q = qs[i % qs.len];
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_gather_qmm(&y, xi[0], q[0], q[1], q[2], .{ .ctx = null }, xi[1], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
    return y;
}

fn mlxMatmul(x: mlx.mlx_array, ws: []const mlx.mlx_array, i: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&y, x, ws[i % ws.len], s));
    return y;
}

fn mlxQmv(x: mlx.mlx_array, qs: []const [3]mlx.mlx_array, i: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const q = qs[i % qs.len];
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(&y, x, q[0], q[1], q[2], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
    return y;
}

fn copiesFor(bytes: usize) usize {
    return std.math.clamp(WORKING_SET / bytes, 1, MAX_COPIES);
}

/// Optional arg: comma separated type names to run (default all), `zig build bench -Dtypes=q6_k,iq4_nl`.
var iters: usize = ITERS_TOKENS;

/// Args: [types|all] [M] [shape name substring]. M > 1 times the prefill (mat-mat) path, ms are per call.
pub fn main(init: std.process.Init) !void {
    const a = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const m: c_int = if (args.len > 2) try std.fmt.parseInt(c_int, args[2], 10) else 1;
    iters = @max(4, ITERS_TOKENS / @as(usize, @intCast(m)));
    const s = mlx.mlx_default_gpu_stream_new();
    var prng = std.Random.DefaultPrng.init(1);

    // Per-op floor: a custom kernel over one block against MLX's own matmul on one row.
    {
        var tiny_bytes: [18]u8 = undefined;
        prng.random().bytes(&tiny_bytes);
        const tiny = [_]mlx.mlx_array{mlx.mlx_array_new_data(&tiny_bytes, &[_]c_int{ 1, 18 }, 2, .uint8)};
        const xv1: [32]f32 = @splat(1);
        const x1 = mlx.mlx_array_new_data(&xv1, &[_]c_int{ 1, 32 }, 2, .float32);
        const custom_ms = try time(ggufMatvec, .{ .{ GgmlType.iq4_nl, x1 }, @as([]const mlx.mlx_array, &tiny), s }, s);
        var wd = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&wd, &[_]c_int{ 32, 32 }, 2, .float32, 0, 1, .{ .ctx = null }, s));
        const native = [_]mlx.mlx_array{wd};
        const native_ms = try time(mlxMatmul, .{ x1, @as([]const mlx.mlx_array, &native), s }, s);
        std.debug.print("per-op floor: custom kernel {d:.4} ms, mlx matmul {d:.4} ms\n", .{ custom_ms, native_ms });
    }
    for (shapes) |sh| {
        if (m > 1 and sh.out > 100_000) continue; // no lm_head in prefill
        if (args.len > 3 and std.mem.indexOf(u8, sh.name, args[3]) == null) continue;
        // MoE: m tokens x TOP_K pairs, x rows gathered and ids sorted, like a prefill.
        const pairs = m * TOP_K;
        const x_rows = if (sh.experts > 0) pairs else m;
        const xv = try a.alloc(f32, @intCast(sh.in * x_rows));
        defer a.free(xv);
        for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
        const x_shape: []const c_int = if (sh.experts > 0) &.{ pairs, 1, sh.in } else &.{ m, sh.in };
        const x32 = mlx.mlx_array_new_data(xv.ptr, x_shape.ptr, @intCast(x_shape.len), .float32);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(x));
        const idv = try a.alloc(u32, @intCast(pairs));
        defer a.free(idv);
        for (idv, 0..) |*id, i| id.* = @intCast(i * @as(usize, @intCast(@max(sh.experts, 1))) / @as(usize, @intCast(pairs)));
        const ids = mlx.mlx_array_new_data(idv.ptr, &[_]c_int{pairs}, 1, .uint32);
        const n_banks: usize = @intCast(@max(sh.experts, 1));

        // Reference: MLX affine 4-bit gs64 on random dense weights.
        const ref_bytes = @as(usize, @intCast(sh.in)) * @as(usize, @intCast(sh.out)) * 9 / 16 * n_banks;
        const qs = try a.alloc([3]mlx.mlx_array, copiesFor(ref_bytes));
        defer a.free(qs);
        for (qs) |*q| {
            var wd = mlx.mlx_array_new();
            const w_shape: []const c_int = if (sh.experts > 0) &.{ sh.experts, sh.out, sh.in } else &.{ sh.out, sh.in };
            try mlx.check(mlx.mlx_random_normal(&wd, w_shape.ptr, @intCast(w_shape.len), .bfloat16, 0, 1, .{ .ctx = null }, s));
            var qv = mlx.mlx_vector_array_new();
            try mlx.check(mlx.mlx_quantize(&qv, wd, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
            for (q, 0..) |*p, i| {
                p.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(p, qv, i));
                try mlx.check(mlx.mlx_array_eval(p.*));
            }
            _ = mlx.mlx_vector_array_free(qv);
            _ = mlx.mlx_array_free(wd);
        }
        const ref_ms = if (sh.experts > 0) try time(mlxGatherQmm, .{ .{ x, ids }, @as([]const [3]mlx.mlx_array, qs), s }, s) else try time(mlxQmv, .{ x, @as([]const [3]mlx.mlx_array, qs), s }, s);
        std.debug.print("{s}\n  {s:<8} {d:>7.3} ms   (MLX affine 4-bit, reference, {d} copies)\n", .{ sh.name, "mlx-q4", ref_ms, qs.len });
        for (qs) |q| for (q) |p| {
            _ = mlx.mlx_array_free(p);
        };

        for (types) |ty| {
            if (@mod(sh.in, @as(c_int, @intCast(ty.blockElems()))) != 0) continue;
            if (args.len > 1 and !std.mem.eql(u8, args[1], "all")) {
                var wanted = std.mem.splitScalar(u8, args[1], ',');
                while (wanted.next()) |want| {
                    if (std.mem.eql(u8, want, @tagName(ty))) break;
                } else continue;
            }
            const row_bytes = @as(usize, @intCast(sh.in)) / ty.blockElems() * ty.blockBytes();
            const bytes = try a.alloc(u8, row_bytes * @as(usize, @intCast(sh.out)) * n_banks);
            defer a.free(bytes);
            const ws = try a.alloc(mlx.mlx_array, copiesFor(bytes.len));
            defer a.free(ws);
            for (ws) |*w| {
                prng.random().bytes(bytes);
                const w_shape: []const c_int = if (sh.experts > 0) &.{ sh.experts, sh.out, @intCast(row_bytes) } else &.{ sh.out, @intCast(row_bytes) };
                w.* = mlx.mlx_array_new_data(bytes.ptr, w_shape.ptr, @intCast(w_shape.len), .uint8);
            }
            defer for (ws) |w| {
                _ = mlx.mlx_array_free(w);
            };
            const ms = if (sh.experts > 0) try time(ggufGather, .{ .{ ty, x, ids }, @as([]const mlx.mlx_array, ws), s }, s) else try time(ggufMatvec, .{ .{ ty, x }, @as([]const mlx.mlx_array, ws), s }, s);
            const bpw = @as(f64, @floatFromInt(ty.blockBytes() * 8)) / @as(f64, @floatFromInt(ty.blockElems()));
            std.debug.print("  {s:<8} {d:>7.3} ms   {d:>5.2}x ref   ({d:.2} bpw)", .{ @tagName(ty), ms, ms / ref_ms, bpw });
            if (m > kernels.MATVEC_MAX_ROWS and sh.experts == 0) {
                const dq_ms = try time(ggufDequantMatmul, .{ .{ ty, x }, @as([]const mlx.mlx_array, ws), s }, s);
                std.debug.print("   dequant+gemm {d:>7.3} ms {d:>5.2}x ref", .{ dq_ms, dq_ms / ref_ms });
            }
            std.debug.print("\n", .{});
        }
    }
}
