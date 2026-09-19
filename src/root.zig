//! mlx-serve-gguf: serve GGUF files as is on MLX, a module of mlx-serve.
//! Needs an `mlx_host` import whose root exposes `pub const mlx` (mlx-serve's
//! src/mlx.zig). mlx-serve passes its own main module, see build.zig for standalone.
const core = @import("core.zig");
pub const gguf = core.gguf;
pub const quants = core.quants;
pub const meta = core.meta;
pub const kernels = @import("kernels.zig");
pub const weights = @import("weights.zig");

test {
    _ = core;
    _ = kernels;
    _ = weights;
}
