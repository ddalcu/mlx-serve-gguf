//! The model architectures we serve, picked by `general.architecture`. An arch
//! module brings three things: `Hparams.read` (GGUF metadata), `configJson`
//! (the HF config.json mlx-serve parses) and `mapTensor` (GGUF -> HF names).
const std = @import("std");
const gguf = @import("gguf.zig");
pub const qwen35 = @import("arch/qwen35.zig");
pub const gemma4 = @import("arch/gemma4.zig");

/// What loading has to do to a tensor's bytes. Most of them undo llama.cpp's
/// value-head tiling of the GatedDeltaNet, see arch/qwen35.zig.
pub const Transform = union(enum) {
    /// Bytes go through untouched.
    none,
    /// Un-tile value heads along the rows: `unit` rows per head, starting at `start`.
    untile_rows: struct { start: u32, unit: u32 },
    /// F32 vector with one entry per value head.
    untile_vec,
    /// ssm_a = -exp(A_log), per value head.
    a_log,
    /// [channels, kernel] depthwise conv, value channels un-tiled, stored [channels, kernel, 1].
    conv1d,
    /// The INPUT side is tiled and quant blocks straddle heads, so the bytes
    /// stay and the matmul tiles its activations instead.
    tiled_input,
    /// The shared expert's gate, a [hidden] vector in GGUF and a [1, hidden] linear in HF.
    shared_gate,
};

pub const Mapped = struct { name: []const u8, transform: Transform };

pub const Arch = union(enum) {
    qwen35: qwen35.Hparams,
    gemma4: gemma4.Hparams,

    /// error.UnsupportedArch for anything not in the table.
    pub fn read(f: *const gguf.File) !Arch {
        const name = f.getString("general.architecture") orelse return error.MissingMetadata;
        if (qwen35.handles(name)) return .{ .qwen35 = try qwen35.Hparams.read(f, name) };
        if (std.mem.eql(u8, name, gemma4.arch_name)) return .{ .gemma4 = try gemma4.Hparams.read(f) };
        return error.UnsupportedArch;
    }

    /// The config.json an mlx-community conversion of the same model would carry.
    pub fn configJson(self: Arch, allocator: std.mem.Allocator) ![]u8 {
        return switch (self) {
            .qwen35 => |hp| qwen35.configJson(allocator, hp),
            .gemma4 => |hp| gemma4.configJson(allocator, hp),
        };
    }

    /// HF name + load transform for a GGUF tensor, null when the tensor is
    /// unknown. An empty name (`skip`) = known and not needed.
    pub fn mapTensor(self: Arch, buf: []u8, name: []const u8) ?Mapped {
        return switch (self) {
            .qwen35 => |hp| qwen35.mapTensor(buf, name, hp),
            .gemma4 => |hp| gemma4.mapTensor(buf, name, hp),
        };
    }
};

/// For tensors mlx-serve derives itself (rope frequency tables).
pub const skip = Mapped{ .name = "", .transform = .none };

pub fn u32Key(f: *const gguf.File, key: []const u8) !u32 {
    return std.math.cast(u32, f.getInt(key) orelse return error.MissingMetadata) orelse error.BadMetadata;
}

/// Parses "blk.<n>.<leaf>" into its layer and leaf, null for anything else.
pub fn splitLayer(name: []const u8, n_layers: u32) ?struct { layer: u32, leaf: []const u8 } {
    if (!std.mem.startsWith(u8, name, "blk.")) return null;
    const rest = name[4..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    if (layer >= n_layers) return null;
    return .{ .layer = layer, .leaf = rest[dot + 1 ..] };
}

test {
    _ = qwen35;
    _ = gemma4;
}
