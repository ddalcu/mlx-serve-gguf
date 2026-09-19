//! Everything mlx-serve normally reads from sidecar JSON files, rebuilt in
//! memory from the GGUF metadata: config.json, tokenizer.json,
//! tokenizer_config.json, generation_config.json.
const std = @import("std");
const gguf = @import("gguf.zig");
const Arch = @import("arch.zig").Arch;

pub const Error = error{ UnsupportedArch, UnsupportedTokenizer, UnsupportedTensor, MissingMetadata, BadMetadata } || std.mem.Allocator.Error;

/// Why we can't serve this file, null when we can. The caller falls back to
/// another engine on a non-null answer, so this has to be cheap and complete.
pub fn unsupportedReason(f: *const gguf.File, buf: []u8) ?[]const u8 {
    const arch = Arch.read(f) catch |err| return switch (err) {
        error.UnsupportedArch => std.fmt.bufPrint(buf, "arch {s}", .{f.getString("general.architecture").?}) catch "arch",
        else => "metadata",
    };
    if (TokenizerKind.of(f) == null) return "tokenizer";
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        if (!e.value_ptr.ty.supported()) return std.fmt.bufPrint(buf, "tensor type of {s}", .{e.key_ptr.*}) catch "tensor type";
        if (arch.mapTensor(&name_buf, e.key_ptr.*) == null) return std.fmt.bufPrint(buf, "tensor {s}", .{e.key_ptr.*}) catch "tensor";
    }
    return null;
}

pub fn configJson(allocator: std.mem.Allocator, f: *const gguf.File) ![]u8 {
    return (try Arch.read(f)).configJson(allocator);
}

/// The tokenizer.json blocks around the BPE vocab, by llama.cpp tokenizer
/// model + pre-tokenizer id. mlx-serve keys its encode path on the
/// pre_tokenizer (ByteLevel or not), the rest mirrors the HF files.
const TokenizerKind = struct {
    /// Everything before "model".
    head: []const u8,
    /// Extra BPE model fields, in front of "vocab".
    model_fields: []const u8 = "",
    /// A token HF lists as special that the GGUF types as a normal one.
    also_special: []const u8 = "",

    fn of(f: *const gguf.File) ?TokenizerKind {
        const model = f.getString("tokenizer.ggml.model") orelse return null;
        if (std.mem.eql(u8, model, "gemma4")) return .{
            .head =
            \\"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"\u2581"},
            \\"pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
            \\"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"\u2581"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
            ,
            .model_fields =
            \\"unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
            ,
            .also_special = "<eos>",
        };
        const pre = f.getString("tokenizer.ggml.pre") orelse return null;
        if (std.mem.eql(u8, model, "gpt2") and std.mem.eql(u8, pre, "qwen35")) return .{ .head =
        \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        };
        return null;
    }
};

// llama.cpp token types.
const TOKEN_CONTROL = 3;
const TOKEN_USER_DEFINED = 4;

/// An HF tokenizer.json equivalent: BPE vocab + merges, control and
/// user-defined tokens as added_tokens (control = special).
pub fn tokenizerJson(allocator: std.mem.Allocator, f: *const gguf.File) ![]u8 {
    const kind = TokenizerKind.of(f) orelse return error.UnsupportedTokenizer;
    const tokens = f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata;
    const types = f.getArray("tokenizer.ggml.token_type") orelse return error.MissingMetadata;
    const merges = f.getArray("tokenizer.ggml.merges") orelse return error.MissingMetadata;
    if (types.len != tokens.len) return error.BadMetadata;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var added: std.ArrayList(u8) = .empty;
    defer added.deinit(allocator);

    try out.append(allocator, '{');
    try out.appendSlice(allocator, kind.head);
    try out.appendSlice(allocator, "\"model\":{\"type\":\"BPE\",");
    try out.appendSlice(allocator, kind.model_fields);
    try out.appendSlice(allocator, "\"vocab\":{");
    var it = tokens.strings();
    var id: usize = 0;
    var first = true;
    while (it.next()) |tok| : (id += 1) {
        const ty = types.int(id);
        const special = ty == TOKEN_CONTROL or (kind.also_special.len > 0 and std.mem.eql(u8, tok, kind.also_special));
        if (special or ty == TOKEN_USER_DEFINED) {
            if (added.items.len > 0) try added.append(allocator, ',');
            try added.print(allocator, "{{\"id\":{d},\"special\":{},\"content\":", .{ id, special });
            try appendJsonString(allocator, &added, tok);
            try added.append(allocator, '}');
            continue;
        }
        if (!first) try out.append(allocator, ',');
        first = false;
        try appendJsonString(allocator, &out, tok);
        try out.print(allocator, ":{d}", .{id});
    }
    try out.appendSlice(allocator, "},\"merges\":[");
    var mit = merges.strings();
    first = true;
    while (mit.next()) |m| {
        if (!first) try out.append(allocator, ',');
        first = false;
        try appendJsonString(allocator, &out, m);
    }
    try out.appendSlice(allocator, "]},\"added_tokens\":[");
    try out.appendSlice(allocator, added.items);
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

/// The tokenizer_config.json fields mlx-serve reads: template + named special tokens.
pub fn tokenizerConfigJson(allocator: std.mem.Allocator, f: *const gguf.File) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"chat_template\":");
    try appendJsonString(allocator, &out, f.getString("tokenizer.chat_template") orelse "");
    const named = [_]struct { []const u8, []const u8 }{
        .{ "eos_token", "tokenizer.ggml.eos_token_id" },
        .{ "bos_token", "tokenizer.ggml.bos_token_id" },
        .{ "pad_token", "tokenizer.ggml.padding_token_id" },
    };
    for (named) |n| {
        try out.print(allocator, ",\"{s}\":", .{n[0]});
        if (tokenById(f, f.getInt(n[1]))) |tok| try appendJsonString(allocator, &out, tok) else try out.appendSlice(allocator, "null");
    }
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator);
}

/// The sampling defaults the model author recommends, null when the GGUF has none.
pub fn generationConfigJson(allocator: std.mem.Allocator, f: *const gguf.File) !?[]u8 {
    const temp = f.getFloat("general.sampling.temp");
    const top_p = f.getFloat("general.sampling.top_p");
    const top_k = f.getInt("general.sampling.top_k");
    if (temp == null and top_p == null and top_k == null) return null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"do_sample\":true");
    if (temp) |v| try out.print(allocator, ",\"temperature\":{d}", .{v});
    // f32 in the file: 0.95 reads back as 0.949999988, round it to what the author wrote.
    if (top_p) |v| try out.print(allocator, ",\"top_p\":{d}", .{@round(v * 1e4) / 1e4});
    if (top_k) |v| try out.print(allocator, ",\"top_k\":{d}", .{v});
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

fn tokenById(f: *const gguf.File, id: ?i64) ?[]const u8 {
    const want = std.math.cast(usize, id orelse return null) orelse return null;
    const tokens = f.getArray("tokenizer.ggml.tokens") orelse return null;
    if (want >= tokens.len) return null;
    var it = tokens.strings();
    for (0..want) |_| _ = it.next();
    return it.next();
}

fn appendJsonString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(allocator, "\\\""),
        '\\' => try out.appendSlice(allocator, "\\\\"),
        '\n' => try out.appendSlice(allocator, "\\n"),
        '\r' => try out.appendSlice(allocator, "\\r"),
        '\t' => try out.appendSlice(allocator, "\\t"),
        else => if (c < 0x20) try out.print(allocator, "\\u{x:0>4}", .{c}) else try out.append(allocator, c),
    };
    try out.append(allocator, '"');
}

test "appendJsonString round-trips through the JSON parser" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    const raw = "a\"b\\c\nd\x01\xc4\xa0";
    try appendJsonString(std.testing.allocator, &out, raw);
    const parsed = try std.json.parseFromSlice([]const u8, std.testing.allocator, out.items, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(raw, parsed.value);
}

test "real model: synthesized JSON parses and matches the GGUF (set MLX_SERVE_GGUF_TEST_MODEL)" {
    const path = std.mem.span(std.c.getenv("MLX_SERVE_GGUF_TEST_MODEL") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    var f = try gguf.File.open(a, path);
    defer f.deinit();
    var buf: [200]u8 = undefined;
    try std.testing.expectEqual(@as(?[]const u8, null), unsupportedReason(&f, &buf));

    const cfg = try configJson(a, &f);
    defer a.free(cfg);
    const cfg_parsed = try std.json.parseFromSlice(std.json.Value, a, cfg, .{});
    defer cfg_parsed.deinit();

    const tj = try tokenizerJson(a, &f);
    defer a.free(tj);
    const tok = try std.json.parseFromSlice(std.json.Value, a, tj, .{});
    defer tok.deinit();
    const n_vocab = tok.value.object.get("model").?.object.get("vocab").?.object.count();
    const n_added = tok.value.object.get("added_tokens").?.array.items.len;
    try std.testing.expectEqual(f.getArray("tokenizer.ggml.tokens").?.len, n_vocab + n_added);

    const tc = try tokenizerConfigJson(a, &f);
    defer a.free(tc);
    const tc_parsed = try std.json.parseFromSlice(std.json.Value, a, tc, .{});
    defer tc_parsed.deinit();
    try std.testing.expect(tc_parsed.value.object.get("eos_token").?.string.len > 0);
    try std.testing.expect(tc_parsed.value.object.get("chat_template").?.string.len > 100);

    if (try generationConfigJson(a, &f)) |gc| {
        defer a.free(gc);
        const gc_parsed = try std.json.parseFromSlice(std.json.Value, a, gc, .{});
        defer gc_parsed.deinit();
        try std.testing.expect(gc_parsed.value.object.get("do_sample").?.bool);
    }
}
