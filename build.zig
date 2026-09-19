const std = @import("std");

/// Standalone build, for testing this repo on its own. mlx-serve does not use
/// this file: it roots a module at src/root.zig and passes itself as `mlx_host`.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const model = b.option([]const u8, "model", "GGUF file for the real-file tests");
    // The MLX bindings + staged libs come from an mlx-serve checkout (sibling
    // repo by default, pass ../.. when this is the lib/ submodule).
    const host = b.option([]const u8, "mlx-serve", "Path to an mlx-serve checkout") orelse "../mlx-serve";

    // `test-core`: reader + reference dequant, no MLX needed.
    const core = b.createModule(.{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const core_step = b.step("test-core", "Run the MLX-free tests");
    core_step.dependOn(&runTests(b, core, model).step);

    // `test`: everything, kernels run on the GPU through mlx-serve's MLX.
    const mlx = b.createModule(.{
        .root_source_file = b.path(b.pathJoin(&.{ host, "src/mlx.zig" })),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const mlx_lib = b.path(b.pathJoin(&.{ host, "lib/mlx/lib" }));
    mlx.addLibraryPath(mlx_lib);
    mlx.addRPath(mlx_lib);
    mlx.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    mlx.linkSystemLibrary("c++", .{});
    for ([_][]const u8{ "Metal", "Foundation", "CoreFoundation", "IOKit", "IOSurface" }) |f| mlx.linkFramework(f, .{});

    const mlx_host = b.createModule(.{
        .root_source_file = b.path("standalone/mlx_host.zig"),
        .imports = &.{.{ .name = "mlx", .module = mlx }},
    });
    const root = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = mlx_host }},
    });
    const bench = b.addExecutable(.{
        .name = "matvec-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/matvec.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{ .{ .name = "mlx_serve_gguf", .module = root }, .{ .name = "mlx_host", .module = mlx_host } },
        }),
    });
    const run_bench = b.addRunArtifact(bench);
    run_bench.addArg(b.option([]const u8, "types", "bench: comma separated type names, default all") orelse "all");
    const bench_m = b.option([]const u8, "m", "bench: activation rows, default 1 (decode)");
    const bench_shapes = b.option([]const u8, "shapes", "bench: only shapes whose name contains this, e.g. e2b");
    if (bench_m orelse if (bench_shapes != null) @as(?[]const u8, "1") else null) |m| run_bench.addArg(m);
    if (bench_shapes) |sh| run_bench.addArg(sh);
    b.step("bench", "Time the matvec kernels against MLX's 4-bit matvec (-Dtypes=q6_k,iq4_nl)").dependOn(&run_bench.step);

    const test_step = b.step("test", "Run all tests (needs mlx-serve with lib/mlx staged)");
    test_step.dependOn(&runTests(b, root, model).step);
}

fn runTests(b: *std.Build, mod: *std.Build.Module, model: ?[]const u8) *std.Build.Step.Run {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = mod }));
    if (model) |m| run.setEnvironmentVariable("MLX_SERVE_GGUF_TEST_MODEL", m);
    return run;
}
