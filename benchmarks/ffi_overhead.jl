#=
TVMFFI FFI Overhead Benchmark Suite

Measures the overhead of the Julia ↔ TVM FFI layer using BenchmarkTools.

## How to run:
    julia --project=. benchmarks/ffi_overhead.jl

## Requirements:
    using Pkg; Pkg.add("BenchmarkTools")
=#

using TVMFFI
using TVMFFI: TVMAny, TVMAnyView, take_value, copy_value, raw_data
using TVMFFI: TVMString, TVMObject, TensorView, DLTensor
using TVMFFI.LibTVMFFI
using BenchmarkTools
using Printf

# Disable interpolation warnings for cleaner output
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10000

# ============================================================================
# Utility
# ============================================================================

function print_bench(name::String, trial::BenchmarkTools.Trial; baseline_ns::Float64 = 0.0)
    t = median(trial)
    ns = time(t)
    allocs = BenchmarkTools.allocs(t)
    mem = BenchmarkTools.memory(t)

    overhead_str = baseline_ns > 0 ? @sprintf("%.1fx", ns / baseline_ns) : "-"

    @printf("  %-42s %8.1f ns  allocs=%d  mem=%d B  [%s]\n",
        name, ns, allocs, mem, overhead_str)
end

# ============================================================================
# Section 1: Baselines (Pure Julia)
# ============================================================================

println("\n" * "="^80)
println("TVMFFI FFI Overhead Benchmark (BenchmarkTools)")
println("="^80)

println("\n" * "-"^80)
println("Section 1: Baselines (Pure Julia)")
println("-"^80)

# 1.1 Empty Julia function
julia_noop() = nothing
baseline_noop = @benchmark $julia_noop()
print_bench("Julia empty function call", baseline_noop)
baseline_noop_ns = time(median(baseline_noop))

# 1.2 Julia Int64 identity
julia_id_int(x) = x
baseline_int = @benchmark $julia_id_int(42)
print_bench("Julia Int64 identity", baseline_int)
baseline_int_ns = time(median(baseline_int))

# 1.3 Julia Float64 identity
julia_id_float(x) = x
baseline_float = @benchmark $julia_id_float(3.14)
print_bench("Julia Float64 identity", baseline_float)

# 1.4 Julia Array identity
test_array = rand(Float32, 1024)
baseline_array = @benchmark $julia_id_int($test_array)
print_bench("Julia Array identity (1024 elems)", baseline_array)

# ============================================================================
# Section 2: Type Conversion (Julia → TVMAny)
# ============================================================================

println("\n" * "-"^80)
println("Section 2: Type Conversion Overhead (Julia → TVMAny)")
println("-"^80)

# POD types
b_int = @benchmark TVMAny(Int64(42))
print_bench("TVMAny(Int64)", b_int; baseline_ns = baseline_noop_ns)

b_float = @benchmark TVMAny(3.14)
print_bench("TVMAny(Float64)", b_float; baseline_ns = baseline_noop_ns)

b_bool = @benchmark TVMAny(true)
print_bench("TVMAny(Bool)", b_bool; baseline_ns = baseline_noop_ns)

# String (requires C API)
b_string = @benchmark TVMAny("hello")
print_bench("TVMAny(String)", b_string; baseline_ns = baseline_noop_ns)

# TensorView creation
b_tensorview = @benchmark TensorView($test_array)
print_bench("TensorView(Array{Float32,1024})", b_tensorview; baseline_ns = baseline_noop_ns)

# TensorView → TVMAny
prealloc_view = TensorView(test_array)
b_view_any = @benchmark TVMAny($prealloc_view)
print_bench("TVMAny(TensorView)", b_view_any; baseline_ns = baseline_noop_ns)

# ============================================================================
# Section 3: Reference Counting
# ============================================================================

println("\n" * "-"^80)
println("Section 3: Reference Counting Overhead")
println("-"^80)

echo_func = get_global_func("testing.echo")
if echo_func !== nothing
    handle = echo_func.handle

    # IncRef + DecRef pair (always balanced)
    b_refcount = @benchmark begin
        LibTVMFFI.TVMFFIObjectIncRef($handle)
        LibTVMFFI.TVMFFIObjectDecRef($handle)
    end
    print_bench("IncRef + DecRef pair", b_refcount; baseline_ns = baseline_noop_ns)
else
    println("  [SKIP] testing.echo not available")
end

# ============================================================================
# Section 4: Raw ccall
# ============================================================================

println("\n" * "-"^80)
println("Section 4: Raw ccall Overhead")
println("-"^80)

b_version = @benchmark LibTVMFFI.TVMFFIGetVersion()
print_bench("TVMFFIGetVersion() [minimal ccall]", b_version; baseline_ns = baseline_noop_ns)

# ============================================================================
# Section 5: End-to-End TVM Function Call
# ============================================================================

println("\n" * "-"^80)
println("Section 5: End-to-End TVM Function Call")
println("-"^80)

# Register test functions
register_global_func("julia.bench.noop", () -> nothing; override = true)
register_global_func("julia.bench.identity_int", (x::Int64) -> x; override = true)
register_global_func("julia.bench.identity_float", (x::Float64) -> x; override = true)
register_global_func("julia.bench.sum", (a, b) -> a + b; override = true)
register_global_func("julia.bench.identity_array", (x) -> x; override = true)
register_global_func("julia.bench.add_one_inplace", (x) -> (x .+= 1; x); override = true)

tvm_noop = get_global_func("julia.bench.noop")
tvm_id_int = get_global_func("julia.bench.identity_int")
tvm_id_float = get_global_func("julia.bench.identity_float")
tvm_sum = get_global_func("julia.bench.sum")
tvm_id_array = get_global_func("julia.bench.identity_array")

# 5.1 Empty function
if tvm_noop !== nothing
    b = @benchmark $tvm_noop()
    print_bench("TVM func() [no args]", b; baseline_ns = baseline_noop_ns)
end

# 5.2 Int64 identity
if tvm_id_int !== nothing
    b = @benchmark $tvm_id_int(Int64(42))
    print_bench("TVM func(Int64) → Int64", b; baseline_ns = baseline_int_ns)
end

# 5.3 Float64 identity
if tvm_id_float !== nothing
    b = @benchmark $tvm_id_float(3.14)
    print_bench("TVM func(Float64) → Float64", b; baseline_ns = baseline_int_ns)
end

# 5.4 Two args
if tvm_sum !== nothing
    b = @benchmark $tvm_sum(Int64(1), Int64(2))
    print_bench("TVM func(Int64, Int64) → Int64", b; baseline_ns = baseline_int_ns)
end

# 5.5 Array round-trip (auto TensorView creation)
if tvm_id_array !== nothing
    for size in [64, 256, 1024, 4096]
        arr = rand(Float32, size)
        local trial = @benchmark $tvm_id_array($arr) samples=1000
        print_bench("TVM func(Float32[$size]) → Array", trial; baseline_ns = baseline_int_ns)
    end
end

# 5.6 Pre-created TensorView (avoids TensorView creation overhead)
if tvm_id_array !== nothing
    println()
    println("  With pre-created TensorView (identity):")
    for size in [64, 256, 1024, 4096]
        arr = rand(Float32, size)
        view = TensorView(arr)
        local trial = @benchmark $tvm_id_array($view) samples=1000
        print_bench("TVM func(TensorView[$size]) → Array", trial; baseline_ns = baseline_int_ns)
    end
end

# 5.7 In-place array operation (x .+= 1) - should trigger identity optimization
tvm_add_one = get_global_func("julia.bench.add_one_inplace")
if tvm_add_one !== nothing
    println()
    println("  In-place (x .+= 1) - returns same array:")
    for size in [64, 256, 1024, 4096]
        arr = rand(Float32, size)
        local trial = @benchmark $tvm_add_one($arr) samples=1000
        print_bench("TVM func(Float32[$size] .+= 1) → Array", trial; baseline_ns = baseline_int_ns)
    end
end

# ============================================================================
# Section 6: Callback Path
# ============================================================================

println("\n" * "-"^80)
println("Section 6: Callback Path Breakdown")
println("-"^80)

# Dict lookup (callback registry pattern)
const _bench_registry = Dict{Ptr{Cvoid}, Any}()
for i in 1:1000
    _bench_registry[Ptr{Cvoid}(UInt(i))] = () -> i
end
lookup_key = Ptr{Cvoid}(UInt(500))

b_dict = @benchmark get($_bench_registry, $lookup_key, nothing)
print_bench("Dict lookup (callback registry)", b_dict; baseline_ns = baseline_noop_ns)

# With lock
const _bench_lock = ReentrantLock()
b_dict_lock = @benchmark lock($_bench_lock) do
    get($_bench_registry, $lookup_key, nothing)
end
print_bench("Dict lookup with ReentrantLock", b_dict_lock; baseline_ns = baseline_noop_ns)

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^80)
println("Summary")
println("="^80)

if tvm_noop !== nothing && tvm_id_int !== nothing
    noop_ns = time(median(@benchmark $tvm_noop()))
    int_ns = time(median(@benchmark $tvm_id_int(Int64(42))))

    println("""

    Key Findings:

    1. Empty FFI call overhead: $(round(noop_ns, digits=0)) ns (~$(round(noop_ns/baseline_noop_ns, digits=0))x baseline)
    2. Scalar FFI call overhead: $(round(int_ns, digits=0)) ns (~$(round(int_ns/baseline_int_ns, digits=0))x baseline)

    Practical Impact:
    - For 1ms inference: FFI overhead is ~0.02% (negligible)
    - For 1μs micro-ops: FFI overhead is ~20-50% (consider batching)
    - For 100ns hot loops: FFI overhead dominates (use raw ccall)
    """)
end

println("\nBenchmark complete.")
