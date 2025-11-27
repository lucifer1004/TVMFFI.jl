#=
Test CUDA support for zero-copy optimization

Run: julia --project=. benchmarks/cuda_test.jl
=#

using TVMFFI
using TVMFFI: TensorView
using CUDA

println("CUDA available: ", CUDA.functional())
println("CUDA device: ", CUDA.device())
println()

# Register test functions
register_global_func("julia.cuda.identity", (x) -> x; override = true)
register_global_func("julia.cuda.add_one", (x) -> (x .+= 1; x); override = true)

tvm_identity = get_global_func("julia.cuda.identity")
tvm_add_one = get_global_func("julia.cuda.add_one")

# Test 1: Basic CuArray round-trip
println("="^60)
println("Test 1: Basic CuArray identity")
println("="^60)

x = CUDA.rand(Float32, 16)
println("Input CuArray: ", typeof(x), " size=", size(x))
println("Input data (first 4): ", Array(x)[1:4])

try
    result = tvm_identity(x)
    println("Output type: ", typeof(result))
    if result isa CUDA.CuArray
        println("Output data (first 4): ", Array(result)[1:4])
        println("Same pointer: ", pointer(x) == pointer(result))
    elseif result isa Array
        println("Output data (first 4): ", result[1:4])
        println("WARNING: Returned CPU Array instead of CuArray!")
    end
catch e
    println("ERROR: ", e)
    println(sprint(showerror, e, catch_backtrace()))
end

# Test 2: In-place operation
println()
println("="^60)
println("Test 2: In-place CuArray (x .+= 1)")
println("="^60)

x = CUDA.ones(Float32, 16)
println("Input: all ones")

try
    result = tvm_add_one(x)
    println("Output type: ", typeof(result))
    if result isa CUDA.CuArray
        println("Output data (first 4): ", Array(result)[1:4])
    elseif result isa Array
        println("Output data (first 4): ", result[1:4])
    end
catch e
    println("ERROR: ", e)
    println(sprint(showerror, e, catch_backtrace()))
end

# Test 3: Benchmark
println()
println("="^60)
println("Test 3: Benchmark")
println("="^60)

using BenchmarkTools

for sz in [64, 1024, 4096]
    local arr = CUDA.rand(Float32, sz)

    # Warmup
    tvm_identity(arr)
    CUDA.synchronize()

    # Benchmark
    local t = @benchmark begin
        $tvm_identity($arr)
        CUDA.synchronize()
    end samples=100

    println("CuArray[$sz] identity: ", round(median(t).time / 1000, digits = 1), " μs")
end

println()
println("In-place benchmark:")
# NOTE: @benchmark causes segfaults with functions returning new GPU arrays.
# Root cause: BenchmarkTools' internal loop discards return values after first iteration,
# then gcscrub() triggers GC which reclaims CuArrays and crashes in CUDA finalizers.
# Even saving return value doesn't help because the internal loop still discards.
# Solution: Use manual timing with explicit result retention.
for sz in [64, 1024, 4096]
    local arr = CUDA.rand(Float32, sz)
    local result  # Keep return value alive across iterations

    # Warmup
    for _ in 1:10
        result = tvm_add_one(arr)
        CUDA.synchronize()
    end

    # Manual timing - keeps each result alive until next iteration
    N = 100
    t0 = time_ns()
    for _ in 1:N
        result = tvm_add_one(arr)  # Previous result can now be GC'd safely
        CUDA.synchronize()
    end
    t1 = time_ns()

    println("CuArray[$sz] .+= 1: ", round((t1 - t0) / N / 1000, digits = 1), " μs")
end

println()
println("CUDA test complete.")
