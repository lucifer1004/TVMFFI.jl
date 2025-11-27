#=
Benchmark API overhead of Julia FFI API calling overhead through DLPack API.

Mirrors the Python bench_example.py for direct comparison.

## How to run:
    cd TVMFFI/benchmarks
    julia --project=. bench_example.jl

## Summary of takeaways from Python version:
- numpy.add roughly takes 0.36 µs per call
- torch.add on gpu takes about 3.7 µs per call

The goal is to compare Julia FFI overhead against Python FFI overhead.
=#

using TVMFFI
using TVMFFI: TensorView, TVMTensor, DLTensor
using TVMFFI.LibTVMFFI
using Printf

# ============================================================================
# Utility Functions
# ============================================================================

function print_speed(name::String, speed::Float64)
    @printf("%-60s %.2e sec/call\n", name, speed)
end

function print_error(name::String, error)
    @printf("%-60s %s\n", name, error)
end

"""
Run a benchmark with warmup.
Returns: average time per call in seconds.
"""
function bench(f::Function, repeat::Int; warmup::Int = 100)
    # Warmup
    for _ in 1:warmup
        f()
    end
    
    # Benchmark
    start = time_ns()
    for _ in 1:repeat
        f()
    end
    elapsed = time_ns() - start
    
    return elapsed / 1e9 / repeat  # seconds per call
end

# ============================================================================
# Baseline Benchmarks (corresponds to numpy.add, torch.add)
# ============================================================================

function baseline_julia_broadcast_add(repeat::Int)
    """Julia broadcast add with one element array (like numpy.add)."""
    x = Float64[1.0]
    y = Float64[1.0]
    z = Float64[0.0]
    
    speed = bench(repeat) do
        z .= x .+ y
    end
    print_speed("Julia broadcast add [like numpy.add]", speed)
    return speed
end

function baseline_julia_inplace_add(repeat::Int)
    """Julia in-place add with one element array."""
    x = Float64[1.0]
    y = Float64[1.0]
    z = Float64[0.0]
    
    speed = bench(repeat) do
        @inbounds z[1] = x[1] + y[1]
    end
    print_speed("Julia inplace add (z[1] = x[1] + y[1])", speed)
    return speed
end

function baseline_cuda_broadcast_add(repeat::Int)
    """CUDA.jl broadcast add with one element array (like torch.add[cuda])."""
    if !CUDA_AVAILABLE
        return NaN
    end
    
    x = CUDA.ones(Float32, 1)
    y = CUDA.ones(Float32, 1)
    z = CUDA.zeros(Float32, 1)
    
    # Warmup
    z .= x .+ y
    CUDA.synchronize()
    
    start = time_ns()
    for _ in 1:repeat
        z .= x .+ y
    end
    CUDA.synchronize()
    speed = (time_ns() - start) / 1e9 / repeat
    
    print_speed("CUDA.jl broadcast add [like torch.add[cuda]]", speed)
    return speed
end

# ============================================================================
# TVM FFI NOP Benchmarks
# ============================================================================

function tvm_ffi_nop(repeat::Int)
    """Overhead of TVM FFI Julia call via calling a NOP.
    
    testing.nop is defined in C++ and does nothing.
    """
    nop = get_global_func("testing.nop")
    x = TVMTensor(Float64[1.0])
    y = TVMTensor(Float64[1.0])
    z = TVMTensor(Float64[1.0])
    
    speed = bench(repeat) do
        nop(x, y, z)
    end
    print_speed("tvm_ffi.nop(TVMTensor, TVMTensor, TVMTensor)", speed)
end

function bench_ffi_nop_from_dlpack(name::String, x, y, z, repeat::Int)
    """Run dlpack conversion + tvm_ffi.nop.
    
    Measures overhead of running from_dlpack for each arg then invoke.
    """
    nop = get_global_func("testing.nop")
    
    # Warmup
    tx = TVMTensor(x)
    ty = TVMTensor(y)
    tz = TVMTensor(z)
    nop(tx, ty, tz)
    
    speed = bench(repeat) do
        tx = TVMTensor(x)
        ty = TVMTensor(y)
        tz = TVMTensor(z)
        nop(tx, ty, tz)
    end
    print_speed(name, speed)
end

function tvm_ffi_nop_from_julia_array(repeat::Int)
    """Run dlpack conversion + tvm_ffi.nop from Julia Array."""
    x = Float64[1.0]
    y = Float64[1.0]
    z = Float64[1.0]
    bench_ffi_nop_from_dlpack("tvm_ffi.nop+TVMTensor(Julia Array)", x, y, z, repeat)
end

function tvm_ffi_nop_from_tvm_tensor(repeat::Int)
    """Run nop with pre-converted TVMTensor (no conversion overhead)."""
    nop = get_global_func("testing.nop")
    x = TVMTensor(Float64[1.0])
    y = TVMTensor(Float64[1.0])
    z = TVMTensor(Float64[1.0])
    
    # Warmup
    nop(x, y, z)
    
    speed = bench(repeat) do
        nop(x, y, z)
    end
    print_speed("tvm_ffi.nop(TVMTensor) [pre-converted, no alloc]", speed)
end

function bench_tvm_ffi_nop_autodlpack(name::String, x, y, z, repeat::Int)
    """Measures overhead of running dlpack via auto convert by directly
    taking Julia Array as inputs.
    """
    nop = get_global_func("testing.nop")
    
    # Warmup
    nop(x, y, z)
    
    speed = bench(repeat) do
        nop(x, y, z)
    end
    print_speed(name, speed)
end

function tvm_ffi_nop_autodlpack_from_julia_array(repeat::Int)
    """Measures overhead with auto DLPack from Julia Array (CPU)."""
    x = Float64[1.0]
    y = Float64[1.0]
    z = Float64[1.0]
    bench_tvm_ffi_nop_autodlpack("tvm_ffi.nop.autodlpack(Julia Array)", x, y, z, repeat)
end

function tvm_ffi_nop_tensorview(repeat::Int)
    """Measures overhead using TensorView (lightweight, no refcount)."""
    nop = get_global_func("testing.nop")
    x = Float64[1.0]
    y = Float64[1.0]
    z = Float64[1.0]
    
    # Warmup
    GC.@preserve x y z begin
        vx = TensorView(x)
        vy = TensorView(y)
        vz = TensorView(z)
        nop(vx, vy, vz)
    end
    
    speed = bench(repeat) do
        GC.@preserve x y z begin
            vx = TensorView(x)
            vy = TensorView(y)
            vz = TensorView(z)
            nop(vx, vy, vz)
        end
    end
    print_speed("tvm_ffi.nop(TensorView) [manual GC.@preserve]", speed)
end

# ============================================================================
# Scalar Argument Benchmarks
# ============================================================================

function tvm_ffi_echo_int(repeat::Int)
    """Measures overhead of passing Int64 to TVM function."""
    echo = get_global_func("testing.echo")
    
    speed = bench(repeat) do
        echo(42)
    end
    print_speed("tvm_ffi.echo(Int64)", speed)
end

function tvm_ffi_echo_float(repeat::Int)
    """Measures overhead of passing Float64 to TVM function."""
    echo = get_global_func("testing.echo")
    
    speed = bench(repeat) do
        echo(3.14)
    end
    print_speed("tvm_ffi.echo(Float64)", speed)
end

function tvm_ffi_echo_string(repeat::Int)
    """Measures overhead of passing String to TVM function."""
    echo = get_global_func("testing.echo")
    s = "hello"
    
    speed = bench(repeat) do
        echo(s)
    end
    print_speed("tvm_ffi.echo(String)", speed)
end

function tvm_ffi_nop_no_args(repeat::Int)
    """Measures overhead of calling TVM function with no arguments."""
    nop = get_global_func("testing.nop")
    
    speed = bench(repeat) do
        nop()
    end
    print_speed("tvm_ffi.nop() [no args]", speed)
end

# ============================================================================
# Argument Count Scaling
# ============================================================================

function tvm_ffi_nop_arg_scaling(repeat::Int)
    """Measures how overhead scales with argument count."""
    nop = get_global_func("testing.nop")
    
    println("\n  Argument count scaling:")
    
    for n in [1, 2, 3, 4, 5, 10, 11, 15, 20]
        args = ntuple(_ -> 42, n)
        
        speed = bench(repeat) do
            nop(args...)
        end
        @printf("    %2d args: %.2e sec/call\n", n, speed)
    end
end

# ============================================================================
# Identity Function (Return Value Overhead)
# ============================================================================

function tvm_ffi_identity_int(repeat::Int)
    """Measures round-trip overhead for Int64."""
    identity = get_global_func("testing.echo")
    
    speed = bench(repeat) do
        identity(42)
    end
    print_speed("tvm_ffi.identity(Int64) [round-trip]", speed)
end

function tvm_ffi_identity_array(repeat::Int)
    """Measures round-trip overhead for Array (identity optimization)."""
    identity = get_global_func("testing.echo")
    arr = Float32[1.0, 2.0, 3.0, 4.0]
    
    speed = bench(repeat) do
        identity(arr)
    end
    print_speed("tvm_ffi.identity(Array) [identity optimization]", speed)
end

# ============================================================================
# GPU Benchmarks (if available)
# ============================================================================

# Check CUDA availability at load time
const CUDA_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

function gpu_bench_nop_tvmtensor(repeat::Int)
    """GPU: NOP with pre-converted TVMTensor."""
    nop = get_global_func("testing.nop")
    x = TVMTensor(CUDA.zeros(Float32, 1))
    y = TVMTensor(CUDA.zeros(Float32, 1))
    z = TVMTensor(CUDA.zeros(Float32, 1))
    
    # Warmup
    nop(x, y, z)
    CUDA.synchronize()
    
    start = time_ns()
    for _ in 1:repeat
        nop(x, y, z)
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) / 1e9 / repeat
    
    print_speed("  GPU: tvm_ffi.nop(TVMTensor) [pre-converted]", elapsed)
end

function gpu_bench_nop_autodlpack(repeat::Int)
    """GPU: NOP with auto DLPack from CuArray."""
    nop = get_global_func("testing.nop")
    x = CUDA.zeros(Float32, 1)
    y = CUDA.zeros(Float32, 1)
    z = CUDA.zeros(Float32, 1)
    
    # Warmup
    nop(x, y, z)
    CUDA.synchronize()
    
    start = time_ns()
    for _ in 1:repeat
        nop(x, y, z)
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) / 1e9 / repeat
    
    print_speed("  GPU: tvm_ffi.nop.autodlpack(CuArray)", elapsed)
end

function gpu_bench_nop_conversion(repeat::Int)
    """GPU: NOP with TVMTensor conversion each call."""
    nop = get_global_func("testing.nop")
    x = CUDA.zeros(Float32, 1)
    y = CUDA.zeros(Float32, 1)
    z = CUDA.zeros(Float32, 1)
    
    # Warmup
    tx = TVMTensor(x)
    ty = TVMTensor(y)
    tz = TVMTensor(z)
    nop(tx, ty, tz)
    CUDA.synchronize()
    
    start = time_ns()
    for _ in 1:repeat
        tx = TVMTensor(x)
        ty = TVMTensor(y)
        tz = TVMTensor(z)
        nop(tx, ty, tz)
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) / 1e9 / repeat
    
    print_speed("  GPU: tvm_ffi.nop+TVMTensor(CuArray)", elapsed)
end

function gpu_bench_identity(repeat::Int)
    """GPU: Identity (round-trip) with CuArray."""
    identity = get_global_func("testing.echo")
    arr = CUDA.zeros(Float32, 4)
    
    # Warmup
    identity(arr)
    CUDA.synchronize()
    
    start = time_ns()
    for _ in 1:repeat
        identity(arr)
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) / 1e9 / repeat
    
    print_speed("  GPU: tvm_ffi.identity(CuArray)", elapsed)
end

function gpu_bench_cuda_add(repeat::Int)
    """Baseline: CUDA.jl broadcast add."""
    x = CUDA.zeros(Float32, 1)
    y = CUDA.zeros(Float32, 1)
    z = CUDA.zeros(Float32, 1)
    
    # Warmup
    z .= x .+ y
    CUDA.synchronize()
    
    start = time_ns()
    for _ in 1:repeat
        z .= x .+ y
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) / 1e9 / repeat
    
    print_speed("  GPU: CUDA.jl broadcast add (z .= x .+ y)", elapsed)
end

function gpu_bench_current_stream(repeat::Int)
    """Benchmark: CUDA.jl current_stream() query."""
    # Warmup
    CUDA.stream()
    
    start = time_ns()
    for _ in 1:repeat
        CUDA.stream()
    end
    elapsed = (time_ns() - start) / 1e9 / repeat
    
    print_speed("  GPU: CUDA.stream() [current stream query]", elapsed)
end

function try_gpu_benchmarks(repeat::Int)
    """Run GPU benchmarks if CUDA is available."""
    println("\n" * "-"^70)
    println("GPU Benchmarks (CUDA)")
    println("-"^70)
    
    if !CUDA_AVAILABLE
        println("  CUDA not available, skipping GPU benchmarks")
        println("  To enable: using Pkg; Pkg.add(\"CUDA\")")
        return
    end
    
    # Print GPU info
    println("  Device: $(CUDA.name(CUDA.device()))")
    println()
    
    try
        gpu_bench_cuda_add(repeat)
        gpu_bench_nop_tvmtensor(repeat)
        gpu_bench_nop_autodlpack(repeat)
        gpu_bench_nop_conversion(repeat)
        gpu_bench_identity(repeat)
        gpu_bench_current_stream(repeat)
    catch e
        println("  Error during GPU benchmarks: $e")
        showerror(stdout, e, catch_backtrace())
    end
end

# ============================================================================
# Python Reference Values (for comparison)
# ============================================================================

# Python benchmark results (actual, from bench_example.py on same machine)
const PYTHON_BASELINES = Dict{String, Float64}(
    "numpy.add" => 2.07e-07,
    "torch.add[cpu]" => 5.44e-07,
    "torch.add[cuda]" => 1.51e-05,
    "tvm_ffi.nop" => 7.22e-08,
    "tvm_ffi.nop.autodlpack(numpy)" => 2.98e-07,
    "tvm_ffi.nop.autodlpack(torch[cuda])" => 9.02e-07,
    "torch.cuda.current_stream[cpp]" => 8.54e-08,
    "torch.cuda.current_stream[python]" => 1.05e-06,
)

function print_comparison(julia_results::Dict{String, Float64})
    println("\n" * "="^70)
    println("Julia vs Python Comparison")
    println("="^70)
    
    comparisons = [
        # Baselines
        ("Broadcast add (CPU)", "broadcast_add", "numpy.add"),
        ("Broadcast add (CUDA)", "cuda_broadcast_add", "torch.add[cuda]"),
        # TVM FFI
        ("TVM NOP (no args)", "nop_no_args", "tvm_ffi.nop"),
        ("TVM NOP (pre-converted TVMTensor)", "nop_tvmtensor", "tvm_ffi.nop"),
        ("TVM autodlpack (CPU Array)", "autodlpack_array", "tvm_ffi.nop.autodlpack(numpy)"),
        ("TVM GPU autodlpack (CuArray)", "gpu_autodlpack", "tvm_ffi.nop.autodlpack(torch[cuda])"),
        ("CUDA stream query", "cuda_stream", "torch.cuda.current_stream[cpp]"),
    ]
    
    @printf("  %-40s %12s %12s %12s\n", "Operation", "Julia", "Python", "Speedup")
    println("  " * "-"^80)
    
    for (name, julia_key, python_key) in comparisons
        julia_val = get(julia_results, julia_key, NaN)
        python_val = get(PYTHON_BASELINES, python_key, NaN)
        
        if !isnan(julia_val) && !isnan(python_val)
            speedup = python_val / julia_val
            speedup_str = speedup >= 1.0 ? @sprintf("%.1fx faster", speedup) : @sprintf("%.1fx slower", 1/speedup)
            # Use appropriate unit
            julia_unit = julia_val >= 1e-6 ? (@sprintf("%.1f µs", julia_val * 1e6)) : (@sprintf("%.0f ns", julia_val * 1e9))
            python_unit = python_val >= 1e-6 ? (@sprintf("%.1f µs", python_val * 1e6)) : (@sprintf("%.0f ns", python_val * 1e9))
            @printf("  %-40s %12s %12s %s\n", name, julia_unit, python_unit, speedup_str)
        elseif !isnan(julia_val)
            julia_unit = julia_val >= 1e-6 ? (@sprintf("%.1f µs", julia_val * 1e6)) : (@sprintf("%.0f ns", julia_val * 1e9))
            @printf("  %-40s %12s %12s %12s\n", name, julia_unit, "N/A", "-")
        end
    end
    println()
end

function print_python_reference()
    println("""
  
  Python Reference Values (actual, from bench_example.py on same machine):
  -----------------------------------------------------------------------
  Baselines:
    numpy.add                                    2.07e-07 sec/call (207 ns)
    torch.add[cpu]                               5.44e-07 sec/call (544 ns)
    torch.add[cuda]                              1.51e-05 sec/call (15.1 µs)
  
  TVM FFI (pre-converted, no DLPack overhead):
    tvm_ffi.nop (TVMTensor)                      7.22e-08 sec/call (72 ns)
    tvm_ffi.nop.autodlpack(DLTensorTestWrapper)  1.38e-07 sec/call (138 ns)
    tvm_ffi.nop.autodlpack(TestFFITensor)        1.66e-07 sec/call (166 ns)
  
  TVM FFI (with DLPack conversion):
    tvm_ffi.nop+from_dlpack(numpy)               5.13e-07 sec/call (513 ns)
    tvm_ffi.nop+from_dlpack(tvm)                 5.22e-07 sec/call (522 ns)
    tvm_ffi.nop.autodlpack(numpy)                2.98e-07 sec/call (298 ns)
    tvm_ffi.nop.autodlpack(torch[cpu])           8.09e-07 sec/call (809 ns)
    tvm_ffi.nop.autodlpack(torch[cuda])          9.02e-07 sec/call (902 ns)
    tvm_ffi.nop+from_dlpack(torch)               3.73e-06 sec/call (3.73 µs)
  
  __dlpack__ overhead:
    tvm.__dlpack__                               4.77e-08 sec/call (48 ns)
    numpy.__dlpack__                             7.12e-08 sec/call (71 ns)
    torch.utils.dlpack.to_dlpack                 1.62e-07 sec/call (162 ns)
    torch.__dlpack__                             1.06e-06 sec/call (1.06 µs)
  
  CUDA stream query:
    torch.cuda.current_stream[cpp-extension]     8.54e-08 sec/call (85 ns)
    torch.cuda.current_stream[python]            1.05e-06 sec/call (1.05 µs)
    """)
end

# ============================================================================
# Main
# ============================================================================

# Modified benchmark functions that return results
function bench_and_record!(results::Dict{String, Float64}, key::String, name::String, f::Function, repeat::Int)
    speed = bench(f, repeat)
    print_speed(name, speed)
    results[key] = speed
    return speed
end

function main()
    repeat = 10000
    results = Dict{String, Float64}()
    
    println("="^70)
    println("Benchmark f(x, y, z) overhead - Julia TVMFFI")
    println("="^70)
    
    println("\n" * "-"^70)
    println("Baselines (like numpy.add / torch.add)")
    println("-"^70)
    results["broadcast_add"] = baseline_julia_broadcast_add(repeat)
    baseline_julia_inplace_add(repeat)
    if CUDA_AVAILABLE
        results["cuda_broadcast_add"] = baseline_cuda_broadcast_add(repeat)
    end
    
    println("\n" * "-"^70)
    println("TVM FFI Scalar Operations")
    println("-"^70)
    
    # NOP no args
    nop = get_global_func("testing.nop")
    bench_and_record!(results, "nop_no_args", "tvm_ffi.nop() [no args]", () -> nop(), repeat)
    
    tvm_ffi_echo_int(repeat)
    tvm_ffi_echo_float(repeat)
    tvm_ffi_echo_string(repeat)
    
    println("\n" * "-"^70)
    println("TVM FFI Tensor Operations (3 args)")
    println("-"^70)
    
    # NOP with pre-converted TVMTensor
    x = TVMTensor(Float64[1.0])
    y = TVMTensor(Float64[1.0])
    z = TVMTensor(Float64[1.0])
    bench_and_record!(results, "nop_tvmtensor", "tvm_ffi.nop(TVMTensor) [pre-converted]", 
        () -> nop(x, y, z), repeat)
    
    tvm_ffi_nop_from_julia_array(repeat)
    
    # Autodlpack from Julia Array
    arr_x = Float64[1.0]
    arr_y = Float64[1.0]
    arr_z = Float64[1.0]
    bench_and_record!(results, "autodlpack_array", "tvm_ffi.nop.autodlpack(Julia Array)",
        () -> nop(arr_x, arr_y, arr_z), repeat)
    
    tvm_ffi_nop_tensorview(repeat)
    
    println("\n" * "-"^70)
    println("TVM FFI Identity (Round-trip)")
    println("-"^70)
    tvm_ffi_identity_int(repeat)
    tvm_ffi_identity_array(repeat)
    
    println("\n" * "-"^70)
    println("Argument Count Scaling")
    println("-"^70)
    tvm_ffi_nop_arg_scaling(repeat)
    
    # GPU benchmarks
    println("\n" * "-"^70)
    println("GPU Benchmarks (CUDA)")
    println("-"^70)
    
    if CUDA_AVAILABLE
        println("  Device: $(CUDA.name(CUDA.device()))")
        println()
        
        try
            gpu_bench_cuda_add(repeat)
            gpu_bench_nop_tvmtensor(repeat)
            
            # GPU autodlpack
            cu_x = CUDA.zeros(Float32, 1)
            cu_y = CUDA.zeros(Float32, 1)
            cu_z = CUDA.zeros(Float32, 1)
            nop(cu_x, cu_y, cu_z)
            CUDA.synchronize()
            
            start = time_ns()
            for _ in 1:repeat
                nop(cu_x, cu_y, cu_z)
            end
            CUDA.synchronize()
            gpu_autodlpack_time = (time_ns() - start) / 1e9 / repeat
            print_speed("  GPU: tvm_ffi.nop.autodlpack(CuArray)", gpu_autodlpack_time)
            results["gpu_autodlpack"] = gpu_autodlpack_time
            
            gpu_bench_nop_conversion(repeat)
            gpu_bench_identity(repeat)
            
            # CUDA stream query
            CUDA.stream()  # warmup
            start = time_ns()
            for _ in 1:repeat
                CUDA.stream()
            end
            cuda_stream_time = (time_ns() - start) / 1e9 / repeat
            print_speed("  GPU: CUDA.stream() [current stream query]", cuda_stream_time)
            results["cuda_stream"] = cuda_stream_time
            
        catch e
            println("  Error during GPU benchmarks: $e")
            showerror(stdout, e, catch_backtrace())
        end
    else
        println("  CUDA not available, skipping GPU benchmarks")
        println("  To enable: using Pkg; Pkg.add(\"CUDA\")")
    end
    
    # Print comparison
    print_comparison(results)
    
    print_python_reference()
    
    println("-"^70)
    println("Done")
    println("-"^70)
    
    # Note: CUDAExt registers an atexit hook for automatic cleanup,
    # so manual GC is no longer needed here.
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

