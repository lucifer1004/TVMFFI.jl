#=
Breakdown analysis of CUDA FFI overhead

Run: julia --project=. cuda_breakdown.jl
=#

using TVMFFI
using TVMFFI: TensorView, TVMAny, TVMTensor, dldevice, raw_data, from_dlpack
using TVMFFI: LibTVMFFI
using CUDA
using BenchmarkTools

println("="^70)
println("CUDA FFI Overhead Breakdown Analysis")
println("="^70)
println()

# Test array
arr = CUDA.rand(Float32, 1024)
println("Test array: CuArray{Float32}(1024)")
println()

# ============================================================================
# Step-by-step breakdown
# ============================================================================

println("Step-by-step breakdown:")
println("-"^70)

# 1. Device detection
print("1. dldevice(arr)                       ")
t1 = @benchmark dldevice($arr) samples=1000
println(round(median(t1).time, digits = 1), " ns")

# 2. TensorView creation (for comparison)
print("2. TensorView(arr)                     ")
t2 = @benchmark TensorView($arr) samples=1000
println(round(median(t2).time, digits = 1), " ns")

# 3. TVMTensor creation (includes DLPack)
print("3. TVMTensor(arr)                      ")
t3 = @benchmark TVMTensor($arr) samples=1000
println(round(median(t3).time, digits = 1), " ns")

# 4. TVMAny from TensorView
view = TensorView(arr)
print("4. TVMAny(TensorView)                  ")
t4 = @benchmark TVMAny($view) samples=1000
println(round(median(t4).time, digits = 1), " ns")

# 5. TVMAny from TVMTensor
tensor = TVMTensor(arr)
print("5. TVMAny(TVMTensor)                   ")
t5 = @benchmark TVMAny($tensor) samples=1000
println(round(median(t5).time, digits = 1), " ns")

# 6. from_dlpack (zero-copy conversion)
print("6. from_dlpack(TVMTensor)              ")
t6 = @benchmark from_dlpack($tensor) samples=1000
println(round(median(t6).time, digits = 1), " ns")

# 7. CUDA.synchronize overhead
print("7. CUDA.synchronize()                  ")
t7 = @benchmark CUDA.synchronize() samples=1000
println(round(median(t7).time, digits = 1), " ns")

println()
println("="^70)
println("Full path comparison:")
println("-"^70)

# Register identity function
register_global_func("julia.cuda.breakdown.identity", (x) -> x; override = true)
tvm_identity = get_global_func("julia.cuda.breakdown.identity")

# A. Full TVM call (GPU path: TVMTensor)
print("A. Full TVM call (TVMTensor path)      ")
tA = @benchmark begin
    $tvm_identity($arr)
    CUDA.synchronize()
end samples=500
println(round(median(tA).time / 1000, digits = 2), " μs")

# B. Full TVM call without sync
print("B. TVM call (no sync)                  ")
tB = @benchmark $tvm_identity($arr) samples=500
println(round(median(tB).time / 1000, digits = 2), " μs")

# C. Just the conversion overhead (no actual FFI)
print("C. TVMTensor + from_dlpack only        ")
tC = @benchmark begin
    local t = TVMTensor($arr)
    from_dlpack(t)
end samples=500
println(round(median(tC).time / 1000, digits = 2), " μs")

# D. CPU comparison
cpu_arr = rand(Float32, 1024)
print("D. CPU Array TVM call                  ")
tD = @benchmark $tvm_identity($cpu_arr) samples=500
println(round(median(tD).time, digits = 1), " ns")

println()
println("="^70)
println("Analysis:")
println("-"^70)

sync_time = median(t7).time
tvmtensor_time = median(t3).time
from_dlpack_time = median(t6).time
full_call_time = median(tA).time
call_no_sync_time = median(tB).time

println("CUDA.synchronize():     ", round(sync_time, digits = 1), " ns")
println("TVMTensor creation:     ", round(tvmtensor_time, digits = 1), " ns")
println("from_dlpack:            ", round(from_dlpack_time, digits = 1), " ns")
println()
println("Full call:              ", round(full_call_time / 1000, digits = 2), " μs")
println("Call (no sync):         ", round(call_no_sync_time / 1000, digits = 2), " μs")
println()
println("Estimated breakdown of full call:")
println("  - CUDA.synchronize:   ~", round(sync_time / 1000, digits = 2), " μs")
println("  - FFI dispatch:       ~",
    round((call_no_sync_time - tvmtensor_time - from_dlpack_time) / 1000, digits = 2),
    " μs")
println("  - TVMTensor + DLPack: ~",
    round((tvmtensor_time + from_dlpack_time) / 1000, digits = 2), " μs")

println()
println("Done.")
