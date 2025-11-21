#=
GPU Device Detection Test

This example tests the GPU device detection and device ID extraction.

Requirements:
- At least one GPU backend installed (CUDA.jl, AMDGPU.jl, Metal.jl, or oneAPI.jl)
- For multi-GPU testing: system with multiple GPUs

Usage:
    julia --project=. examples/test_gpu_device_detection.jl
=#

using TVMFFI

println("="^60)
println("GPU Device Detection Test")
println("="^60)

# Test 1: Check available backends
println("\n[Test 1] Available GPU Backends:")
println("-"^40)
TVMFFI.print_gpu_info()

# Test 2: Test CUDA (if available)
println("\n[Test 2] CUDA Device Detection:")
println("-"^40)
try
    using CUDA

    if CUDA.functional()
        println("✓ CUDA.jl is functional")

        # Test with default device
        x = CUDA.CuArray(Float32[1, 2, 3, 4])
        backend, dev_id = TVMFFI.detect_gpu_backend(x)
        println("  Array on device: $backend, device_id=$dev_id")
        println("  Current CUDA device: ", CUDA.device())
        @assert backend == :CUDA "Expected :CUDA, got $backend"
        @assert dev_id == CUDA.deviceid(CUDA.device()) "Device ID mismatch!"
        println("  ✓ Device ID correctly extracted!")

        # Test with multiple devices (if available)
        if length(CUDA.devices()) > 1
            println("\n  Testing multi-GPU:")
            for i in 0:(length(CUDA.devices()) - 1)
                CUDA.device!(i)
                y = CUDA.CuArray(Float32[1, 2, 3])
                backend, dev_id = TVMFFI.detect_gpu_backend(y)
                println("    Device $i: detected device_id=$dev_id")
                @assert dev_id == i "Expected device_id=$i, got $dev_id"
            end
            println("  ✓ Multi-GPU device detection works!")
        else
            println("  (Single GPU system - skipping multi-GPU test)")
        end

        # Test DLTensor creation
        println("\n  Testing DLTensor creation:")
        x_cuda = CUDA.CuArray(Float32[1, 2, 3, 4])
        holder = TVMFFI.from_julia_array(x_cuda)
        println("    DLTensor device type: ", holder.tensor.device.device_type)
        println("    DLTensor device id: ", holder.tensor.device.device_id)
        @assert Int(holder.tensor.device.device_type) == Int(TVMFFI.LibTVMFFI.kDLCUDA)
        @assert holder.tensor.device.device_id == CUDA.deviceid(CUDA.device())
        println("  ✓ DLTensor has correct device info!")

        # Test with wrapped arrays (OffsetArrays, SubArrays, etc.)
        println("\n  Testing wrapped arrays:")
        try
            using OffsetArrays

            # Test OffsetArray wrapper
            x_base = CUDA.CuArray(Float32[1, 2, 3, 4])
            x_offset = OffsetArray(x_base, -1:2)  # Wrap with custom indexing

            backend, dev_id = TVMFFI.detect_gpu_backend(x_offset)
            println("    OffsetArray: backend=$backend, device_id=$dev_id")
            @assert backend == :CUDA "Failed to detect CUDA through OffsetArray wrapper"
            @assert dev_id == CUDA.deviceid(CUDA.device()) "Device ID mismatch for OffsetArray"
            println("    ✓ OffsetArray wrapper handled correctly!")

            # Test DLTensor creation with wrapped array
            holder = TVMFFI.from_julia_array(x_offset)
            @assert Int(holder.tensor.device.device_type) == Int(TVMFFI.LibTVMFFI.kDLCUDA)
            @assert holder.tensor.device.device_id == CUDA.deviceid(CUDA.device())
            println("    ✓ DLTensor creation works with wrapped arrays!")

        catch e
            if e isa ArgumentError && occursin("OffsetArrays", string(e))
                println("    ℹ OffsetArrays.jl not available - skipping wrapper tests")
                println("      Install with: using Pkg; Pkg.add(\"OffsetArrays\")")
            else
                rethrow(e)
            end
        end

        # Test SubArray (view)
        println("\n  Testing SubArray (view):")
        x_full = CUDA.CuArray(Float32[1, 2, 3, 4, 5, 6])
        x_view = view(x_full, 2:5)

        backend, dev_id = TVMFFI.detect_gpu_backend(x_view)
        println("    SubArray: backend=$backend, device_id=$dev_id")
        @assert backend == :CUDA "Failed to detect CUDA through SubArray"
        @assert dev_id == Int(CUDA.device()) "Device ID mismatch for SubArray"
        println("    ✓ SubArray (view) handled correctly!")

        # Test ReshapedArray
        println("\n  Testing ReshapedArray:")
        x_matrix = CUDA.CuArray(Float32[1, 2, 3, 4, 5, 6])
        x_reshaped = reshape(x_matrix, 2, 3)

        backend, dev_id = TVMFFI.detect_gpu_backend(x_reshaped)
        println("    ReshapedArray: backend=$backend, device_id=$dev_id")
        @assert backend == :CUDA "Failed to detect CUDA through ReshapedArray"
        @assert dev_id == Int(CUDA.device()) "Device ID mismatch for ReshapedArray"
        println("    ✓ ReshapedArray handled correctly!")

    else
        println("⚠ CUDA.jl installed but not functional")
    end
catch e
    if e isa ArgumentError && occursin("CUDA", string(e))
        println("ℹ CUDA.jl not installed - skipping CUDA tests")
        println("  Install with: using Pkg; Pkg.add(\"CUDA\")")
    else
        println("✗ CUDA test failed: $e")
        rethrow(e)
    end
end

# Test 3: Test AMDGPU (if available)
println("\n[Test 3] AMDGPU Device Detection:")
println("-"^40)
try
    using AMDGPU

    if AMDGPU.functional()
        println("✓ AMDGPU.jl is functional")

        # Test device detection
        x = AMDGPU.ROCArray(Float32[1, 2, 3, 4])
        backend, dev_id = TVMFFI.detect_gpu_backend(x)
        println("  Array on device: $backend, device_id=$dev_id")
        println("  Current AMDGPU device: ", AMDGPU.device_id())
        @assert backend == :ROCm "Expected :ROCm, got $backend"
        # Note: AMDGPU uses 1-indexed, we convert to 0-indexed
        @assert dev_id == AMDGPU.device_id() - 1 "Device ID conversion failed!"
        println("  ✓ Device ID correctly extracted (converted from 1-indexed)!")

        # Test DLTensor creation
        println("\n  Testing DLTensor creation:")
        x_roc = AMDGPU.ROCArray(Float32[1, 2, 3, 4])
        holder = TVMFFI.from_julia_array(x_roc)
        println("    DLTensor device type: ", holder.tensor.device.device_type)
        println("    DLTensor device id: ", holder.tensor.device.device_id)
        @assert Int(holder.tensor.device.device_type) == Int(TVMFFI.LibTVMFFI.kDLROCM)
        println("  ✓ DLTensor has correct device info!")

    else
        println("⚠ AMDGPU.jl installed but not functional")
    end
catch e
    if e isa ArgumentError && occursin("AMDGPU", string(e))
        println("ℹ AMDGPU.jl not installed - skipping AMDGPU tests")
        println("  Install with: using Pkg; Pkg.add(\"AMDGPU\")")
    else
        println("✗ AMDGPU test failed: $e")
        rethrow(e)
    end
end

# Test 4: Test Metal (if available)
println("\n[Test 4] Metal Device Detection:")
println("-"^40)
try
    using Metal

    if Metal.functional()
        println("✓ Metal.jl is functional")

        x = Metal.MtlArray(Float32[1, 2, 3, 4])
        backend, dev_id = TVMFFI.detect_gpu_backend(x)
        println("  Array on device: $backend, device_id=$dev_id")
        @assert backend == :Metal "Expected :Metal, got $backend"
        @assert dev_id == 0 "Metal should always report device 0"
        println("  ✓ Device ID correctly extracted!")

        # Test DLTensor creation
        println("\n  Testing DLTensor creation:")
        x_mtl = Metal.MtlArray(Float32[1, 2, 3, 4])
        holder = TVMFFI.from_julia_array(x_mtl)
        println("    DLTensor device type: ", holder.tensor.device.device_type)
        println("    DLTensor device id: ", holder.tensor.device.device_id)
        @assert Int(holder.tensor.device.device_type) == Int(TVMFFI.LibTVMFFI.kDLMetal)
        println("  ✓ DLTensor has correct device info!")

    else
        println("⚠ Metal.jl installed but not functional")
    end
catch e
    if e isa ArgumentError && occursin("Metal", string(e))
        println("ℹ Metal.jl not installed or not on macOS - skipping Metal tests")
        println("  Install on macOS with: using Pkg; Pkg.add(\"Metal\")")
    else
        println("✗ Metal test failed: $e")
        rethrow(e)
    end
end

# Test 5: Test oneAPI (if available)
println("\n[Test 5] oneAPI Device Detection:")
println("-"^40)
try
    using oneAPI

    if oneAPI.functional()
        println("✓ oneAPI.jl is functional")

        x = oneAPI.oneArray(Float32[1, 2, 3, 4])
        backend, dev_id = TVMFFI.detect_gpu_backend(x)
        println("  Array on device: $backend, device_id=$dev_id")
        @assert backend == :oneAPI "Expected :oneAPI, got $backend"
        println("  ✓ Device backend correctly detected!")
        println("  ⚠ Note: Device ID extraction for oneAPI needs verification")

    else
        println("⚠ oneAPI.jl installed but not functional")
    end
catch e
    if e isa ArgumentError && occursin("oneAPI", string(e))
        println("ℹ oneAPI.jl not installed - skipping oneAPI tests")
        println("  Install with: using Pkg; Pkg.add(\"oneAPI\")")
    else
        println("✗ oneAPI test failed: $e")
        rethrow(e)
    end
end

println("\n" * "="^60)
println("GPU Device Detection Test Complete!")
println("="^60)
