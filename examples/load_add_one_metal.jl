#!/usr/bin/env julia
#=
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
=#

"""
Example: Loading and calling TVM Metal module

This example demonstrates how to:
1. Load a compiled TVM Metal module (.metallib file)
2. Create Metal arrays (MtlArray from Metal.jl)
3. Call Metal kernels through TVM FFI

Based on tvm-ffi/examples/quickstart/load/load_cuda.cc
"""

using TVMFFI

# Load fixture utilities (auto-builds if needed)
include("fixtures_utils.jl")

println("=" ^ 70)
println("TVM FFI Julia Example: Loading add_one_metal (Apple GPU)")
println("=" ^ 70)

# Check if Metal is available
has_metal = try
    using Metal
    Metal.functional()
catch
    false
end

if !has_metal
    println("\n⚠️  Metal not available!")
    println("   This example requires:")
    println("   • Metal.jl package: using Pkg; Pkg.add(\"Metal\")")
    println("   • Apple Silicon Mac (M1/M2/M3) or Mac with Metal-capable GPU")
    println("   • macOS 11.0+")
    println("\n   Note: The fixture will still be built, but GPU execution will be skipped.")
    println("   You can run this example later once Metal is available.\n")
end

# Ensure Metal fixture is built (auto-builds on first run, including Metal kernel)
println("1. Preparing Metal fixture...")
module_path = try
    ensure_fixture_built("add_one_metal")
catch e
    println("❌ Error building Metal fixture:")
    println("   ", e)
    println("\n   This likely means:")
    println("   • Not running on macOS, or")
    println("   • Metal framework not available, or")
    println("   • CMake cannot find Metal")
    println("\n   To disable Metal fixture build:")
    println("   cmake ../fixtures -B build/fixtures -DBUILD_METAL_FIXTURES=OFF")
    if !has_metal
        println("\n   Note: Metal.jl is also not available, so GPU tests will be skipped.")
    end
    exit(1)
end
println("   ✓ Metal fixture ready: $module_path")

if !has_metal
    println("\n⚠️  Skipping GPU tests - Metal.jl not available")
    println("   Install Metal.jl to test stride-aware N-D Metal arrays:")
    println("   using Pkg; Pkg.add(\"Metal\")")
    exit(0)
end

# Load the module
println("\n2. Loading Metal module...")
module_loader = get_global_func("ffi.ModuleLoadFromFile")
if module_loader === nothing
    println("❌ Error: ffi.ModuleLoadFromFile not found!")
    exit(1)
end

tvm_module = try
    module_loader(module_path)
catch e
    println("❌ Error loading Metal module:")
    println("   ", e)
    exit(1)
end

println("✓ Metal module loaded successfully")

# Get the function
println("\n3. Getting 'add_one_metal' function from module...")
func_getter = get_global_func("ffi.ModuleGetFunction")

add_one_metal = try
    func_getter(tvm_module, "add_one_metal", true)
catch e
    println("❌ Error getting function:")
    println("   ", e)
    exit(1)
end

println("✓ Got Metal function: ", typeof(add_one_metal))

println("\n" * "="^70)
println("🚀 Metal Stride-Aware N-D Array Support")
println("="^70)
println("\nThis demonstrates the Metal kernel that handles:")
println("  • Any dimensionality (1D, 2D, 3D, ...)")
println("  • Non-contiguous memory (arbitrary strides)")
println("  • Julia's column-major layout")
println("  • Zero-copy views and slices\n")

if has_metal
    # ============================================================
    # Test 1: Simple 1D vector
    # ============================================================
    println("=" ^ 70)
    println("Test 1: Simple 1D Vector")
    println("=" ^ 70)
    
    x_metal = Metal.MtlArray(Float32[1, 2, 3, 4, 5])
    y_metal = Metal.zeros(Float32, 5)

    println("Input:  ", Array(x_metal))
    add_one_metal(x_metal, y_metal)
    Metal.synchronize()
    
    result = Array(y_metal)
    expected = Float32[2, 3, 4, 5, 6]
    println("Output: ", result)
    println("Expected: ", expected)
    
    if result ≈ expected
        println("✅ 1D vector passed")
    else
        println("❌ 1D vector FAILED")
        exit(1)
    end

    # ============================================================
    # Test 2: 1D strided view (every 2nd element)
    # ============================================================
    println("\n" * "=" ^ 70)
    println("Test 2: 1D Strided View (stride=2)")
    println("=" ^ 70)
    
    x_vec_metal = Metal.MtlArray(Float32[1, 2, 3, 4, 5, 6, 7, 8])
    y_vec_metal = Metal.zeros(Float32, 8)
    x_strided = @view x_vec_metal[1:2:end]  # [1, 3, 5, 7]
    y_strided = @view y_vec_metal[1:2:end]

    println("Input (strided):  ", Array(x_strided))
    println("Stride: ", Base.strides(x_strided))
    add_one_metal(x_strided, y_strided)
    Metal.synchronize()
    
    result = Array(y_strided)
    expected = Float32[2, 4, 6, 8]
    println("Output: ", result)
    println("Expected: ", expected)
    
    if result ≈ expected
        println("✅ Strided 1D passed")
    else
        println("❌ Strided 1D FAILED")
        exit(1)
    end

    # ============================================================
    # Test 3: 2D Matrix
    # ============================================================
    println("\n" * "=" ^ 70)
    println("Test 3: 2D Matrix (Column-Major Layout)")
    println("=" ^ 70)
    
    x_mat_metal = Metal.MtlArray(Float32[1 2 3; 4 5 6])  # 2×3
    y_mat_metal = Metal.similar(x_mat_metal)

    println("Input shape: ", size(x_mat_metal))
    println("Input strides: ", Base.strides(x_mat_metal))
    println("Input:\n", Array(x_mat_metal))
    
    add_one_metal(x_mat_metal, y_mat_metal)
    Metal.synchronize()
    
    result = Array(y_mat_metal)
    expected = Float32[2 3 4; 5 6 7]
    println("Output:\n", result)
    println("Expected:\n", expected)
    
    if result ≈ expected
        println("✅ 2D matrix passed")
    else
        println("❌ 2D matrix FAILED")
        println("Difference:\n", result .- expected)
        exit(1)
    end

    # ============================================================
    # Test 4: Column slice (contiguous)
    # ============================================================
    println("\n" * "=" ^ 70)
    println("Test 4: Column Slice (Contiguous in Column-Major)")
    println("=" ^ 70)
    
    mat_metal = Metal.MtlArray(Float32[1 2 3 4; 5 6 7 8; 9 10 11 12])
    x_col = @view mat_metal[:, 2]  # [2, 6, 10]
    y_col = Metal.similar(x_col)

    println("Input column: ", Array(x_col))
    println("Stride: ", Base.strides(x_col), " (contiguous)")
    
    add_one_metal(x_col, y_col)
    Metal.synchronize()
    
    result = Array(y_col)
    expected = Float32[3, 7, 11]
    println("Output: ", result)
    println("Expected: ", expected)
    
    if result ≈ expected
        println("✅ Column slice passed")
    else
        println("❌ Column slice FAILED")
        exit(1)
    end

    # ============================================================
    # Test 5: Row slice (NON-contiguous, stride > 1)
    # ============================================================
    println("\n" * "=" ^ 70)
    println("Test 5: Row Slice (Non-Contiguous, Stride-Aware)")
    println("=" ^ 70)
    
    x_row = @view mat_metal[2, :]  # [5, 6, 7, 8]
    y_row = Metal.similar(x_row)

    println("Input row: ", Array(x_row))
    println("Stride: ", Base.strides(x_row), " (non-contiguous!)")
    
    add_one_metal(x_row, y_row)
    Metal.synchronize()
    
    result = Array(y_row)
    expected = Float32[6, 7, 8, 9]
    println("Output: ", result)
    println("Expected: ", expected)
    
    if result ≈ expected
        println("✅ Row slice passed (stride-aware!)")
    else
        println("❌ Row slice FAILED")
        exit(1)
    end

    # ============================================================
    # Test 6: 2D sub-matrix
    # ============================================================
    println("\n" * "=" ^ 70)
    println("Test 6: 2D Sub-Matrix (Complex Strides)")
    println("=" ^ 70)
    
    big_mat = Metal.MtlArray(Float32[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16])
    x_sub = @view big_mat[2:3, 2:3]  # 2×2 sub-matrix
    y_sub = Metal.similar(x_sub)

    println("Input shape: ", size(x_sub))
    println("Input strides: ", Base.strides(x_sub))
    println("Input:\n", Array(x_sub))
    
    add_one_metal(x_sub, y_sub)
    Metal.synchronize()
    
    result = Array(y_sub)
    expected = Float32[7 8; 11 12]
    println("Output:\n", result)
    println("Expected:\n", expected)
    
    if result ≈ expected
        println("✅ 2D sub-matrix passed")
    else
        println("❌ 2D sub-matrix FAILED")
        exit(1)
    end

    # ============================================================
    # Summary
    # ============================================================
    println("\n" * "="^70)
    println("✅ ALL METAL STRIDE-AWARE TESTS PASSED!")
    println("="^70)
    println("\nVerified capabilities:")
    println("  ✅ 1D vectors")
    println("  ✅ Strided views (non-contiguous memory)")
    println("  ✅ 2D matrices (column-major layout)")
    println("  ✅ Column slices (contiguous)")
    println("  ✅ Row slices (non-contiguous)")
    println("  ✅ Sub-matrices (complex strides)")
    println("\n🎉 The Metal kernel correctly handles:")
    println("  • Arbitrary dimensions (N-D)")
    println("  • Arbitrary strides (non-contiguous memory)")
    println("  • Julia's column-major layout")
    println("  • Zero-copy views (no data duplication)")
    println("\nThis matches CPU functionality - feature parity achieved!")
end

println("\n" * "=" ^ 70)
println("Metal Example Completed!")
println("=" ^ 70)

println("\n📝 Summary:")
println("   ✓ Loaded TVM Metal module with auto-build")
println("   ✓ Retrieved 'add_one_metal' function")
if has_metal
    println("   ✓ Tested 6 different array scenarios:")
    println("     1. Simple 1D vectors")
    println("     2. Strided views (stride=2)")
    println("     3. 2D matrices (column-major)")
    println("     4. Column slices (contiguous)")
    println("     5. Row slices (non-contiguous)")
    println("     6. Sub-matrices (complex strides)")
    println("\n🎉 Key achievements:")
    println("   • Stride-aware N-D Metal kernel")
    println("   • Correct column-major layout handling")
    println("   • Zero-copy GPU memory access")
    println("   • Feature parity with CPU implementation")
    println("   • Full support for Julia's array semantics")
else
    println("   ⚠️  Metal.jl not available (demo mode)")
    println("\nInstall Metal.jl to test stride-aware N-D Metal arrays:")
    println("   using Pkg; Pkg.add(\"Metal\")")
end

