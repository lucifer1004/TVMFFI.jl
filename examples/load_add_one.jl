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
Example: Loading and calling TVM compiled module

This example demonstrates how to:
1. Load a compiled TVM module (.so file)
2. Get a function from the module
3. Create tensors and call the function

Based on tvm-ffi/examples/quickstart/load/load_numpy.py
"""

using TVMFFI

# Load fixture utilities (auto-builds if needed)
include("fixtures_utils.jl")

println("="^60)
println("TVM FFI Julia Example: Loading add_one_cpu")
println("="^60)

# Ensure fixture is built (auto-builds on first run)
println("\n1. Preparing fixture...")
module_path = ensure_fixture_built("add_one_cpu")
println("   ✓ Fixture ready: $module_path")

# Load the module
println("\n2. Loading module...")
module_loader = get_global_func("ffi.ModuleLoadFromFile")
if module_loader === nothing
    println("❌ Error: ffi.ModuleLoadFromFile not found!")
    println("   Make sure TVM FFI runtime library is properly loaded.")
    exit(1)
end

tvm_module = try
    module_loader(module_path)
catch e
    println("❌ Error loading module:")
    println("   ", e)
    exit(1)
end

println("✓ Module loaded successfully: ", typeof(tvm_module))

# Get the function from the module
println("\n3. Getting 'add_one_cpu' function from module...")
func_getter = get_global_func("ffi.ModuleGetFunction")

if func_getter === nothing
    println("❌ Error: ffi.ModuleGetFunction not found!")
    exit(1)
end

# Get the function: ModuleGetFunction(module, name, query_imports)
add_one_cpu = try
    func_getter(tvm_module, "add_one_cpu", true)
catch e
    throw(e)
    # println("❌ Error getting function:")
    # println("   ", e)
    # exit(1)
end

println("✓ Got function: ", typeof(add_one_cpu))

# Create input and output arrays
println("\n4. Creating tensors...")
x = Float32[1, 2, 3, 4, 5]
y = zeros(Float32, 5)

println("   Input (x):  ", x)
println("   Output (y): ", y)

# NEW: Direct array passing! (Auto-conversion)
# Arrays are automatically converted to TensorView
println("\n5. Calling add_one_cpu(x, y) - direct array passing!")
println("   (Arrays auto-converted to views internally)")
try
    add_one_cpu(x, y)  # ← Pass arrays directly!
    println("✓ Function call succeeded!")
catch e
    println("❌ Error calling function:")
    println("   ", e)
    exit(1)
end

# Check results
println("\n7. Results:")
println("   Input (x):   ", x)
println("   Output (y):  ", y)
println("   Expected:    ", x .+ 1)

# Verify
if y ≈ x .+ 1
    println("\n✅ SUCCESS! Output matches expected values!")
else
    println("\n❌ FAILED! Output does not match expected values")
    println("   Difference: ", y .- (x .+ 1))
end

# ============================================================
# NEW: Zero-copy slice support with stride-aware kernel!
# ============================================================
println("\n" * "="^60)
println("🚀 BONUS: Zero-Copy Slice Support (Stride-Aware!)")
println("="^60)

# Create a vector for contiguous slice demo
println("\n8. Creating vector for slice demo...")
vector = Float32[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
println("   Vector: ", vector)

# Test 1: Contiguous slice
println("\n9. Testing contiguous slice (zero-copy!)...")
slice = @view vector[3:7]  # Elements 3-7
slice_output = zeros(Float32, 5)

println("   Input slice:  ", slice)
println("   Stride: ", Base.strides(slice), " (contiguous)")

# NEW: Direct slice passing! Auto-converted to view
add_one_cpu(slice, slice_output)  # ← Pass slices directly!

println("   Output:       ", slice_output)
println("   Expected:     ", collect(slice) .+ 1)

if slice_output ≈ collect(slice) .+ 1
    println("   ✅ Contiguous slice works!")
else
    println("   ❌ Slice failed!")
    exit(1)
end

# Test 2: Column slice (contiguous in column-major)
println("\n10. Testing column slice (zero-copy!)...")
matrix = Float32[1 2 3 4
                 5 6 7 8
                 9 10 11 12]
col = @view matrix[:, 3]  # Third column (contiguous!)
col_output = zeros(Float32, 3)

println("   Input column:  ", col)
println("   Stride: ", Base.strides(col), " (contiguous)")

# Direct slice passing - auto-converted!
add_one_cpu(col, col_output)

println("   Output:       ", col_output)
println("   Expected:     ", collect(col) .+ 1)

if col_output ≈ collect(col) .+ 1
    println("   ✅ Column slice works!")
else
    println("   ❌ Column slice failed!")
    exit(1)
end

# Test 3: Row slice (non-contiguous in column-major - stride-aware!)
println("\n11. Testing row slice (non-contiguous, stride-aware!)...")
row = @view matrix[2, :]  # Second row (non-contiguous in column-major!)
row_output = zeros(Float32, 4)

println("   Input row:     ", row)
println("   Stride: ", Base.strides(row), " (non-contiguous)")
println("   Note: In column-major layout, rows have stride > 1")

# Direct slice passing - stride-aware kernel handles it!
add_one_cpu(row, row_output)

println("   Output:       ", row_output)
println("   Expected:     ", collect(row) .+ 1)

if row_output ≈ collect(row) .+ 1
    println("   ✅ Non-contiguous row slice works!")
else
    println("   ❌ Row slice failed!")
    exit(1)
end

# Test 4: Strided view (e.g., every other element)
println("\n12. Testing strided view (every other element)...")
strided = @view vector[1:2:9]  # Elements 1, 3, 5, 7, 9
strided_output = zeros(Float32, 5)

println("   Input strided: ", strided)
println("   Stride: ", Base.strides(strided), " (stride=2)")

# Stride-aware kernel handles arbitrary strides!
add_one_cpu(strided, strided_output)

println("   Output:       ", strided_output)
println("   Expected:     ", collect(strided) .+ 1)

if strided_output ≈ collect(strided) .+ 1
    println("   ✅ Strided view works!")
else
    println("   ❌ Strided view failed!")
    exit(1)
end

println("\n" * "="^60)
println("✅ STRIDE-AWARE SLICE SUPPORT VERIFIED!")
println("="^60)

# ============================================================
# High-dimensional array support!
# ============================================================
println("\n" * "="^60)
println("🚀 HIGH-DIMENSIONAL ARRAY SUPPORT")
println("="^60)

# Test 5: 3D array
println("\n13. Testing 3D array...")
arr_3d = reshape(Float32[1:24...], 2, 3, 4)
arr_3d_output = zeros(Float32, size(arr_3d))

println("   Input shape:  ", size(arr_3d), " (3D)")
println("   Input strides: ", Base.strides(arr_3d))

add_one_cpu(arr_3d, arr_3d_output)

println("   Output shape: ", size(arr_3d_output))
if arr_3d_output ≈ arr_3d .+ 1
    println("   ✅ 3D array works!")
else
    println("   ❌ 3D array failed!")
    println("   Difference: ", maximum(abs.(arr_3d_output .- (arr_3d .+ 1))))
    exit(1)
end

# Test 6: 4D array
println("\n14. Testing 4D array...")
arr_4d = reshape(Float32[1:48...], 2, 3, 2, 4)
arr_4d_output = zeros(Float32, size(arr_4d))

println("   Input shape:  ", size(arr_4d), " (4D)")
println("   Input strides: ", Base.strides(arr_4d))

add_one_cpu(arr_4d, arr_4d_output)

println("   Output shape: ", size(arr_4d_output))
if arr_4d_output ≈ arr_4d .+ 1
    println("   ✅ 4D array works!")
else
    println("   ❌ 4D array failed!")
    println("   Difference: ", maximum(abs.(arr_4d_output .- (arr_4d .+ 1))))
    exit(1)
end

# Test 7: 5D array
println("\n15. Testing 5D array...")
arr_5d = reshape(Float32[1:120...], 2, 3, 2, 2, 5)
arr_5d_output = zeros(Float32, size(arr_5d))

println("   Input shape:  ", size(arr_5d), " (5D)")
println("   Input strides: ", Base.strides(arr_5d))

add_one_cpu(arr_5d, arr_5d_output)

println("   Output shape: ", size(arr_5d_output))
if arr_5d_output ≈ arr_5d .+ 1
    println("   ✅ 5D array works!")
else
    println("   ❌ 5D array failed!")
    println("   Difference: ", maximum(abs.(arr_5d_output .- (arr_5d .+ 1))))
    exit(1)
end

# Test 8: High-dimensional slice (3D array slice)
println("\n16. Testing 3D array slice...")
arr_3d_slice = @view arr_3d[:, :, 2]  # 3rd dimension slice (2D result)
arr_3d_slice_output = zeros(Float32, size(arr_3d_slice))

println("   Input slice shape:  ", size(arr_3d_slice), " (2D slice from 3D)")
println("   Input slice strides: ", Base.strides(arr_3d_slice))

add_one_cpu(arr_3d_slice, arr_3d_slice_output)

println("   Output shape: ", size(arr_3d_slice_output))
if arr_3d_slice_output ≈ arr_3d_slice .+ 1
    println("   ✅ 3D array slice works!")
else
    println("   ❌ 3D array slice failed!")
    exit(1)
end

# Test 9: Non-contiguous high-dimensional slice
println("\n17. Testing non-contiguous high-dimensional slice...")
# Slice along first dimension in a 3D array (non-contiguous)
arr_3d_slice2 = @view arr_3d[1, :, :]  # First dimension slice
arr_3d_slice2_output = zeros(Float32, size(arr_3d_slice2))

println("   Input slice shape:  ", size(arr_3d_slice2), " (2D slice)")
println("   Input slice strides: ", Base.strides(arr_3d_slice2), " (non-contiguous)")

add_one_cpu(arr_3d_slice2, arr_3d_slice2_output)

println("   Output shape: ", size(arr_3d_slice2_output))
if arr_3d_slice2_output ≈ arr_3d_slice2 .+ 1
    println("   ✅ Non-contiguous high-dimensional slice works!")
else
    println("   ❌ Non-contiguous high-dimensional slice failed!")
    exit(1)
end

println("\n" * "="^60)
println("✅ HIGH-DIMENSIONAL ARRAY SUPPORT VERIFIED!")
println("="^60)
println("\nKey Points:")
println("  • ✅ Contiguous slices: Full zero-copy support")
println("  • ✅ Column slices: Contiguous in column-major layout")
println("  • ✅ Row slices: Non-contiguous, stride-aware kernel handles it!")
println("  • ✅ Strided views: Arbitrary strides supported (e.g., arr[::2])")
println("  • ✅ Safe: views keep parent arrays alive")
println("  • ✅ Multi-dimensional: Handles any number of dimensions (1D, 2D, 3D, 4D, 5D+)")
println("  • ✅ High-dimensional slices: Works with slices from N-D arrays")

println("\n" * "="^60)
println("TVM FFI Julia Example - Completed Successfully!")
println("="^60)

println("\n📝 Summary:")
println("   ✓ Loaded TVM module from: $module_path")
println("   ✓ Retrieved 'add_one_cpu' function")
println("   ✓ Created DLTensor views of Julia arrays")
println("   ✓ Called TVM function successfully")
println("   ✓ Verified correct results (element-wise add one)")
println("   ✓ Tested contiguous slices (zero-copy)")
println("   ✓ Tested non-contiguous slices (stride-aware!)")
println("   ✓ Tested strided views (arbitrary strides)")
println("   ✓ Tested 3D, 4D, and 5D arrays")
println("   ✓ Tested high-dimensional slices")
println("   ✓ Demonstrated zero-copy views")
println("\nThis demonstrates:")
println("   • Module loading")
println("   • Function retrieval")
println("   • Zero-copy tensor passing")
println("   • Successful execution on CPU")
println("   • 🆕 Slice support (contiguous and non-contiguous)")
println("   • 🆕 Stride-aware kernel (handles arbitrary memory layouts)")
println("   • 🆕 Zero-copy views with proper strides")
println("   • 🆕 High-dimensional array support (N-D tensors)")
