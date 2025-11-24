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
    _get_root_array(arr) -> AbstractArray

Unwrap array wrappers to find the underlying root array.

Design Philosophy (Linus-style):
- Good taste: Use Julia's standard `parent()` interface
- Simple: Recursively unwrap until we can't anymore
- Handles all wrappers: OffsetArray, SubArray, ReshapedArray, etc.

# Arguments
- `arr`: Potentially wrapped array

# Returns
- `AbstractArray`: The unwrapped underlying array

# Why This Matters
GPU arrays are often wrapped by:
- OffsetArrays.OffsetArray (custom indexing)
- SubArray (views)
- ReshapedArray (reshape)
- PermutedDimsArray (permutedims)
- ReinterpretArray (reinterpret)

All these wrappers implement `parent()` to access the underlying array.

# Examples
```julia
using CUDA, OffsetArrays

x = CuArray([1, 2, 3])
y = OffsetArray(x, -1:1)

unwrapped = _get_root_array(y)  # Returns x (the CuArray)
```
"""
function _get_root_array(arr)
    current = arr

    # Keep unwrapping while parent() gives us something different
    while hasmethod(parent, Tuple{typeof(current)})
        next = parent(current)
        # Stop if parent() returns the same object (we've reached the bottom)
        if next === current
            break
        end
        current = next
    end

    return current
end

"""
    _navigate_to_root_module(arr, target_module_name::Symbol) -> Module

Navigate from an array's type to its root GPU package module.

Design Philosophy (Linus-style):
- Good taste: One function handles all GPU backends
- Simple: Just walk up the module tree
- No special cases: Same logic for CUDA, AMDGPU, oneAPI, Metal

# Arguments
- `arr`: GPU array object (will be unwrapped if it's a wrapper)
- `target_module_name::Symbol`: Name of the root module to find (:CUDA, :AMDGPU, :oneAPI, :Metal)

# Returns
- `Module`: The root module of the GPU package

# Algorithm
1. Unwrap array wrappers to find the actual GPU array
2. Start from `parentmodule(typeof(arr))`
3. Walk up the module tree via repeated `parentmodule()` calls
4. Stop when we reach a module with the target name OR the root (module is its own parent)

# Examples
```julia
using CUDA
x = CUDA.CuArray([1, 2, 3])
cuda_module = _navigate_to_root_module(x, :CUDA)
# Returns: CUDA module

# Also works with wrappers!
using OffsetArrays
y = OffsetArray(x, -1:1)
cuda_module = _navigate_to_root_module(y, :CUDA)  # Still returns CUDA module
```
"""
function _navigate_to_root_module(arr, target_module_name::Symbol)
    # First, unwrap any array wrappers
    unwrapped = _get_root_array(arr)

    arr_module = parentmodule(typeof(unwrapped))

    # Walk up the module tree to find the root module
    while parentmodule(arr_module) !== arr_module
        arr_module = parentmodule(arr_module)
        # Early exit if we found the target module
        if nameof(arr_module) == target_module_name
            break
        end
    end

    return arr_module
end


