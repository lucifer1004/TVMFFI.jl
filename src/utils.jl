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

# _navigate_to_root_module has been DELETED!
#
# Design Philosophy (Linus-style):
# - It was a hack to work around not having proper type dispatch
# - Now we use DLPack.dldevice() for device detection - no duplication!
# - DLPack.jl already handles type dispatch for all GPU backends
#
# If you're looking for how GPU backends are detected, see:
# - DLPack.jl/ext/CUDAExt.jl: dldevice(::CUDA.CuArray)
# - TVMFFI/ext/AMDGPUExt.jl: DLPack.dldevice(::AMDGPU.ROCArray)
# - TVMFFI/ext/MetalExt.jl: DLPack.dldevice(::Metal.MtlArray)
