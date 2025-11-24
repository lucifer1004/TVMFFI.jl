/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Test fixture: add_one_metal
// Metal Shading Language kernel for testing Julia bindings with Metal arrays

#include <metal_stdlib>
using namespace metal;

/*! \brief Metal kernel: y[i] = x[i] + 1 (N-D, stride-aware)
 * 
 * Each thread handles one element using linear indexing.
 * Converts linear index to multi-dimensional coordinates (column-major).
 * 
 * @param x Input buffer (device)
 * @param y Output buffer (device)
 * @param total_elements Total number of elements
 * @param x_strides Stride array for input (device)
 * @param y_strides Stride array for output (device)
 * @param shape Shape array (device)
 * @param ndim Number of dimensions
 */
kernel void add_one_kernel_nd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const int32_t* x_strides [[buffer(2)]],
    device const int32_t* y_strides [[buffer(3)]],
    device const int32_t* shape [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    constant int32_t& total_elements [[buffer(6)]],
    uint linear_idx [[thread_position_in_grid]]
) {
    if (linear_idx >= total_elements) return;
    
    // Convert linear index to multi-dimensional indices (column-major)
    int32_t x_offset = 0;
    int32_t y_offset = 0;
    int32_t remaining = static_cast<int32_t>(linear_idx);
    
    for (int i = 0; i < ndim; ++i) {
        int32_t coord = remaining % shape[i];
        remaining /= shape[i];
        x_offset += coord * x_strides[i];
        y_offset += coord * y_strides[i];
    }

    y[y_offset] = x[x_offset] + 1.0f;
}

