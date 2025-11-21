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

// Test fixture: add_one_cuda
// CUDA kernel for testing Julia bindings with GPU arrays

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <cuda_runtime.h>

namespace tvmffi_test_fixtures {

/*! \brief CUDA kernel: y[i] = x[i] + 1 (N-D, stride-aware)
 * 
 * Each thread handles one element using linear indexing.
 * Converts linear index to multi-dimensional coordinates (column-major).
 */
__global__ void add_one_kernel_nd(const float* x, float* y, 
                                   int64_t total_elements,
                                   const int64_t* x_strides,
                                   const int64_t* y_strides,
                                   const int64_t* shape,
                                   int ndim) {
  int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx >= total_elements) return;
  
  // Convert linear index to multi-dimensional indices (column-major)
  int64_t x_offset = 0;
  int64_t y_offset = 0;
  int64_t remaining = linear_idx;
  
  for (int i = 0; i < ndim; ++i) {
    int64_t coord = remaining % shape[i];
    remaining /= shape[i];
    x_offset += coord * x_strides[i];
    y_offset += coord * y_strides[i];
  }
  
  // Perform computation
  y[y_offset] = x[x_offset] + 1.0f;
}

/*! \brief Perform N-D add one: y = x + 1 (float32, stride-aware)
 * 
 * This CUDA version correctly handles:
 * - Any number of dimensions (1D, 2D, 3D, ...)
 * - Non-contiguous tensors with arbitrary strides
 * - Slices and strided views in any dimension
 * - Zero-copy device memory access
 */
void AddOneCUDA(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int ndim = x.ndim();
  int64_t total_elements = x.numel();
  
  // Get device pointers
  float* x_data = static_cast<float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  
  // Get strides and shape (need to copy to device)
  auto x_strides_view = x.strides();
  auto y_strides_view = y.strides();
  auto shape_view = x.shape();
  
  // Copy metadata to device
  int64_t* d_x_strides;
  int64_t* d_y_strides;
  int64_t* d_shape;
  cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_y_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_shape, ndim * sizeof(int64_t));
  
  // Copy stride/shape data (ShapeView provides data() method)
  cudaMemcpy(d_x_strides, x_strides_view.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_strides, y_strides_view.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_shape, shape_view.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  
  // Launch kernel
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  
  add_one_kernel_nd<<<num_blocks, threads_per_block>>>(
    x_data, y_data, total_elements, d_x_strides, d_y_strides, d_shape, ndim
  );
  
  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_x_strides);
    cudaFree(d_y_strides);
    cudaFree(d_shape);
    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + 
                           cudaGetErrorString(err));
  }
  
  // Synchronize to ensure kernel completion
  cudaDeviceSynchronize();
  
  // Free device memory
  cudaFree(d_x_strides);
  cudaFree(d_y_strides);
  cudaFree(d_shape);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, tvmffi_test_fixtures::AddOneCUDA);

}  // namespace tvmffi_test_fixtures

