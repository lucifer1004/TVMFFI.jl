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

// Test fixture: add_one_cpu
// A simple TVM FFI function for testing Julia bindings

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

namespace tvmffi_test_fixtures {

/*! \brief Perform element-wise add one: y = x + 1 (N-D float32, stride-aware)
 * 
 * This version correctly handles:
 * - Any number of dimensions (1D, 2D, 3D, ...)
 * - Non-contiguous tensors (arbitrary strides)
 * - Slices in any dimension
 * - Row/column slices in matrices
 * - Strided views (e.g., arr[::2, ::3])
 * 
 * Implementation uses recursive multi-dimensional indexing to handle arbitrary
 * stride patterns efficiently.
 */
void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int ndim = x.ndim();
  float* x_data = static_cast<float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  
  // Get strides using ShapeView (in element units per DLPack standard)
  auto x_strides = x.strides();
  auto y_strides = y.strides();
  
  // Compute total number of elements
  int64_t total_elements = x.numel();
  
  // Multi-dimensional index buffer
  std::vector<int64_t> indices(ndim, 0);
  
  // Iterate through all elements using multi-dimensional indexing
  for (int64_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Compute memory offsets using strides (ShapeView supports operator[])
    int64_t x_offset = 0;
    int64_t y_offset = 0;
    for (int i = 0; i < ndim; ++i) {
      x_offset += indices[i] * x_strides[i];
      y_offset += indices[i] * y_strides[i];
    }
    
    // Perform computation
    y_data[y_offset] = x_data[x_offset] + 1.0f;
    
    // Increment indices (column-major order for compatibility with Julia)
    for (int i = 0; i < ndim; ++i) {
      if (++indices[i] < x.size(i)) {
        break;  // No carry
      }
      indices[i] = 0;  // Reset and carry to next dimension
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, tvmffi_test_fixtures::AddOne);

}  // namespace tvmffi_test_fixtures

