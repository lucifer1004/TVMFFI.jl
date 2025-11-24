# TVMFFI.jl

Julia bindings for the TVM Deep Learning Compiler FFI.

## Overview

`TVMFFI.jl` provides a bridge between Julia and the TVM runtime. It allows you to:
- Load compiled TVM modules (CPU, CUDA, Metal, etc.).
- Manage TVM arrays (`TVMTensor`) with zero-copy where possible.
- Call TVM functions from Julia.
- Register Julia functions to be called by TVM.

## Installation

```julia
using Pkg
Pkg.add("TVMFFI")
```

## Quick Start

### Loading a Module

```julia
using TVMFFI

# 1. Load a compiled library
# mod = load_module("compiled_lib.so")

# 2. Get a function
# func = mod["my_function"]

# 3. Create input tensor
input = TVMTensor(Float32[1, 2, 3])
output = TVMTensor(zeros(Float32, 3))

# 4. Call the function
# func(input, output)
```

### Working with GPU

If you have a CUDA-enabled TVM build:

```julia
using TVMFFI, CUDA

# Create a TVMTensor from a CuArray (zero-copy)
cu_arr = CuArray(Float32[1, 2, 3])
tvm_gpu_arr = TVMTensor(cu_arr)

# Use it in TVM functions
# func(tvm_gpu_arr)
```

