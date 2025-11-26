# API Reference

```@docs
TVMFFI
```

## Module & Function Management

Functions for loading modules and managing functions.

```@docs
load_module
get_function
system_lib
write_to_file
inspect_source
get_module_kind
implements_function
get_global_func
register_global_func
Base.getindex
```

## TensorViews & Data

Core data structures for exchanging data with TVM.

```@docs
TVMTensorView
shape
dtype
device
to_julia_array
TensorView
dtype_to_julia_type
DLDataType
DLDataTypeCode
Base.size
Base.ndims
Base.length
```

## Devices

Device management and creation.

```@docs
DLDevice
DLDeviceType
cpu
cuda
opencl
vulkan
metal
rocm
```

## GPU Utilities

Helper functions for GPU support.

```@docs
supports_gpu_backend
list_available_gpu_backends
print_gpu_info
gpu_array_info
```

## Types

Core types used in the FFI.

```@docs
TVMFunction
TVMModule
TVMObject
TVMError
TVMString
TVMBytes
register_object
get_type_index
Base.String
Base.Vector
```

## Error Handling

```@docs
check_call
TVMErrorKind
```

## Low Level API

Direct mappings to C API structures. Use `TVMTensorView` and `TVMFunction` for high-level access.

```@docs
DLTensor
TensorView
tvm_ffi_version
TVMFFI.LibTVMFFI
```
