# API Reference

This page provides a comprehensive reference for all public APIs in TVMFFI.jl.

## Main Module

```@docs
TVMFFI
```

## Version Information

```@docs
tvm_ffi_version
```

## Module System

Functions for loading compiled TVM modules and accessing their contents.

Modules represent compiled TVM code that can be loaded from shared libraries or other formats.

```@docs
load_module
get_function
system_lib
get_global_func
register_global_func
```

## Tensors & Data Types

Core data structures for efficient data exchange between Julia and TVM. TVMFFI.jl provides both reference-counted tensors (`TVMTensor`) and lightweight views (`TensorView`) for different use cases.

```@docs
TVMTensor
TensorView
from_dlpack
dldevice
shape
dtype
device
DLDataType
```

## Device Management

Functions for creating and working with TVM devices. TVM supports multiple device types including CPU, CUDA, Metal, ROCm, and OpenCL.

```@docs
DLDevice
cpu
cuda
opencl
vulkan
metal
rocm
```

## GPU Support

Utilities for detecting and working with GPU backends. TVMFFI.jl provides automatic GPU detection and optimized paths for CUDA, Metal, and ROCm.

```@docs
supports_gpu_backend
list_available_gpu_backends
```

## Core Types

Main types exported by TVMFFI.jl for working with TVM objects, functions, and modules.

```@docs
TVMFunction
TVMModule
TVMError
TVMErrorKind
```

## Object Registration

Macros for registering Julia types as TVM objects.

```@docs
TVMFFI.@register_object
TVMFFI.type_index
TVMFFI.type_key
```


---

# Advanced API

These APIs are available but not exported by default. Use `TVMFFI.xxx` to access them. They provide lower-level control and introspection capabilities.

## Low-Level Types

Primitive types and structures that map directly to TVM's C API.

```@docs
TVMFFI.DLTensor
TVMFFI.DLDeviceType
TVMFFI.DLDataTypeCode
TVMFFI.TVMObject
TVMFFI.TVMString
TVMFFI.TVMBytes
```

## Any Containers (Internal)

Types for handling TVM's dynamic `TVMValue` and `TVMValue*` types. These are used internally for type-erased value passing.

```@docs
TVMFFI.TVMAny
TVMFFI.TVMAnyView
TVMFFI.take_value
TVMFFI.copy_value
TVMFFI.raw_data
```

## Object Registration (Advanced)

Advanced macros and functions for registering Julia types as TVM objects and managing the type system.

```@docs
TVMFFI.register_object
TVMFFI.get_type_index
TVMFFI.@register_object_simple
```

## Reflection API

Functions for runtime introspection of TVM objects, including field access, method discovery, and type information.

```@docs
TVMFFI.get_type_info
TVMFFI.get_fields
TVMFFI.get_methods
TVMFFI.FieldInfo
TVMFFI.MethodInfo
TVMFFI.get_field_value
TVMFFI.set_field_value!
TVMFFI.call_method
TVMFFI.get_method_function
TVMFFI.has_ffi_init
TVMFFI.ffi_init
```

## Module Introspection

Functions for examining compiled TVM modules, including source inspection and capability queries.

```@docs
TVMFFI.write_to_file
TVMFFI.inspect_source
TVMFFI.get_module_kind
TVMFFI.implements_function
```

## Internal Utilities

Helper functions used internally by TVMFFI.jl for error handling, type conversion, and debugging.

```@docs
TVMFFI.check_call
TVMFFI.dtype_to_julia_type
TVMFFI.print_gpu_info
TVMFFI.gpu_array_info
```

## Low-Level C Bindings

Direct bindings to the TVM C API functions. These are automatically generated and provide the foundation for the higher-level Julia APIs.

```@docs
TVMFFI.LibTVMFFI
```
