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
get_global_func
register_global_func
```

## Tensors & Data

Core data structures for exchanging data with TVM.

```@docs
TVMTensor
TensorView
from_dlpack
shape
dtype
device
DLDataType
```

## Devices

Device management and creation.

```@docs
DLDevice
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
```

## Types

Core types used in the FFI.

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

## Version

```@docs
tvm_ffi_version
```

---

# Advanced API

These APIs are available but not exported. Use `TVMFFI.xxx` to access them.

## Low-Level Types

```@docs
TVMFFI.DLTensor
TVMFFI.DLDeviceType
TVMFFI.DLDataTypeCode
TVMFFI.TVMObject
TVMFFI.TVMString
TVMFFI.TVMBytes
```

## Any Containers (Internal)

```@docs
TVMFFI.TVMAny
TVMFFI.TVMAnyView
TVMFFI.take_value
TVMFFI.copy_value
TVMFFI.raw_data
```

## Object Registration (Advanced)

```@docs
TVMFFI.register_object
TVMFFI.get_type_index
TVMFFI.@register_object_simple
```

## Reflection API

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

```@docs
TVMFFI.write_to_file
TVMFFI.inspect_source
TVMFFI.get_module_kind
TVMFFI.implements_function
```

## Internal Utilities

```@docs
TVMFFI.check_call
TVMFFI.dtype_to_julia_type
TVMFFI.print_gpu_info
TVMFFI.gpu_array_info
```

## Low-Level C Bindings

```@docs
TVMFFI.LibTVMFFI
```
