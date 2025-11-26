# Agent Guide: TVMFFI.jl

This document outlines the workflow and goals for working on the `TVMFFI.jl` Julia package.

## Version Control: Jujutsu (`jj`)

This repository uses **Jujutsu (jj)** for version control. Do **NOT** use standard `git` commands for creating commits or branches.

### Common Commands
-   **Start work**: `jj new` (creates a new anonymous change on top of the current one).
-   **Save progress**: `jj describe -m "Description of changes"` (gives the current change a commit message).
-   **View status**: `jj st` (shows the current working copy status and parent chain).
-   **View log**: `jj log` (shows the commit graph).
-   **Push**: `jj git push` (pushes changes to the remote).

**Important**: You are always in a "working copy" commit. You don't need to "stage" files. Just modify them and `jj describe` to name the commit.

## Development Workflow

### Environment
-   Activate the environment: `julia --project=.`
-   Instantiate dependencies: `using Pkg; Pkg.instantiate()`

### Testing
-   Run tests: `using Pkg; Pkg.test()`
-   Run specific test file: `include("test/runtests.jl")` or specific files in `test/`.

### Code Style
-   This project uses `.JuliaFormatter.toml`.
-   Format code: `using JuliaFormatter; format(".")`

## Roadmap & Tasks

The immediate goal is to reach feature parity with Python/Rust bindings. See `../STATUS.md` for a detailed matrix.

### Priority 1: Function Registration ✅ COMPLETE
We need to allow Julia functions to be called by TVM.
1.  ✅ **`CallbackFunctionObjImpl`**: Implemented via `safe_call` C callback with Julia function registry.
2.  ✅ **`register_global_func`**: Fully functional API that registers Julia functions into TVM global registry.
3.  ✅ **Argument Conversion**: Both `from_tvm_any` and `to_tvm_any` work for all major types (int, float, bool, string, objects).
4.  ✅ **Exception Handling**: Julia exceptions are properly caught and translated to TVM errors.

### Priority 2: Object Registration ⚠️ Partially Complete
Allow defining custom TVM objects in Julia.
1.  ✅ **Basic `register_object` function**: Implemented. Can register type keys and allocate type indices.
2.  ✅ **Type queries**: `get_type_index` works for looking up registered types.
3.  **TODO: `@register_object` Macro**: Create a macro to generate full boilerplate (vtable, field accessors, methods) for a Julia struct to be a complete `TVMObject`.

### Priority 3: System Library ✅ COMPLETE
1.  ✅ **`system_lib()`**: Access statically linked TVM modules.
2.  ✅ **Module introspection**: `inspect_source()`, `get_module_kind()`.
3.  ✅ **Module export**: `write_to_file()` for saving compiled modules.
4.  ✅ **Function checking**: `implements_function()` to verify function availability.

### Priority 4: Documentation ✅ COMPLETE
1.  ✅ **Basic Structure**: `index.md` and `api.md` created.
2.  ✅ **API Reference**: Comprehensive docstrings for exported types and functions.
3.  ✅ **Examples**: Practical examples in `README.md` and `examples/` directory.

## Directory Structure
-   `src/LibTVMFFI.jl`: Low-level C bindings (generated/maintained manually).
-   `src/TVMFFI.jl`: Main entry point.
-   `src/function.jl`: Function wrappers. **Focus here for Priority 1.**
-   `src/object.jl`: Object wrappers. **Focus here for Priority 2.**
-   `docs/`: Documentation source and build scripts.

---

## Memory Safety Guidelines

### Critical Rules for C FFI Code

When working with C API bindings, **ALWAYS** follow these rules:

#### 1. GC Safety: Use `GC.@preserve` for ALL Pointer Passing

**Rule**: Any time you extract a pointer from a Julia object and pass it to C, you **MUST** use `GC.@preserve`.

```julia
# ❌ WRONG - GC can collect str during C call
str = "hello"
byte_array = LibTVMFFI.TVMFFIByteArray(pointer(str), sizeof(str))
ret = some_c_function(byte_array)  # CRASH!

# ✅ CORRECT
str = "hello"
GC.@preserve str begin
    byte_array = LibTVMFFI.TVMFFIByteArray(
        Ptr{UInt8}(pointer(str)), UInt(sizeof(str))
    )
    ret = some_c_function(byte_array)
end
```

**Why**: Julia's GC can run at any time, even during C calls. Without `@preserve`, the string/array may be freed while C is reading it → segfault.

**Watch out for**:
- String → `TVMFFIByteArray` conversions
- Array → pointer conversions  
- Field access: `GC.@preserve obj.field` ❌ → Extract to local first: `x = obj.field; GC.@preserve x` ✅

#### 2. Reference Counting: Understand Ownership

**The Golden Rule**: Every `IncRef` must have a matching `DecRef`.

**Ownership Model**:
```julia
# Scenario 1: Taking ownership (C gives us a reference)
function from_c_api_returns_new_ref(any::TVMFFIAny)
    handle = reinterpret(TVMFFIObjectHandle, any.data)
    return TVMObject(handle; own=false)  # Don't IncRef, we already own it
    # Finalizer will DecRef → balanced
end

# Scenario 2: Borrowing (C lends us a reference)
function from_c_api_borrowed_ref(any::TVMFFIAny)
    handle = reinterpret(TVMFFIObjectHandle, any.data)
    return TVMObject(handle; own=true)   # IncRef to copy reference
    # Finalizer will DecRef → C's reference untouched
end

# Scenario 3: Passing to C (we create a temporary reference)
function pass_to_c(obj::TVMObject)
    LibTVMFFI.TVMFFIObjectIncRef(obj.handle)  # +1 for C
    result = some_c_function(obj.handle)
    LibTVMFFI.TVMFFIObjectDecRef(obj.handle)  # -1 cleanup
    return result
end
```

**Quick Reference**:
- `borrowed=true`: IncRef immediately, DecRef in finalizer (copy reference)
- `borrowed=false`: Don't IncRef, DecRef in finalizer (take ownership)

#### 3. Cross-Language Verified Reference Semantics

> **Audit Date**: 2025-11-26
> **Verified Against**: C++ (`object.h`, `function.h`), Python (`object.pxi`, `function.pxi`), Rust (`object.rs`, `any.rs`, `function.rs`)

**C API Return Semantics** (when Julia receives from C):

| C API Function | Returns | Julia `borrowed` | Rationale |
|----------------|---------|------------------|-----------|
| `TVMFFIFunctionGetGlobal` | New reference | `false` | Caller owns, don't IncRef |
| `TVMFFIFunctionCall` result | New reference | `false` | Caller owns the result |
| `TVMFFIErrorMoveFromRaised` | Moved ownership | `false` | "Move" = take ownership |
| `TVMFFIStringFromByteArray` | New reference | N/A | Direct use, finalizer handles |
| `TVMFFIBytesFromByteArray` | New reference | N/A | Direct use, finalizer handles |
| Callback arguments | Borrowed | `true` | Must IncRef to keep alive |

**Julia → C Passing Pattern**:

```julia
# When passing TVMObject/TVMFunction to C API:
function (func::TVMFunction)(args...)
    # 1. IncRef before passing (to_tvm_any does this)
    args_array[i] = to_tvm_any(arg)  # IncRef inside
    
    # 2. Call C API
    ret = LibTVMFFI.TVMFFIFunctionCall(...)
    
    # 3. DecRef after call (cleanup)
    for arg_any in args_array
        if is_object(arg_any)
            LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
        end
    end
end
```

**Cross-Language Equivalents**:

| Julia | C++ | Rust | Python |
|-------|-----|------|--------|
| `TVMObject(h; borrowed=false)` | `ObjectPtrFromOwned(h)` | `ObjectArc::from_raw(h)` | Direct assign to `.chandle` |
| `TVMObject(h; borrowed=true)` | `ObjectPtrFromUnowned(h)` | Clone after `from_raw` | `TVMFFIAnyViewToOwnedAny` |
| `from_tvm_any(; borrowed=false)` | Return from function | `Any::from_raw_ffi_any` | `make_ret(result)` |
| `from_tvm_any(; borrowed=true)` | Callback arg handling | `T::copy_from_any_view` | `TVMFFIAnyViewToOwnedAny` |

#### 4. Common Pitfalls

| Mistake | Impact | How to Avoid |
|---------|--------|--------------|
| Missing `GC.@preserve` | Segfault, data corruption | Always wrap `pointer()` → C calls |
| Wrong `borrowed` parameter | Double-free, use-after-free | Check if C API returns new ref or borrows |
| Forgetting `DecRef` | Memory leak | Every `IncRef` needs matching `DecRef` |
| Preserving fields directly | Syntax error | Extract to local: `x = obj.field` |

#### 5. Code Review Checklist

Before committing C FFI code, verify:

- [ ] Every `pointer(x)` is inside `GC.@preserve x`
- [ ] Every `TVMFFIByteArray` construction preserves the source data
- [ ] Every `IncRef` has a corresponding `DecRef`
- [ ] `borrowed` parameter matches reference source (new ref vs borrowed ref)
- [ ] Finalizers are registered for all heap objects
- [ ] NULL pointer checks before dereferencing

#### 6. Testing for Memory Safety

Add tests that stress memory management:

```julia
@testset "Memory Safety" begin
    # Force GC between operations
    for i in 1:1000
        result = some_function(data)
        GC.gc()  # Try to trigger use-after-free bugs
        @test result == expected
    end
    
    # Test with exception paths
    @test_throws TVMError error_function()
    GC.gc()  # Ensure error cleanup is correct
end
```

---

## Design Principles (Linus Style)

### 1. Good Taste - Eliminate Special Cases

Use proper data structures to avoid if/else chains:

```julia
# Good: Unified handling via generic type parameter
mutable struct DLTensorHolder{T, S}
    source::S  # Works for Array, CuArray, ROCArray, etc.
end

# No need for:
if source isa CuArray
    # special GPU handling
elseif source isa ROCArray
    # another special case
```

### 2. Simplicity - Direct Mapping

```julia
# Struct layout must match C exactly
struct TVMFFIObject
    combined_ref_count::UInt64  # Match C layout
    type_index::Int32
    __padding::UInt32
    deleter::Ptr{Cvoid}
end
```

No intermediate wrappers. One Julia struct = One C struct.

### 3. Practical - Solve Real Problems

- If C API doesn't support it, don't hack around it
- If existing code works, verify and document it (don't rewrite)
- Defer advanced features until they're actually needed

---

## Future Design: GC Pooling for Julia Object Exposure

> Reference: MWORKS.Syslab团队的GC Pooling技术（同元软控）

### Problem Statement

When exposing Julia objects to external languages (like TVM/C++), we face a challenge:
- Julia GC may collect objects that are still referenced by the external language
- External language has no way to participate in Julia's GC

### Current Implementation

`function.jl` uses a simple registry approach:

```julia
const _callback_registry = Dict{Ptr{Cvoid}, Any}()

function register_global_func(name, func)
    func_ref = Ref{Any}(func)
    resource_handle = Base.unsafe_convert(Ptr{Cvoid}, func_ref)
    _callback_registry[resource_handle] = func_ref  # Keep alive
    # ... register with TVM ...
end

function callback_deleter(resource_handle)
    delete!(_callback_registry, resource_handle)  # Allow GC
end
```

**Limitations**:
- Dict uses pointer as key → hash overhead
- No slot reuse after deletion → memory fragmentation
- Pointer debugging is harder than integer indices

### GC Pooling Pattern

Core idea: Use a **slot pool** as GC root, expose **integer indices** instead of pointers.

```julia
# Object pool as GC root
const JULIA_OBJECT_POOL = Vector{Any}(undef, 1024)
const POOL_FREELIST = Vector{Int}()
const POOL_LOCK = ReentrantLock()

function pool_insert(obj)::Int
    lock(POOL_LOCK) do
        idx = if isempty(POOL_FREELIST)
            push!(JULIA_OBJECT_POOL, obj)
            length(JULIA_OBJECT_POOL)
        else
            pop!(POOL_FREELIST)
            JULIA_OBJECT_POOL[idx] = obj
            idx
        end
        return idx
    end
end

function pool_get(idx::Int)
    @inbounds JULIA_OBJECT_POOL[idx]
end

function pool_release(idx::Int)
    lock(POOL_LOCK) do
        JULIA_OBJECT_POOL[idx] = nothing  # Allow GC
        push!(POOL_FREELIST, idx)
    end
end
```

**Benefits**:
- O(1) slot reuse via freelist
- Integer indices are easier to debug
- No hash computation
- Compact memory layout

### When to Implement

**Do NOT implement now** - current `_callback_registry` works fine for:
- Global function registration (infrequent)
- Long-lived callbacks

**Implement when**:
1. **`@register_object` macro** (Priority 2) - Julia objects exposed as TVM objects need lifecycle management
2. **Callback returns Julia arrays** - If TVM needs to hold array references beyond callback scope
3. **Performance issues** - If profiling shows registry operations as bottleneck

### Integration Points

When implementing, modify:
1. `src/function.jl`: Replace `_callback_registry` Dict with pool
2. `src/object.jl`: Use pool for `@register_object` macro
3. C callbacks: Pass slot index instead of pointer
