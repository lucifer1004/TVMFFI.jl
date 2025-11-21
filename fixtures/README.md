# Test Fixtures

Compiled test fixtures for TVMFFI.jl testing.

## Available Fixtures

### CPU Fixtures

**`add_one_cpu`** - Element-wise add one kernel (CPU, stride-aware)
- Supports arbitrary dimensions (1D, 2D, 3D, ...)
- Handles non-contiguous tensors with arbitrary strides
- Works with slices, sub-arrays, and strided views
- Implementation: `add_one_cpu.cc`

### GPU Fixtures (Optional)

**`add_one_cuda`** - Element-wise add one kernel (CUDA GPU, stride-aware)
- Same functionality as CPU version, but runs on GPU
- Requires CUDA toolkit and GPU hardware
- Implementation: `add_one_cuda.cu`

## Building Fixtures

Fixtures are automatically built when first used in tests. Manual build:

```bash
cd fixtures
mkdir -p ../build/fixtures
cmake . -B ../build/fixtures
cmake --build ../build/fixtures
```

### Building with CUDA Support

CUDA fixtures are built automatically if CUDA is detected:

```bash
# CUDA will be detected automatically
cmake . -B ../build/fixtures
cmake --build ../build/fixtures
```

To explicitly disable CUDA:

```bash
cmake . -B ../build/fixtures -DBUILD_CUDA_FIXTURES=OFF
cmake --build ../build/fixtures
```

### CUDA Requirements

For CUDA fixture to build:
- CUDA toolkit installed (nvcc compiler)
- CMake 3.20+ with CUDA support
- GPU with compute capability 6.0+ (Pascal or newer)

For CUDA tests to run:
- CUDA.jl package: `using Pkg; Pkg.add("CUDA")`
- CUDA-capable GPU available
- CUDA fixture successfully built

## Testing

Run tests (CUDA tests skipped automatically if unavailable):

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Test output will show:
- `Pass` - CPU tests always run
- `Broken` - CUDA tests skipped (no CUDA available)
- `Pass` - CUDA tests run (CUDA available)

## Architecture

All fixtures follow the TVM FFI C API:
- Use `TensorView` for zero-copy tensor access
- Handle strides correctly (DLPack standard: strides in elements)
- Support both contiguous and non-contiguous memory layouts
- Exported via `TVM_FFI_DLL_EXPORT_TYPED_FUNC` macro

## Adding New Fixtures

1. Create source file (`.cc` for CPU, `.cu` for CUDA)
2. Implement function using `tvm::ffi::TensorView`
3. Export with `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, func)`
4. Add to `CMakeLists.txt`
5. Add tests to `test/test_fixtures.jl`
