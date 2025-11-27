# Agent Guide: TVMFFI.jl

本文档为 AI Agent 和开发者提供 TVMFFI.jl 项目的技术指南。

## 快速入门

### 版本控制：Jujutsu (`jj`)

本仓库使用 **Jujutsu (jj)** 而非 git。

```bash
jj new                      # 创建新变更
jj describe -m "message"    # 添加提交信息
jj st                       # 查看状态
jj log                      # 查看历史
jj git push                 # 推送到远程
```

### 开发环境

```bash
julia --project=.           # 激活环境
```

```julia
using Pkg; Pkg.instantiate()  # 安装依赖
using Pkg; Pkg.test()         # 运行测试
using JuliaFormatter; format(".")  # 格式化代码
```

---

## 项目架构

### 目录结构

```
src/
├── LibTVMFFI.jl       # C API 绑定（底层）
├── TVMFFI.jl          # 主入口
├── any.jl             # TVMAny/TVMAnyView 所有权容器
├── conversion.jl      # ABI 边界层（to_tvm_any, take_value, copy_value）
├── function.jl        # 函数包装器
├── object.jl          # 对象包装器
├── tensor.jl          # TensorView 实现
├── dlpack.jl          # DLPack 零拷贝交换
└── gpuarrays_support.jl  # GPU 数组支持

ext/
├── CUDAExt.jl         # 占位符（DLPack.jl 提供 CUDA 支持）
├── AMDGPUExt.jl       # AMD ROCm 支持
└── MetalExt.jl        # Apple Metal 支持
```

### 核心类型

| 类型 | 用途 | type_index |
|------|------|------------|
| `TVMObject` | 通用 TVM 对象包装 | 变化 |
| `TVMFunction` | 可调用函数 | 16 |
| `TVMTensor` | 引用计数张量 | 70 |
| `TensorView` | 轻量指针视图 | 7 |
| `TVMAny` | 拥有所有权的值容器 | - |
| `TVMAnyView` | 借用的值视图 | - |

---

## 内存安全规范

### 1. GC 安全：指针传递必须用 `GC.@preserve`

```julia
# ❌ 错误 - GC 可能在 C 调用期间回收 str
str = "hello"
byte_array = LibTVMFFI.TVMFFIByteArray(pointer(str), sizeof(str))
ret = some_c_function(byte_array)  # 崩溃！

# ✅ 正确
str = "hello"
GC.@preserve str begin
    byte_array = LibTVMFFI.TVMFFIByteArray(
        Ptr{UInt8}(pointer(str)), UInt(sizeof(str))
    )
    ret = some_c_function(byte_array)
end
```

**注意**：`GC.@preserve obj.field` 无效，必须先提取到局部变量。

### 2. 引用计数：所有权模型

**黄金法则**：每个 `IncRef` 必须有匹配的 `DecRef`。

```julia
# 场景 1：接管所有权（C 返回新引用）
TVMObject(handle; borrowed=false)  # 不 IncRef，finalizer 会 DecRef

# 场景 2：借用（C 借给我们）
TVMObject(handle; borrowed=true)   # IncRef，finalizer 会 DecRef
```

**C API 返回语义**：

| C API 函数 | 返回类型 | Julia `borrowed` |
|------------|----------|------------------|
| `TVMFFIFunctionGetGlobal` | 新引用 | `false` |
| `TVMFFIFunctionCall` 结果 | 新引用 | `false` |
| 回调参数 | 借用 | `true` |

> **设计决策**：`borrowed` 参数**无默认值**。强制显式指定语义，防止误用。

### 3. TVMAny / TVMAnyView 类型系统

```julia
# TVMAnyView - 借用视图（回调参数用）
view = TVMAnyView(raw_any)
value = copy_value(view)  # 复制引用（对象会 IncRef）

# TVMAny - 拥有所有权（函数返回用）
owned = TVMAny(raw_any)
value = take_value(owned)  # 取走所有权，owned 失效
```

### 4. 代码审查清单

- [ ] 每个 `pointer(x)` 都在 `GC.@preserve x` 内
- [ ] 每个 `TVMFFIByteArray` 构造都保护了源数据
- [ ] 每个 `IncRef` 有对应的 `DecRef`
- [ ] `borrowed` 参数与引用来源匹配
- [ ] 所有堆对象都注册了 finalizer

---

## DLPack 张量交换

### API

```julia
# Julia → TVM（零拷贝）
arr = rand(Float32, 3, 4)
tensor = TVMTensor(arr)

# TVM → Julia（零拷贝）
arr2 = from_dlpack(tensor)

# 轻量视图（需要手动管理生命周期）
view = TensorView(arr)
GC.@preserve arr begin
    tvm_func(view)
end
```

### 类型对比

| 类型 | type_index | 引用计数 | 适用场景 |
|------|------------|----------|----------|
| `TVMTensor` | 70 | ✅ 有 | GPU 数组，跨边界传递 |
| `TensorView` | 7 | ❌ 无 | CPU 数组，短期使用 |

### GPU 支持

| 后端 | 扩展 | 数组类型 |
|------|------|----------|
| NVIDIA CUDA | DLPack.jl/CUDAExt | CuArray |
| Apple Metal | TVMFFI/MetalExt | MtlArray |
| AMD ROCm | TVMFFI/AMDGPUExt | ROCArray |

---

## 已知限制

### 1. BenchmarkTools + GPU 数组

**问题**：`@benchmark` 会在迭代间调用 `GC.gc()`，导致返回的 GPU 数组被回收时触发段错误。

```julia
# ❌ 崩溃
@benchmark my_gpu_func($arr)

# ✅ 使用手动计时
n = 10000
t_start = time_ns()
for _ in 1:n
    func(arr)
end
CUDA.synchronize()
t_avg = (time_ns() - t_start) / n
```

**原因**：BenchmarkTools 的 `gcscrub()` 与 CUDA.jl finalizer 的交互问题，非 TVMFFI bug。

### 2. GPU 数组开销

GPU 数组通过 `kTVMFFITensor`（type_index=70）传递，有完整引用计数开销：

- CPU 数组 identity：~600 ns
- GPU 数组 identity：~14-26 μs

对于计算密集操作，此开销可忽略。

---

## 设计原则

### 1. 消除特殊情况

```julia
# ❌ 字符串匹配和模块导航 hack
function detect_backend(arr)
    type_name = string(typeof(arr).name.name)
    if occursin("Cu", type_name)
        ...
    end
end

# ✅ 使用 DLPack.jl 的类型派发
function _dlpack_to_tvm_device(arr)
    dlpack_dev = DLPack.dldevice(arr)  # DLPack 处理一切
    return DLDevice(Int32(dlpack_dev.device_type), Int32(dlpack_dev.device_id))
end
```

### 2. 直接映射

```julia
# Julia struct 布局必须精确匹配 C
struct TVMFFIObject
    combined_ref_count::UInt64
    type_index::Int32
    __padding::UInt32
    deleter::Ptr{Cvoid}
end
```

### 3. 实用主义

- C API 不支持的功能，不要 hack
- 现有代码能工作，验证并文档化（不要重写）
- 推迟高级功能直到真正需要

---

## 未来工作（可选）

### GC Pooling

当前使用 `Dict{Ptr{Cvoid}, Any}` 作为回调注册表。如果以下场景出现性能瓶颈，考虑实现 slot pool：

- 大量短生命周期回调
- 高频函数注册/注销

核心思路：用 `Vector{Any}` + freelist 替代 Dict，暴露整数索引而非指针。

**当前状态**：不需要实现。现有实现对全局函数注册足够高效。
