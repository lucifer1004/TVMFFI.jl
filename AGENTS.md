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

### Priority 1: Function Registration
We need to allow Julia functions to be called by TVM.
1.  **Implement `CallbackFunctionObjImpl`**: Create a structure to hold a Julia closure and expose it as a TVM object.
2.  **`register_global_func`**: Implement the API to register these objects into the global TVM registry.
3.  **Argument Conversion**: Implement `from_tvm_any` and `to_tvm_any` logic specifically for handling callback arguments (handling `TVMValue` and `type_code` arrays from C).

### Priority 2: Object Registration
Allow defining custom TVM objects in Julia.
1.  **`register_object` Macro**: Create a macro to generate the necessary boilerplate (type index registration, vtable, etc.) for a Julia struct to be a `TVMObject`.

### Priority 3: System Library
1.  **`system_lib`**: Implement support for loading statically linked TVM modules.

## Directory Structure
-   `src/LibTVMFFI.jl`: Low-level C bindings (generated/maintained manually).
-   `src/TVMFFI.jl`: Main entry point.
-   `src/function.jl`: Function wrappers. **Focus here for Priority 1.**
-   `src/object.jl`: Object wrappers. **Focus here for Priority 2.**
