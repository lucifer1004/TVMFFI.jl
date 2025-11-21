using TVMFFI
using TVMFFI.LibTVMFFI
using TVMFFI_jll

println("Checking TVM version...")
version = Ref{Int32}(0)
ret = @ccall libtvm_ffi.TVMFFIGetVersion(version::Ptr{Int32})::Cint

println("Return code: $ret")
println("Version: $(version[])")
