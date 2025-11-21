using TVMFFI
using TVMFFI.LibTVMFFI

println("Looking up 'ffi.Object'...")

type_key = "ffi.Object"
GC.@preserve type_key begin
    key_bytes = LibTVMFFI.TVMFFIByteArray(
        Ptr{UInt8}(pointer(type_key)), UInt(sizeof(type_key))
    )
    ret, idx = LibTVMFFI.TVMFFITypeKeyToIndex(key_bytes)
    
    println("Return code: $ret")
    println("Index: $idx")
end
