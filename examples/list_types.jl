using TVMFFI
using TVMFFI.LibTVMFFI

println("Listing registered type keys...")

# According to c_api.h:
# - POD types: 0-12 (kTVMFFINone to kTVMFFISmallBytes)
# - Static objects start at 64 (kTVMFFIStaticObjectBegin)

for i in vcat(0:12, 64:74, 128:139)
    info_ptr = LibTVMFFI.TVMFFIGetTypeInfo(Int32(i))
    
    if info_ptr != C_NULL
        info = unsafe_load(info_ptr)
        key = info.type_key
        if key.data != C_NULL
            str = unsafe_string(key.data, key.size)
            println("  Index $i: $str")
        end
    end
end

println("\nDone!")
