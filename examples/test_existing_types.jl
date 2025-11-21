using TVMFFI

println("Testing type lookup with existing types...")

# Try to look up some built-in types that should exist
type_keys = [
    "ffi.String",
    "ffi.Module",  
    "ffi.PackedFunc"
]

for key in type_keys
    println("\nLooking up '$key'...")
    try
        idx = get_type_index(key)
        println("  Found index: $idx")
    catch e
        println("  Not found or error: $e")
    end
end

println("\nTest complete!")
