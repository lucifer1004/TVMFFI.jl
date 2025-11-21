using TVMFFI

println("Testing tvm_ffi_version() function...")

# Get version as VersionNumber
v = tvm_ffi_version()

println("TVM FFI Version: ", v)
println("Type: ", typeof(v))

# Test version comparison
println("\nVersion comparisons:")
println("  v >= v\"0.1.0\": ", v >= v"0.1.0")
println("  v == v\"0.1.2\": ", v == v"0.1.2")
println("  v < v\"1.0.0\": ", v < v"1.0.0")

# Display components
println("\nVersion components:")
println("  Major: ", v.major)
println("  Minor: ", v.minor)
println("  Patch: ", v.patch)

println("\n✓ tvm_ffi_version() test passed!")

