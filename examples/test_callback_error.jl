using TVMFFI

function my_error_func()
    error("This is a Julia error")
end

println("Registering error function...")
register_global_func("test.error", my_error_func)
func = get_global_func("test.error")

println("Calling error function...")
try
    func()
    error("Should have thrown an error")
catch e
    println("Caught expected error: $e")
    if e isa TVMError
        println("TVMError kind: $(e.kind)")
        println("TVMError message: $(e.message)")
        
        if !occursin("This is a Julia error", e.message)
            error("Error message does not contain original Julia error message")
        end
        
        if e.kind != "RuntimeError"
            error("Error kind is not RuntimeError")
        end
    else
        error("Caught error is not a TVMError: $(typeof(e))")
    end
end

println("Success!")
