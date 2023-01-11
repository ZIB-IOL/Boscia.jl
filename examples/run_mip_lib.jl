include("mip-examples.jl")

example="22433"

bo_mode="boscia"
for v_num in [4:6:1]
    for seed in 1:3
        @show seed, v_num
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "as"
for v_num in [4:6:1]
    for seed in 1:3
        @show seed, v_num
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "ss"
for v_num in [4:6:1]
    for seed in 1:3
        @show seed, v_num
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "as_ss"
for v_num in [4:6:1]
    for seed in 1:3
        @show seed, v_num
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "afw"
for v_num in [4:6:1]
    for seed in 1:3
        @show seed, v_num
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "scip_oa"
for v_num in [4:6:1]
    for seed in 1:3
        @show seed, v_num
        try 
            mip_lib_scip(seed, num_v; example=example)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end
