include("mip-examples.jl")

example="neos5" #"ran14x18-disj-8" "pg5_34" "neos5" "22433"
#=
bo_mode = "boscia"
for num_v in [5]#[4:1:8;]
    for seed in [1] #1:3
        @show seed, num_v
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
    end
end
=#

bo_mode = "strong_convexity"
for example in ["neos5"] #, "pg5_34", "ran14x18-disj-8"]
    for num_v in [4:1:8;]#[4:1:8;]
        for seed in 1:3 #1:3
            @show seed, num_v
            try 
                mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
            catch 
                println(e)
                open("mip_lib_" * example * "_errors.txt","a") do io
                    println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
                end
            end
        end
    end
end
#=
bo_mode = "as"
for num_v in [4:1:8;]
    for seed in 1:3
        @show seed, num_v
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "ss"
for num_v in [4:1:8;]
    for seed in 1:3
        @show seed, num_v
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "as_ss"
for num_v in [4:1:8;]
    for seed in 1:3
        @show seed, num_v
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "afw"
for num_v in [4:1:8;]
    for seed in 1:3
        @show seed, num_v
        try 
            mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
        catch e
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
    end
end=#

#=
bo_mode = "scip_oa"
for num_v in [4:1:8;]
   for seed in 1:3
       @show seed, num_v
       try 
           mip_lib_scip(seed, num_v; example=example)
       catch e
           println(e)
           open("mip_lib_" * example * "_errors.txt","a") do io
               println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
           end
       end
   end
end
=#


#=
bo_mode = "ipopt"
for num_v in [4:1:8;]#[4:1:8;]
    for seed in 1:3#1:3
        @show seed, num_v
        try
            mip_lib_ipopt(seed, num_v; example=example)
        catch 
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
    end
end
=#
#=
bo_mode = "boscia"
num_v = 6
seed = 1

for num_v in [7]
     for seed in [1]
        try 
            mip_lib(seed, num_v; example =example, bo_mode=bo_mode)
        catch  
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
        #mip_lib_scip(seed, num_v; example=example)
    end
end 
=#


# Full runs for images
#=
examples = ["22433", "pg5_34", "neos5", "ran14x18-disj-8"]

for example in examples
    for num_v in [4,8]
        seed = 1
        try
            mip_lib(seed, num_v, true; example=example, bo_mode ="boscia")
           # mip_lib_ipopt(seed, num_v, true; example=example)
        catch
            println(e)
            open("mip_lib_" * example * "_errors.txt","a") do io
                println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
            end
        end
    end
end=#

