include("sparse_reg.jl")
#==

bo_mode="boscia"
for dimension in [16,18,20]
    for seed in 3:5
        @show seed, dimension
        try 
            sparse_reg(seed, dimension, 1, true; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_reg_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "local_tightening"
for dimension in [16,18,20]
    for seed in 3:5
        @show seed, dimension
        try 
            sparse_reg(seed, dimension, 1, true; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_reg_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end

bo_mode = "global_tightening"
for dimension in [16,18,20]
    for seed in 3:5
        @show seed, dimension
        try 
            sparse_reg(seed, dimension, 1, true; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_reg_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end
=#
bo_mode = "no_tightening"
for dimension in [16,18,20]
    for seed in 3:5
        @show seed, dimension
        try 
            sparse_reg(seed, dimension, 1, true; bo_mode=bo_mode)
        catch e
            println(e)
            open("sparse_reg_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end 

#=bo_mode = "as"
for dimension in [15:1:30;]
   for seed in 1:10
       @show seed, dimension
       try 
           sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
       catch e
           println(e)
           open("sparse_reg_errors.txt","a") do io
               println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
           end
       end
   end
end

bo_mode = "ss"
for dimension in [15:1:30;]
   for seed in 1:10
       @show seed, dimension
       try 
           sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
       catch e
           println(e)
           open("sparse_reg_errors.txt","a") do io
               println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
           end
       end
   end
end

bo_mode = "as_ss"
for dimension in [15:1:30;]
   for seed in 1:10
       @show seed, dimension
       try 
           sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
       catch e
           println(e)
           open("sparse_reg_errors.txt","a") do io
               println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
           end
       end
   end
end =#

# bo_mode = "afw"
# for dimension in [15:1:30;]
#    for seed in 1:10
#        @show seed, dimension
#        try 
#            sparse_reg(seed, dimension, 1; bo_mode=bo_mode)
#        catch 
#            println(e)
#            open("sparse_reg_errors.txt","a") do io
#                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#            end
#        end
#    end
# end

#=
bo_mode = "afw"
try 
    sparse_reg(4, 15, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(5, 15, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(7, 15, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(8, 15, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 16, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(8, 16, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(10, 17, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 18, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(5, 18, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(5, 19, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 20, 1; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 20, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 20, 5; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 20, 7; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 20, 8; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(1, 21, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(10, 22, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(10, 23, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(10, 24, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(10, 25, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(6, 26, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    sparse_reg(6, 28, 2; bo_mode=bo_mode)
catch 
    println(e)
    open("sparse_reg_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
=#
#=bo_mode = "scip_oa"
for dimension in [28:1:30;]
    for seed in 1:10
        @show seed, dimension
        try 
            sparse_reg_scip(seed, dimension, 1; tol=1e-9)
        catch e
            println(e)
            open("sparse_reg_errors.txt","a") do io
                println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
            end
        end
    end
end=#

# bo_mode = "ipopt"
# for dimension in [15:1:30;]#[28:1:30;] #[15:1:30;]
#     for seed in 1:10 #1:10
#         @show seed, dimension
#         try
#             sparse_reg_ipopt(seed, dimension, 1)
#         catch e
#             println(e)
#             open("sparse_reg_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end
