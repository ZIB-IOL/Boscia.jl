include("portfolio.jl")

mode = "integer"

bo_mode="boscia"
try 
    portfolio(5, 110; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
bo_mode="boscia"
try 
    portfolio(10, 110; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
endbo_mode="boscia"
try 
    portfolio(1, 115; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end


# bo_mode="boscia"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end 

# bo_mode="ss"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="ss"
try 
    portfolio(10, 110; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(1, 115; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(5, 115; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(3, 120; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end

# bo_mode="as"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="as"
try 
    portfolio(8, 120; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end

# bo_mode="as_ss"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="as_ss"
try 
    portfolio(6, 115; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end

# bo_mode="afw"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="afw"
try 
    portfolio(5, 110; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(6, 110; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(7, 120; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end


mode = "mixed"

# bo_mode="boscia"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end 

bo_mode="boscia"
try 
    portfolio(7, 75; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(4, 115; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end

# bo_mode="ss"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="ss"
try 
    portfolio(5, 70; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(6, 75; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(1, 100; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(3, 120; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end

# bo_mode="as"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="as"
try 
    portfolio(6, 80; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(4, 115; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end

# bo_mode="as_ss"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="as_ss"
try 
    portfolio(8, 80; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(4, 90; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(5, 110; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(3, 120; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end

# bo_mode="afw"
# for dimension in [120:1:120;]
#     for seed in 4:10
#         @show seed, dimension
#         try 
#             portfolio(seed, dimension; bo_mode=bo_mode, mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

bo_mode="afw"
try 
    portfolio(10, 80; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(1, 100; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(10, 115; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(4, 120; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end
try 
    portfolio(5, 120; bo_mode=bo_mode, mode=mode)
catch e
    println(e)
    open("portfolio_" * mode * "_errors.txt","a") do io
        println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
    end
end


# bo_mode = "scip_oa"
# for dimension in [20:5:120;]
#     for seed in 1:10
#         @show seed, dimension
#         try 
#             portfolio_scip(seed, dimension; mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end =#


# bo_mode = "ipopt"
# for dimension in [20:5:120;] #[20:5:120;]
#     for seed in 1:10#1:10
#         @show seed, dimension
#         try 
#             portfolio_ipopt(seed, dimension; mode=mode)
#         catch e
#             println(e)
#             open("portfolio_" * mode * "_errors.txt","a") do io
#                 println(io, seed, " ", dimension, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end
