include("mip-examples.jl")

example="22433"

bo_mode="boscia"
for num_v in [4:1:6;]
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

# bo_mode = "as"
# for num_v in [4:1:6;]
#     for seed in 1:3
#         @show seed, num_v
#         try 
#             mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
#         catch e
#             println(e)
#             open("mip_lib_" * example * "_errors.txt","a") do io
#                 println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

# bo_mode = "ss"
# for num_v in [4:1:6;]
#     for seed in 1:3
#         @show seed, num_v
#         try 
#             mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
#         catch e
#             println(e)
#             open("mip_lib_" * example * "_errors.txt","a") do io
#                 println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

# bo_mode = "as_ss"
# for num_v in [4:1:6;]
#     for seed in 1:3
#         @show seed, num_v
#         try 
#             mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
#         catch e
#             println(e)
#             open("mip_lib_" * example * "_errors.txt","a") do io
#                 println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

# bo_mode = "afw"
# for num_v in [4:1:6;]
#     for seed in 1:3
#         @show seed, num_v
#         try 
#             mip_lib(seed, num_v; example=example, bo_mode=bo_mode)
#         catch e
#             println(e)
#             open("mip_lib_" * example * "_errors.txt","a") do io
#                 println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end

# bo_mode = "scip_oa"
# for num_v in [4:1:6;]
#     for seed in 1:3
#         @show seed, num_v
#         try 
#             mip_lib_scip(seed, num_v; example=example)
#         catch e
#             println(e)
#             open("mip_lib_" * example * "_errors.txt","a") do io
#                 println(io, seed, " ", num_v, " ", bo_mode, " : ", e)
#             end
#         end
#     end
# end
