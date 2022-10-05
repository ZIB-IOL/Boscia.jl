function test()
    seed, dimension, iter = parse.(Int,ARGS)
    @show seed, dimension, iter
end

test()