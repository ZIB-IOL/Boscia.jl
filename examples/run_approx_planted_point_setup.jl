include("approx_planted_point.jl")

mode = ARGS[1]
dimension = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
problem_type = ARGS[4]

@show mode, dimension, seed

try
	approx_planted_point_boscia(seed, dimension, mode = mode, problem_type = problem_type)
catch e
	println(e)
	file = "Boscia_birkhoff_" * mode * "_" * string(seed) * "_" * string(dimension) * problem_type
	open(file * ".txt", "a") do io
		println(io, e)
	end
end
