modes = [
	"blmo_lazy",
	"mip_lazy",
	"blmo_dicg",
	"blmo_dicg_lazy",
	"blmo_dicg_ws",
	"blmo_dicg_lazy_ws",
	"mip_dicg",
	"mip_dicg_lazy",
	"mip_dicg_ws",
	"mip_dicg_lazy_ws",
]


problem_types = ["integer", "mixed"]




for problem_type in problem_types
	if problem_type == "mixed"
		for mode in modes
			@show mode
			for dimension in 40:10:190
				for seed in 1:10
					@show seed, dimension
					run(`sbatch batch_approx_planted_point.sh $mode $dimension $seed $problem_type`)
				end
			end
		end
	else
		for mode in modes
			@show mode
			for dimension in 55:70
				for seed in 1:10
					@show seed, dimension
					run(`sbatch batch_approx_planted_point.sh $mode $dimension $seed $problem_type`)
				end
			end
		end

	end

end

