modes = [
	"custom",
	"custom_lazy",
	"mip_lazy",
	"mip",
	"custom_dicg",
	"custom_dicg_lazy",
	"custom_dicg_ws",
	"custom_dicg_lazy_ws",
	"mip_dicg",
	"mip_dicg_lazy",
	"mip_dicg_ws",
	"mip_dicg_lazy_ws",
]




for mode in modes
	@show mode
	for dimension in 3:15
		for seed in 1:10
			@show seed, dimension
			run(`sbatch batch_birkhoff.sh $mode $dimension $seed`)
		end
	end
end

