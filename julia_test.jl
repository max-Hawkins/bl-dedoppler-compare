### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 86e4ca24-ec07-11ec-015b-43ce95996a23
begin
	using Pkg
	Pkg.activate(".")
	using NPZ
	using BenchmarkTools
	using FastDMTransform
	using Dedisp
	using Plots
	using CUDA
end

# ╔═╡ ec7df75b-bec6-4e26-a321-9eb4781555e3
pulse = npzread("/home/mhawkins/.julia/packages/FastDMTransform/wpJW3/pulse.npz")

# ╔═╡ 5b664451-f9b1-455b-9984-f72485197281
heatmap(pulse)

# ╔═╡ b30d57cf-4b96-45ee-8987-2752ed02c5ff
begin
	f_max   = 1500
	f_min   = 1200
	t_samp  = 1e-3
	dm_min  = 0
	dm_max  = 2000
end

# ╔═╡ 9b4d14d3-b71e-423a-8055-d084a5bd8d26
begin
	fdmt_out = fdmt(pulse,f_max,f_min,t_samp,dm_min,dm_max)
end

# ╔═╡ cb6abf58-67ef-4bb6-8838-50cb88c7eb58
fdmt_plot = heatmap(rotl90(fdmt_out), 
	title="FDMT Output",
	ylabel="DM",
	xlabel="Time of Arrival (High Freq)",
	yflip=true)

# ╔═╡ 64159b6f-f6ab-4513-8384-bf625e77ffce
@benchmark fdmt($pulse,$f_max,$f_min,$t_samp,$dm_min,$dm_max)

# ╔═╡ ac26c90c-179f-4d70-ae96-9965e39aa5f9
plan = plan_dedisp(range(1500,stop=1200,length=4096),1500,range(0,stop=2000,length=2076),1e-3)

# ╔═╡ 78359b02-1965-4567-96be-8c87b2b7007a
@benchmark Dedisp.dedisp($pulse,$plan)

# ╔═╡ 87d3198c-4c4c-45a3-ad04-f8a3c9601257
begin
	output = CUDA.zeros(3000,2076)
	plan_cu = cu(plan)
	pulse_cu = cu(pulse)
end

# ╔═╡ fc076c7b-975e-47a4-83ea-2ea987ac83a2
@benchmark CUDA.@sync dedisp!($output,$pulse_cu,$plan_cu)

# ╔═╡ 441f9746-eed5-4222-be81-94992522f5dc
size(collect(range(1500,stop=1200,length=4096)))

# ╔═╡ Cell order:
# ╠═86e4ca24-ec07-11ec-015b-43ce95996a23
# ╠═ec7df75b-bec6-4e26-a321-9eb4781555e3
# ╠═5b664451-f9b1-455b-9984-f72485197281
# ╠═b30d57cf-4b96-45ee-8987-2752ed02c5ff
# ╠═9b4d14d3-b71e-423a-8055-d084a5bd8d26
# ╠═cb6abf58-67ef-4bb6-8838-50cb88c7eb58
# ╠═64159b6f-f6ab-4513-8384-bf625e77ffce
# ╠═ac26c90c-179f-4d70-ae96-9965e39aa5f9
# ╠═78359b02-1965-4567-96be-8c87b2b7007a
# ╠═87d3198c-4c4c-45a3-ad04-f8a3c9601257
# ╠═fc076c7b-975e-47a4-83ea-2ea987ac83a2
# ╠═441f9746-eed5-4222-be81-94992522f5dc
