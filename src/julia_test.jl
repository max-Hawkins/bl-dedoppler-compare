### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 86e4ca24-ec07-11ec-015b-43ce95996a23
begin
	using Pkg
	Pkg.activate("..")
	using NPZ
	using BenchmarkTools
	using FastDMTransform
	using Dedisp
	using Plots
	using CUDA
	using Test
end

# ╔═╡ ec7df75b-bec6-4e26-a321-9eb4781555e3
pulse = npzread("../dispersed_pulse.npz");

# ╔═╡ 5b664451-f9b1-455b-9984-f72485197281
heatmap(rotl90(pulse),
	title="Dispersed Pulse",
	xlabel="Time",
	ylabel="Frequency")

# ╔═╡ b30d57cf-4b96-45ee-8987-2752ed02c5ff
begin
	f_max   = 1500
	f_min   = 1200
	t_samp  = 1e-3
	dm_min  = 0
	dm_max  = 2000;
end

# ╔═╡ 88a4bc03-073a-49a9-9e3f-0838c5eff43f
md"""
# FastDMTransform.jl
"""

# ╔═╡ 9b4d14d3-b71e-423a-8055-d084a5bd8d26
fdmt_out = fdmt(pulse,f_max,f_min,t_samp,dm_min,dm_max);

# ╔═╡ cb6abf58-67ef-4bb6-8838-50cb88c7eb58
heatmap(rotl90(fdmt_out), 
	title="FastDMTransform.jl Output",
	ylabel="DM",
	xlabel="Time of Arrival (High Freq)",
	yflip=true)

# ╔═╡ b7970161-36aa-4214-b8bd-d2ee585d4b5f
md"""
## Benchmarking results:
"""

# ╔═╡ 64159b6f-f6ab-4513-8384-bf625e77ffce
# The `$` symbol interpolates the variable for more accurate benchmarking
@benchmark fdmt($pulse,$f_max,$f_min,$t_samp,$dm_min,$dm_max)

# ╔═╡ adc33310-882d-4e4b-9b3d-dd222c6551b0
md"""
# Dedisp.jl - Brute Force Dedispersion (CPU)
"""

# ╔═╡ b4d62380-2a8b-4f01-b5d3-1416ea80d39e
begin
	plan = plan_dedisp(range(1500,stop=1200,length=4096),1500,range(0,stop=2000,length=2076),1e-3);
	dedisp_cpu_out = Dedisp.dedisp(pulse,plan);
	nothing
end

# ╔═╡ 5129ecc1-204d-4d15-a5ac-c1d3ded296ab
heatmap(rotl90(Array(dedisp_cpu_out)), 
	title="Dedisp.jl CPU Output",
	ylabel="DM",
	xlabel="Time of Arrival (High Freq)",
	yflip=true)

# ╔═╡ 121b1e5f-9d64-4df8-8d07-962a2fa36a2e
md"""
 ## Benchmark results:
"""

# ╔═╡ 78359b02-1965-4567-96be-8c87b2b7007a
@benchmark Dedisp.dedisp($pulse,$plan)

# ╔═╡ 876d6927-df24-42f8-8e33-699d63c8a81f
md"""
# Dedisp.jl - Brute Force Dedipsersion (GPU)
"""

# ╔═╡ 87d3198c-4c4c-45a3-ad04-f8a3c9601257
begin
	dedisp_gpu_out = CUDA.zeros(3000,2076)
	plan_cu = cu(plan)
	pulse_cu = cu(pulse)
	dedisp!(dedisp_gpu_out, pulse_cu, plan_cu)
	nothing
end

# ╔═╡ 3de98352-7fc1-4a64-bbc7-99ff5a38ec42
heatmap(rotl90(Array(dedisp_gpu_out)), 
	title="Dedisp.jl GPU Output",
	ylabel="DM",
	xlabel="Time of Arrival (High Freq)",
	yflip=true,
	colorbar=false)

# ╔═╡ 678b01e8-bf8b-4358-b777-d66f8e376482
md"""
## Benchmark results using $(CUDA.device()):
"""

# ╔═╡ fc076c7b-975e-47a4-83ea-2ea987ac83a2
@benchmark CUDA.@sync dedisp!($dedisp_gpu_out,$pulse_cu,$plan_cu)

# ╔═╡ 441f9746-eed5-4222-be81-94992522f5dc
md"""
### Now let's verify that the outputs of these methods all numerically agree.
"""

# ╔═╡ 502ae924-47f3-4383-a497-bd71be71f254
@test Array(dedisp_gpu_out) ≈ dedisp_cpu_out

# ╔═╡ 7bfeec4a-d704-4606-9c1e-82984de35d05
@test fdmt_out ≈ dedisp_cpu_out

# ╔═╡ 6105bef3-3374-4935-aea4-6a79b09dabd0
@test Array(dedisp_gpu_out) ≈ dedisp_cpu_out

# ╔═╡ 43031f61-5fa2-4a14-95e5-55b4125d4bb2
reduce(max, fdmt_out)

# ╔═╡ 925e898e-97c3-4d6c-a82c-5038b505d6c3
heatmap(abs.(dedisp_cpu_out ./ reduce(max, dedisp_cpu_out) .- (Array(dedisp_gpu_out) ./reduce(max, dedisp_gpu_out))))

# ╔═╡ 7a6e9334-cddd-4ea6-9739-935089294912
dedisp_cpu_out ./ reduce(max, dedisp_cpu_out)

# ╔═╡ 7ab99351-0039-4451-b00f-e16eb4ce81ac
dedisp_gpu_out ./reduce(max, dedisp_gpu_out)

# ╔═╡ 64c672a0-e118-4649-b8ad-abc196ff3e5b


# ╔═╡ 8b9927e5-aac1-473a-af87-473c832d9a8b
fdmt_out

# ╔═╡ bb9a827f-8bf4-4b59-a6ea-03f18a8fe24a
dedisp_cpu_out

# ╔═╡ 04f18251-f565-4e46-8813-5d2f586660c4
dedisp_gpu_out

# ╔═╡ 6a59848c-3dde-4f9c-81f9-62b096bf4718


# ╔═╡ Cell order:
# ╠═86e4ca24-ec07-11ec-015b-43ce95996a23
# ╠═ec7df75b-bec6-4e26-a321-9eb4781555e3
# ╟─5b664451-f9b1-455b-9984-f72485197281
# ╠═b30d57cf-4b96-45ee-8987-2752ed02c5ff
# ╟─88a4bc03-073a-49a9-9e3f-0838c5eff43f
# ╠═9b4d14d3-b71e-423a-8055-d084a5bd8d26
# ╠═cb6abf58-67ef-4bb6-8838-50cb88c7eb58
# ╟─b7970161-36aa-4214-b8bd-d2ee585d4b5f
# ╠═64159b6f-f6ab-4513-8384-bf625e77ffce
# ╟─adc33310-882d-4e4b-9b3d-dd222c6551b0
# ╠═b4d62380-2a8b-4f01-b5d3-1416ea80d39e
# ╟─5129ecc1-204d-4d15-a5ac-c1d3ded296ab
# ╟─121b1e5f-9d64-4df8-8d07-962a2fa36a2e
# ╠═78359b02-1965-4567-96be-8c87b2b7007a
# ╟─876d6927-df24-42f8-8e33-699d63c8a81f
# ╠═87d3198c-4c4c-45a3-ad04-f8a3c9601257
# ╠═3de98352-7fc1-4a64-bbc7-99ff5a38ec42
# ╟─678b01e8-bf8b-4358-b777-d66f8e376482
# ╠═fc076c7b-975e-47a4-83ea-2ea987ac83a2
# ╟─441f9746-eed5-4222-be81-94992522f5dc
# ╠═502ae924-47f3-4383-a497-bd71be71f254
# ╠═7bfeec4a-d704-4606-9c1e-82984de35d05
# ╠═6105bef3-3374-4935-aea4-6a79b09dabd0
# ╠═43031f61-5fa2-4a14-95e5-55b4125d4bb2
# ╠═925e898e-97c3-4d6c-a82c-5038b505d6c3
# ╠═7a6e9334-cddd-4ea6-9739-935089294912
# ╠═7ab99351-0039-4451-b00f-e16eb4ce81ac
# ╠═64c672a0-e118-4649-b8ad-abc196ff3e5b
# ╠═8b9927e5-aac1-473a-af87-473c832d9a8b
# ╠═bb9a827f-8bf4-4b59-a6ea-03f18a8fe24a
# ╠═04f18251-f565-4e46-8813-5d2f586660c4
# ╠═6a59848c-3dde-4f9c-81f9-62b096bf4718
