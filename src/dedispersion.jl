ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

using Pkg
Pkg.activate(".")
using PyCall
using Plots
using Dedisp
using FastDMTransform
using CUDA
using Distributions
using JLD

# Import necessary python packages (will need to have previously setup with Conda(.jl))
pyfdmt = pyimport("pyfdmt")
np     = pyimport("numpy")

# Helper display function
function disp_times(times)
    println("    Avg: $(mean(times)) seconds
    Min: $(reduce(min,times))
    Max: $(reduce(max,times))")
end

default(show=false)

# Test example pulse
pulse = np.load("./dispersed_pulse.npz")

# Examples
f_max   = 1500
f_min   = 1200
t_samp  = 1e-3
dm_min  = 0
dm_max  = 2000

save_plots = true


### pyfdmt

println("\nGenerating pyfdmt Output...")

function bench_pyfdmt(pulse, n_iters)
    f_max   = 1500
	f_min   = 1200
	t_samp  = 1e-3
	dm_min  = 0
	dm_max  = 2000

    pyfdmt.transform(pulse,f_max,f_min,t_samp,0,dm_max)

    # bench = @benchmark fdmt($pulse,$f_max,$f_min,$t_samp,$dm_min,$dm_max)
    # display(bench)

    times = zeros(Float32, n_iters)
    for i in 1:n_iters
        stats = @timed pyfdmt.transform(pulse,f_max,f_min,t_samp,0,dm_max)
        times[i] = stats.time
    end
    disp_times(times)
    return times
end

pyfdmt_out = pyfdmt.transform(transpose(pulse),f_max,f_min,t_samp,0,dm_max)

if save_plots
    pyfdmt_plot = heatmap(pyfdmt_out.data,
                            title="pyfdmt Output",
                            ylabel="DM",
                            xlabel="Time of Arrival (High Freq)",
                            yflip=false)
    savefig(pyfdmt_plot, "./reports/pyfdmt_plot.png")
end

pyfdmt_times = bench_pyfdmt(transpose(pulse), 10)



### FastDMTransform CPU

println("\nGenerating FastDMTransform.jl Output...")

function bench_fdmt(pulse, n_iters)
    f_max   = 1500
	f_min   = 1200
	t_samp  = 1e-3
	dm_min  = 0
	dm_max  = 2000

    fdmt(pulse,f_max,f_min,t_samp,dm_min,dm_max);

    # bench = @benchmark fdmt($pulse,$f_max,$f_min,$t_samp,$dm_min,$dm_max)
    # display(bench)

    times = zeros(Float32, n_iters)
    for i in 1:n_iters
        stats = @timed fdmt(pulse,f_max,f_min,t_samp,dm_min,dm_max)
        times[i] = stats.time
    end
    disp_times(times)
    return times
end

fdmt_jl_out = fdmt(pulse,f_max,f_min,t_samp,dm_min,dm_max)

if save_plots
    fdmt_jl_plot = heatmap(rotl90(fdmt_jl_out)[end:-1:1, :],
                            title="FastDMTransform.jl CPU Output",
                            ylabel="DM",
                            xlabel="Time of Arrival (High Freq)",
                            yflip=false)
    savefig(fdmt_jl_plot, "./reports/fdmt_jl_plot.png")
end

fdmt_times = bench_fdmt(pulse, 10)


### Dedisp CPU

println("\nGenerating Dedisp CPU output...")


function bench_dedisp_cpu(pulse, n_iters)
    plan = plan_dedisp(range(1500,stop=1200,length=4096),1500,range(0,stop=2000,length=2076),1e-3)
    Dedisp.dedisp(pulse,plan)

    # bench = @benchmark Dedisp.dedisp(pulse,plan)
    # display(bench)
    times = zeros(Float32, n_iters)
    for i in 1:n_iters
        stats = @timed Dedisp.dedisp(pulse,plan)
        times[i] = stats.time
    end
    disp_times(times)
    return times
end

plan = plan_dedisp(range(f_max,stop=f_min,length=4096),f_max,range(0,stop=dm_max,length=2076),t_samp)
dedisp_cpu_out = Dedisp.dedisp(pulse,plan)

if save_plots
    dedisp_cpu_plot = heatmap(rotl90(dedisp_cpu_out)[end:-1:1,:],
                            title="Dedisp.jl CPU Output",
                            ylabel="DM",
                            xlabel="Time of Arrival (High Freq)",
                            yflip=false)
    savefig(dedisp_cpu_plot, "./reports/dedisp_cpu_plot.png")
end

dedisp_cpu_times = bench_dedisp_cpu(pulse, 1)



### Dedisp GPU

println("\nGenerating Dedisp GPU Output...")
println("\tUsing $(CUDA.device())")

function bench_dedisp_gpu(pulse, n_iters)
    plan = plan_dedisp(range(1500,stop=1200,length=4096),1500,range(0,stop=2000,length=2076),1e-3)
    output = CUDA.zeros(3000,2076)
	plan_cu = cu(plan)
	pulse_cu = cu(pulse)
    CUDA.@sync dedisp!(output,pulse_cu,plan_cu)

    # bench = @benchmark CUDA.@sync dedisp!($output,$pulse_cu,$plan_cu)
    # display(bench)
    times = zeros(Float32, n_iters)
    for i in 1:n_iters
        stats = @timed CUDA.@sync dedisp!(output,pulse_cu,plan_cu)
        times[i] = stats.time
    end
    disp_times(times)
    return times
end

plan = plan_dedisp(range(f_max,stop=f_min,length=4096),f_max,range(0,stop=dm_max,length=2076),t_samp)
dedisp_gpu_out = CUDA.zeros(3000,2076)
plan_cu = cu(plan)
pulse_cu = cu(pulse)
CUDA.@sync dedisp!(dedisp_gpu_out,pulse_cu,plan_cu)

if save_plots
    dedisp_gpu_plot = heatmap(rotl90(Array(dedisp_gpu_out))[end:-1:1,:],
                            title="Dedisp.jl GPU Output",
                            ylabel="DM",
                            xlabel="Time of Arrival (High Freq)",
                            yflip=false)
    savefig(dedisp_gpu_plot, "./reports/dedisp_gpu_plot.png")
end

dedisp_gpu_times = bench_dedisp_gpu(pulse, 10)

# Generate runtime comparison plot
using StatsPlots

labels = repeat(["FDMT Python", "FDMT Julia", "Dedisp CPU", "Dedisp GPU"], outer = 2)
grouped_times = [mean(pyfdmt_times)     reduce(min,pyfdmt_times);
                 mean(fdmt_times)       reduce(min,fdmt_times);
                 mean(dedisp_cpu_times) reduce(min,dedisp_cpu_times);
                 mean(dedisp_gpu_times) reduce(min,dedisp_gpu_times)]
ctg = repeat(["Mean", "Min"], inner = 4)

if save_plots
    runtime_plot = groupedbar(labels,
                            grouped_times,
                            yaxis=(:log10, (0.01,120)),
                            group=ctg,
                            ylabel="Runtime in Seconds",
                            xlabel="Implementation",
                            title="Dedispersion Runtime Comparison")
    savefig(runtime_plot, "./reports/dedisp_runtimes_plot.png")
end