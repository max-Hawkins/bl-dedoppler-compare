using NPZ
using BenchmarkTools
using FastDMTransform
using Dedisp
using CUDA

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
    return times
end

function bench_dedisp(pulse, n_iters)
    plan = plan_dedisp(range(1500,stop=1200,length=4096),1500,range(0,stop=2000,length=2076),1e-3)
    Dedisp.dedisp(pulse,plan)

    # bench = @benchmark Dedisp.dedisp(pulse,plan)
    # display(bench)
    times = zeros(Float32, n_iters)
    for i in 1:n_iters
        stats = @timed Dedisp.dedisp(pulse,plan)
        times[i] = stats.time
    end
    return times
end

function bench_dedisp_gpu(pulse, n_iters)
    plan = plan_dedisp(range(1500,stop=1200,length=4096),1500,range(0,stop=2000,length=2076),1e-3)
    output = CUDA.zeros(3000,2076)
	plan_cu = cu(plan)
	pulse_cu = cu(pulse)
    CUDA.@sync dedisp!(output,pulse_cu,plan_cu)

    println("Dedisp GPU Benchmark using $(CUDA.device()):")
    # bench = @benchmark CUDA.@sync dedisp!($output,$pulse_cu,$plan_cu)
    # display(bench)
    times = zeros(Float32, n_iters)
    for i in 1:n_iters
        stats = @timed CUDA.@sync dedisp!(output,pulse_cu,plan_cu)
        times[i] = stats.time
    end
    return times
end

function disp_times(times)
    println("    Avg: $(mean(times)) seconds
    Min: $(reduce(min,times))
    Max: $(reduce(max,times))")
end

pulse = npzread("/home/mhawkins/.julia/packages/FastDMTransform/wpJW3/pulse.npz")
println("Starting Benchmarks...\n")

times_fdmt       = bench_fdmt(pulse, 10)
# times_dedisp_cpu = bench_dedisp(pulse, 1)
times_dedisp_gpu = bench_dedisp_gpu(pulse, 10)

println("\nFastDMTransform Benchmark:")
disp_times(times_fdmt)

println("\nDedisp Benchmark:")
# println("CPU:")
# disp_times(times_dedisp_cpu)
println("GPU:")
disp_times(times_dedisp_gpu)
