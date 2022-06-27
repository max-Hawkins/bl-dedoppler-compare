import pyfdmt
import numpy as np
import time

def bench_pyfdmt(data, f_max, f_min, t_samp, dm_min, dm_max, n_iters):
    times = np.empty(n_iters, dtype=float)

    # Warmup
    out = pyfdmt.transform(data, f_max, f_min, t_samp, dm_min, dm_max)

    for i in range(n_iters):
        tic = time.perf_counter()
        out = pyfdmt.transform(data, f_max, f_min, t_samp, dm_min, dm_max)
        toc = time.perf_counter()
        times[i] = toc - tic

    return (times, out.data)

n_iters = 10
f_max   = 1500
f_min   = 1200
t_samp  = 1e-3
dm_min  = 0
dm_max  = 2000

pulse = np.load("/home/mhawkins/.julia/packages/FastDMTransform/wpJW3/pulse.npz")

(times_pyfdmt, out) = bench_pyfdmt(pulse, f_max, f_min, t_samp, dm_min, dm_max, n_iters)

# Save output data to verify
out_filename = "/home/mhawkins/fdmt/pyfdmt_out.npy"
np.save(out_filename,out)

print(f"\n\nPyFDMT Benchmark ({n_iters} iterations):\n\
    Frame size: ({pulse.shape})   Max dedoppler: {dm_max}\n\
    \nCPU:\n\
        Avg: {np.mean(times_pyfdmt)} seconds\n\
        Min: {np.min(times_pyfdmt)}\n\
        Max: {np.max(times_pyfdmt)}")