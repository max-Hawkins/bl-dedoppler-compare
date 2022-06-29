import math
import time
import numpy as np
import bifrost as bf
from bifrost.fdmt import Fdmt
import cupy
import matplotlib.pyplot as plt


def run_test(pulse, f_max, f_min, dt, max_dm, exponent, n_iters, batch_shape=()):
    # Make null stream to force synching of all other streams (IDK what FDMT Bifrost is actually doing to launch the kernels)
    cu_stream = cupy.cuda.Stream(null=True)
    cu_stream.use()

    (nchan, ntime) =  pulse.shape

    bw        = f_max - f_min
    df        = bw / nchan

    # Calculate max delay from max dm
    rel_delay = (4.148741601e3 / dt * max_dm * (f_min**-2 - (f_min + nchan * df)**-2))
    max_delay = int(math.ceil(abs(rel_delay)))

    fdmt = Fdmt()
    fdmt.init(nchan, max_delay, f_min, df, exponent, 'cuda')

    ishape = batch_shape + (nchan, ntime)
    print(ishape)
    oshape = batch_shape + (max_delay, ntime)

    # Random input
    in_data = bf.asarray(pulse, space='cuda')
    # Pre-allocate output
    out_data = bf.asarray(np.zeros(oshape, np.float32),
                        space='cuda')

    # Warmup
    fdmt.execute(in_data, out_data)

    times = np.empty(n_iters, dtype=float)

    for i in range(n_iters):
        tic = time.perf_counter()
        fdmt.execute(in_data, out_data)
        cu_stream.synchronize()
        toc = time.perf_counter()
        times[i] = toc - tic

    out_data = out_data.copy('system')

    return (times, out_data)

# Load the data
# Transpose is necessary because otherwise numpy treats the data as transposed from what we
# want but with column-major storage. Transpose doesn't change actual memory layout but
# rather how numpy treats the data.
pulse = np.transpose(np.load("./dispersed_pulse.npz"))

n_iters = 100
f_max   = 1500
f_min   = 1200
t_samp  = 1e-3
dm_min  = 0
dm_max  = 2000

(times, out_data) = run_test(pulse,
                            f_max=f_min,
                            f_min=f_max,
                            dt=t_samp,
                            max_dm=dm_max,
                            exponent=-2.0,
                            n_iters=n_iters)

np.save("./reports/fdmt_gpu_out.npy", out_data)

# Save times for later displaying
np.save("./reports/fdmt_gpu_times.npy", times)

print(f"\n\nBifrost FDMT GPU Dedispersion Benchmark ({n_iters} iterations):\n\
    Frame size: {pulse.shape}   Max dm: {dm_max}\n\
    Avg: {np.mean(times)} seconds\n\
    Min: {np.min(times)}\n\
    Max: {np.max(times)}")
