import math
import time
import numpy as np
import bifrost as bf
from bifrost.fdmt import Fdmt
import cupy


def run_test(pulse, f0, f_min, dt, max_dm, exponent, n_iters, batch_shape=()):
    # Make null stream to force synching of all other streams (IDK what FDMT Bifrost is actually doing to launch the kernels)
    cu_stream = cupy.cuda.Stream(null=True)
    cu_stream.use()

    (nchan, ntime) = pulse.shape

    bw        = f0 - f_min
    df        = bw / nchan

    # Calculate max delay from max dm
    rel_delay = (4.148741601e3 / dt * max_dm * (f0**-2 - (f0 + nchan * df)**-2))
    max_delay = int(math.ceil(abs(rel_delay)))

    fdmt = Fdmt()
    fdmt.init(nchan, max_delay, f0, df, exponent, 'cuda')

    ishape = batch_shape + (nchan, ntime)
    oshape = batch_shape + (max_delay, ntime)

    # Random input
    in_data = bf.asarray(np.random.normal(size=ishape)
                        .astype(np.float32), space='cuda')
    # Pre-allocate output
    out_data = bf.asarray(-999 * np.ones(oshape, np.float32),
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

    if max_delay > 1:
        min = out_data.min()
        if min != -999:
            print("Error in output data minimum value!\n")
            exit()
    # TODO: Need better tests
    # max = out_data.max
    # if max >= 100.:
    #     print("Error in odata maximum\n")
    #     exit()
    return (times, out_data)

pulse = np.load("/home/mhawkins/.julia/packages/FastDMTransform/wpJW3/pulse.npz")

n_iters = 100
f_max   = 1500
f_min   = 1200
t_samp  = 1e-3
dm_min  = 0
dm_max  = 2000

(times, out_data) = run_test(pulse,
                            f0=f_max,
                            f_min=f_min,
                            dt=t_samp,
                            max_dm=dm_max,
                            exponent=1.0,
                            n_iters=n_iters)

# Save output for testing
out_filename = "/home/mhawkins/fdmt/fdmt_bifrost_out.npy"
np.save(out_filename,out_data)

# print(f"\n\nBifrost FDMT GPU Dedispersion Benchmark ({n_iters} iterations):\n\
#     Frame size: {pulse.shape}   Max dm: {dm_max}\n\
#     Avg: {np.mean(times)} seconds\n\
#     Min: {np.min(times)}\n\
#     Max: {np.max(times)}")
