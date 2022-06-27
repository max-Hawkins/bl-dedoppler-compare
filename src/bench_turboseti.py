import numpy as np
import time
import logging

from astropy import units as u
import setigen as stg
import turbo_seti
import matplotlib.pyplot as pyplot

def generate_data(fchans, tchans, filename):
    # Create data using setigen
    frame = stg.Frame(fchans=fchans*u.pixel,
                    tchans=tchans*u.pixel,
                    df=2*u.Hz,
                    dt=10*u.s,
                    fch1=1420*u.MHz)
    noise = frame.add_noise(x_mean=10, noise_type='chi2')
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=1000),
                                                drift_rate=2*u.Hz/u.s),
                            stg.constant_t_profile(level=frame.get_intensity(snr=50)),
                            stg.gaussian_f_profile(width=10*u.Hz),
                            stg.constant_bp_profile(level=1))
    # Make data 32 bit floating point
    frame.data = frame.data.astype('float32')
    frame.save_hdf5(filename=filename)
    return frame

def bench_turboseti_dedoppler_cpu(data_file, fchans, tchans, max_dd, n_iters):

    times = np.empty(n_iters, dtype=float)

    # Warmup
    turbo_seti.find_doppler.find_doppler.FindDoppler(data_file, max_drift=max_dd, min_drift=1e-04, snr=25.0, out_dir='./', coarse_chans=None, obs_info=None, flagging=False, n_coarse_chan=1, kernels=None, gpu_backend=False, gpu_id=0, precision=1, append_output=False, log_level_int=20, blank_dc=True)

    for i in range(n_iters):
        tic = time.perf_counter()
        turbo_seti.find_doppler.find_doppler.FindDoppler(data_file, max_drift=max_dd, min_drift=1e-04, snr=25.0, out_dir='./', coarse_chans=None, obs_info=None, flagging=False, n_coarse_chan=1, kernels=None, gpu_backend=False, gpu_id=0, precision=1, append_output=False, log_level_int=logging.DEBUG, blank_dc=True)
        toc = time.perf_counter()
        times[i] = toc - tic

    return times

def bench_turboseti_dedoppler_gpu(data_file, fchans, tchans, max_dd, n_iters):

    times = np.empty(n_iters, dtype=float)

    kernel = turbo_seti.find_doppler.kernels.Kernels(gpu_backend=True, precision=1, gpu_id=0)

    # Warmup
    turbo_seti.find_doppler.FindDoppler(data_file, max_drift=max_dd, min_drift=1e-04, snr=25.0, out_dir='./', coarse_chans=None, obs_info=None, flagging=False, n_coarse_chan=1, kernels=kernel, gpu_backend=True, gpu_id=0, precision=1, append_output=False, log_level_int=20, blank_dc=True)

    for i in range(n_iters):
        tic = time.perf_counter()
        turbo_seti.find_doppler.FindDoppler(data_file, max_drift=max_dd, min_drift=1e-04, snr=25.0, out_dir='./', coarse_chans=None, obs_info=None, flagging=False, n_coarse_chan=1, kernels=kernel, gpu_backend=True, gpu_id=0, precision=1, append_output=False, log_level_int=20, blank_dc=True)
        toc = time.perf_counter()
        times[i] = toc - tic

    return times


filename = 'turboseti_bench_frame.h5'
fchans = 1048576 #4096
tchans = 16 #512
max_dedopp = 8.0
n_iters = 100
hyperseti_boxcar_size = 1
hyperseti_threshold = 100
hyperseti_min_fdist = 10

# test_frame = generate_data(fchans, tchans, filename)

times_turboseti_cpu = bench_turboseti_dedoppler_cpu(filename, fchans, tchans, max_dedopp, n_iters)
times_turboseti_gpu = bench_turboseti_dedoppler_gpu(filename, fchans, tchans, max_dedopp, n_iters)

print(f"\n\nTurboSeti Dedoppler Benchmark ({n_iters} iterations):\n\
    Frame size: ({fchans}, {tchans})   Max dedoppler: {max_dedopp}\n\
    \nCPU:\n\
        Avg: {np.mean(times_turboseti_cpu)} seconds\n\
        Min: {np.min(times_turboseti_cpu)}\n\
        Max: {np.max(times_turboseti_cpu)}\
    \n\nGPU:\n\
        Avg: {np.mean(times_turboseti_gpu)} seconds\n\
        Min: {np.min(times_turboseti_gpu)}\n\
        Max: {np.max(times_turboseti_gpu)}")
