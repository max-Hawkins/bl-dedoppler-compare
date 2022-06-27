import numpy as np
import time
import logging

from astropy import units as u
import setigen as stg
import turbo_seti
import matplotlib.pyplot as pyplot
from hyperseti.io import from_setigen
from hyperseti import dedoppler
from hyperseti import find_et
from hyperseti.plotting import imshow_waterfall, imshow_dedopp
from hyperseti import  hitsearch



def bench_hyperseti_find_et(file,config, n_iters):
    times = np.empty(n_iters, dtype=float)

    for i in range(n_iters):
        tic = time.perf_counter()
        dframe = find_et(file, config, gulp_size=2**18, filename_out='/mnt_home/mhawkins/hyperseti_hits.csv')
        toc = time.perf_counter()
        times[i] = toc - tic

    return times

def generate_data(fchans, tchans):
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
    return frame

def bench_hyperseti_dedoppler(frame, boxcar_size, max_dd, n_iters):
    d = from_setigen(frame)
    meta = {'time_step': 2, 'frequency_step':10}
    print(f"\nshape: {d.shape}")
    # Warmup
    dedopp, md = dedoppler(d, boxcar_size=boxcar_size, min_dd=1e-4, max_dd=max_dd)

    times = np.empty(n_iters, dtype=float)

    for i in range(n_iters):
        tic = time.perf_counter()
        dedopp, md = dedoppler(d, boxcar_size=boxcar_size, max_dd=max_dd)
        toc = time.perf_counter()
        times[i] = toc - tic

    return (times, dedopp)

def bench_hyperseti_hitsearch(dedopp, thresh, min_fdist, n_iters):

    #Warmup
    hits = hitsearch(dedopp, threshold=thresh, min_fdistance=min_fdist)

    times = np.empty(n_iters, dtype=float)

    for i in range(n_iters):
        tic = time.perf_counter()
        hits = hitsearch(dedopp, threshold=thresh, min_fdistance=min_fdist)
        toc = time.perf_counter()
        times[i] = toc - tic

    return times



def bench_pipeline():
    config = {
        'preprocess': {
            'sk_flag': False,
            'normalize': False,
        },
        'sk_flag': {
            'n_sigma': 3,
        },
        'dedoppler': {
            'boxcar_mode': 'sum',
            'kernel': 'dedoppler',
            'max_dd': 8.0,
            'min_dd': 0.0001,
            'apply_smearing_corr': False,
            'beam_id': 0
        },
        'hitsearch': {
            'threshold': 3,
            'min_fdistance': 100
        },
        'pipeline': {
            'n_boxcar': 4,
            'merge_boxcar_trials': True,
            'n_blank': 2
        }
    }

    n_iters = 1

    find_et_file = '/datag/lacker/perf/1x77777.h5'
    times_hyperseti_find_et = bench_hyperseti_find_et(find_et_file, config, n_iters)
    print(f"\n\nHyperSETI Find ET (pipeline) Benchmark ({n_iters} iterations):\n\
        File: {find_et_file} \n\
        \nDedoppler:\n\
            Avg: {np.mean(times_hyperseti_find_et)} seconds\n\
            Min: {np.min(times_hyperseti_find_et)}\n\
            Max: {np.max(times_hyperseti_find_et)}")


def microbench():
    n_iters = 1
    fchans = 1048576 #4096
    tchans = 16 #512
    max_dedopp = 8.0
    hyperseti_boxcar_size = 1
    hyperseti_threshold = 100
    hyperseti_min_fdist = 10

    test_frame = generate_data(fchans, tchans)

    times_hyperseti, dedopp   = bench_hyperseti_dedoppler(test_frame, hyperseti_boxcar_size, max_dedopp, n_iters)
    times_hyperseti_hitsearch = bench_hyperseti_hitsearch(dedopp, hyperseti_threshold, hyperseti_min_fdist, n_iters)

    print(f"\n\nHyperSETI Dedoppler Benchmark ({n_iters} iterations):\n\
        Frame size: ({fchans}, {tchans})   Max dedoppler: {max_dedopp}\n\
        \nDedoppler:\n\
            Avg: {np.mean(times_hyperseti)} seconds\n\
            Min: {np.min(times_hyperseti)}\n\
            Max: {np.max(times_hyperseti)}\
        \n\nHitsearch:\n\
            Avg: {np.mean(times_hyperseti_hitsearch)} seconds\n\
            Min: {np.min(times_hyperseti_hitsearch)}\n\
            Max: {np.max(times_hyperseti_hitsearch)}")

microbench()
