# bl-dedoppler-compare

This is a collection and summary of benchmarks and comparisons of dedoppler/dedispersion
algorithms.

## Algorithms Benchmarked

DeDoppler Pipelines:
- turboSETI
- hyperSETI
- SETIcore

DeDispersion Algorithms:
- CPU Fast Dispersion Measure Transform (FDMT) in Python (pyfdmt)
- GPU FDMT in Python (from Bifrost)
- CPU FDMT in Julia (FastDMTransform.jl)
- CPU Brute-Force Dedispersion in Julia (Dedisp.jl)
- GPU Brute-Force Dedispersion in Julia (Dedisp.jl)
- GPU Brute-Force Dedispersion in C/C++ (Dedisp library)
- GPU Fourier Domain Dedispersion in C/C++ (Dedisp library)

## Summary:

TODO


## Methodology

For more information, look at the comprehensive benchmark results document (to be released
later).

### turboSETI

We used the installed turbsoSETI command line utility on the public Breakthrough Listen
servers.

### hyperSETI

Codebase: https://github.com/UCBerkeleySETI/hyperseti \
Specific Python package versions will be found in the comprehensive document.

### SETIcore

We used the installed seticore command line utility on the public Breakthrough Listen
 servers.

### CPU Fast Dispersion Measure Transform (FDMT) in Python (pyfdmt)

Codebase: https://bitbucket.org/vmorello/pyfdmt/src/master/

### GPU FDMT in Python (from Bifrost)

Codebase: https://github.com/ledatelescope/bifrost

### CPU FDMT in Julia (FastDMTransform.jl)

Codebase: https://github.com/max-Hawkins/FastDMTransform.jl
Note: This fork is used to test against a single-threaded FDMT implementation that's a more
fair comparison to pyfdmt.

### CPU Brute-Force Dedispersion in Julia (Dedisp.jl)

Codebase: https://github.com/max-Hawkins/Dedisp.jl
Note: This fork is used to remove the loop vectorization helper to make a more fair comparison.

### GPU Brute-Force Dedispersion in Julia (Dedisp.jl)

Codebase: https://github.com/max-Hawkins/Dedisp.jl
Note: This fork is used to remove the normalization step after dedispersion.

### GPU Brute-Force Dedispersion in C/C++ (Dedisp library)

Codebase: https://github.com/kiranshila/dedisp

### GPU Fourier Domain Dedispersion in C/C++ (Dedisp library)

Codebase: https://github.com/kiranshila/dedisp
