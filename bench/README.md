# Paired core benchmark

Baseline `627f07a898376e12ddba1e40d1d84c0610af639a` and candidate source hashes are recorded in
[raw results](results/core_v0.1.1.json), including all samples, validation,
settings, dependency versions and profiles. macOS arm64, CPython 3.13.3, one
worker and one BLAS/OpenMP thread. 3 warm repetitions per case.

| Case | Baseline median s | Candidate median s | Ratio | Python peak B (old/new) | Process RSS B (old/new) |
|---|---:|---:|---:|---:|---:|
| small | 0.0993505 | 0.0554015 | 1.79x | 3807074 / 3770016 | 45383680 / 44449792 |
| medium | 0.187535 | 0.119523 | 1.57x | 5700108 / 5663119 | 47955968 / 46907392 |
| large | 21.5331 | 18.411 | 1.17x | 3807029 / 3770022 | 45252608 / 47054848 |

Reuse exact spectral brackets and cache immutable budget predicates. Store diagnostic extrema squared until export. Default proof precision remains 220224 bits; diagnostic runs use 4096 bits and do not inherit the general proof guarantee. The native build cache was already warm: compilation is excluded. No superiority over LAPACK is claimed.

Setup/import, first call, warm calls, separate allocation and separate profiling
passes are distinguished. Validation is outside timed regions; process RSS
includes imports, input and validation. Python tracemalloc excludes untraced
native allocations. Native subprocess high-water RSS is separately reported in the JSON; it is the largest child, not a simultaneous aggregate. These finite cases do not prove an asymptotic or universal speed/memory gain.
Timing noise and hardware matter; small RSS changes are not interpreted as gains.
The remaining arithmetic/BLAS kernels dominate larger workloads.

```sh
git worktree add --detach ../rational-bidiagonal-svd-baseline 627f07a898376e12ddba1e40d1d84c0610af639a
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/benchmark_core.py \
  --repeats 3 --baseline-source ../rational-bidiagonal-svd-baseline/src \
  --output /tmp/rational-bidiagonal-svd-core.json
```

The output path must be new. Install the package's development dependencies first;
SVD also needs Boost >=1.89 and C++17. Each variant imports the specified source
in a fresh worker. Performance thresholds are not enforced on shared CI.
