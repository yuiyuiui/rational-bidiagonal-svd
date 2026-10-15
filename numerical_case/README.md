# Reproducible fixed cases

`benchmark.py` runs two deterministic matrices, all ranks, each with `auto`,
explicit `proof` and explicit 4096-bit `diagnostic` configurations. Default
and explicit proof outputs must be identical arrays. Every run requests the
separate exact rational output check; a failed check stops the script.
`results.json` contains the actual settings, reasons, values/carriers, metrics,
validation reports, dependency versions and separated timings. No seed or
input search is used. This is a finite validation set, not an asymptotic study.

```sh
python -m pip install '.[test,figures]'
python numerical_case/benchmark.py --output numerical_case/raw/run.json
python numerical_case/figures.py --input numerical_case/raw/run.json --output numerical_case/raw/figures
```

The reference is NumPy/LAPACK SVD on the same stored matrix. Production uses
no dense decomposition. Exact rational inertia and residual checks provide a
separate check against the original bidiagonal input; their cost is excluded
from production complexity and reported separately. The raw-array comparison
bound is `256*n*2**(-p)*||B||2`, with separate orthogonality bound
`256*n*2**(-p)` and squared-norm-defect bound `8*2**(-p)`. All norms in the
residual figures are Euclidean; matrix norms are spectral.

![Residuals divided by the raw bound](figures/residuals.svg)

Smaller residual ratios are better. The linear vertical axis preserves zero
as zero; the dashed line is the bound 1. The profiles have different general
precision guarantees, even when their exact finite checks all pass.

![Default values and the reference](figures/values.svg)

Values use the units of the matrix entries and descending zero-based ranks.
The zero singular values in the rank-deficient case are displayed as zeros.

![Left and right carriers of the rank-deficient case](figures/directions.svg)

Carrier components show the default output for the three-dimensional singular
case. A repeated zero singular value permits other orthogonal bases. A small
component difference between runs is therefore not by itself an error.

Build/cache inspection, production solve and exact validation times are
reported separately. First compilation and warm cache observations must not
be conflated. Diagnostic timings have a different general accuracy contract;
their shorter times are not a like-for-like speedup theorem. The conservative
default is intended to realize the stated precision bound, not to compete with
LAPACK in practical speed.
