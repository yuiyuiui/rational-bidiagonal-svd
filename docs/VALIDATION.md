# Validation and reproduction record

Version 0.1.0 was installed from the source package in two new virtual
environments on 2026-10-15. The final installed package passed all 26 API tests,
and the README's input-only nonzero default call ran at the actual 220224-bit
backend precision. Imports resolved to the installed package, not the source
tree. Tests cover independent NumPy singular values and both equations,
orthogonality, exact finite-output checks, zero/signed/repeated inputs, the
accuracy/dimension boundary, incompatible settings, resource safeguards,
backend absence/identity mismatch and API/CLI agreement.

The [six matched cases](../numerical_case/results.json) all passed their
separate exact rational checks. Default and explicit proof outputs agree as
arrays. Diagnostic outputs are explicitly distinguished from the general
proof-precision contract. The benchmark uses a warm compiler cache; the
`build_seconds` field on a cache hit measures inspection/reuse rather than a
fresh compile. Production solve and exact-check times are separate.

Recorded environment: Python 3.13.3, NumPy 2.5.4, mpmath 1.3.0, pytest 9.1.1,
Matplotlib 3.11.2, Apple clang 17.0.0 (clang-1700.6.4.2), Boost 1.89.0,
macOS 26.5.1 arm64, Apple M4 Pro. Compilation used `-O2 -std=c++17`, with
the precision and source fingerprint supplied explicitly by the library.
BLAS/OpenMP thread limits were 1. Builds/runs used one numerical job at a time, with
a local process-group RSS safeguard of 1.5 GiB. The sampled matched-case peak
was about 109 MB; this is an observation, not a general memory bound.

The tested Boost archive is available from the
[official release archive](https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.bz2),
SHA-256 `85a33fa22621b4f314f8e85e1a5e2a9363d22e4f4992925d4bb3bc631b5a0c7a`.
Boost headers are a separately installed dependency and are not vendored here.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduce.txt
.venv/bin/python -m pip install .
export RBSVD_BOOST_INCLUDE=/absolute/directory/containing/boost
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q
.venv/bin/python numerical_case/benchmark.py --output numerical_case/raw/run.json
.venv/bin/python numerical_case/figures.py --input numerical_case/raw/run.json --output numerical_case/raw/figures
```

The finite observations and tests support this software adoption. They are
not a universal proof, a scaling study, formal verification of the software
stack, independent human review, or a claim of practical speed superiority.
