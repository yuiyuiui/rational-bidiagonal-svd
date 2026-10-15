# Public API and parameter policy (version 0.1)

```python
selected_svd(d, e, indices, *, target_bits=53, profile="auto",
             kernel_bits=None, verify=False, max_workspace_bytes=2**31,
             timeout=None, cache_dir=None, work_dir=None) -> SVDResult
```

This release curates an existing rational-filter solver into an installed
library. The original solver required a run directory and external precision
build scripts. Version 0.1 adds an input-only default call, one API/CLI resolver,
source-bound backend identification, resource admission, packaging and tests;
the underlying numerical formulas are retained.

## Mathematical inputs

The stored matrix is real upper bidiagonal, with diagonal `d` of length
`n >= 1` and superdiagonal `e` of length `n-1`. Both accept one-dimensional
finite real numeric sequences/arrays exactly representable in binary64.
Integer inputs that change on binary64 conversion, complex arrays, NaN and
infinity are rejected. Float32 inputs embed exactly. The result concerns this
stored matrix; no claim is made about unknown input measurement error.

`indices` is a nonempty sequence of `k` distinct integer ranks in `[0,n)`.
Rank 0 is the largest singular value; columns and values preserve the caller's
order. Index order does not require a contiguous rank interval. Repeated
singular values permit any orthogonal directions in the corresponding subspace.
Dimensions and ranks are problem inputs, not tuning knobs.

All input matrix entries and returned singular values have the same units;
directions, residuals divided by the matrix norm, and orthogonality are
dimensionless. No random seed is used.

## Method parameters

| Parameter | Type, units and supported range | Default and selection | Consequences and recommendation |
|---|---|---|---|
| `target_bits` | Integer `p` from 16 to 53, bits; require `n*2**(-p) <= 1/64` | 53; never changed from the requested value | Defines the requested absolute error scale `epsilon=2**(-p)`. Smaller values explicitly request a looser result and reduce internal precision; do not lower it to hide a failed run. At fixed input format this is a target accuracy, rather than a change to the stored matrix. |
| `profile` | String `auto`, `proof`, or `diagnostic` | `auto`: zero matrix uses exact coordinate vectors; all other inputs use `proof` | Use `auto` for the stated arithmetic-model bound. `proof` makes that choice explicit. `diagnostic` uses an explicitly smaller core budget and has no general theorem guarantee; use it only for experiments or with separately paid instance verification. Zero inputs always return the exact solution. |
| `kernel_bits` | Optional integer real-significand precision, 4096 to `2**22` bits | `None`: proof uses `67584+2880*p`; diagnostic uses 4096 | Proof/auto overrides must be at least the proof budget. Diagnostic overrides must be below it. Larger precision increases arithmetic time and storage without changing the stated error target. Missing compiler/headers or incompatible precision raises an error; no fallback occurs. |
| `verify` | Boolean | `False` | `True` adds exact rational inertia, residual, norm and normalized-inner-product checks for this finite output. Failure raises `ArithmeticError`. Validation may cost more than production and is not part of the linear-output complexity bound. It does not certify all inputs or formally verify the implementation. |
| `max_workspace_bytes` | Positive integer, bytes | `2**31` (2 GiB) | A planning safeguard. Checks output/controller storage before compilation and a conservative estimate before allocating each filter. Exceeding it raises `ResourceLimitError`; it never reduces precision or truncates ranks. Increase only with adequate memory. This is not an OS-enforced memory bound or a claim of exact peak RSS. |
| `timeout` | `None` or positive finite real seconds per C++ kernel call | `None` | No automatic timeout. Set a finite value for an application deadline; expiration raises `subprocess.TimeoutExpired`. Compilation and Python preprocessing are outside this per-call limit. A timeout gives no mathematical output and never changes the profile. |
| `cache_dir` | Optional path | `None`: `${XDG_CACHE_HOME:-~/.cache}/rational-bidiagonal-svd` | Reuses a compiled backend for the same source, compiler command, Boost include path and precision. A cache hit rechecks binary checksum and executable build metadata. Use a writable private directory on a local filesystem with working locks. It changes build cost, not accuracy. |
| `work_dir` | Optional new/empty directory path | `None`: temporary directory, deleted after return/error | A supplied directory retains effective configuration, inputs, outputs, packet decisions and verification data. Choose a new directory for each call; nonempty directories are rejected. These files can be large. Returning carriers and metadata does not require retaining this directory. |

The coefficient precision is `1024+16*q` and the internal Gram target is
`q=3*p+32`. These are determined by the same resolver for API and CLI; they
have no independent override in this version. Integer count grids, the cut and
packet refinement are deterministic. Each packet chooses a rational sign
iteration count by a downward integer recurrence, then rank `2**(J+1)`.
This automatic choice depends on count-certified packet geometry, never on an
assumed or unchecked input gap. The result records the actual `J`, rank and
precision of every nontrivial packet, and the cut/count details.

For a packet with dimension `n`, rank `r` and core precision `b`, admission
uses `2*(ceil(b/8)+32)*(4*r*r+20*n*r+30*n+20*r)` bytes. This intentionally
includes generous array multiplicities. Python objects, allocator overhead,
compiler memory and OS behavior prevent interpreting it as a strict RSS cap.
If it rejects a case, the exception reports the estimated and allowed bytes.

## Returned object and representation

`SVDResult` contains:

- `values`: nonnegative binary64 array of shape `(k,)`;
- `left`, `right`: read-only binary64 carrier arrays of shape `(n,k)`;
- `indices`: immutable requested-rank tuple;
- `config`: effective target, requested/resolved profile, precision requirements,
  actual executable precision/Boost/source hash, parameter reasons and packet settings;
- `diagnostics`: build/cache, production solve and optional validation times in
  seconds, workspace plans and mathematical branch records;
- `verification`: exact-instance report when requested, otherwise `None`.

`left_directions` and `right_directions` are tuples of `UnitDirection` views.
The mathematical meaning of a nonzero carrier `x` is `x/sqrt(x.T@x)` over the
reals. This representation defines an exactly unit mathematical direction;
it does not assert that a stored floating array has exact norm one.
`exact_squared_norm()` explicitly computes a rational sum of squares in
`O(n)` rational operations. `to_float()` performs an approximate binary64
normalization in `O(n)` floating operations. Neither provides a free exact
irrational scalar primitive. Output materialization and representation work
are charged in the method's cost.

For the **proof** profile, under the stated arithmetic model, represented unit
directions have ordered value errors and both residual norms at most
`128*n*2**(-p)*||B||2`, and separate left/right off-diagonal inner products at
most `128*n*2**(-p)`. The raw returned arrays have the conservative constant
256 for those errors, and squared-norm defects at most `8*2**(-p)`.
The model assumes correctly rounded real basic operations and square roots,
protected exact indices and power-of-two scaling, with overflow/underflow
excluded. The software stack is not formally verified.

`verify=True` checks a stronger sufficient bound with `C=128` using
`rho=max(abs(d),abs(e)) <= ||B||2`. It uses exact rational input/output
dyadics, exact tridiagonal inertia for the requested ordered values, sparse
residual sums, norm defects and normalized inner products. The finite report
describes only this instance. Diagnostic results remain labeled diagnostic
after a successful check; `instance_verified` is a distinct field.

## Cost, backends and failures

Production performs no dense eigensolve, full spectrum, or pairwise vector
reorthogonalization. Its precision-explicit bound is `O(p**4*n*k)` arithmetic
operations on `O(p)`-bit numbers and, conservatively, `O(p**6*n*k)` bit work.
It is `O(n*k)` in dimension and requested count only for a fixed format.
Storage is `O(n*k+n*r+r*r)` high-precision words including the output.
At `p=53`, the proof core has **220224 bits**. Practical speed superiority over
LAPACK is not claimed. The small reproducible cases are validation and timing
observations, not a scaling study.

Compilation is a one-time cached setup cost, reported separately. All input
scaling, count/packet selection, coefficient generation, solves, recovery and
output conversion are included in `solve_seconds`. Exact instance validation
is separately reported. `total_seconds` also includes backend inspection/build
and result construction after input checking; input checking itself is `O(n+k)`.

The public core uses a C++17 compiler and Boost `cpp_bin_float` >=1.89 headers;
the tested version is 1.89.0. `CXX` may supply a compiler command (parsed as
arguments, never through a shell); `RBSVD_BOOST_INCLUDE` may identify the
directory containing `boost/`. Defaults are `c++` and the normal compiler
include search path. No compiler flags or approximate numerical backend are
selected automatically. No network download occurs during library calls.

`ValueError` reports input/parameter/domain errors. `BackendUnavailable`
reports missing compiler/Boost or a source/precision/cache-identity mismatch.
`ResourceLimitError` reports a planned workspace excess. `ArithmeticError`
reports a failed numerical guard or requested exact-instance check. File and
process errors propagate with their paths. Correct input does not guarantee
that finite hardware has enough address space, memory or time.

Calls in a process are serialized because mpmath precision is a global context.
The library creates no worker pool. If using multiple processes, the caller
must budget resources; compiler cache writes are protected by an OS file lock.
The current implementation targets POSIX 64-bit hosts (Linux/macOS).

## CLI

```sh
rational-bidiagonal-svd input.json output-directory --verify
rational-bidiagonal-svd input.json diagnostic-directory --profile diagnostic --verify
```

Input JSON has keys `d`, `e`, `indices`. The CLI calls `selected_svd` directly
and exposes the same method options. It prints effective settings and writes
the output arrays/configuration into the requested empty directory.
The installed [tests](../tests/test_api.py) exercise default nonzero input
against an independent dense reference, explicit accuracy/profile choices,
zero/repeated/signed cases, dimensional boundaries, invalid settings,
unavailable backends, resource admission and CLI/API agreement.
