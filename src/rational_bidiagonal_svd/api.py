"""Selected singular values, floating carriers, and represented unit directions."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from contextlib import nullcontext
import json
import math
import numbers
import tempfile
import threading
import time

import numpy as np

from ._budget import allocate_gram
from ._compiler import BackendUnavailable, build_backend
from ._gram import RationalGramOracle
from ._reduction import solve
from ._validation import exact_output_certificate

_LOCK = threading.RLock()


class ResourceLimitError(RuntimeError):
    """A planned allocation exceeds the user's workspace safeguard."""


def _integer(value, name, lower, upper=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f'{name} must be in [{lower}, {upper if upper is not None else "infinity"}]')
    return value


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be a one-dimensional real numeric array')
    out = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f'{name} must contain finite binary64-representable numbers')
    if raw.dtype.kind in 'iu' and any(int(a) != int(b) for a, b in zip(raw, out)):
        raise ValueError(f'{name} contains an integer not exactly representable in binary64')
    if raw.dtype.kind == 'f' and raw.dtype.itemsize > 8 and any(a != b for a, b in zip(raw, out)):
        raise ValueError(f'{name} contains values not exactly representable in binary64')
    return out


def _resolve(d, e, indices, target_bits, profile, kernel_bits, verify,
             max_workspace_bytes, timeout):
    p = _integer(target_bits, 'target_bits', 16, 53)
    d = _real_array(d, 'd'); e = _real_array(e, 'e')
    n = len(d)
    if n < 1 or len(e) != n-1:
        raise ValueError('Require len(d)=n>=1 and len(e)=n-1')
    if n > 1 << (p-6):
        raise ValueError('Require n*2**(-target_bits) <= 1/64; increase target_bits or reduce n')
    raw = np.asarray(indices)
    if raw.ndim != 1 or len(raw) == 0 or raw.dtype.kind not in 'iu':
        raise ValueError('indices must be a nonempty sequence of distinct integer ranks')
    ranks = [int(x) for x in raw]
    if min(ranks) < 0 or max(ranks) >= n or len(set(ranks)) != len(ranks):
        raise ValueError('indices must be distinct, zero based, and less than n; rank 0 is largest')
    if profile not in ('auto', 'proof', 'diagnostic'):
        raise ValueError('profile must be auto, proof, or diagnostic')
    if not isinstance(verify, bool):
        raise ValueError('verify must be a boolean')
    limit = _integer(max_workspace_bytes, 'max_workspace_bytes', 1)
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, numbers.Real)
                                or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout must be None or a positive finite number of seconds per kernel call')
    q = 3*p+32
    budget = allocate_gram(n, q)
    zero = not np.any(d) and not np.any(e)
    resolved = 'exact_zero' if zero else ('proof' if profile == 'auto' else profile)
    if kernel_bits is not None:
        bits = _integer(kernel_bits, 'kernel_bits', 4096, 1 << 22)
    else:
        bits = 4096 if profile == 'diagnostic' else budget.scan_bits
    if profile == 'diagnostic':
        if bits >= budget.scan_bits:
            raise ValueError('diagnostic kernel_bits must be below proof precision; use profile=proof')
    elif bits < budget.scan_bits:
        raise ValueError(f'proof/auto requires kernel_bits >= {budget.scan_bits}; no automatic downgrade')
    # Output and controller storage must be admitted before expensive compilation.
    if 16*n*len(ranks) + 64*n > limit:
        raise ResourceLimitError('Output/controller storage alone exceeds max_workspace_bytes')
    config = dict(target_bits=p, epsilon=f'2^-{p}', requested_profile=profile,
                  resolved_profile=resolved, inner_bits=q, coefficient_bits=budget.coefficient_bits,
                  required_kernel_bits=budget.scan_bits, kernel_bits=0 if zero else bits,
                  verify=verify, max_workspace_bytes=limit, timeout=timeout,
                  index_convention='zero-based descending; return columns in requested order',
                  reasons=[('The matrix is zero; coordinate vectors are exact.' if zero else
                            'auto selects proof precision; diagnostic must be explicitly requested.'),
                           'q=3*p+32 resolves the safe-cut reduction error.',
                           'Kernel bound is 67584+2880*p bits; no input-gap estimate is assumed.'],
                  accuracy_basis='exact_zero' if zero else ('arithmetic_model' if resolved == 'proof' else 'diagnostic'),
                  unit_direction_constant=128, raw_carrier_constant=256,
                  squared_norm_defect_bound=f'8*2^-{p}')
    return d, e, ranks, config


@dataclass(frozen=True)
class UnitDirection:
    """Exact mathematical direction carrier/sqrt(carrier.T@carrier).

    The carrier is a binary64 array. Materialization is an approximation; its
    exact squared norm is available at an explicitly paid O(n) rational cost.
    """
    carrier: np.ndarray

    def exact_squared_norm(self) -> Fraction:
        return sum((Fraction.from_float(float(x))**2 for x in self.carrier), Fraction(0))

    def to_float(self) -> np.ndarray:
        scale = float(np.max(np.abs(self.carrier)))
        if scale == 0:
            raise ArithmeticError('Zero direction')
        a = self.carrier / scale
        return a / np.linalg.norm(a)


@dataclass(frozen=True)
class SVDResult:
    values: np.ndarray
    left: np.ndarray
    right: np.ndarray
    indices: tuple[int, ...]
    config: dict
    diagnostics: dict
    verification: dict | None

    @property
    def left_directions(self) -> tuple[UnitDirection, ...]:
        return tuple(UnitDirection(self.left[:, j]) for j in range(len(self.indices)))

    @property
    def right_directions(self) -> tuple[UnitDirection, ...]:
        return tuple(UnitDirection(self.right[:, j]) for j in range(len(self.indices)))


def selected_svd(d, e, indices, *, target_bits=53, profile='auto', kernel_bits=None,
                 verify=False, max_workspace_bytes=2**31, timeout=None,
                 cache_dir=None, work_dir=None) -> SVDResult:
    """Compute selected triplets of the stored real upper bidiagonal matrix.

    See docs/API.md for all parameters, representation, arithmetic assumptions,
    explicit precision costs, and the optional exact-instance verification cost.
    No dense eigensolver or full spectrum is used in the production path.
    """
    d, e, ranks, config = _resolve(d, e, indices, target_bits, profile, kernel_bits,
                                  verify, max_workspace_bytes, timeout)
    clock = time.monotonic()
    estimates = []

    def admit(n, r, bits):
        # A conservative planning estimate, not an OS-enforced memory bound.
        estimate = 2*((bits+7)//8+32)*(4*r*r+20*n*r+30*n+20*r)
        estimates.append(dict(n=n, rank=r, bits=bits, estimated_bytes=estimate))
        if estimate > config['max_workspace_bytes']:
            raise ResourceLimitError(f'Packet rank {r} plans {estimate} workspace bytes, above '
                                     f'{config["max_workspace_bytes"]}; increase the safeguard explicitly')

    # mpmath's global precision context requires serializing calls in this process.
    with _LOCK:
        backend = None
        if config['resolved_profile'] != 'exact_zero':
            backend = build_backend(config['kernel_bits'], None if cache_dir is None else Path(cache_dir))
        context = tempfile.TemporaryDirectory(prefix='rational-bidiagonal-svd-') if work_dir is None else nullcontext(str(work_dir))
        with context as directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            if any(directory.iterdir()):
                raise ValueError('work_dir must be empty; use a new directory for each call')
            oracle = RationalGramOracle(directory/'gram_calls', diagnostic_bits=config['kernel_bits'] or None,
                                        timeout=timeout, backend=None if backend is None else backend.path, admit=admit)
            values, left, right, details = solve(d, e, ranks, directory, p=target_bits,
                oracle=oracle, diagnostic_bits=config['kernel_bits'] or None, timeout=timeout)
            if (not np.isfinite(values).all() or not np.isfinite(left).all()
                    or not np.isfinite(right).all()):
                raise OverflowError('Binary64 output is nonfinite; overflow/underflow are outside the model')
            validation_start = time.monotonic()
            validation = exact_output_certificate(directory/'output.npz', C=128, p=target_bits) if verify else None
            validation_seconds = time.monotonic()-validation_start if verify else 0.0
            if validation is not None and not validation['passed']:
                raise ArithmeticError('Exact instance validation failed; no certified result is returned')
            config['backend'] = None if backend is None else backend.info
            config['packet_parameters'] = [
                {key: group[key] for key in ('J', 'rank', 'kernel_bits', 'coefficient_bits', 'population') if key in group}
                for side in ('right_oracle', 'left_oracle')
                for group in details.get(side, {}).get('groups', [])]
            diagnostics = dict(engine=details, workspace_plans=estimates,
                               build_seconds=0.0 if backend is None else backend.build_seconds,
                               cache_hit=None if backend is None else backend.cache_hit,
                               solve_seconds=details['seconds'], validation_seconds=validation_seconds,
                               total_seconds=time.monotonic()-clock,
                               instance_verified=validation is not None and validation['passed'])
            if work_dir is not None:
                (directory/'effective_config.json').write_text(json.dumps(config, indent=2)+'\n')
                if validation is not None:
                    (directory/'verification.json').write_text(json.dumps(validation, indent=2)+'\n')
        for array in (values, left, right):
            array.setflags(write=False)
        return SVDResult(values, left, right, tuple(ranks), config, diagnostics, validation)
