"""Diagnostic and default-proof cases checked against NumPy and exact certificates."""
import hashlib
import numpy as np
from rational_bidiagonal_svd import selected_svd


def make(case):
    if case == 'medium':
        diagonal, off = [1., 0., 0.], [.25, 0.]
    else:
        diagonal, off = [2., 1.], [.25]
    profile = 'auto' if case == 'large' else 'diagnostic'
    n = len(diagonal)
    matrix = np.diag(diagonal) + np.diag(off, 1)
    reference = np.linalg.svd(matrix, compute_uv=False)
    tolerance = 256 * n * 2.**-53 * np.linalg.norm(matrix, 2)
    settings = {'n': n, 'profile': profile, 'target_bits': 53, 'verify': True}

    def call():
        result = selected_svd(diagonal, off, list(range(n)), profile=profile, verify=True)
        if 'first_build_seconds' not in settings:
            settings.update(first_build_seconds=result.diagnostics['build_seconds'],
                            first_backend_cache_hit=result.diagnostics['cache_hit'],
                            kernel_bits=result.config['kernel_bits'])
        return result

    def validate(result):
        right = np.linalg.norm(matrix @ result.right - result.left * result.values)
        left = np.linalg.norm(matrix.T @ result.left - result.right * result.values)
        if not result.verification['passed'] or max(right, left) > tolerance:
            raise ArithmeticError('Exact certificate or independent residual check failed')
        if np.max(np.abs(result.values - reference)) > tolerance:
            raise ArithmeticError('Independent NumPy singular values disagree')
        digest = hashlib.sha256()
        for a in (result.values, result.left, result.right):
            digest.update(a.tobytes())
        return {'passed': True, 'exact_instance_certificate': True,
                'array_sha256': digest.hexdigest(), 'maximum_residual': max(right, left)}

    return call, validate, settings
