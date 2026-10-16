"""Bounded, matched benchmarks; run each implementation in a fresh process."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc


def source_hash(source):
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.cpp'):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(source, case, repeats):
    sys.path.insert(0, str(source))
    started = time.perf_counter()
    from core_cases import make
    call, validate, settings = make(case)
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = call()
    cold_seconds = time.perf_counter() - started
    validation = validate(result)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
        if validate(result) != validation:
            raise ArithmeticError('A timed result changed its validated output')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if sys.platform == 'darwin' else rss * 1024)
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    child_rss = int(child_rss if sys.platform == 'darwin' else child_rss * 1024)
    tracemalloc.start()
    allocation_result = call()
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if validate(allocation_result) != validation:
        raise ArithmeticError('Allocation check changed the result')
    profile = cProfile.Profile()
    profile.runcall(call)
    report = io.StringIO()
    pstats.Stats(profile, stream=report).strip_dirs().sort_stats('cumulative').print_stats(14)
    return dict(case=case, settings=settings, validation=validation,
                source_sha256=source_hash(source), setup_seconds=setup_seconds,
                first_call_seconds=cold_seconds, warm_seconds=samples,
                median_seconds=statistics.median(samples), minimum_seconds=min(samples),
                maximum_seconds=max(samples), python_traced_peak_bytes=python_peak,
                fresh_worker_peak_rss_bytes=rss_bytes, largest_child_peak_rss_bytes=child_rss, profile=report.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'src')
    parser.add_argument('--cases', nargs='+', default=['small', 'medium', 'large'])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 3 or args.repeats > 15 or len(args.cases) > 4:
        parser.error('Use 3..15 repeats and at most four bounded cases')
    if args.worker:
        print(json.dumps(worker(args.source.resolve(), args.cases[0], args.repeats)))
        return
    if args.output is None or args.output.exists():
        parser.error('--output must name a new JSON file')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
    implementations = [('candidate', args.source)]
    if args.baseline_source:
        implementations.insert(0, ('baseline', args.baseline_source))
    rows = []
    for index, case in enumerate(args.cases):
        for label, source in (implementations if index % 2 == 0 else implementations[::-1]):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--source', str(source.resolve()), '--cases', case,
                       '--repeats', str(args.repeats)]
            process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
            if process.returncode:
                raise RuntimeError(f'{label}/{case} failed: {process.stderr}')
            row = json.loads(process.stdout)
            row['implementation'] = label
            rows.append(row)
    versions = {}
    for name in ('numpy', 'scipy', 'mpmath', 'sympy', 'python-flint'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    payload = dict(format_version=1, preflight='ASSESSED_RUN', evidence='CLAIM_FACING',
                   placement='LOCAL_EXECUTION; one worker and one BLAS/OpenMP thread',
                   python=platform.python_version(), platform=platform.platform(),
                   machine=platform.machine(), dependencies=versions,
                   timer='perf_counter; validation and profiling excluded from timings',
                   memory='Fresh-worker absolute peak RSS after timing includes imports, inputs, outputs and validation. '
                          'Child RSS is the largest completed subprocess high-water mark, not a simultaneous aggregate. Python traced peak is one separate call and excludes untraced native allocations.',
                   cost='Setup/import, first call and warm public calls are separated. Internal caches may be reused.',
                   results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    for row in rows:
        print(row['implementation'], row['case'], row['median_seconds'], row['fresh_worker_peak_rss_bytes'])


if __name__ == '__main__':
    main()
