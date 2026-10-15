"""Build and identify the fixed precision C++ backend without network access."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time


class BackendUnavailable(RuntimeError):
    """The requested compiler, Boost headers, or compiled backend is unavailable."""


def default_cache() -> Path:
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'rational-bidiagonal-svd'


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).with_name('core.cpp').read_bytes()).hexdigest()


@dataclass(frozen=True)
class Backend:
    path: Path
    info: dict
    build_seconds: float
    cache_hit: bool


def inspect_backend(path: Path, bits: int, expected_hash: str) -> dict:
    try:
        p = subprocess.run([str(path), '--build-info'], text=True, capture_output=True,
                           timeout=10, check=True)
        info = json.loads(p.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        raise BackendUnavailable(f'Cannot identify backend {path}: {e}') from e
    if (info.get('precision_bits') != bits or info.get('source_sha256') != expected_hash
            or info.get('index_bits', 0) < 63 or info.get('boost_version', 0) < 108900):
        raise BackendUnavailable(f'Backend precision/source/platform mismatch: {info}')
    return info


def build_backend(bits: int, cache_dir: Path | None = None) -> Backend:
    """Build once per precision/source/compiler; a failed build never downgrades."""
    start = time.monotonic()
    digest = source_hash()
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler or shutil.which(compiler[0]) is None:
        raise BackendUnavailable('A C++17 compiler is required. Set CXX to its command.')
    boost_include = os.environ.get('RBSVD_BOOST_INCLUDE', '')
    identity = json.dumps([compiler, boost_include, digest, bits], sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    folder = Path(cache_dir) if cache_dir is not None else default_cache()
    folder = folder / key
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / f'driver_{bits}'
    manifest = folder / 'build.json'
    with (folder / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binary.exists() and manifest.exists():
            saved = json.loads(manifest.read_text())
            actual = hashlib.sha256(binary.read_bytes()).hexdigest()
            if saved.get('binary_sha256') != actual:
                raise BackendUnavailable('Cached binary checksum mismatch; remove this cache entry and rebuild.')
            info = inspect_backend(binary, bits, digest)
            return Backend(binary, info, time.monotonic()-start, True)
        with tempfile.TemporaryDirectory(prefix='build-', dir=folder) as tmp:
            pending = Path(tmp) / 'backend'
            command = compiler + ['-O2', '-std=c++17', f'-DWORK_BITS={bits}',
                                  f'-DRBSVD_SOURCE_SHA256="{digest}"']
            if boost_include:
                command += ['-I', boost_include]
            command += [str(Path(__file__).with_name('core.cpp')), '-o', str(pending)]
            try:
                proc = subprocess.run(command, capture_output=True, text=True)
            except OSError as e:
                raise BackendUnavailable(f'Cannot invoke compiler: {e}') from e
            (folder / 'compiler.log').write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                raise BackendUnavailable('Backend compilation failed. Install Boost >=1.89 headers; '
                    'set RBSVD_BOOST_INCLUDE to the directory containing boost/. '
                    f'Compiler output: {folder / "compiler.log"}')
            info = inspect_backend(pending, bits, digest)
            os.replace(pending, binary)
            record = {'source_sha256': digest, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                      'command': command, 'info': info}
            manifest.write_text(json.dumps(record, indent=2) + '\n')
        return Backend(binary, info, time.monotonic()-start, False)
