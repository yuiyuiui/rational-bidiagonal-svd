"""CLI adapter; all method selection remains in selected_svd."""
import argparse
import json
from pathlib import Path
import numpy as np
from .api import selected_svd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path, help='JSON with d, e, indices')
    ap.add_argument('output', type=Path, help='new/empty result directory')
    ap.add_argument('--target-bits', type=int, default=53)
    ap.add_argument('--profile', choices=['auto', 'proof', 'diagnostic'], default='auto')
    ap.add_argument('--kernel-bits', type=int)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--max-workspace-bytes', type=int, default=2**31)
    ap.add_argument('--timeout', type=float)
    ap.add_argument('--cache-dir', type=Path)
    args = ap.parse_args(argv)
    data = json.loads(args.input.read_text())
    try:
        r = selected_svd(data['d'], data['e'], data['indices'], target_bits=args.target_bits,
             profile=args.profile, kernel_bits=args.kernel_bits, verify=args.verify,
             max_workspace_bytes=args.max_workspace_bytes, timeout=args.timeout,
             cache_dir=args.cache_dir, work_dir=args.output)
    except (ValueError, RuntimeError, ArithmeticError, OSError) as e:
        ap.exit(2, f'{type(e).__name__}: {e}\n')
    print(json.dumps({'values': r.values.tolist(), 'config': r.config,
                      'instance_verified': r.diagnostics['instance_verified']}, indent=2))


if __name__ == '__main__':
    main()
