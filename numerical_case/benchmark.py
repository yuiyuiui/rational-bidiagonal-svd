"""Fixed, matched cases; no random search and no speed-superiority claim."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import mpmath
import numpy as np
from rational_bidiagonal_svd import selected_svd, __version__


def metrics(d,e,indices,r):
    b=np.diag(d)+np.diag(e,1)
    ref=np.linalg.svd(b,compute_uv=False)[indices]
    rho=np.linalg.norm(b,2)
    n=len(d);eps=2.**-r.config['target_bits']
    ug=r.left.T@r.left;vg=r.right.T@r.right
    off=lambda x:float(np.max(abs(x-np.diag(np.diag(x)))))
    return dict(value_error=float(np.max(abs(r.values-ref))),
                left_residual=float(np.max(np.linalg.norm(b@r.right-r.left*r.values,axis=0))),
                right_residual=float(np.max(np.linalg.norm(b.T@r.left-r.right*r.values,axis=0))),
                left_orthogonality=off(ug),right_orthogonality=off(vg),
                squared_norm_defect=float(max(np.max(abs(np.diag(ug)-1)),np.max(abs(np.diag(vg)-1)))),
                residual_bound_raw=256*n*eps*rho,orthogonality_bound_raw=256*n*eps,
                norm_defect_bound=8*eps,
                reference='NumPy/LAPACK SVD for finite comparison; exact rational certificate separately supplied')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    cases=[('coupled2',[2.,1.],[.25],[0,1]),
           ('rank_deficient3',[1.,0.,0.],[.125,0.],[0,1,2])]
    records=[]
    for name,d,e,indices in cases:
        previous=None
        for profile in ('auto','proof','diagnostic'):
            r=selected_svd(d,e,indices,profile=profile,verify=True)
            same=None if previous is None else bool(np.array_equal(r.values,previous.values)
                and np.array_equal(r.left,previous.left) and np.array_equal(r.right,previous.right))
            if profile=='proof':assert same,'auto and explicit proof must reproduce exact output arrays'
            if profile=='auto':previous=r
            rec=dict(case=name,d=d,e=e,indices=indices,profile=profile,config=r.config,
                     timings={k:r.diagnostics[k] for k in ('build_seconds','cache_hit','solve_seconds','validation_seconds','total_seconds')},
                     validation=r.verification,metrics=metrics(d,e,indices,r),
                     values=r.values.tolist(),left=r.left.tolist(),right=r.right.tolist(),
                     same_arrays_as_auto=same,workspace_plans=r.diagnostics['workspace_plans'])
            records.append(rec)
            data=dict(package_version=__version__,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      environment=dict(python=sys.version.split()[0],numpy=np.__version__,mpmath=mpmath.__version__,
                                       platform=platform.platform(),machine=platform.machine()),
                      seed=None,randomness='none',records=records,
                      scope='Fixed small cases; timings are observations, not asymptotic or superiority evidence.')
            args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'case':name,'profile':profile,'seconds':r.diagnostics['solve_seconds'],
                              'exact_check':r.verification['passed']}),flush=True)


if __name__=='__main__':main()
