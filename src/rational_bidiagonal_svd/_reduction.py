"""Internal numerical implementation; use the public API."""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import math, json, time
import mpmath as mp
from ._gram import RawGrid, RationalGramOracle
from ._gaps import GapFinder, floor_power_two
from ._budget import allocate_gram, reduction_checks
from ._filter import bounded_sum, mf

def validate(d,e,indices,p):
    d=np.asarray(d,dtype=float);e=np.asarray(e,dtype=float);ix=np.asarray(indices)
    if d.ndim!=1 or len(d)<1 or e.shape!=(len(d)-1,):raise ValueError('invalid bidiagonal shape')
    if not np.isfinite(d).all() or not np.isfinite(e).all():raise ValueError('nonfinite input')
    if ix.ndim!=1 or len(ix)<1 or not np.issubdtype(ix.dtype,np.integer):raise ValueError('integer requested ranks')
    ix=list(map(int,ix))
    if len(set(ix))!=len(ix) or min(ix)<0 or max(ix)>=len(d):raise ValueError('distinct descending zero-based indices')
    if not 16<=p<=53 or len(d)>1<<(p-6):raise ValueError('p=16..53 and n*2^-p<=1/64')
    return d,e,ix

def scale_and_grams(d,e):
    rho=max([abs(float(x)) for x in d]+[abs(float(x)) for x in e])
    if rho==0:return None
    exponent=math.frexp(rho)[1]+1
    scale=F(2)**exponent
    a=[F.from_float(float(x))/scale for x in d];b=[F.from_float(float(x))/scale for x in e]
    n=len(a)
    right=[a[i]*a[i]+(b[i-1]*b[i-1] if i else 0) for i in range(n)]
    roff=[a[i]*b[i] for i in range(n-1)]
    left=[a[i]*a[i]+(b[i]*b[i] if i<n-1 else 0) for i in range(n)]
    loff=[b[i]*a[i+1] for i in range(n-1)]
    return a,b,right,roff,left,loff,exponent

def safe_cut(diag,off,p):
    n=len(diag);eps=F(1,1<<p);D=n*eps;q=3*p+32
    counter=RawGrid([2+F(x) for x in diag],off,q);Q=counter.Q
    lo=int(D*D*Q);hi=2*lo
    h=floor_power_two((hi-lo)//(32*n))
    finder=GapFinder(lambda z:counter.count(z+2*Q)[0],n,counter.beta)
    gap=finder.find(lo,hi,h)
    record={'theta':str(F(gap.center,Q)),'gamma':str(F(h,Q)),'delta':str(D),
      'low_count':gap.count,'positive_count':n-gap.count,'depth':gap.depth,'count_calls':finder.calls,
      'work':counter.work,'events':finder.events,'backward_radius':str(F(counter.beta,Q)),
      'uniform_gamma_lower':str(D*D/(64*n)),'cut_interval':[str(D*D),str(2*D*D)]}
    return F(gap.center,Q),F(h,Q),gap.count,record

def bmul(a,b,x,transpose=False):
    n=len(a)
    return [a[i]*x[i]+((b[i-1]*x[i-1] if i else 0) if transpose else
               (b[i]*x[i+1] if i<n-1 else 0)) for i in range(n)]

def trmul(a,b,x):
    return [a[i]*x[i]+(b[i-1]*x[i-1] if i else 0)+(b[i]*x[i+1] if i<len(a)-1 else 0) for i in range(len(a))]

def norm(x):return mp.sqrt(bounded_sum(y*y for y in x))

def unit(x):
    n=norm(x)
    if not n:raise ArithmeticError('zero direction')
    return [v/n for v in x]

def pack_mp(x):
    s,m,e,_=mp.mpf(x)._mpf_;return [str(-m if s else m),e]

def solve(d,e,indices,directory,*,p=53,oracle=None,diagnostic_bits=None,timeout=900):
    d,e,idx=validate(d,e,indices,p);n=len(d);k=len(idx);start=time.monotonic()
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'input.npz',d=d,e=e,idx=np.array(idx))
    scaled=scale_and_grams(d,e)
    if scaled is None:
        U=np.zeros((n,k));U[idx,np.arange(k)]=1.;V=U.copy();s=np.zeros(k)
        meta={'mode':'exact_zero','n':n,'k':k,'p':p,'proof_precision':True,'seconds':time.monotonic()-start,'full_vector_reorthogonalizations':0}
        np.savez_compressed(folder/'output.npz',d=d,e=e,idx=idx,s=s,U=U,V=V)
        (folder/'summary.json').write_text(json.dumps(meta,indent=2));return s,U,V,meta
    a,b,gr,er,gl,el,sexp=scaled;q=3*p+32;budget=allocate_gram(n,q)
    theta,gamma,m0,cut=safe_cut(gr,er,p)
    (folder/'safe_cut.json').write_text(json.dumps(cut,indent=2))
    near=[j>=n-m0 for j in idx];low_indices=[j for j,z in zip(idx,near) if z]
    if oracle is None:oracle=RationalGramOracle(folder/'gram_calls',diagnostic_bits=diagnostic_bits,timeout=timeout)
    (folder/'progress.json').write_text(json.dumps({'stage':'right Gram','q':q,'near_zero_requested':len(low_indices)}))
    rvalues,rcols,rm=oracle(gr,er,idx,q,'right')
    (folder/'progress.json').write_text(json.dumps({'stage':'left Gram','q':q,'near_zero_requested':len(low_indices)}))
    lvalues,lcols,lm=oracle(gl,el,low_indices,q,'left') if low_indices else ([],[],{'requested':0,'proof_precision':True})
    s=np.empty(k);U=np.empty((n,k));V=np.empty((n,k));checkpoints=[];internal=[]
    with mp.workprec(budget.coefficient_bits):
        aa,bb=list(map(mf,a)),list(map(mf,b));rr,re,ll,le=map(lambda x:list(map(mf,x)),(gr,er,gl,el))
        tdelta=mf(n*F(1,1<<p));lp=0
        for pos,j in enumerate(idx):
            v=unit(rcols[pos]);rv=trmul(rr,re,v)
            eigres_r=norm([x-rvalues[pos]*y for x,y in zip(rv,v)])
            if near[pos]:
                u=unit(lcols[lp]);lval=lvalues[lp];lp+=1;ss=mp.mpf(0)
                lu=trmul(ll,le,u);eigres_l=norm([x-lval*y for x,y in zip(lu,u)])
            else:
                cv=bmul(aa,bb,v);ss=norm(cv)
                if ss<tdelta/2:raise ArithmeticError('positive branch norm lower bound violated')
                u=[x/ss for x in cv];lval=None;eigres_l=None
            cv=bmul(aa,bb,v);ctu=bmul(aa,bb,u,True)
            leftres=norm([x-ss*y for x,y in zip(cv,u)])
            rightres=norm([x-ss*y for x,y in zip(ctu,v)])
            s[pos]=float(mp.ldexp(ss,sexp));U[:,pos]=[float(x) for x in u];V[:,pos]=[float(x) for x in v]
            checkpoints.append({'position':pos,'requested_descending_index':j,'near_zero':near[pos],
               'right_center':mp.nstr(rvalues[pos],85),'left_center':None if lval is None else mp.nstr(lval,85),
               'right_eigen_residual':mp.nstr(eigres_r,40),'left_eigen_residual':None if eigres_l is None else mp.nstr(eigres_l,40),
               'normalized_sigma':mp.nstr(ss,85),'left_SVD_residual':mp.nstr(leftres,40),'right_SVD_residual':mp.nstr(rightres,40)})
            internal.append({'u':[pack_mp(x) for x in u],'v':[pack_mp(x) for x in v],'sigma_scaled':pack_mp(ss)})
    meta={'mode':'general_safe_Gram_reduction','n':n,'k':k,'p':p,'inner_q':q,'scale_exponent':sexp,
      'cut':cut,'requested_near_zero':len(low_indices),'right_oracle':rm,'left_oracle':lm,
      'proof_precision':rm.get('proof_precision',False) and lm.get('proof_precision',False),
      'eigen_oracle_backend':rm.get('mode','unknown'),'checkpoints':checkpoints,'seconds':time.monotonic()-start,
      'full_vector_reorthogonalizations':0,'reduction_scalar_checks':reduction_checks(n,p),
      'scope':'Rational spectral reduction; effective configuration distinguishes the arithmetic-model and diagnostic profiles.'}
    (folder/'internal_vectors.json').write_text(json.dumps(internal))
    (folder/'summary.json').write_text(json.dumps(meta,indent=2));(folder/'progress.json').write_text(json.dumps({'stage':'complete'}))
    np.savez_compressed(folder/'output.npz',d=d,e=e,idx=idx,s=s,U=U,V=V)
    return s,U,V,meta
