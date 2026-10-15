"""Internal numerical implementation; use the public API."""
from __future__ import annotations
from fractions import Fraction as F
import numpy as np

def ff(x):return F.from_float(float(x))

def inertia_tridiagonal(diag,off,z):
    """Exact 1x1/2x2 symmetric pivots, with zero eigenvalues retained."""
    a=[F(x)-z for x in diag];b=list(map(F,off));neg=zero=pos=0;i=0
    while i<len(a):
        if a[i]:
            if a[i]<0:neg+=1
            else:pos+=1
            if i+1<len(a):a[i+1]-=b[i]*b[i]/a[i]
            i+=1
        elif i+1<len(a) and b[i]:
            neg+=1;pos+=1;i+=2
        else:zero+=1;i+=1
    return neg,zero,pos

def exact_output_certificate(path,C=128,p=53):
    z=np.load(path);d=list(map(ff,z['d']));e=list(map(ff,z['e']));idx=list(map(int,z['idx']))
    ss=list(map(ff,z['s']));U=[[ff(x) for x in z['U'][:,j]] for j in range(len(idx))];V=[[ff(x) for x in z['V'][:,j]] for j in range(len(idx))]
    n=len(d);eps=F(1,1<<p);rho=max(map(abs,d+e),default=F(0));unit_tol=C*n*eps;valtol=unit_tol*rho
    gd=[d[i]**2+(e[i-1]**2 if i else 0) for i in range(n)];go=[d[i]*e[i] for i in range(n-1)]
    records=[];ok=True
    for j,rank in enumerate(idx):
        u,v,s=U[j],V[j],ss[j];un=sum(x*x for x in u);vn=sum(x*x for x in v)
        dn_u=abs(un-1);dn_v=abs(vn-1)
        r1=[d[i]*v[i]+(e[i]*v[i+1] if i<n-1 else 0)-s*u[i] for i in range(n)]
        r2=[d[i]*u[i]+(e[i-1]*u[i-1] if i else 0)-s*v[i] for i in range(n)]
        b1=valtol-2*rho*dn_v-abs(s)*dn_u;b2=valtol-2*rho*dn_u-abs(s)*dn_v
        checks={'nonnegative':s>=0,'norms':dn_u<=8*eps and dn_v<=8*eps,
           'left_residual_after_unitization':b1>=0 and sum(x*x for x in r1)<=b1*b1,
           'right_residual_after_unitization':b2>=0 and sum(x*x for x in r2)<=b2*b2}
        lower=max(F(0),s-valtol);upper=s+valtol
        nl,zl,_=inertia_tridiagonal(gd,go,lower*lower);nu,zu,_=inertia_tridiagonal(gd,go,upper*upper)
        r=n-rank;checks['ordered_value']=nl<r and nu+zu>=r
        ok &= all(checks.values());records.append({'index':rank,'checks':checks,'lower_count':nl,'upper_count_including_equal':nu+zu})
    orth=[]
    for i in range(len(idx)):
        for j in range(i):
            sides=[]
            for Q in (U,V):
                dot=sum(x*y for x,y in zip(Q[i],Q[j]));ni=sum(x*x for x in Q[i]);nj=sum(x*x for x in Q[j])
                sides.append(dot*dot<=unit_tol**2*ni*nj)
            ok &= all(sides);orth.append({'i':i,'j':j,'left':sides[0],'right':sides[1]})
    return {'passed':bool(ok),'C':C,'p':p,'n':n,'k':len(idx),'triplets':records,'orthogonality':orth,
      'method':'exact rational inertia, sparse residuals, and normalized inner products',
      'production_algorithm_cost':False}
