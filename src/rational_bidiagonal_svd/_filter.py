"""Internal numerical implementation; use the public API."""
from __future__ import annotations
from fractions import Fraction as F
import mpmath as mp

def positive_backtrack(scales):
    if not scales: raise ValueError('at least one scale is required')
    alpha=scales[-1]
    pairs=[(1/alpha,2/alpha)]
    for alpha in reversed(scales[:-1]):
        children=[]
        for b,W in pairs:
            s=mp.sqrt(1+b*b)
            low=b/(alpha*(1+s))
            high=(1+s)/(alpha*b)
            # Derivative magnitudes of the two inverse branches.
            wl=W/(alpha*s*(1+s))
            wh=W*(1+s)/(alpha*s*b*b)
            children.extend([(low,wl),(high,wh)])
        pairs=children
    return pairs

def sign_eval(x,design):
    y=mp.mpf(x)
    for a in design['scales']:
        ay=a*y;y=2*ay/(1+ay*ay)
    return y

def mf(x):
    x=F(x);return mp.mpf(x.numerator)/x.denominator

def finite_design(mu,a,g,delta,J):
    mu,a,g,delta=map(mf,(mu,a,g,delta));h=(g-a)/(g+a);initial=h;scales=[]
    for _ in range(J):
        rt=mp.sqrt(h);scales.append(1/rt);h=2*rt/(1+h)
    pairs=positive_backtrack(scales)
    sd={'scales':scales}
    theta=delta/4;scale=1+2*theta;root=mp.sqrt(a*g)
    constant=((1-sign_eval(1,sd))/2+theta)/scale
    poles=[];weights=[]
    for t,W in pairs:
        st=mp.sqrt(1+t*t);xr=root/st;yi=xr*t
        wr=-W*root/(4*scale*st**3);wi=wr*t
        for z,w in ((mp.mpc(mu+xr,yi),mp.mpc(wr,wi)),(mp.mpc(mu-xr,yi),mp.mpc(-wr,wi))):
            poles.extend((z,mp.conj(z)));weights.extend((w,mp.conj(w)))
    return {'constant':constant,'poles':poles,'weights':weights,'rank':len(poles),
            'ell':initial,'positive_floor':theta/scale,'scalar_endpoint_error':1-h}

def bounded_sum(xs):
    # Keep every intermediate rounded to the current fixed precision.
    # An exact/superaccumulator sum can grow with the exponent spread in n.
    total=mp.mpf(0)
    for x in xs: total=total+x
    return total

def emit_real(f,x):
    x=mp.mpf(x);sgn,man,exp,bc=x._mpf_
    f.write(f'{-man if sgn else man} {exp}\n')
