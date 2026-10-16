"""Internal numerical implementation; use the public API."""
from __future__ import annotations
from fractions import Fraction as F
from dataclasses import dataclass, asdict
from math import isqrt
from functools import lru_cache

def ceil_log2_int(x: int) -> int:
    if not isinstance(x,int) or x<1: raise ValueError('positive integer required')
    return (x-1).bit_length()

def pow2(k:int)->F:
    return F(1<<k) if k>=0 else F(1,1<<(-k))

@dataclass(frozen=True)
class Budget:
    p:int
    n:int
    delta_design_exponent:int
    coefficient_bits:int
    supply_budget_bits:int
    scan_bits:int
    generator_log2_bound:int
    sign_steps_max:int
    rank_max:int
    packet_refinement_bits:int
    def dictionary(self): return asdict(self)
    @property
    def epsilon(self): return pow2(-self.p)
    @property
    def delta(self): return pow2(-self.delta_design_exponent)

def lower_sign_steps(h:F,b:Budget):
    """Choose J using a directed DOWNWARD dyadic recurrence, no mp backend.

    L/Q <= ell_j is an invariant. Each step uses one exact integer sqrt.
    Absolute local downward defect <3/Q.
    """
    h=F(h)
    if not F(1,512*b.n)<=h<1:raise ValueError('invalid certified mapped gap')
    Q=1<<b.supply_budget_bits
    L=h.numerator*Q//h.denominator
    threshold=Q-(Q>>(b.delta_design_exponent+4)) # 1-delta/16
    trace=[]
    for j in range(1,b.sign_steps_max+1):
        root=isqrt(L*Q)
        L=(2*root*Q)//(Q+L)
        trace.append({'iteration':j,'lower_numerator':str(L),'denominator_exponent':b.supply_budget_bits})
        if L>=threshold:
            return j,trace
    raise AssertionError('proved step bound failed')

def factor_two_trace_bound(n:int,s:int,t:int)->F:
    if not 0<=t<=s<=n:raise ValueError('0<=t<=s<=n')
    q=F(2*s-1,2)
    return n*((q+1)*(q+2)/((q-t+1)*(q-t+2))-1)

def allocate_gram(n:int,q:int)->Budget:
    if not isinstance(n,int) or n<1 or not 16<=q<=244:raise ValueError('n>=1;16<=q<=244')
    if n>(1<<(q-6)):raise ValueError('n*2^-q>1/64')
    J=ceil_log2_int(q+3)+ceil_log2_int(6*q+27)
    ell=224+6*q
    return Budget(q,n,20+6*q,1024+16*q,768+16*q,1024+160*ell,ell,J,1<<(J+1),12*q)

@lru_cache(maxsize=64)
def _check_items(b):
    q,n=b.p,b.n;e=b.epsilon;d=b.delta;B=pow2(b.generator_log2_bound)
    us=pow2(-b.supply_budget_bits);uc=pow2(-b.scan_bits);uco=pow2(-b.coefficient_bits)
    f=4*n**3
    return tuple({
      'domain':n*e<=F(1,64),'J_le_19':b.sign_steps_max<=19,'rank_le_2_20':b.rank_max<=1<<20,
      'real_pole_bound':F(1,4096*3**b.sign_steps_max)>pow2(-44),
      'coefficient_operator':uco*pow2(136)*e**-2<=d/256,
      'gram_operator':us*pow2(104)*e**-2<=d/256,
      'supplied_operator':us*pow2(274)*e**-7<=d/256,
      'small_u_n':us<=F(1,8192*n),'small_u_rank':us<=F(1,1024*b.rank_max),
      'small_u_pole':us<=pow2(-119)*e**2,
      'pole_read_margin':uco*pow2(28)<=pow2(-49)*e,
      'scan_condition':uc<=pow2(-128)/B**160,
      'negative_h_check':pow2(96)*uc*B**128<pow2(-250),
      'column_error':pow2(90)*uc*B**96<=n*e/F(4096),
      'shorting_condition':4*d*f<F(1,2),
      'projected_direction':(128*d)**2*n**9<=(n*e/F(4096))**2,
      'scalar_downward':16*(b.sign_steps_max+1)*us*512*n<d/F(64),
      'text_export_100_digits':F(10)**-99*n <=n*e/F(4096),
      'twice_greedy_bound':factor_two_trace_bound(n,n,n)<=f,
    }.items())

def checks(b):
    """Return detached checks for an immutable budget; cache booleans only."""
    return dict(_check_items(b))

def reduction_checks(n,p):
    eps=pow2(-p);D=n*eps;xi=n*eps**3/pow2(20);g=D*D/(64*n)
    return {'dimension':D<=F(1,64),'oracle_tolerance':512*n*pow2(-(3*p+32))<=xi/8,
      'rank_margin':xi<=g/4,'positive_norm_margin':2*xi<g,
      'right_orth':xi<=D/1024,'positive_left_orth':3*xi/(D*D)<=D/1024,
      'cross_orth':2*xi/g<=D/1024,'positive_residual':xi/D<=D/1024}
