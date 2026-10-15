"""Internal numerical implementation; use the public API."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable

@dataclass(frozen=True)
class Gap:
    center: int
    halfwidth: int
    count: int
    depth: int
    calls: int

def floor_power_two(x: int) -> int:
    if x < 1:
        raise ValueError('a positive grid quantity is required')
    return 1 << (x.bit_length()-1)

class GapFinder:
    """find() is proved to succeed if the stated soft-count contract holds.

    An exception in a supplied oracle is deliberately NOT swallowed.  The
    supplied count oracle must satisfy its stated soft-count contract.
    """
    def __init__(self, count: Callable[[int], int], dimension: int, beta: int=0):
        if dimension < 1 or beta < 0:
            raise ValueError('invalid dimension/backward radius')
        self.oracle=count; self.dimension=dimension; self.beta=beta
        self.cache: dict[int,int]={}; self.calls=0; self.events: list[dict]=[]

    def query(self, z: int) -> int:
        if z not in self.cache:
            c=int(self.oracle(z))
            if not 0 <= c <= self.dimension:
                raise ArithmeticError('count outside dimensional range')
            self.cache[z]=c; self.calls+=1
        return self.cache[z]

    def find(self, left: int, right: int, h: int) -> Gap:
        if h < max(1,4*self.beta):
            raise ValueError('h >= max(1,4 beta) is required')
        before=self.calls
        cl=self.query(left-self.beta); cr=self.query(right+self.beta)
        r=cr-cl
        if r < 0 or right-left < (12*r+4)*h:
            raise ValueError('root length >= (12 upper_count+4)h required')
        depth=0
        # At every failed midpoint test r_next <= floor((r-1)/2).
        while True:
            if right-left < (12*r+4)*h:
                raise AssertionError('length/population invariant violated')
            mid=(left+right)//2
            if r == 0:
                gap=Gap(mid,h,cl,depth,self.calls-before)
                self.events.append(dict(stage='empty_parent',left=str(left),right=str(right),
                                        upper_count=r,**asdict(gap)))
                return gap
            cminus=self.query(mid-h-self.beta)
            cplus=self.query(mid+h+self.beta)
            if not cl <= cminus <= cplus <= cr:
                raise ArithmeticError('soft-count ordering contract violated')
            event=dict(stage='gap_test',left=str(left),right=str(right),mid=str(mid),
                       halfwidth=str(h),upper_count=r,count_minus=cminus,count_plus=cplus,depth=depth)
            self.events.append(event)
            if cminus == cplus:
                return Gap(mid,h,cminus,depth,self.calls-before)
            # Remove a larger central strip; its soft counts absorb >=1 value.
            lstop=mid-2*h; rstart=mid+2*h
            cleft=self.query(lstop+self.beta)
            cright=self.query(rstart-self.beta)
            nl=cleft-cl; nr=cr-cright
            if min(nl,nr) < 0 or nl+nr > r-1:
                raise ArithmeticError('population reduction contract violated')
            event.update(left_count=nl,right_count=nr,removed_lower_bound=r-nl-nr)
            if nl <= nr:
                right=lstop;cr=cleft;r=nl;event['chosen']='left'
            else:
                left=rstart;cl=cright;r=nr;event['chosen']='right'
            depth+=1
            if depth > self.dimension.bit_length()+1:
                raise AssertionError('proved logarithmic depth bound violated')

class LazyPacketGrid:
    """One deterministic, shared boundary per grid index; distinct cells disjoint.

    macro_width W is a power of two, measured in integer grid units.  A
    boundary lives in [jW-W/4,jW+W/4], so any cell has width <= 3W/2.
    """
    def __init__(self, finder: GapFinder, macro_width: int):
        W=macro_width
        if W < 8 or W & (W-1):
            raise ValueError('macro width must be a power of two >=8')
        self.finder=finder;self.W=W
        self.h=floor_power_two((W//2)//(12*finder.dimension+4))
        if self.h < max(1,4*finder.beta):
            raise ValueError('macro width too small for the count uncertainty')
        self.boundaries: dict[int,Gap]={}

    def boundary(self, j: int) -> Gap:
        if j not in self.boundaries:
            anchor=j*self.W
            self.boundaries[j]=self.finder.find(anchor-self.W//4,anchor+self.W//4,self.h)
        return self.boundaries[j]

    def assign(self, rank: int, center: int, error: int) -> tuple[int,Gap,Gap]:
        if not 0 <= rank < self.finder.dimension:
            raise ValueError('ascending rank outside range')
        if error < 0 or error >= self.W//4:
            raise ValueError('located eigenvalue error must be < W/4')
        j=center//self.W
        l=self.boundary(j);r=self.boundary(j+1)
        if rank < l.count:
            j-=1;r=l;l=self.boundary(j)
        elif rank >= r.count:
            j+=1;l=r;r=self.boundary(j+1)
        if not l.count <= rank < r.count:
            raise ArithmeticError('located rank is not inside its proved neighboring cell')
        if r.center-l.center > 3*self.W//2:
            raise AssertionError('cell diameter bound violated')
        return j,l,r
