"""Internal numerical implementation; use the public API."""
from __future__ import annotations
from fractions import Fraction as F

def nearest(a: int, b: int) -> int:
    if not b: raise ZeroDivisionError
    if b < 0: a,b=-a,-b
    s=-1 if a<0 else 1
    q,r=divmod(abs(a),b)
    if 2*r>b or (2*r==b and q&1):q+=1
    return s*q

def grid(x: F, Q: int) -> int:
    return nearest(x.numerator*Q,x.denominator)

class GridGram:
    def count(self, z: int, *, details=False):
        """Exact inertia of a nearby matrix, not claimed exact inertia of G."""
        last = None
        c = 0
        clamps = 0
        mx = 0
        piv = [] if details else None
        for i, a in enumerate(self.a):
            raw = a - z - (nearest(self.b[i - 1] ** 2, last) if i else 0)
            cur = raw
            if abs(cur) < self.tau:
                cur = self.tau if cur > 0 else -self.tau
                clamps += 1
            c += cur < 0
            last = cur
            mx = max(mx, abs(cur))
            if details:
                piv.append(cur)
        self.work['sturm_rows'] += self.n
        self.work['count_calls'] += 1
        rec = {'count': int(c), 'clamped_pivots': clamps, 'max_pivot_bits': mx.bit_length(), 'backward_radius': str(F(self.beta, self.Q))}
        if details:
            rec['pivots'] = piv
        return (int(c), rec)

    def domain(self):
        lo = self.Q // 2 + self.beta
        hi = 4 * self.Q - self.beta
        c0, r0 = self.count(lo)
        c1, r1 = self.count(hi)
        return {'applicable': c0 == 0 and c1 == self.n, 'n': self.n, 'certified_squared_domain': '[1/2,4)', 'lower': r0, 'upper': r1, 'guaranteed_acceptance_domain': '[3/4,13/4]'}

    def locate(self, descending_index: int):
        j = self.n - descending_index
        lo = self.Q // 2
        hi = 4 * self.Q
        trace = []
        for it in range(2 * self.p + 10):
            mid = (lo + hi) // 2
            c, r = self.count(mid)
            if c < j:
                lo = max(lo, mid - self.beta)
            else:
                hi = min(hi, mid + self.beta)
            trace.append({'iteration': it + 1, 'count': c, 'clamps': r['clamped_pivots'], 'lo_grid': str(lo), 'hi_grid': str(hi)})
        if hi - lo > self.eta // 64:
            raise AssertionError('proved bracket width violated')
        mu = (lo + hi) // 2
        return (mu, {'descending_index': descending_index, 'lo_grid': str(lo), 'hi_grid': str(hi), 'mu_grid': str(mu), 'width': str(F(hi - lo, self.Q)), 'steps': len(trace), 'trace': trace})
