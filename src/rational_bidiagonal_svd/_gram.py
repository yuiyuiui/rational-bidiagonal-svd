"""Internal numerical implementation; use the public API."""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import json, subprocess, os, time
import mpmath as mp
from ._spectral_grid import GridGram, grid
from ._gaps import GapFinder, LazyPacketGrid, floor_power_two
from ._budget import allocate_gram, checks, lower_sign_steps
from ._filter import finite_design, emit_real, bounded_sum, mf

class RawGrid(GridGram):
    def __init__(self,diag,off,q):
        self.n=len(diag);self.p=q;self.Q=1<<(12*q);self.eps=F(1,1<<q)
        self.eta=self.Q>>(2*q);self.tau=self.Q>>(4*q);self.beta=self.tau+4
        self.a_exact=list(map(F,diag));self.b_exact=list(map(F,off))
        if len(off)!=self.n-1:raise ValueError('tridiagonal shape')
        self.a=[grid(x,self.Q) for x in self.a_exact];self.b=[grid(x,self.Q) for x in self.b_exact]
        self.work={'sturm_rows':0,'count_calls':0}

def packet_plan(g,indices):
    n=g.n;Q=g.Q;W=floor_power_two(n*(Q>>g.p)//4)
    finder=GapFinder(lambda z:g.count(z)[0],n,g.beta);pack=LazyPacketGrid(finder,W)
    groups={};locations=[]
    for pos,j in enumerate(indices):
        mu,loc=g.locate(j);locations.append(loc)
        err=max(mu-int(loc['lo_grid']),int(loc['hi_grid'])-mu)
        cell,l,r=pack.assign(n-1-j,mu,err)
        item=groups.setdefault(cell,dict(cell=cell,left=str(F(l.center,Q)),right=str(F(r.center,Q)),
            population=r.count-l.count,count_lo=l.count,count_hi=r.count,indices=[],positions=[]))
        item['indices'].append(j);item['positions'].append(pos)
    return {'groups':[groups[i] for i in sorted(groups)],'boundary_clearance':str(F(pack.h,Q)),
       'locations':locations,'gap_events':finder.events,'work':dict(g.work),
       'boundary_count':len(pack.boundaries),'q':g.p,'matrix_dimension':n}

class RationalGramOracle:
    def __init__(self,run_dir,*,diagnostic_bits=None,timeout=None,backend=None,admit=None):
        self.root=Path(run_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.backend=backend;self.admit=admit;self.calls=[];self.diagnostic_bits=diagnostic_bits;self.timeout=timeout
    def __call__(self,diag,off,indices,q,label):
        if not indices:return [],[],{'calls':0}
        n=len(diag);b=allocate_gram(n,q);ck=checks(b)
        if not all(ck.values()):raise AssertionError(ck)
        # H=2I+G, with G a Gram of a matrix of norm at most one.
        hdiag=[F(2)+F(x) for x in diag];g=RawGrid(hdiag,off,q)
        dom=g.domain()
        if not dom['applicable']:raise ValueError('positive raw Gram outside domain')
        plan=packet_plan(g,indices);folder=self.root/label;folder.mkdir(exist_ok=True)
        (folder/'packet_plan.json').write_text(json.dumps(plan,indent=2))
        bits=self.diagnostic_bits or b.scan_bits
        exe=self.backend
        if exe is None or not exe.exists():raise FileNotFoundError(f'build kernel at {bits} bits')
        vals=[None]*len(indices);vec=[None]*len(indices);groups=[];cache={}
        def locate(j):
            if j not in cache:
                _,rec=g.locate(j);cache[j]=(F(int(rec['lo_grid']),g.Q),F(int(rec['hi_grid']),g.Q))
            return cache[j]
        with mp.workprec(b.coefficient_bits):
            for gi,gr in enumerate(plan['groups']):
                m=len(gr['positions']);l,u=F(gr['left']),F(gr['right']);h=F(plan['boundary_clearance'])
                mu=(l+u)/2;A=(u-l)/2;a=A-h/2;ext=A+h/2
                if gr['population']==n:
                    cols=[[mp.mpf(i==j) for i in range(n)] for j in range(m)]
                    rec={'rank':0,'mode':'whole-space','requested':m,'population':n}
                else:
                    distances=[]
                    if gr['count_lo']>0:distances.append(mu-locate(n-gr['count_lo'])[1])
                    if gr['count_hi']<n:distances.append(locate(n-1-gr['count_hi'])[0]-mu)
                    if distances:ext=max(ext,min(distances))
                    if not 0<a<ext<=8:raise AssertionError('bad packet geometry')
                    J,tr=lower_sign_steps((ext-a)/(ext+a),b)
                    if self.admit is not None:self.admit(n,1<<(J+1),bits)
                    des=finite_design(mu,a,ext,b.delta,J)
                    if des['rank']>b.rank_max:raise AssertionError('rank bound')
                    sub=folder/f'packet_{gi}';sub.mkdir(exist_ok=True)
                    inp=sub/'input.txt';out=sub/'kernel.json'
                    with inp.open('w') as f:
                        f.write(f"{n} {m} {des['rank']}\n")
                        for x in list(map(mf,hdiag))+list(map(mf,off))+[des['constant']]:emit_real(f,x)
                        for z,w in zip(des['poles'],des['weights']):
                            for x in [z.real,z.imag,w.real,w.imag]:emit_real(f,x)
                    design={'mu':str(mu),'a':str(a),'g':str(ext),'q':q,'design_delta':str(b.delta),
                      'J':J,'rank':des['rank'],'kernel_bits':bits,'required_kernel_bits':b.scan_bits,
                      'proof_precision':bits>=b.scan_bits,'coefficient_bits':b.coefficient_bits,
                      'gamma_observed':mp.nstr(min(z.imag for z in des['poles'] if z.imag>0),60)}
                    (sub/'design.json').write_text(json.dumps(design,indent=2))
                    (folder/'progress.json').write_text(json.dumps({'packet':gi,**design}))
                    t=time.monotonic();env=dict(os.environ)
                    # The public core has no environment-driven audit branches.
                    proc=subprocess.run([str(exe),str(inp),str(out)],capture_output=True,text=True,env=env,timeout=self.timeout)
                    (sub/'stderr.txt').write_text(proc.stderr)
                    if proc.returncode:raise ArithmeticError(proc.stderr)
                    krec=json.loads(out.read_text())
                    if krec['precision_bits']!=bits:raise ArithmeticError('backend precision mismatch')
                    cols=[[mp.mpf(v) for v in item['vector']] for item in krec['columns']]
                    rec={**design,'requested':m,'population':gr['population'],'seconds':time.monotonic()-t,
                         'pivots':[x['pivot'] for x in krec['columns']],'work':krec['work']}
                for pos,col in zip(gr['positions'],cols):
                    norm=mp.sqrt(bounded_sum(x*x for x in col));vec[pos]=[x/norm for x in col]
                    vals[pos]=mf(mu-2)
                groups.append(rec)
        meta={'mode':'rational_joint','q':q,'requested':len(indices),'groups':groups,'bits':bits,
              'proof_precision':bits>=b.scan_bits,'required_bits':b.scan_bits,'checks':ck,
              'counter_work':g.work,'full_vector_reorthogonalizations':0,'dense_decomposition':False}
        (folder/'summary.json').write_text(json.dumps(meta,indent=2));self.calls.append(meta)
        return vals,vec,meta
