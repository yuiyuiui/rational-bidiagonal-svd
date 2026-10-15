"""Regenerate vector figures from the compact validated fixed-case data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();data=json.loads(args.input.read_text())
    out=args.output or args.input.parent/'figures';out.mkdir(parents=True,exist_ok=True)
    records=data['records'];assert len(records)==6 and all(r['validation']['passed'] for r in records)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    colors=['#2369a1','#b95b18','#328046']
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.2),layout='constrained')
    for ax,name in zip(axes,['coupled2','rank_deficient3']):
        rows=[r for r in records if r['case']==name]
        for j,r in enumerate(rows):
            m=r['metrics'];v=[m[k]/m['residual_bound_raw'] for k in ('left_residual','right_residual')]
            ax.plot([0,1],v,'o-',label=r['profile'],color=colors[j],alpha=.8)
        ax.axhline(1,color='black',ls='--',lw=1)
        ax.set_xticks([0,1],['left equation','right equation']);ax.set_ylim(-.025,1.08)
        ax.set_title(name);ax.set_ylabel('residual / raw bound (linear scale)');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    save(fig,out/'residuals')

    fig,axes=plt.subplots(1,2,figsize=(9.4,3.2),layout='constrained')
    for ax,name in zip(axes,['coupled2','rank_deficient3']):
        r=next(r for r in records if r['case']==name and r['profile']=='auto')
        b=np.diag(r['d'])+np.diag(r['e'],1);ref=np.linalg.svd(b,compute_uv=False)
        x=np.arange(len(ref));ax.bar(x-.14,ref,width=.28,color='#666666',label='NumPy reference')
        ax.bar(x+.14,r['values'],width=.28,color=colors[0],label='default output')
        ax.set_xticks(x,[str(v) for v in r['indices']]);ax.set_xlabel('descending zero-based rank')
        ax.set_ylabel('singular value');ax.set_title(name);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,out/'values')

    r=next(r for r in records if r['case']=='rank_deficient3' and r['profile']=='auto')
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.4),layout='constrained')
    for ax,key in zip(axes,['left','right']):
        a=np.array(r[key]);im=ax.imshow(a,cmap='RdBu_r',vmin=-1,vmax=1)
        ax.set_xticks(range(3));ax.set_yticks(range(3));ax.set_xlabel('requested column');ax.set_ylabel('coordinate')
        ax.set_title(key+' carriers: rank-deficient case')
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='black' if abs(v)<.7 else 'white')
    fig.colorbar(im,ax=list(axes),shrink=.85,label='carrier component')
    save(fig,out/'directions')


def save(fig,path):
    fig.savefig(path.with_suffix('.svg'));fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


if __name__=='__main__':main()
