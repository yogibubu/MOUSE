"""Identical selected-CID matrix extraction for newly registered cohorts.

Body copied unchanged from run_selected_cid.py; only idempotent skip becomes
function return. No orbital, coupling, projector or assignment rule is changed.
"""
"""Read-only selected-CID subspace control using completed matrix checkpoints."""
from pathlib import Path
import sys,json,ast,socket
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from nosi_spectrum import spectrum
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');BASE=Path('/home/enzo/MOUSE-frontier-cisd-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101')
def select(cohort,name):
    if (ROOT/'selected_cid'/cohort/name/'result.json').exists():return
    source=BASE/cohort/name;report=json.loads((source/'result.json').read_text());labels=report['labels']
    with np.load(NATIVE/cohort/name/'coherent_common/ground/stationary.npz') as z:no=int(np.count_nonzero(z['result_occupations'][0]))
    keep=[]
    for i,label in enumerate(labels):
        rank=0
        if label.startswith('open_'):
            _,paired,opens=label.split('_',2);rank=2*(int(paired)>=no)+sum(j>=no for j in ast.literal_eval(opens))
        if rank!=1:keep.append(i)
    with np.load(source/'checkpoint.npz') as z:fulls=z['metric'].copy();h=z['matrix'][np.ix_(keep,keep)].copy();s=z['metric'][np.ix_(keep,keep)].copy()
    old=NATIVE/'nosi_coherent'/cohort/name;oldreport=json.loads((old/'result.json').read_text());rows=oldreport['predictions']
    with np.load(old/'checkpoint.npz') as z:reference=np.column_stack([z['coefficients'][:,r['root']] for r in rows])
    # Expand old target vectors into the selected basis; retained old block is intact.
    pad=np.zeros((len(keep),len(rows)));pad[:len(reference)]=reference
    primary,e,c=spectrum(h,s,pad,rows,True);control,fe,fc=spectrum(h,s,pad,rows,False)
    conventional=[]
    for row in control['predictions']:conventional.append(dict(row,gap_ev=row['gap_ev']-float(fe[0]*27.211386245988)))
    out=ROOT/'selected_cid'/cohort/name;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'checkpoint.npz',matrix=h,metric=s,indices=keep,energies=e,coefficients=c,full_energies=fe,full_coefficients=fc)
    (out/'result.json').write_text(json.dumps(dict(primary=primary,full_control=control,conventional=conventional,labels=[labels[i] for i in keep]),indent=2,default=lambda a:a.tolist())+'\n')
    print(name,primary['predictions'],control['predictions'],flush=True)
