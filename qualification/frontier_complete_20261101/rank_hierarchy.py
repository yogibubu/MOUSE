"""Excitation-rank convergence controls; no fitted matrix elements."""
from pathlib import Path
import sys,ast,json,socket
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-frontier-complete-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101');EV=27.211386245988
with threadpool_limits(2):
 for cohort,name in json.loads((ROOT/'protocol.json').read_text())['cases']:
    folder=ROOT/cohort/name;report=json.loads((folder/'result.json').read_text());labels=report['labels'];base=next(i for i,label in enumerate(labels) if label.startswith('pair_') or label.startswith('open_') or label.startswith('four_spin'))
    z=np.load(folder/'checkpoint.npz');s=z['metric'];h=z['matrix'];old= NATIVE/'nosi_coherent'/cohort/name
    oldreport=json.loads((old/'result.json').read_text());oldz=np.load(old/'checkpoint.npz');oldc=oldz['coefficients'];oldpred=oldreport['predictions'];reference=np.column_stack([oldc[:,p['root']] for p in oldpred]);result=[]
    # Frontier index follows old CSF labels and contiguous canonical indices.
    with np.load(NATIVE/cohort/name/'coherent_common/ground/stationary.npz') as ground:
        no=int(np.count_nonzero(ground['result_occupations'][0]))
    ranks=[]
    for label in labels:
        if label.startswith('pair_'):rank=sum(2*(i>=no) for i in ast.literal_eval(label[5:]))
        elif label.startswith('open_'):
            _,paired,opens=label.split('_',2);rank=2*(int(paired)>=no)+sum(i>=no for i in ast.literal_eval(opens))
        elif label.startswith('four_spin'):rank=2
        else:rank=0
        ranks.append(rank)
    for maximum in [1,2,3,4]:
        ids=np.flatnonzero(np.array(ranks)<=maximum);ss=s[np.ix_(ids,ids)];hh=h[np.ix_(ids,ids)];w,u=eigh(ss);keep=w>1e-8;x=u[:,keep]/np.sqrt(w[keep]);e,c=eigh(x.T@hh@x);c=x@c
        overlap=abs(reference.T@s[:base,ids]@c);ii,jj=linear_sum_assignment(-overlap[:,1:]);pred=[]
        for i,j in zip(ii,jj+1):
            ov=float(overlap[i,j]);margin=ov-np.max(np.delete(overlap[i,1:],j-1),initial=0.)
            pred.append(dict(id=oldpred[i]['id'],gap_ev=float(e[j]*EV),overlap=ov,margin=float(margin),qualified=bool(ov>=.7 and margin>=.05)))
        result.append(dict(maximum_rank=maximum,predictions=pred,dimension=len(ids),aux_ground_ev=float(e[0]*EV)))
    (folder/'rank_hierarchy.json').write_text(json.dumps(result,indent=2));print(name,result,flush=True)
