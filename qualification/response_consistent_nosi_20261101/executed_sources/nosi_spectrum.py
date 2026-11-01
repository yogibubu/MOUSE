import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
EV=27.211386245988

def spectrum(h,s,reference,rows,project):
    p=np.eye(len(s))
    if project:p=p[:,1:];p[0]-=s[0,1:]/s[0,0]
    ss=p.T@s@p;w,u=eigh(ss);keep=w>1e-8;x=p@u[:,keep]/np.sqrt(w[keep]);e,c=eigh(x.T@h@x);c=x@c
    overlap=abs(reference.T@s[:reference.shape[0]]@c);start=0 if project else 1
    ii,jj=linear_sum_assignment(-overlap[:,start:]);pred=[]
    for i,j in zip(ii,jj+start):
        ov=float(overlap[i,j]);margin=ov-float(np.max(np.delete(overlap[i,start:],j-start),initial=0.))
        pred.append(dict(id=rows[i]['id'],root=int(j),gap_ev=float(e[j]*EV),overlap=ov,margin=margin,qualified=bool(ov>=.7 and margin>=.05 and e[j]>0)))
    return dict(predictions=pred,all_eigenvalues_ev=e*EV,metric_eigenvalues=w),e,c

