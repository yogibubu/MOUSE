"""Response-consistent SI diagnostic; no TDA eigensolver or fitted energies."""
from pathlib import Path
import sys,json
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');BASE=Path('/home/enzo/MOUSE-frontier-cisd-20261101')
# Avoid shadowing the Python standard-library operator module.
sys.path.insert(0,str(ROOT))
from basis import *
from transition import vectors_in_frame
from response_operator import projected_response

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

def run(cohort,name):
    out=ROOT/cohort/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    d=prepare_basis(cohort,name);case=d['case'];no=d['no'];basis=d['basis'];canonical=d['canonical'];rows=d['rows']
    with np.load(BASE/cohort/name/'checkpoint.npz') as z:s=z['metric'].copy();h=z['matrix'].copy()
    assert json.loads((BASE/cohort/name/'result.json').read_text())['labels']==d['labels']
    t=vectors_in_frame(d['sr']@canonical,basis,no)
    fg,_=d['ctx'].fock_and_energy(np.array([canonical[:,:no]@canonical[:,:no].T]*2));levels=np.diag(canonical.T@fg[0]@canonical)
    model,hf=projected_response(d['ctx'],case['s'],canonical,levels,no,t,case['eri'],case['eri_lr'],d['sr']@d['h']@d['sr'])
    delta=model-hf;diag=np.diag(delta)/np.diag(s);corrected=h+delta-.5*s*(diag[:,None]+diag[None,:])
    assert np.max(abs(np.diag(corrected)-np.diag(h)))<1e-10
    old=d['old'];oldreport=json.loads((old/'result.json').read_text());oldpred={r['id']:r for r in oldreport['predictions']}
    with np.load(old/'checkpoint.npz') as z:reference=np.column_stack([z['coefficients'][:,oldpred[row['id']]['root']] for row in rows])
    projected,pe,pc=spectrum(corrected,s,reference,rows,True);full,fe,fc=spectrum(corrected,s,reference,rows,False)
    report=dict(primary=projected,full_control=full,labels=d['labels'],transition_vector_norms=np.linalg.norm(t,axis=1),largest_coupling_change_ev=float(np.max(abs(corrected-h))*EV),diagonal_energy_change_ev=float(np.max(abs(np.diag(corrected-h)))*EV),ground_reference='moving stationary ground projector, Eg=0',model_uses_reference_excitation_energies=False,lambda_exchange=.9,weights=[.5,.5],new_DFT_calls=0,new_SCC_calls=0,full_TDA_matrix_constructed=False,TDA_eigensolver_calls=0,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'))
    np.savez_compressed(out/'checkpoint.npz',matrix=corrected,metric=s,old_matrix=h,response_delta=delta,transition_vectors=t,projected_model_operator=model,projected_HF_operator=hf,energies=pe,coefficients=pc,full_energies=fe,full_coefficients=fc)
    write(out/'result.json',report);print(name,projected['predictions'],flush=True)
if __name__=='__main__':
    with threadpool_limits():
        p=json.loads((BASE/'protocol.json').read_text())
        for cohort,name in p['cases']+[['prospective_cisd',r['molecule']] for r in p['prospective_cases']]:run(cohort,name)
