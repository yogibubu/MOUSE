"""Contract the model and physical HF single-excitation operators on few CSFs."""
import numpy as np

def contract_exchange(eri,p):
    direct=np.einsum('pqrs,apr,bqs->ab',eri,p,p,optimize=True)
    crossed=np.einsum('pqrs,aps,brq->ab',eri,p,p,optimize=True)
    return direct,crossed

def projected_response(context,state,canonical,levels,no,transition_vectors,eri,eri_lr,h_ao):
    t=transition_vectors.reshape(len(transition_vectors),no,len(canonical)-no);co=canonical[:,:no];cv=canonical[:,no:]
    p=np.einsum('mi,aij,nj->amn',co,t,cv,optimize=True)
    trial=type('Reference',(),dict(vars(state)))();trial.coefficients=canonical;trial.fock=context.fock_and_energy(np.array([co@co.T]*2))[0][0];trial.orbital_energies_hartree=levels
    kernel=context.density_response_kernel(trial);symmetric=.5*(p+p.transpose(0,2,1))
    fields=kernel.response_batch(symmetric) if hasattr(kernel,'response_batch') else np.array([kernel(x) for x in symmetric])
    static=np.einsum('pia,ia,qia->pq',t,levels[no:][None,:]-levels[:no,None],t,optimize=True)
    static+=2*np.einsum('aij,bij->ab',p,fields)
    exchange=.9*(.19*eri+.46*eri_lr);direct,crossed=contract_exchange(exchange,p);model=static+.5*(crossed-direct)
    pg=2*co@co.T;f=h_ao+np.einsum('pqrs,rs->pq',eri,pg,optimize=True)-.5*np.einsum('prqs,rs->pq',eri,pg,optimize=True)
    foo=co.T@f@co;fvv=cv.T@f@cv
    one=np.einsum('pia,ab,qib->pq',t,fvv,t,optimize=True)-np.einsum('pia,ji,qja->pq',t,foo,t,optimize=True)
    direct,_=contract_exchange(eri,p);coulomb=np.einsum('pqrs,apq,brs->ab',eri,p,p,optimize=True)
    hf=one+2*coulomb-direct
    assert max(np.max(abs(model-model.T)),np.max(abs(hf-hf.T)))<1e-8
    return .5*(model+model.T),.5*(hf+hf.T)
