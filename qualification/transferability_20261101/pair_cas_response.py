"""Analytic orbital gradient of the research pair-CAS functional closure."""
import numpy as np
from types import SimpleNamespace
from pair_cas import pair_cas_matrix

def pair_cas_response(context,c,no):
    n=len(c);left,right=np.triu_indices(n,1);d=len(left);h=0;p=no
    ground=np.zeros((2,n));ground[:,:no]=1
    mixed=ground.copy();mixed[0,h]=0;mixed[0,p]=1
    triplet=np.array([np.maximum(*mixed),np.minimum(*mixed)])
    double=ground.copy();double[:,h]=0;double[:,p]=1
    dc=np.zeros((d,n,n));dc[np.arange(d),:,right]=c[:,left].T;dc[np.arange(d),:,left]=-c[:,right].T
    energies=[];gradients=[];focks=[];couplings=[];coupling_gradients=[]
    for index,occ in enumerate([ground,mixed,triplet,double]):
        density=np.array([(c*x)@c.T for x in occ]);f,e=context.fock_and_energy(density)
        q=sum((c.T@ff@c)*o[None,:] for ff,o in zip(f,occ))
        energies.append(e);gradients.append(2*(q[left,right]-q[right,left]));focks.append(f)
        if index in [0,3]:
            state=SimpleNamespace(**vars(context.state));state.coefficients=c
            state.alpha_occupations,state.beta_occupations=occ
            kernel=context.density_response_kernel(state)
            perturbations=[]
            for o in occ:
                dp=(dc*o)@c.T;perturbations.append(dp+dp.transpose(0,2,1))
            perturbations=np.stack(perturbations,axis=1);total=perturbations.sum(1)
            charge=kernel.response_batch(total) if hasattr(kernel,'response_batch') else np.asarray([kernel(x) for x in total])
            spin=context.polarization.response_batch(perturbations) if hasattr(context.polarization,'response_batch') else np.asarray([context.polarization.evaluate(x)[0] for x in perturbations])
            ff=.5*(f[0]+f[1]);df=charge+spin.mean(1)
            couplings.append(float(c[:,h]@ff@c[:,p]))
            coupling_gradients.append(np.einsum('ai,ij,j->a',dc[:,:,h],ff,c[:,p])+np.einsum('i,ij,aj->a',c[:,h],ff,dc[:,:,p])+np.einsum('i,aij,j->a',c[:,h],df,c[:,p]))
    matrix=pair_cas_matrix(*energies,*couplings);values,z=np.linalg.eigh(matrix)
    dg=np.zeros((d,3,3));dg[:,0,0]=gradients[0];dg[:,1,1]=2*gradients[1]-gradients[2];dg[:,2,2]=gradients[3]
    dg[:,0,2]=dg[:,2,0]=gradients[1]-gradients[2]
    dg[:,0,1]=dg[:,1,0]=np.sqrt(2)*coupling_gradients[0]
    dg[:,1,2]=dg[:,2,1]=np.sqrt(2)*coupling_gradients[1]
    gradients=np.einsum('ir,aij,jr->ra',z,dg,z,optimize=True)
    return values,z,gradients,matrix,left,right
