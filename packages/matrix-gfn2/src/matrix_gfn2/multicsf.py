"""Experimental common-orbital, spin-singlet state interaction with analytic jets.

Diagonal: existing ROKS functionals. Off-diagonal: bare projected Slater-Condon
2(ia|jb)-(ij|ab), as a wavefunction coupling rather than a TDA kernel insertion.
This is an explicitly different multistate model. No fitted mixing scale.
This module provides orbital derivatives. Nuclear jets and their limited
molecular qualification are in multicsf_nuclear_response.py and the frame
response modules. Every constituent CSF and their linear combination
has S=0. The CI-eigenvector contribution to orbital curvature is included.
"""
from types import SimpleNamespace
import numpy as np
from scipy.linalg import eigh
from .roks_orbital_response import orbital_residual_jacobian


def quartic_jet(eri,c,terms,dc=None):
    """Value, d/dC, and its directional derivative for quartic ERI terms."""
    n=len(c);g=np.zeros_like(c);dg=None if dc is None else np.zeros_like(dc);value=0.
    for weight,ids in terms:
        vec=[c[:,i] for i in ids]
        value+=weight*np.einsum('abcd,a,b,c,d->',eri,*vec,optimize=True)
        for k in range(4):
            args=[eri];subs=['abcd']
            for l in range(4):
                if l!=k:args.append(vec[l]);subs.append('abcd'[l])
            g[:,ids[k]]+=weight*np.einsum(','.join(subs)+'->'+'abcd'[k],*args,optimize=True)
            if dc is not None:
                for l in range(4):
                    if l==k:continue
                    args=[eri];subs=['abcd']
                    for m in range(4):
                        if m not in [k,l]:args.append(vec[m]);subs.append('abcd'[m])
                    block=weight*np.einsum(','.join(subs)+'->'+'abcd'[k]+'abcd'[l],*args,optimize=True)
                    dg[:,:,ids[k]]+=dc[:,:,ids[l]]@block.T
    return float(value),g,dg

class MultiCSFObjective:
    def __init__(self,context,eri,noccupied,rank,reference_coefficients,reference_transition):
        self.context=context;self.eri=np.asarray(eri);self.no=int(noccupied);self.rank=int(rank)
        self.s=context.state.overlap;self.n=len(self.s)
        if not 1<=rank<=min(self.no,self.n-self.no):raise ValueError('invalid CSF rank')
        self.occupations=[];self.triplets=[]
        for i in range(rank):
            o=np.zeros((2,self.n));o[:,:self.no]=1;o[0,i]=0;o[0,self.no+i]=1
            self.occupations.append(o);self.triplets.append(np.array([np.maximum(*o),np.minimum(*o)]))
        left,right=np.triu_indices(self.n,1);labels=np.vstack(self.occupations+self.triplets)
        active=np.any(labels[:,left]!=labels[:,right],axis=0);self.left,self.right=left[active],right[active]
        self.dimension=len(self.left);self.reference_coefficients=np.asarray(reference_coefficients);self.reference_transition=np.asarray(reference_transition)
        self.rank_cap=float(np.sqrt(np.sum(np.linalg.svd(reference_transition,compute_uv=False)[:rank]**2))/np.linalg.norm(reference_transition));self.cache={}
    def target_vector(self,c):
        # Reference occupied/virtual spaces are the column blocks of the fixed chart.
        cr=self.reference_coefficients
        h=cr[:,:self.no].T@self.s@c[:,:self.rank]
        p=cr[:,self.no:].T@self.s@c[:,self.no:self.no+self.rank]
        return np.einsum('ik,ij,jk->k',h,self.reference_transition,p)
    def evaluate(self,c,curvature=True):
        if 'point' not in self.cache or not np.array_equal(c,self.cache['point']):
            r=self.rank;h=np.zeros((r,r));dh=np.zeros((r,r,self.dimension));diagonal=[];off=[]
            for i,(occ,triplet) in enumerate(zip(self.occupations,self.triplets)):
                g=np.zeros(self.dimension);e=0.;focks=[]
                for w,o in [(2.,occ),(-1.,triplet)]:
                    dm=np.array([(c*x)@c.T for x in o]);f,ee=self.context.fock_and_energy(dm);e+=w*ee;focks.append(f)
                    q=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));g+=2*w*(q[self.left,self.right]-q[self.right,self.left])
                h[i,i]=e;dh[i,i]=g;diagonal.append(focks)
            for i in range(r):
                for j in range(i):
                    terms=[(2.,(i,self.no+i,j,self.no+j)),(-1.,(i,j,self.no+i,self.no+j))]
                    e,g,_=quartic_jet(self.eri,c,terms);q=c.T@g;grad=q[self.left,self.right]-q[self.right,self.left]
                    h[i,j]=h[j,i]=e;dh[i,j]=dh[j,i]=grad;off.append((i,j,terms,g))
            energies,z=eigh(h);target=self.target_vector(c);scores=abs(target@z)/max(np.linalg.norm(target),1e-300);root=int(np.argmax(scores));coeff=z[:,root]
            if coeff@target<0:coeff=-coeff
            gradient=np.einsum('i,ija,j->a',coeff,dh,coeff)
            self.cache=dict(point=c.copy(),energy=float(energies[root]),g=gradient,energies=energies,eigenvectors=z,root=root,ci=coeff,matrix=h,dh=dh,diagonal=diagonal,off=off,coefficient_overlap=float(scores[root]),coefficient_margin=float(scores[root]-np.max(np.delete(scores,root),initial=0.)))
        a=self.cache
        if curvature and 'h' not in a:
            total=np.zeros((self.dimension,self.dimension));state=SimpleNamespace(**vars(self.context.state));state.coefficients=c
            for i,(occ,triplet,focks) in enumerate(zip(self.occupations,self.triplets,a['diagonal'])):
                hh=np.zeros_like(total)
                for w,o,f in [(2.,occ,focks[0]),(-1.,triplet,focks[1])]:
                    state.alpha_occupations,state.beta_occupations=o
                    kernel=self.context.density_response_kernel(state)
                    hh+=w*orbital_residual_jacobian(c,o,f,kernel,self.context.polarization,self.left,self.right)
                total+=a['ci'][i]**2*.5*(hh+hh.T)
            dc=np.zeros((self.dimension,self.n,self.n))
            for b,(i,j) in enumerate(zip(self.left,self.right)):dc[b,:,j]=c[:,i];dc[b,:,i]=-c[:,j]
            for i,j,terms,g in a['off']:
                _,_,dg=quartic_jet(self.eri,c,terms,dc);dq=dc.transpose(0,2,1)@g+c.T@dg
                hh=(dq[:,self.left,self.right]-dq[:,self.right,self.left]).T
                total+=2*a['ci'][i]*a['ci'][j]*.5*(hh+hh.T)
            # Exact response of the selected CI root, not frozen coefficients.
            for k in range(self.rank):
                if k==a['root']:continue
                gap=a['energy']-a['energies'][k]
                if abs(gap)<1e-8:raise ValueError('unresolved CI degeneracy')
                dg=np.einsum('i,ija,j->a',a['eigenvectors'][:,k],a['dh'],a['ci'])
                total+=2*np.outer(dg,dg)/gap
            a['h']=.5*(total+total.T)
        return a['energy'],a['g'],a.get('h')
    def fidelity(self,c):
        self.evaluate(c,False);cr=self.reference_coefficients;h=cr[:,:self.no].T@self.s@c[:,:self.rank];p=cr[:,self.no:].T@self.s@c[:,self.no:self.no+self.rank]
        transition=(h*self.cache['ci'])@p.T;n=np.linalg.norm(transition)
        raw=abs(np.sum(transition*self.reference_transition))/max(n*np.linalg.norm(self.reference_transition),1e-300)
        return float(raw/self.rank_cap),float(raw)
    def signed_overlap(self,c,ci,d,dj):
        overlap=c.T@self.s@d;amplitude=0.
        for i,oi in enumerate(self.occupations):
            phasei=(-1)**(self.no-1+i)
            for j,oj in enumerate(self.occupations):
                phasej=(-1)**(self.no-1+j);v=0.
                for a in [oi,oi[::-1]]:
                    for b in [oj,oj[::-1]]:
                        v+=.5*np.linalg.det(overlap[np.ix_(np.flatnonzero(a[0]),np.flatnonzero(b[0]))])*np.linalg.det(overlap[np.ix_(np.flatnonzero(a[1]),np.flatnonzero(b[1]))])
                amplitude+=ci[i]*dj[j]*phasei*phasej*v
        return float(amplitude)
