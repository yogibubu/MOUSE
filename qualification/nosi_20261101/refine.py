"""Tighten retained orbital roots for independent derivative validation."""
import numpy as np
from types import SimpleNamespace
from scipy.linalg import expm,eigh
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.multicsf import MultiCSFObjective
from orthogonal_singlet import ground_overlap_jet,ground_overlap_hessian

def normalize(c,s):
    w,u=eigh(c.T@s@c);return c@(u/w**.5)@u.T

def retract(c,x,left,right):
    k=np.zeros_like(c);k[left,right]=x;k[right,left]=-x;return c@expm(k)

def ground(context,c,tolerance=2e-13):
    c=normalize(c,context.state.overlap);n=len(c);no=context.state.electron_count//2
    occ=np.zeros((2,n));occ[:,:no]=1;left,right=np.triu_indices(n,1);keep=occ[0,left]!=occ[0,right];left,right=left[keep],right[keep]
    for iteration in range(40):
        dm=np.array([(c*x)@c.T for x in occ]);f,e=context.fock_and_energy(dm)
        g=sum(2*(c.T@ff@c)[left,right]*(x[right]-x[left]) for ff,x in zip(f,occ))
        if np.max(abs(g))<tolerance:return c,float(e),float(np.max(abs(g))),iteration
        state=SimpleNamespace(**vars(context.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occ
        h=orbital_residual_jacobian(c,occ,f,context.density_response_kernel(state),context.polarization,left,right);h=.5*(h+h.T)
        assert np.min(np.linalg.eigvalsh(h))>1e-8
        step=np.linalg.solve(h,-g);assert np.linalg.norm(step)<.2
        c=retract(c,step,left,right)
    raise RuntimeError('retained ground did not reach derivative-validation tolerance')

def roks(context,eri,c,cg,transition,multiplier,tolerance=2e-13):
    c=normalize(c,context.state.overlap);no=context.state.electron_count//2;mu=float(multiplier)
    objective=MultiCSFObjective(context,eri,no,1,cg,transition);left,right=objective.left,objective.right
    for iteration in range(50):
        e,g,h=objective.evaluate(c,True)
        ov,j=ground_overlap_jet(cg,c,context.state.overlap,objective.occupations[0],left,right)
        residual=np.r_[g-mu*j,ov]
        if np.max(abs(residual))<tolerance:return c,mu,float(e),float(np.max(abs(residual))),iteration
        hc=ground_overlap_hessian(cg,c,context.state.overlap,objective.occupations[0],left,right)
        matrix=np.block([[h-mu*hc,-j[:,None]],[j[None,:],np.zeros((1,1))]])
        step=np.linalg.solve(matrix,-residual);assert np.linalg.norm(step[:-1])<.2
        c=retract(c,step[:-1],left,right);mu+=step[-1]
        assert objective.fidelity(c)[1]>=.7
    raise RuntimeError('retained ROKS root did not reach derivative-validation tolerance')
