"""Analytic first/second orbital response of ground-orthogonal ROKS.

The ground reference, AO metric, nonlinear overlap constraint and multiplier
all respond. Second response differentiates the full KKT residual using the
analytic mixed Fock and singular-safe determinant adjugate jets. No displaced
SCF or orbital-optimization calculation enters this implementation.
"""
import numpy as np
from types import SimpleNamespace
from scipy.linalg import lu_factor,lu_solve
from matrix_gfn2._mixed_jet import MixedJet as Jet,stack
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.multicsf import MultiCSFObjective
from mixed_fock import MixedFock
from determinant_mixed import ground_constraint
from orthogonal_singlet import ground_overlap_hessian

def coefficient_jet(c,s,kappa,direction,mixed_rotation=None):
    a=c.T@s.a@c;b=c.T@s.b@c;ab=c.T@s.ab@c
    kb=np.einsum('a,aij->ij',direction.ravel(),kappa)
    first=c@(-.5*a+kappa);tangent=c@(-.5*b+kb)
    mixed=.375*(a@b+b@a)-.5*ab-.5*a@kb-.5*b@kappa+.5*(kappa@kb+kb@kappa)
    if mixed_rotation is not None:mixed=mixed+mixed_rotation
    return Jet(c,first,tangent,c@mixed)

def densities(c,occupations):return [(c*o)@c.T for o in occupations]

def energy_gradient(c,focks,occupations,left,right):
    value=Jet.constant(np.zeros(len(left)),c.n)
    for f,o in zip(focks,occupations):
        mo=c.T@f@c;value=value+2*mo[left,right]*(o[right]-o[left])
    return value

def rotations(values,left,right,n):
    values=np.asarray(values);result=np.zeros((*values.shape[:-1],n,n));result[...,left,right]=values;result[...,right,left]=-values
    return result

class GroundResponse:
    def __init__(self,context,c,fock_engine=None):
        self.context=context;self.c=np.asarray(c);self.n=len(c);self.engine=MixedFock(context) if fock_engine is None else fock_engine
        self.nr=context.state.coordinates_bohr.size;self.zero=np.zeros(self.nr);self.geometry=self.engine.geometry(self.zero)[1]
        self.occupations=np.zeros((2,self.n));self.occupations[:,:context.state.electron_count//2]=1
        left,right=np.triu_indices(self.n,1);keep=self.occupations[0,left]!=self.occupations[0,right];self.left,self.right=left[keep],right[keep]
        dm=np.array([(c*o)@c.T for o in self.occupations]);f,_=context.fock_and_energy(dm)
        state=SimpleNamespace(**vars(context.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=self.occupations
        kernel=context.density_response_kernel(state)
        h=orbital_residual_jacobian(c,self.occupations,f,kernel,context.polarization,self.left,self.right);self.h=.5*(h+h.T)
        self.minimum_curvature=float(np.min(np.linalg.eigvalsh(self.h)));assert self.minimum_curvature>1e-8
        self.factor=lu_factor(self.h)
        initial=coefficient_jet(c,self.geometry,np.zeros((self.nr,self.n,self.n)),self.zero)
        residual=self.residual(initial,self.zero);self.primal_error=float(np.max(abs(residual.v)));assert self.primal_error<1e-7
        response=lu_solve(self.factor,-residual.a.T).T
        self.first_rotations=rotations(response,self.left,self.right,self.n)
        self.first_residual=float(np.max(abs(self.h@response.T+residual.a.T)))
        self.cache={}
    def residual(self,c,direction):
        focks=self.engine.evaluate(densities(c,self.occupations),direction)
        return energy_gradient(c,focks,self.occupations,self.left,self.right)
    def coefficient(self,direction):
        direction=np.asarray(direction).ravel();key=direction.tobytes()
        if key not in self.cache:
            s=self.engine.geometry(direction)[1]
            known=coefficient_jet(self.c,s,self.first_rotations,direction)
            if np.any(direction):
                residual=self.residual(known,direction)
                solution=lu_solve(self.factor,-residual.ab.T).T
                mixed=rotations(solution,self.left,self.right,self.n)
                known=coefficient_jet(self.c,s,self.first_rotations,direction,mixed)
            self.cache[key]=known
        return self.cache[key]

class ROKSResponse:
    def __init__(self,context,eri,c,ground,multiplier,transition):
        self.context=context;self.engine=ground.engine;self.ground=ground;self.c=np.asarray(c);self.n=len(c);self.nr=ground.nr
        self.zero=np.zeros(self.nr);self.multiplier=float(multiplier)
        no=context.state.electron_count//2
        objective=MultiCSFObjective(context,eri,no,1,ground.c,transition)
        self.occupations=objective.occupations[0];self.triplet=objective.triplets[0];self.left,self.right=objective.left,objective.right
        _,_,h=objective.evaluate(c,True)
        metric=self.engine.geometry(self.zero)[1]
        initial=coefficient_jet(c,metric,np.zeros((self.nr,self.n,self.n)),self.zero)
        gzero=ground.coefficient(self.zero)
        overlap,jacobian=ground_constraint(gzero,initial,metric,self.occupations,self.left,self.right)
        hc=ground_overlap_hessian(ground.c,c,context.state.overlap,self.occupations,self.left,self.right)
        self.kkt=np.block([[h-self.multiplier*hc,-jacobian.v[:,None]],[jacobian.v[None,:],np.zeros((1,1))]])
        self.minimum_singular_value=float(np.min(np.linalg.svd(self.kkt,compute_uv=False)))
        if self.minimum_singular_value<1e-10:raise ValueError('unresolved ground-orthogonal KKT response')
        self.factor=lu_factor(self.kkt)
        mu=Jet.constant(self.multiplier,self.nr)
        residual=self.residual(initial,gzero,metric,mu,self.zero)
        self.primal_error=float(np.max(abs(residual.v)));assert self.primal_error<1e-7
        solution=lu_solve(self.factor,-residual.a.T).T
        self.first_rotations=rotations(solution[:,:-1],self.left,self.right,self.n);self.first_multiplier=solution[:,-1]
        self.first_residual=float(np.max(abs(self.kkt@solution.T+residual.a.T)));self.cache={}
    def residual(self,c,ground,metric,multiplier,direction):
        mixed=self.engine.evaluate(densities(c,self.occupations),direction)
        triplet=self.engine.evaluate(densities(c,self.triplet),direction)
        gradient=2*energy_gradient(c,mixed,self.occupations,self.left,self.right)-energy_gradient(c,triplet,self.triplet,self.left,self.right)
        overlap,jacobian=ground_constraint(ground,c,metric,self.occupations,self.left,self.right)
        gradient=gradient-multiplier*jacobian
        # stack cannot concatenate arrays with different rank.
        return Jet(np.r_[gradient.v,overlap.v],np.column_stack([gradient.a,overlap.a]),np.r_[gradient.b,overlap.b],np.column_stack([gradient.ab,overlap.ab]))
    def coefficient(self,direction):
        direction=np.asarray(direction).ravel();key=direction.tobytes()
        if key not in self.cache:
            metric=self.engine.geometry(direction)[1];gs=self.ground.coefficient(direction)
            known=coefficient_jet(self.c,metric,self.first_rotations,direction)
            mu=Jet(self.multiplier,self.first_multiplier,float(direction@self.first_multiplier))
            if np.any(direction):
                residual=self.residual(known,gs,metric,mu,direction)
                solution=lu_solve(self.factor,-residual.ab.T).T
                mixed=rotations(solution[:,:-1],self.left,self.right,self.n)
                known=coefficient_jet(self.c,metric,self.first_rotations,direction,mixed)
                mu=Jet(self.multiplier,self.first_multiplier,float(direction@self.first_multiplier),solution[:,-1])
            self.cache[key]=(known,mu)
        return self.cache[key][0]
