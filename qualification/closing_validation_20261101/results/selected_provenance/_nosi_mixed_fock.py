"""Analytic mixed Fock jets of the native-plus-frozen-response functional.

Native geometry/density algebra is reused from molecular_second_response.
AO-pair charge, spin and one-body corrections use early feature contraction,
including all overlap and co-rotating frame derivatives. No displaced SCF
enters the implementation. This is not yet a complete molecular Hessian.
"""
import numpy as np
from ._mixed_jet import MixedJet as Jet,contract
from .molecular_second_response import _integrals,_fock

class MixedFock:
    def __init__(self,context):
        self.context=context;self.calc=context.calculator;self.state=context.state
        self.fixed=self.calc._prepare_density_response_gradient_context(self.state)
        self.charge=context.charge_correction;self.spin=context.polarization.term
        assert np.max(abs(context.polarization.coupling_hartree))==0
        assert np.array_equal(self.charge.transport.original,self.spin.transport.original)
        self.operators=self.charge.transport.original;self.flat=self.operators.reshape(len(self.operators),-1)
        self.calc._model(self.state.coordinates_bohr,relocate_basis=True)
        self.geometry_cache={}
    def geometry(self,direction):
        direction=np.asarray(direction).reshape(self.state.coordinates_bohr.shape);key=direction.tobytes()
        if key not in self.geometry_cache:
            s,mu,q=_integrals(self.calc,self.state,self.fixed,direction)
            n=self.state.coordinates_bohr.size
            r=Jet(self.state.coordinates_bohr,np.eye(n).reshape(n,*direction.shape),direction)
            transport=self.charge.transport;du,d2r=transport.frame_jet(second=True)
            tangent=np.einsum('a,aij->ij',direction.ravel(),du)
            mixed=np.array([transport.lift(x) for x in np.einsum('b,abij->aij',direction.ravel(),d2r)])
            u=Jet(transport.u,du,tangent,mixed)
            self.geometry_cache[key]=(r,s,mu,q,u)
        return self.geometry_cache[key]
    def moments(self,p):
        return Jet(self.flat@p.v.ravel(),p.a.reshape(p.n,-1)@self.flat.T,self.flat@p.b.ravel(),p.ab.reshape(p.n,-1)@self.flat.T)
    def expand(self,x):
        shape=self.operators.shape[1:]
        return Jet((x.v@self.flat).reshape(shape),(x.a@self.flat).reshape(x.n,*shape),(x.b@self.flat).reshape(shape),(x.ab@self.flat).reshape(x.n,*shape))
    @staticmethod
    def linear(matrix,x):return Jet(matrix@x.v,x.a@matrix.T,matrix@x.b,x.ab@matrix.T)
    def correction(self,p,s,u,term):
        density=.5*(s@p+p@s);pulled=u.T@density@u
        q=self.moments(pulled);field=self.linear(term.k,q-term.q0)+term.v
        a=u@self.expand(field)@u.T
        return .5*(s@a+a@s)
    def evaluate(self,spin_density,direction):
        r,s,mu,q,u=self.geometry(direction)
        native=_fock(self.calc,self.state,self.fixed,r,s,mu,q,spin_density[0]+spin_density[1])
        charge=self.correction(spin_density[0]+spin_density[1],s,u,self.charge)
        spin=self.correction(spin_density[0]-spin_density[1],s,u,self.spin)
        return native+charge+spin,native+charge-spin
