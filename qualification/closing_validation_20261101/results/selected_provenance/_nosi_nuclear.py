"""Portable analytic molecular matrix-response primitives; no external DFT runtime."""
import numpy as np
from scipy.linalg import eigh,solve_sylvester
from ._nosi_mixed import element as determinant_element,Jet
from ._nosi_orbital_response import densities
from .deltascf_analytic_hessian import _fixed_density_geometry,_spin_geometry,_add_charge_response
def square_root_jet(s):
    w,u=eigh(s.v);root=(u*np.sqrt(w))@u.T
    first=np.array([solve_sylvester(root,root,x) for x in s.a]) if s.n else np.empty_like(s.a);tangent=solve_sylvester(root,root,s.b)
    mixed=np.array([solve_sylvester(root,root,x-a@tangent-tangent@a) for x,a in zip(s.ab,first)]) if s.n else np.empty_like(s.ab)
    return Jet(root,first,tangent,mixed)

def closed(c,indices):return [(1.,(c[:,indices],c[:,indices]))]
def opened(c,no):
    a=c[:,list(range(1,no))+[no]];b=c[:,:no]
    return [(2**-.5,(a,b)),(2**-.5,(b,a))]

def csf_element(a,b,h,w):
    result=[Jet.constant(0.,a[0][1][0].n),Jet.constant(0.,a[0][1][0].n)]
    for wa,aa in a:
        for wb,bb in b:
            values=determinant_element(aa,bb,h,w)
            result=[x+wa*wb*y for x,y in zip(result,values)]
    return result

def projected_eigen_jet(h,s,root):
    p=Jet.constant(np.eye(len(s.v))[:,1:],s.n);p[0,:]=p[0,:]-s[0,1:]/s[0,0]
    hp=p.T@h@p;sp=p.T@s@p
    e,c=eigh(hp.v,sp.v);root=int(root);energy=e[root];v=c[:,root]
    a=hp.a-energy*sp.a;b=hp.b-energy*sp.b
    gradient=np.einsum('i,aij,j->a',v,a,v);tangent=float(v@b@v)
    mixed=np.einsum('i,aij,j->a',v,hp.ab-energy*sp.ab,v)
    mixed-=tangent*np.einsum('i,aij,j->a',v,sp.a,v)+gradient*float(v@sp.b@v)
    for other in range(len(e)):
        if other==root:continue
        gap=energy-e[other]
        if abs(gap)<1e-8:raise ValueError('unresolved projected state-interaction degeneracy')
        left=np.einsum('i,aij,j->a',c[:,other],a,v);right=float(c[:,other]@b@v)
        mixed+=2*left*right/gap
    return Jet(np.array(energy),gradient,np.array(tangent),mixed)

class DensityEnergy:
    def __init__(self,context):self.context=context;self.base={};self.spin={}
    def evaluate(self,c,occupations,direction):
        context=self.context;dm=densities(c,occupations);p=np.array([x.v for x in dm]);total=p.sum(0)
        key=(c.v.tobytes(),occupations.sum(0).tobytes())
        if key not in self.base:
            kernel,ds,df,d2s,rr=_fixed_density_geometry(context,p)
            if context.charge_correction is not None:
                charge=context.charge_correction;field,hessian=charge.geometry(total,ds,d2s);df=df+field;rr=rr+hessian;kernel=_add_charge_response(kernel,charge)
            self.base[key]=(kernel,ds,df,d2s,rr)
        kernel,ds,df,d2s,rr=self.base[key];spin_key=p.tobytes()
        if spin_key not in self.spin:
            sf,srr=_spin_geometry(context.polarization,p,ds,d2s)
            f,e=context.fock_and_energy(p);g=context.analytic_gradient(p,np.zeros_like(total)).ravel()
            self.spin[spin_key]=(df[:,None]+sf,rr+srr,f,e,g)
        explicit,curvature,f,e,g=self.spin[spin_key]
        first=np.array([x.a for x in dm]).transpose(1,0,2,3);tangent=np.array([x.b for x in dm]);mixed=np.array([x.ab for x in dm]).transpose(1,0,2,3)
        total_field=explicit.transpose(1,2,3,0)@direction
        total_field+=kernel(tangent.sum(0))[None]+context.polarization.response_batch(tangent[None])[0]
        derivative=g+np.einsum('sij,asij->a',f,first)
        second=curvature@direction+np.einsum('asij,sij->a',explicit,tangent)+np.einsum('sij,asij->a',total_field,first)+np.einsum('sij,asij->a',f,mixed)
        return Jet(np.array(e),derivative,np.array(direction@derivative),second)

