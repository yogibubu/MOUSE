"""Analytic nuclear jets of the experimental common-orbital CSF functional.

Bare ERIs are frozen in a co-rotating orthonormal AO frame. Metric square-root
jets and molecular-frame jets are analytic. The eigenvector and orbital
responses are both included. This module does not implement the two-parent
relative constraints or the ground-reference subtraction.
"""
import numpy as np
from scipy.linalg import solve_sylvester
from matrix_gfn2.nonlocal_response import TransportedOperators,rotation_jet
from matrix_gfn2.deltascf_analytic_hessian import _fixed_density_geometry,_response_blocks
from multicsf_objective import quartic_jet


def metric_power(s,p):
    w,v=np.linalg.eigh(s)
    if w.min()<=1e-10:raise ValueError('singular AO metric')
    return (v*w**p)@v.T

class FrozenLowdinERI:
    def __init__(self,anchor_state,eri):
        self.anchor=anchor_state;inverse=metric_power(anchor_state.overlap,-.5)
        self.eri=np.einsum('abcd,ai,bj,ck,dl->ijkl',eri,inverse,inverse,inverse,inverse,optimize=True)
    def chart(self,state):
        n=len(state.overlap);transport=TransportedOperators(state,self.anchor.coordinates_bohr,np.zeros((0,n,n)))
        root=metric_power(state.overlap,.5);return transport,root
    def ao_eri(self,state):
        t,l=self.chart(state);x=t.u.T@l
        return np.einsum('abcd,ai,bj,ck,dl->ijkl',self.eri,x,x,x,x,optimize=True)
    def nuclear_blocks(self,state,c,rank,left,right,ds,d2s):
        n=len(c);nr=len(ds);nk=len(left);no=int(self.anchor.electron_count)//2
        active=np.r_[np.arange(rank),no+np.arange(rank)];ca=c[:,active]
        transport,l=self.chart(state);u=transport.u
        _,dr,d2r=rotation_jet(state.coordinates_bohr,self.anchor.coordinates_bohr)
        du=np.array([transport.lift(d) for d in dr])
        dl=np.array([solve_sylvester(l,l,d) for d in ds]);cross=dl[:,None]@dl[None,:]
        d2l=np.empty_like(d2s)
        for a in range(nr):
            for b in range(nr):d2l[a,b]=solve_sylvester(l,l,d2s[a,b]-cross[a,b]-cross[b,a])
        metric=c.T@ds@c;cr=-.5*(c@metric)
        ck=np.zeros((nk,n,len(active)));crk=np.zeros((nk,nr,n,len(active)))
        for k,(i,j) in enumerate(zip(left,right)):
            for a,column in enumerate(active):
                if column==j:ck[k,:,a]=c[:,i];crk[k,:,:,a]=cr[:,:,i]
                if column==i:ck[k,:,a]=-c[:,j];crk[k,:,:,a]=-cr[:,:,j]
        y=u.T@l@ca;yk=u.T@l@ck
        first_map=du.transpose(0,2,1)@l+u.T@dl
        yr=first_map@ca+u.T@l@cr[:,:,active]
        ykr=np.einsum('rij,kja->kria',first_map,ck)+u.T@l@crk
        yrr=np.empty((nr,nr,n,len(active)))
        for a in range(nr):
            for b in range(nr):
                crr=c@(.375*(metric[a]@metric[b]+metric[b]@metric[a])-.5*(c.T@d2s[a,b]@c))[:,active]
                d2u=transport.lift(d2r[a,b])
                yrr[a,b]=(d2u.T@l@ca+du[a].T@dl[b]@ca+du[a].T@l@cr[b,:,active].T
                    +du[b].T@dl[a]@ca+u.T@d2l[a,b]@ca+u.T@dl[a]@cr[b,:,active].T
                    +du[b].T@l@cr[a,:,active].T+u.T@dl[b]@cr[a,:,active].T+u.T@l@crr)
        first=np.concatenate([yk,yr]);rows=[]
        for i in range(rank):
            for j in range(i):
                terms=[(2.,(i,rank+i,j,rank+j)),(-1.,(i,j,rank+i,rank+j))]
                value,g,dg=quartic_jet(self.eri,y,terms,first)
                gradient=np.einsum('rij,ij->r',yr,g)
                rr=yr.reshape(nr,-1)@dg[nk:].reshape(nr,-1).T+np.einsum('rsij,ij->rs',yrr,g)
                kr=yk.reshape(nk,-1)@dg[nk:].reshape(nr,-1).T+np.einsum('krij,ij->kr',ykr,g)
                rows.append((i,j,value,gradient,.5*(rr+rr.T),kr))
        return rows


def nuclear_response(obj,c,eri_transport):
    """Return parent total E gradient, frozen/relaxed nuclear Hessian and KR."""
    e,g,h=obj.evaluate(c,True);cache=obj.cache;ci=cache['ci'].copy();rank=obj.rank;ctx=obj.context;nr=ctx.state.coordinates_bohr.size;nk=len(g)
    gradient_matrix=np.zeros((rank,rank,nr));rr=np.zeros((nr,nr));kr=np.zeros((nk,nr));orbital_check=np.zeros((nk,nk));ds=d2s=None
    for i,(occ,triplet,focks) in enumerate(zip(obj.occupations,obj.triplets,cache['diagonal'])):
        pm=np.array([(c*x)@c.T for x in occ]);total=pm.sum(0)
        kernel,ds,df,d2s,explicit=_fixed_density_geometry(ctx,pm)
        if ctx.charge_correction is not None:
            charge=ctx.charge_correction;dfield,efield=charge.geometry(total,ds,d2s);df=df+dfield;explicit=explicit+efield;base=kernel
            kernel=lambda p,base=base,charge=charge:base(p)+charge.response(p)
        rri,kki,kri=_response_blocks([c],[(2.,occ,focks[0]),(-1.,triplet,focks[1])],ds,d2s,[(df,explicit)]*2,[kernel]*2,ctx.polarization,orbital_pairs=(obj.left,obj.right))
        rr+=ci[i]**2*rri;kr+=ci[i]**2*kri;orbital_check+=ci[i]**2*kki
        q=sum(2*(c.T@f@c)*o[None,:] for f,o in zip(focks[0],occ))-sum((c.T@f@c)*o[None,:] for f,o in zip(focks[1],triplet))
        weighted=c@(.5*(q+q.T))@c.T
        pt=np.array([(c*x)@c.T for x in triplet])
        gradient_matrix[i,i]=(2*ctx.analytic_gradient(pm,.5*weighted)-ctx.analytic_gradient(pt,np.zeros_like(weighted))).ravel()
    # Existing orbital terms are already tested; only add nuclear and mixed
    # bare-ERI jets here, then the CI-eigenvector response in the full chart.
    off=eri_transport.nuclear_blocks(ctx.state,c,rank,obj.left,obj.right,ds,d2s)
    for i,j,value,v,rrv,krv in off:
        assert abs(value-cache['matrix'][i,j])<1e-8
        gradient_matrix[i,j]=gradient_matrix[j,i]=v;rr+=2*ci[i]*ci[j]*rrv;kr+=2*ci[i]*ci[j]*krv
    gradient=np.einsum('i,ijr,j->r',ci,gradient_matrix,ci)
    for k in range(rank):
        if k==cache['root']:continue
        denominator=e-cache['energies'][k]
        dr=np.einsum('i,ijr,j->r',cache['eigenvectors'][:,k],gradient_matrix,ci)
        dk=np.einsum('i,ija,j->a',cache['eigenvectors'][:,k],cache['dh'],ci)
        rr+=2*np.outer(dr,dr)/denominator;kr+=2*np.outer(dk,dr)/denominator
    curvature=np.linalg.eigvalsh(h)
    if np.min(abs(curvature))<=1e-7:raise ValueError('unresolved orbital response')
    response=np.linalg.solve(h,-kr);correction=kr.T@response;total=rr+correction
    return dict(energy=e,gradient=gradient.reshape(-1,3),frozen_hessian=.5*(rr+rr.T),orbital_hessian=h,orbital_nuclear_coupling=kr,orbital_response=response,hessian=.5*(total+total.T),maximum_response_residual=float(np.max(abs(h@response+kr))),raw_symmetry_error=float(np.max(abs(total-total.T))))


def parent_gradient(obj,c,eri_transport):
    """First nuclear derivative only, without building any nuclear Hessian."""
    obj.evaluate(c,False);ctx=obj.context;ci=obj.cache['ci'].copy();rank=obj.rank;nr=ctx.state.coordinates_bohr.size;n=len(c);no=obj.no
    fixed=ctx.calculator._prepare_density_response_gradient_context(ctx.state)
    ds=[]
    for a in range(nr):
        direction=np.zeros(nr);direction[a]=1.
        d,_=ctx.calculator._ground_state_directional_derivative_matrices(ctx.state,direction.reshape(-1,3),context=fixed);ds.append(d)
    ds=np.asarray(ds);transport,l=eri_transport.chart(ctx.state);u=transport.u
    _,dr,_=rotation_jet(ctx.state.coordinates_bohr,eri_transport.anchor.coordinates_bohr,second=False)
    du=np.array([transport.lift(d) for d in dr]);dl=np.array([solve_sylvester(l,l,d) for d in ds]);active=np.r_[np.arange(rank),no+np.arange(rank)];ca=c[:,active]
    cr=-.5*(c@(c.T@ds@c));y=u.T@l@ca;yr=(du.transpose(0,2,1)@l+u.T@dl)@ca+u.T@l@cr[:,:,active]
    gradient=np.zeros_like(ctx.state.coordinates_bohr)
    for i,(occ,triplet,focks) in enumerate(zip(obj.occupations,obj.triplets,obj.cache['diagonal'])):
        pm=np.array([(c*x)@c.T for x in occ]);pt=np.array([(c*x)@c.T for x in triplet])
        q=sum(2*(c.T@f@c)*o[None,:] for f,o in zip(focks[0],occ))-sum((c.T@f@c)*o[None,:] for f,o in zip(focks[1],triplet));weighted=c@(.5*(q+q.T))@c.T
        gradient+=ci[i]**2*(2*ctx.analytic_gradient(pm,.5*weighted)-ctx.analytic_gradient(pt,np.zeros_like(weighted)))
    for i in range(rank):
        for j in range(i):
            terms=[(2.,(i,rank+i,j,rank+j)),(-1.,(i,j,rank+i,rank+j))]
            _,g,_=quartic_jet(eri_transport.eri,y,terms);gradient+=(2*ci[i]*ci[j]*np.einsum('rij,ij->r',yr,g)).reshape(-1,3)
    return gradient
