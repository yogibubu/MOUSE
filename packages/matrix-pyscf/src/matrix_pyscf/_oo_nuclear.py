"""Analytic nuclear response of a shared-orbital weighted energy surface.

The generalized energy-weighted density includes off-diagonal multipliers.
A common orbital CP equation replaces independent mixed/triplet CP solutions.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres


def _symmetric(a):return .5*(a+a.T)


def validate_hessian_model(mf):
    if (getattr(mf,'with_df',None) is not None or mf.mol.has_ecp()
        or getattr(mf,'with_solvent',None) is not None or getattr(mf,'with_x2c',None) is not None):
        raise NotImplementedError('OO nuclear Hessian requires full-integral all-electron molecular HF/DFT')
    if getattr(mf,'xc',None):
        from pyscf.dft import gen_grid,radi
        if mf._numint._xc_type(mf.xc) not in ('LDA','GGA'):
            raise NotImplementedError('OO nuclear Hessian supports LDA/GGA, including hybrids')
        if mf.grids.becke_scheme is not gen_grid.original_becke or mf.grids.radii_adjust not in (None,radi.becke_atomic_radii_adjust,radi.treutler_atomic_radii_adjust):
            raise NotImplementedError('OO nuclear Hessian requires the supported Becke partition')


def nuclear_response(surface,c,*,hessian=False,response_tolerance=1e-11):
    if hessian:
        for weight,mf,occ in surface.components:validate_hessian_model(mf)
    from pyscf.hessian.rhf import get_ovlp
    from ._oo_grid import grid_correction
    energy,g,action,focks,kernels,densities=surface.evaluate(c)
    if hessian and len(g) and np.max(abs(g))>1e-7:raise ValueError('analytic nuclear response requires orbital stationarity')
    cs=surface.spin_coefficients(c);mol=surface.components[0][1].mol;n=mol.nao_nr();m=mol.natm*3
    if cs.shape!=(2,n,n):raise NotImplementedError('OO nuclear Hessian requires a complete nonsingular AO orbital frame')
    inverse=cs[0]@cs[0].T;metric=surface.components[0][1].get_ovlp()
    if not np.allclose(inverse@metric,np.eye(n),atol=1e-8,rtol=0):raise ValueError('incomplete OO response metric frame')
    weighted=np.zeros((n,n));gradient=np.zeros((mol.natm,3));h1s=[];partial=np.zeros((m,m));grid_certificates=[]
    for (weight,mf,occ),f,dm in zip(surface.components,focks,densities):
        lagrangian=np.array([_symmetric(inverse@ff@dd) for ff,dd in zip(f,dm)])
        weighted+=weight*lagrangian.sum(axis=0)
        derivative=mf.nuc_grad_method()
        derivative.make_rdm1e=lambda *args,dme=lagrangian,**kwargs:dme
        if hasattr(derivative,'grid_response'):derivative.grid_response=True
        gradient+=weight*np.asarray(derivative.kernel(mo_energy=np.zeros_like(occ),mo_coeff=cs,mo_occ=occ))
        if not hessian:continue
        obj=mf.Hessian();zeros=np.zeros_like(occ)
        bare=obj.partial_hess_elec(zeros,cs,occ)+obj.hess_nuc()
        partial+=weight*bare.transpose(0,2,1,3).reshape(m,m)
        h1=np.asarray(obj.make_h1(cs,occ)).reshape(2,m,n,n)
        correction,df,certificate=grid_correction(mf,cs,occ)
        partial+=weight*correction;h1s.append(h1+df);grid_certificates.append(certificate)
    certificate=dict(orbital_residual=float(np.max(abs(g))) if len(g) else 0.,surface='Common-orbital multiplet sum' if surface.common else 'Unrestricted determinant',
                     off_diagonal_energy_weighted_density=True,finite_differences_in_production=False)
    if not hessian:return gradient,None,certificate
    s_aa,s_ab,s_a=get_ovlp(mol);s1=np.zeros((m,n,n));slices=mol.aoslice_by_atom()
    for atom,part in enumerate(slices):
        p0,p1=part[2:];s1[3*atom:3*atom+3,p0:p1]+=s_a[:,p0:p1]
        s1[3*atom:3*atom+3,:,p0:p1]+=s_a[:,p0:p1].transpose(0,2,1)
    # Second overlap derivative, contracted with the total symmetric multiplier.
    for atom,part in enumerate(slices):
        p0,p1=part[2:];a=slice(3*atom,3*atom+3)
        partial[a,a]-=2*np.einsum('xypq,pq->xy',s_aa[:,:,p0:p1],weighted[p0:p1])
        for other,otherpart in enumerate(slices):
            q0,q1=otherpart[2:];b=slice(3*other,3*other+3)
            partial[a,b]-=2*np.einsum('xypq,pq->xy',s_ab[:,:,p0:p1,q0:q1],weighted[p0:p1,q0:q1])
    dimension=surface.dimension
    operator=LinearOperator((dimension,dimension),matvec=action,dtype=float)
    response=np.empty((dimension,m));maximum=0.;raw=partial.copy()
    for y in range(m):
        sm=cs[0].T@s1[y]@cs[0]
        dc0= -.5*c@sm if surface.common else np.array([-.5*a@(a.T@s1[y]@a) for a in c])
        dcs0=surface.spin_coefficients(dc0);changes0=[];df0=[]
        for (weight,mf,occ),h1,kernel in zip(surface.components,h1s,kernels):
            dp=np.array([(a*o)@b.T for a,b,o in zip(dcs0,cs,occ)]);dp+=dp.transpose(0,2,1)
            changes0.append(dp);df0.append(h1[:,y]+kernel(dp))
        rhs=surface.residual_derivative(c,focks,dc0,df0)
        if dimension:
            step,info=gmres(operator,-rhs,rtol=response_tolerance,atol=1e-12,restart=min(100,dimension),maxiter=20)
            error=float(np.max(abs(action(step)+rhs)));rhs_scale=max(1.,float(np.max(abs(rhs))))
        else:step=np.empty(0);info=0;error=0.;rhs_scale=1.
        maximum=max(maximum,error)
        if error>max(1e-10,10*response_tolerance*rhs_scale):raise RuntimeError(('orbital nuclear response not converged',y,error,info))
        response[:,y]=step;dc=dc0+c@surface.rotation(step);dcs=surface.spin_coefficients(dc)
        dw=np.zeros((n,n))
        for (weight,mf,occ),f,dm,h1,kernel in zip(surface.components,focks,densities,h1s,kernels):
            dp=np.array([(a*o)@b.T for a,b,o in zip(dcs,cs,occ)]);dp+=dp.transpose(0,2,1)
            df=h1[:,y]+kernel(dp)
            for spin in range(2):
                dw+=weight*_symmetric(-inverse@s1[y]@inverse@f[spin]@dm[spin]+inverse@df[spin]@dm[spin]+inverse@f[spin]@dp[spin])
            raw[:,y]+=weight*np.einsum('sxpq,spq->x',h1,dp)
        raw[:,y]-=np.einsum('xpq,pq->x',s1,dw)
    asymmetry=float(np.max(abs(raw-raw.T)))
    if asymmetry>2e-6:raise RuntimeError(('analytic nuclear Hessian is not symmetric',asymmetry))
    certificate.update(orbital_response_maximum_residual=maximum,raw_Hessian_asymmetry=asymmetry,
                       coupled_common_response=surface.common,grid_response=grid_certificates)
    return gradient,.5*(raw+raw.T),certificate
