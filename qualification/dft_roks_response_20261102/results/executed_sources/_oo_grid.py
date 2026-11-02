"""Analytic moving-Becke-grid correction to fixed-grid XC derivatives.

Second-order forward differentiation acts on partition weights. Gaussian AO
spatial derivatives through order three and LibXC vxc/fxc supply the density
chain rule. There are no displaced geometries or finite steps in this module.
"""
import numpy as np


class _Jet:
    __array_priority__=1000
    def __init__(self,v,g,h):self.v=np.asarray(v);self.g=g;self.h=h
    def __add__(a,b):
        if not isinstance(b,_Jet):return _Jet(a.v+b,a.g,a.h)
        return _Jet(a.v+b.v,a.g+b.g,a.h+b.h)
    __radd__=__add__
    def __neg__(a):return _Jet(-a.v,-a.g,-a.h)
    def __sub__(a,b):return a+-b
    def __rsub__(a,b):return -a+b
    def __mul__(a,b):
        if not isinstance(b,_Jet):return _Jet(a.v*b,a.g*b,a.h*b)
        return _Jet(a.v*b.v,a.g*b.v+a.v*b.g,a.h*b.v+a.v*b.h+
                    a.g[:,None]*b.g[None,:]+b.g[:,None]*a.g[None,:])
    __rmul__=__mul__
    def __pow__(a,p):
        return _Jet(a.v**p,p*a.v**(p-1)*a.g,p*a.v**(p-1)*a.h+
                    p*(p-1)*a.v**(p-2)*a.g[:,None]*a.g[None,:])
    def __truediv__(a,b):return a*b**-1


def _distance(vector,factors):
    # factors[x,k] = d vector[k] / d R[x].
    r=np.linalg.norm(vector,axis=1);unit=vector/r[:,None]
    g=np.einsum('xk,gk->xg',factors,unit)
    second=(np.eye(3)[None]-unit[:,:,None]*unit[:,None,:])/r[:,None,None]
    h=np.einsum('xk,gkl,yl->xyg',factors,second,factors)
    return _Jet(r,g,h)


def _weights(mol,grids,center,coordinates,volume):
    from pyscf.dft import gen_grid,radi
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('analytic grid Hessian currently requires original Becke partition')
    n=mol.natm;m=3*n;g=len(volume);positions=mol.atom_coords()
    adjust=None
    if grids.radii_adjust is not None and grids.atomic_radii is not None:
        if grids.radii_adjust not in (radi.treutler_atomic_radii_adjust,radi.becke_atomic_radii_adjust):
            raise NotImplementedError('custom radii adjustment has no analytic grid Hessian')
        adjust=grids.radii_adjust(mol,grids.atomic_radii)
    distances=[]
    for atom in range(n):
        factors=np.zeros((m,3));factors[3*center:3*center+3]+=np.eye(3);factors[3*atom:3*atom+3]-=np.eye(3)
        distances.append(_distance(coordinates-positions[atom],factors))
    partitions=[_Jet(np.ones(g),np.zeros((m,g)),np.zeros((m,m,g))) for a in range(n)]
    for i in range(n):
        for j in range(i):
            factors=np.zeros((m,3));factors[3*i:3*i+3]=np.eye(3);factors[3*j:3*j+3]=-np.eye(3)
            separation=_distance((positions[i]-positions[j])[None],factors)
            value=(distances[i]-distances[j])/separation
            if adjust is not None:value=value+float(adjust(i,j,0.))*(1-value*value)
            for unused in range(3):value=.5*(3*value-value*value*value)
            partitions[i]=partitions[i]*(.5*(1-value));partitions[j]=partitions[j]*(.5*(1+value))
    total=partitions[0]
    for p in partitions[1:]:total=total+p
    return volume*(partitions[center]/total)


_EXPONENTS=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2),
            (3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),(0,3,0),(0,2,1),(0,1,2),(0,0,3)]
_INDEX={e:i for i,e in enumerate(_EXPONENTS)}


def _products(ao,density):
    return np.einsum('spq,kgq->skgp',density,ao,optimize=True)


def _rho_products(a,products):
    result=np.empty((2,4,len(a[0])))
    result[:,0]=np.einsum('gp,sgp->sg',a[0],products[:,0])
    for k in range(1,4):
        result[:,k]=np.einsum('gp,sgp->sg',a[k],products[:,0])+np.einsum('gp,sgp->sg',a[0],products[:,k])
    return result


def _potential(coeff,a,b):
    result=np.einsum('sg,gp,gq->spq',coeff[:,0],a[0],b[0],optimize=True)
    for k in range(1,coeff.shape[1]):
        result+=np.einsum('sg,gp,gq->spq',coeff[:,k],a[k],b[0],optimize=True)
        result+=np.einsum('sg,gp,gq->spq',coeff[:,k],a[0],b[k],optimize=True)
    return result


def grid_correction(mf,coefficients,occupations,block_size=512):
    """Return exact moving-minus-fixed grid E_RR and F_R at fixed AO density."""
    if not getattr(mf,'xc',None):
        n=mf.mol.nao_nr();m=3*mf.mol.natm
        return np.zeros((m,m)),np.zeros((2,m,n,n)),dict(required=False)
    from pyscf.dft import gen_grid
    mol=mf.mol;grids=mf.grids;ni=mf._numint;xctype=ni._xc_type(mf.xc)
    if xctype not in ('LDA','GGA'):
        raise NotImplementedError('moving-grid OO Hessian currently supports LDA and GGA functionals')
    atomic=grids.gen_atomic_grids(mol,grids.atom_grid,radi_method=grids.radi_method,level=grids.level,prune=grids.prune)
    m=3*mol.natm;n=mol.nao_nr();hessian=np.zeros((m,m));fock1=np.zeros((2,m,n,n));positions=mol.atom_coords()
    density=np.array([(c*o)@c.T for c,o in zip(coefficients,occupations)])
    aoatoms=np.empty(n,dtype=int)
    for atom,part in enumerate(mol.aoslice_by_atom()):aoatoms[part[2]:part[3]]=atom
    block_size=min(block_size,max(8,2000000//max(1,m*m*mol.natm)))
    points=0
    for center in range(mol.natm):
        coords,volumes=atomic[mol.atom_symbol(center)]
        for lo in range(0,len(coords),block_size):
            coordinates=coords[lo:lo+block_size]+positions[center];volume=volumes[lo:lo+block_size]
            weights=_weights(mol,grids,center,coordinates,volume);points+=len(volume)
            ao=ni.eval_ao(mol,coordinates,deriv=3);phi=ao[:4];products=_products(phi,density);rho=_rho_products(phi,products)
            width=1 if xctype=='LDA' else 4
            exc,vxc,fxc,*unused=ni.eval_xc_eff(mf.xc,rho[:,:width],deriv=2,xctype=xctype)
            e0=exc*(rho[0,0]+rho[1,0]);vxc=np.asarray(vxc);fxc=np.asarray(fxc)
            # Retain all density-gradient channels in AO contractions for GGA.
            for moving,sign in [(False,-1.),(True,1.)]:
                first=[];first_products=[];rho1=[];factors=[]
                for x in range(m):
                    atom,cart=divmod(x,3);factor=(float(center==atom) if moving else 0.)-(aoatoms==atom).astype(float)
                    factors.append(factor)
                    indices=[]
                    for exponent in _EXPONENTS[:4]:
                        target=list(exponent);target[cart]+=1;indices.append(_INDEX[tuple(target)])
                    phix=ao[indices]*factor[None,None,:];first.append(phix);first_products.append(_products(phix,density))
                    rho1.append(2*_rho_products(phix,products)[:,:width])
                    wx=weights.g[x] if moving else 0.
                    dv=np.einsum('sktlg,tlg->skg',fxc,rho1[-1],optimize=True)
                    fock1[:,x]+=sign*(_potential(wx*vxc+weights.v*dv,phi,phi)+
                        _potential(weights.v*vxc,phix,phi)+_potential(weights.v*vxc,phi,phix))
                for x in range(m):
                    for y in range(x+1):
                        ix=x%3;iy=y%3;indices=[]
                        for exponent in _EXPONENTS[:4]:
                            target=list(exponent);target[ix]+=1;target[iy]+=1;indices.append(_INDEX[tuple(target)])
                        phixy=ao[indices]*(factors[x]*factors[y])[None,None,:]
                        rho2=2*(_rho_products(phixy,products)+_rho_products(first[x],first_products[y]))[:,:width]
                        value=np.sum(weights.v*(np.einsum('skg,skg->g',vxc,rho2)+
                            np.einsum('skg,sktlg,tlg->g',rho1[x],fxc,rho1[y],optimize=True)))
                        if moving:
                            value+=np.sum(weights.h[x,y]*e0+weights.g[x]*np.einsum('skg,skg->g',vxc,rho1[y])+
                                          weights.g[y]*np.einsum('skg,skg->g',vxc,rho1[x]))
                        hessian[x,y]+=sign*value
    hessian=np.tril(hessian)+np.tril(hessian,-1).T
    return hessian,fock1,dict(required=True,scheme='Analytic moving Becke partition, AO derivatives through order 3, LibXC vxc/fxc',
                             points=points,finite_differences=False)
