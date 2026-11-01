"""Exact fixed-bra/common-core Schur contractions for canonical CID columns.

The regular transition density is independent of the canonical active
occupation. Project the remaining pair contractions once per fixed bra,
instead of reading the full AO pair tensor for every configuration.
No orbital, density, or integral singular value is discarded.
"""
import numpy as np
from hashlib import sha256
from ._nosi_slater import determinant


def spectral_cofactors(matrix):
    """First/second determinant derivatives without inverse singular values."""
    n=len(matrix);u,s,vh=np.linalg.svd(matrix);phase=determinant(u)*determinant(vh)
    first_weights=np.array([np.prod(np.delete(s,i)) for i in range(n)])
    weights=np.zeros((n,n))
    for i in range(n):
        for j in range(i):weights[i,j]=weights[j,i]=np.prod(np.delete(s,[i,j]))
    feature=np.einsum('ia,ja->aij',u,vh.T).reshape(n,n*n)
    term=(feature.T@weights@feature).reshape((n,)*4)
    return phase*(u*first_weights)@vh,phase*(term-term.swapaxes(1,3))


class UnusableCore(ValueError):
    pass


class FixedBra:
    def __init__(self,hamiltonian,bra,core,active):
        self.hamiltonian=hamiltonian;self.core=core;self.active=active
        self.nc=core.shape[1];self.sectors=[];features=[];offset=0;self.occupation_cache_hits=0;self.occupation_cache_misses=0
        for spin,a in enumerate(bra):
            if spin and np.array_equal(a,bra[0]):
                self.sectors.append(self.sectors[0]);continue
            u,s,vh=np.linalg.svd(a.T@core,full_matrices=True)
            if len(s) and s.min()<.1:raise UnusableCore('common occupied core is ill-conditioned')
            regular=a@u[:,:self.nc];null=a@u[:,self.nc:];r=null.shape[1]
            if not 1<=r<=12:raise UnusableCore('fixed-core accelerator requires one to twelve active electrons per spin')
            rotated=core@vh.T;inverse=rotated/s[None,:] if len(s) else rotated
            density=inverse@regular.T
            left=active-inverse@(regular.T@active)
            pair=np.einsum('pi,qj->ijpq',left,null).reshape(len(active.T)*r,-1)
            phase=determinant(u)*determinant(vh)
            j=density.ravel()@hamiltonian.j;jk=density.ravel()@hamiltonian.jk
            sector=dict(regular=density,null=null,left=left,r=r,prefactor=phase*float(np.prod(s)),
                        occupation_cache={},offset=offset,count=len(pair),one=float(np.trace(hamiltonian.h@density)),
                        same=float(jk@density.ravel()),j=j,jk=jk,features=pair)
            self.sectors.append(sector);features.append(pair);offset+=len(pair)
        features=np.concatenate(features)
        # Only simultaneous pair transposition is required, as in the
        # parent Hamiltonian contract; separate pair symmetry is unnecessary.
        self.integrals=(features@hamiltonian.j)@features.T
        self.one=features@hamiltonian.h.T.ravel()
        self.j_fields=[features@sector['j'] for sector in self.sectors]
        self.same_fields=[features@sector['jk'] for sector in self.sectors]
        self.regular_cross=float(self.sectors[0]['j']@self.sectors[1]['regular'].ravel())

    def _occupation(self,b,sector):
        if not np.allclose(b[:,:self.nc],self.core,atol=1e-12,rtol=0):raise UnusableCore('canonical determinant has a different ordered core')
        orbitals=b[:,self.nc:];projection=self.active.T@orbitals
        ids=np.argmax(abs(projection),axis=0)
        if not np.allclose(projection,np.eye(len(self.active.T))[:,ids],atol=1e-12,rtol=0):raise UnusableCore('ket active orbitals are not canonical columns')
        key=sha256(np.ascontiguousarray(orbitals).tobytes()).digest();cache=sector['occupation_cache']
        if key in cache:
            self.occupation_cache_hits+=1;return cache[key]
        self.occupation_cache_misses+=1
        block=sector['null'].T@orbitals;first,second=spectral_cofactors(block);r=sector['r']
        q=np.zeros((len(self.active.T),r));q[ids]=first.T
        vector=np.zeros(len(self.one));lo=sector['offset'];hi=lo+sector['count'];vector[lo:hi]=q.ravel()
        index=lo+np.arange(sector['count']).reshape(len(self.active.T),r)[ids].T.ravel()
        tensor=self.integrals[np.ix_(index,index)].reshape((r,)*4)
        null_two=.5*float(np.sum(second*tensor))
        value=(determinant(block),vector,null_two)
        if len(cache)>=16384:cache.pop(next(iter(cache)))
        cache[key]=value
        return value

    def element(self,ket):
        values=[self._occupation(b,sector) for b,sector in zip(ket,self.sectors)]
        total=0.
        for spin,(sector,(d,q,two)) in enumerate(zip(self.sectors,values)):
            other=values[1-spin][0]
            total+=other*(d*sector['one']+float(self.one@q)+.5*d*sector['same']+float(self.same_fields[spin]@q)+two)
        da,qa,_=values[0];db,qb,_=values[1]
        total+=da*db*self.regular_cross+da*float(self.j_fields[0]@qb)+db*float(self.j_fields[1]@qa)+float(qa@self.integrals@qb)
        prefactor=self.sectors[0]['prefactor']*self.sectors[1]['prefactor']
        return prefactor*da*db,prefactor*total


class FixedBraCSF:
    def __init__(self,hamiltonian,core,active):
        self.hamiltonian=hamiltonian;self.core=core;self.active=active;self.engines={}
    def element(self,bra,ket):
        overlap=energy=0.
        for x,a in bra:
            key=tuple(np.ascontiguousarray(part).tobytes() for part in a)
            if key not in self.engines:
                try:self.engines[key]=FixedBra(self.hamiltonian,a,self.core,self.active)
                except UnusableCore:self.engines[key]=None
            engine=self.engines[key]
            for y,b in ket:
                if engine is None:s,h=self.hamiltonian.element(a,b)
                else:
                    try:s,h=engine.element(b)
                    except UnusableCore:s,h=self.hamiltonian.element(a,b)
                overlap+=x*y*s;energy+=x*y*h
        return overlap,energy
