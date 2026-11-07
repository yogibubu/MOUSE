"""Common analytic orbital surface for determinant and multiplet-sum DFT."""
from types import MethodType
import numpy as np
from scipy.linalg import expm


class OrbitalSurface:
    def __init__(self, mean_field, coefficients, occupations, common=False):
        self.statistics=dict(Fock_builds=0,response_builds=0,Jacobian_actions=0,range_integral_builds=0,range_integral_hits=0,range_integral_bytes=0)
        self.common=common
        self.c=np.asarray(coefficients,dtype=float)
        self.occupations=np.asarray(occupations,dtype=float)
        mixed=mean_field.copy()
        mixed.mol=mean_field.mol.copy()
        if getattr(mixed,'xc',None):
            mixed.grids=mean_field.grids.copy()
            mixed.grids.coords=None; mixed.grids.weights=None
            mixed.small_rho_cutoff=0.
        self.components=[(1.,mixed,self.occupations)]
        if common:
            triplet=mean_field.copy();triplet.mol=mean_field.mol.copy();triplet.mol.spin=2
            if getattr(triplet,'xc',None):
                triplet.grids=mean_field.grids.copy()
                triplet.grids.coords=None;triplet.grids.weights=None;triplet.small_rho_cutoff=0.
            t=np.array([np.maximum(*occupations),np.minimum(*occupations)])
            self.components=[(2.,mixed,self.occupations),(-1.,triplet,t)]
        self._range_integrals={}
        for weight,component,occ in self.components:self._cache_range_exchange(component)
        n=self.c.shape[-1];left,right=np.triu_indices(n,1)
        if common:
            labels=np.vstack([o for w,m,o in self.components]);keep=np.any(labels[:,left]!=labels[:,right],axis=0)
            self.pairs=[(left[keep],right[keep])]
        else:
            self.pairs=[(left[o[left]!=o[right]],right[o[left]!=o[right]]) for o in occupations]
        self.dimension=sum(len(i) for i,j in self.pairs)

    def _cache_range_exchange(self,component):
        """Reuse exact range-separated AO ERIs within a bounded memory budget.

        Native PySCF caches full Coulomb ERIs but rebuilds range-separated ERIs
        on each Krylov action. The density-independent integrals can be shared
        by mixed/triplet components without changing any J/K contraction.
        """
        from pyscf.scf.hf import dot_eri_dm
        native=type(component).get_jk;owner_mol=component.mol
        n=owner_mol.nao_nr();pairs=n*(n+1)//2;needed=8*pairs*(pairs+1)//2
        budget=max(0.,float(component.max_memory))*1e6*.25
        if not getattr(component,'xc',None) or needed>budget or 'get_jk' in component.__dict__:return
        cache=self._range_integrals;statistics=self.statistics
        def get_jk(instance,mol=None,dm=None,hermi=1,with_j=True,with_k=True,omega=None):
            target=instance.mol if mol is None else mol
            if not omega or target is not owner_mol:
                return native(instance,mol,dm,hermi,with_j,with_k,omega)
            key=float(omega)
            if key not in cache:
                if sum(value.nbytes for value in cache.values())+needed>budget:
                    return native(instance,mol,dm,hermi,with_j,with_k,omega)
                with target.with_range_coulomb(key):cache[key]=target.intor('int2e',aosym='s8')
                statistics['range_integral_builds']+=1
                statistics['range_integral_bytes']+=cache[key].nbytes
            else:statistics['range_integral_hits']+=1
            return dot_eri_dm(cache[key],instance.make_rdm1() if dm is None else dm,hermi,with_j,with_k)
        component.get_jk=MethodType(get_jk,component)

    def spin_coefficients(self,c):
        return np.array([c,c]) if self.common else c

    def rotation(self,vector):
        n=self.c.shape[-1];rotations=[];offset=0
        for i,j in self.pairs:
            k=np.zeros((n,n));k[i,j]=vector[offset:offset+len(i)];k[j,i]=-k[i,j]
            rotations.append(k);offset+=len(i)
        if offset!=len(vector):raise ValueError('incompatible orbital response direction')
        return rotations[0] if self.common else np.array(rotations)

    def rotate(self,c,vector):
        k=self.rotation(vector)
        return c@expm(k) if self.common else np.array([a@expm(b) for a,b in zip(c,k)])

    def residual(self,c,focks):
        cs=self.spin_coefficients(c);total=[]
        for channel,(left,right) in enumerate(self.pairs):
            value=np.zeros(len(left))
            for (weight,mf,occ),f in zip(self.components,focks):
                for spin in range(2):
                    if not self.common and spin!=channel:continue
                    mo=cs[spin].T@f[spin]@cs[spin]
                    value+=2*weight*mo[left,right]*(occ[spin,right]-occ[spin,left])
            total.append(value)
        return np.concatenate(total)

    def residual_derivative(self,c,focks,dc,dfocks):
        cs=self.spin_coefficients(c);dcs=self.spin_coefficients(dc);total=[]
        for channel,(left,right) in enumerate(self.pairs):
            value=np.zeros(len(left))
            for (weight,mf,occ),f,df in zip(self.components,focks,dfocks):
                for spin in range(2):
                    if not self.common and spin!=channel:continue
                    mo=dcs[spin].T@f[spin]@cs[spin]+cs[spin].T@f[spin]@dcs[spin]+cs[spin].T@df[spin]@cs[spin]
                    value+=2*weight*mo[left,right]*(occ[spin,right]-occ[spin,left])
            total.append(value)
        return np.concatenate(total)

    def evaluate(self,c,*,linearize=True):
        cs=self.spin_coefficients(c);energy=0.;focks=[];kernels=[];densities=[]
        for weight,mf,occ in self.components:
            mf.mo_coeff=cs;mf.mo_occ=occ
            dm=np.array([(a*o)@a.T for a,o in zip(cs,occ)])
            v=mf.get_veff(dm=dm);f=np.asarray(mf.get_hcore())+v
            energy+=weight*float(mf.energy_tot(dm=dm,vhf=v));focks.append(f);densities.append(dm)
            self.statistics["Fock_builds"]+=1
        gradient=self.residual(c,focks)
        data=(energy,gradient,None,focks,None,densities)
        return self.linearize(c,data) if linearize else data

    def linearize(self,c,data):
        energy,gradient,unused,focks,unused_kernels,densities=data
        cs=self.spin_coefficients(c);kernels=[]
        for weight,mf,occ in self.components:
            mf.mo_coeff=cs;mf.mo_occ=occ
            kernels.append(mf.gen_response(cs,occ,hermi=1,with_nlc=False))
            self.statistics['response_builds']+=1
        q=[];diagonal=[]
        for channel,(left,right) in enumerate(self.pairs):
            matrix=np.zeros((c.shape[-1],c.shape[-1]));d=np.zeros(len(left))
            for (weight,mf,occ),f in zip(self.components,focks):
                for spin in range(2):
                    if not self.common and channel!=spin:continue
                    mo=cs[spin].T@f[spin]@cs[spin]
                    matrix+=2*weight*mo*occ[spin][None,:]
                    d+=2*weight*(occ[spin,right]-occ[spin,left])*(np.diag(mo)[left]-np.diag(mo)[right])
            q.append(matrix);diagonal.append(d)
        def action(vector):
            self.statistics['Jacobian_actions']+=1
            dc=c@self.rotation(vector);dcs=self.spin_coefficients(dc);changes=[]
            for (weight,mf,occ),kernel in zip(self.components,kernels):
                dp=np.array([(a*o)@b.T for a,b,o in zip(dcs,cs,occ)]);dp+=dp.transpose(0,2,1)
                changes.append(kernel(dp))
            return self.residual_derivative(c,focks,dc,changes)
        def connection(vector):
            rotations=self.rotation(vector)
            if self.common:rotations=[rotations]
            values=[]
            for k,matrix,(left,right) in zip(rotations,q,self.pairs):
                commutator=k@matrix-matrix@k
                values.append(commutator[left,right]-commutator[right,left])
            return np.concatenate(values)
        # The moving-frame Jacobian is not symmetric away from stationarity.
        # Its skew part is fixed by the orbital-gradient connection.
        action.adjoint=lambda vector:action(vector)+connection(vector)
        action.symmetric=lambda vector:action(vector)+.5*connection(vector)
        action.preconditioner_diagonal=np.concatenate(diagonal)
        return energy,gradient,action,focks,kernels,densities
