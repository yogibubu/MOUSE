"""Spin-pure 50/50 ROKS on a fixed relative active-orbital manifold.

The assigned vertical parents define two squared cross-parent open-orbital
cosines. Their values remain fixed along the PES. No energy term is added:
E=1/2(E_full+E_spectral). At the vertical anchor both separately stationary
parents solve this constrained problem with zero multipliers, preserving all
completed vertical energies. Nuclear response solves the coupled KKT system.
"""
from dataclasses import dataclass,replace,fields
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from .shared_pair_roks import (_SharedPairObjective,_contexts,_coupled_geometry_blocks,
                                SharedPairROKSResult)
from .deltascf import project_delta_orbitals,delta_cross_overlap
from .roks import ROKSSingletResult


def relative_active_targets(components):
    if len(components)!=2:raise ValueError('two assigned parent CSFs are required')
    a,b=components
    if (not np.array_equal(a.occupations,b.occupations)
            or not np.allclose(a.overlap,b.overlap,atol=1e-10,rtol=0)
            or not np.allclose(a.reference_state.coordinates_bohr,b.reference_state.coordinates_bohr,atol=1e-10,rtol=0)):
        raise ValueError('relative targets require the same vertical geometry and occupations')
    opens=np.flatnonzero(a.occupations[0]!=a.occupations[1])
    return np.array([(a.common_coefficients[:,k]@a.overlap@b.common_coefficients[:,k])**2 for k in opens])


class _RelativePairObjective:
    def __init__(self,contexts,occupations,target):
        target=np.asarray(target,dtype=float)
        if target.shape!=(2,) or not np.all(np.isfinite(target)) or np.any(target<=0) or np.any(target>1+1e-10):
            raise ValueError('relative active targets must be two positive squared overlaps at most one')
        self.target=np.minimum(target,1.)
        self.opens=np.flatnonzero(occupations[0]!=occupations[1])
        # At unit overlap the cosine constraint has zero Jacobian. Eliminate
        # that common orbital exactly instead of regularizing a singular KKT.
        shared=tuple(self.opens[self.target>=1-1e-10])
        self.base=_SharedPairObjective(contexts,occupations,shared)
        self.constrained=np.flatnonzero(self.target<1-1e-10)
        self.dimension=self.base.dimension

    def retract(self,point,step):return self.base.retract(point,step)

    def evaluate(self,point,multipliers,curvature=True):
        energy,g,h=self.base.evaluate(point,curvature=curvature)
        multipliers=np.asarray(multipliers)
        if multipliers.shape!=(len(self.constrained),):raise ValueError('wrong number of relative-state multipliers')
        h=None if h is None or not curvature else h.copy()
        maps=self.base.cache['maps'];hs=self.base.cache['hs'];generators=self.base.generators
        mapped=[np.einsum('aj,aik->jik',t,generators,optimize=True) for t in maps]
        ca,cb=point;s=self.base.s;constraints=[];jac=[]
        for index,mu in zip(self.constrained,multipliers):
            k=self.opens[index];r=float(ca[:,k]@s@cb[:,k])
            za,zb=ca.T@s@cb[:,k],cb.T@s@ca[:,k]
            cols=[v[:,:,k] for v in mapped]
            dr=cols[0]@za+cols[1]@zb
            constraints.append(r*r-self.target[index]);jac.append(2*r*dr)
            if curvature and mu!=0.:
                second=np.zeros_like(h)
                for z,v,col in zip([za,zb],mapped,cols):
                    zg=np.einsum('i,aij->aj',z,v);ss=zg@col.T
                    second+=.5*(ss+ss.T)
                cross=(ca@cols[0].T).T@s@(cb@cols[1].T)
                second+=cross+cross.T
                for model,z in enumerate([za,zb]):
                    gg=np.zeros_like(s);gr=generators[:,:,k]@z
                    gg[self.base.left,self.base.right]=gr;gg[self.base.right,self.base.left]=-gr
                    comm=gg@hs[model]-hs[model]@gg
                    correction=.5*comm[:,self.base.left[self.base.private],self.base.right[self.base.private]]
                    columns=self.base.ns+model*self.base.np+np.arange(self.base.np)
                    second[np.ix_(np.arange(self.base.ns),columns)]+=correction
                    second[np.ix_(columns,np.arange(self.base.ns))]+=correction.T
                h+=mu*(2*np.outer(dr,dr)+2*r*second)
        jac=np.asarray(jac).reshape(len(self.constrained),self.dimension)
        residual=np.concatenate([g+jac.T@multipliers,constraints])
        kkt=None if h is None else np.block([[h,jac.T],[jac,np.zeros((len(jac),len(jac)))]])
        return energy,residual,kkt,jac


def _certificate(kkt,jac,dimension):
    if len(jac) and np.min(np.linalg.svd(jac,compute_uv=False))<1e-10:
        raise ValueError('relative-state constraint Jacobian is rank deficient')
    z=null_space(jac);values=np.linalg.eigvalsh(z.T@kkt[:dimension,:dimension]@z)
    return int(np.sum(values<-1e-7)),float(np.min(abs(values)))


@dataclass(frozen=True)
class RelativePairROKSResult:
    components: tuple
    target_squared_overlaps: np.ndarray
    multipliers: np.ndarray
    orbital_residual: float
    orbital_saddle_index: int
    minimum_absolute_orbital_curvature_hartree: float
    evaluations: int
    gradient_hartree_per_bohr: np.ndarray | None = None
    converged: bool = True

    def __post_init__(self):
        if len(self.components)!=2 or not all(isinstance(r,ROKSSingletResult) for r in self.components):
            raise ValueError('relative ROKS needs two parent CSF descriptors')
        a,b=self.components
        if (not np.array_equal(a.occupations,b.occupations) or a.reference_state is None or b.reference_state is None
                or not np.allclose(a.overlap,b.overlap,atol=1e-10,rtol=0)
                or not np.array_equal(a.reference_state.atomic_numbers,b.reference_state.atomic_numbers)
                or not np.allclose(a.reference_state.coordinates_bohr,b.reference_state.coordinates_bohr,atol=1e-10,rtol=0)
                or a.reference_state.model_id!=b.reference_state.model_id or a.reference_state.charge!=b.reference_state.charge
                or not np.array_equal(a.core_electrons,b.core_electrons)):
            raise ValueError('relative parents have different molecular states')
        fa,fb=a.spin_frame_clusters,b.spin_frame_clusters
        if (fa is None)!=(fb is None) or fa is not None and not np.array_equal(fa,fb):raise ValueError('relative parent frames differ')
        target=np.array(self.target_squared_overlaps,dtype=float,copy=True)
        mu=np.array(self.multipliers,dtype=float,copy=True)
        if target.shape!=(2,) or not np.all(np.isfinite(target)) or np.any(target<=0) or np.any(target>1+1e-10):raise ValueError('invalid relative targets')
        if mu.shape!=(np.sum(target<1-1e-10),) or not np.all(np.isfinite(mu)):raise ValueError('invalid relative multipliers')
        if np.max(abs(relative_active_targets(self.components)-target))>1e-8:raise ValueError('relative CSFs violate their frozen state constraint')
        if (not self.converged or not np.isfinite(self.orbital_residual) or not 0<=self.orbital_residual<=1e-8
                or self.orbital_saddle_index is None or self.orbital_saddle_index<0
                or not np.isfinite(self.minimum_absolute_orbital_curvature_hartree)
                or self.minimum_absolute_orbital_curvature_hartree<=1e-7):raise ValueError('unresolved relative-state KKT certificate')
        target.setflags(write=False);mu.setflags(write=False)
        object.__setattr__(self,'target_squared_overlaps',target);object.__setattr__(self,'multipliers',mu)
        if self.gradient_hartree_per_bohr is not None:
            g=np.array(self.gradient_hartree_per_bohr,copy=True)
            if g.shape!=a.reference_state.coordinates_bohr.shape or not np.all(np.isfinite(g)):raise ValueError('invalid relative nuclear gradient')
            g.setflags(write=False);object.__setattr__(self,'gradient_hartree_per_bohr',g)

    @property
    def coefficients(self):return np.array([r.common_coefficients for r in self.components])
    @property
    def occupations(self):return self.components[0].occupations
    @property
    def energy_hartree(self):return .5*sum(r.energy_hartree for r in self.components)
    @property
    def spin_squared(self):return 0.
    @property
    def component_unconstrained_residuals(self):return tuple(r.orbital_residual for r in self.components)
    @property
    def component_csf_overlap(self):return SharedPairROKSResult.component_csf_overlap.fget(self)
    @property
    def component_active_pair_overlap(self):return SharedPairROKSResult.component_active_pair_overlap.fget(self)


def evaluate_relative_pair_roks(calculator,coordinates_angstrom,responses,*,reference_result=None,
        initial_components=None,anchor_components=None,target_squared_overlaps=None,gradient=False,
        orbital_tolerance=1e-10,maximum_iterations=100,maximum_step=.1,minimum_state_overlap=.7,checkpoint=None,initial_multipliers=None):
    """Continue stationary constrained CSFs; target cosines are never refitted.

    A first call may tighten the constraint from the retained parents to the
    original assigned parents by numerical continuation. A warm call solves
    the SAME fixed constraint directly. No homotopy parameter enters energy.
    """
    if reference_result is not None and (not isinstance(reference_result,RelativePairROKSResult) or initial_components is not None):
        raise ValueError('provide one retained relative state or initial parent states')
    if maximum_iterations<1 or not all(np.isfinite(x) and x>0 for x in [orbital_tolerance,maximum_step]) or not 0<=minimum_state_overlap<=1:
        raise ValueError('invalid relative ROKS controls')
    parents=initial_components if reference_result is None else reference_result.components
    if parents is None:raise ValueError('relative ROKS requires assigned parent orbitals')
    target=(relative_active_targets(anchor_components or parents) if reference_result is None else reference_result.target_squared_overlaps)
    if target_squared_overlaps is not None:
        if reference_result is not None and not np.array_equal(target_squared_overlaps,target):raise ValueError('cannot change frozen relative targets on continuation')
        target=np.asarray(target_squared_overlaps,dtype=float)
    contexts=_contexts(calculator,coordinates_angstrom,responses,parents)
    objective=_RelativePairObjective(contexts,parents[0].occupations,target)
    point=objective.base.seed([project_delta_orbitals(parent,ctx)[0] for parent,ctx in zip(parents,contexts)])
    mu=np.zeros(len(objective.constrained)) if reference_result is None else reference_result.multipliers.copy()
    if initial_multipliers is not None:
        if reference_result is not None:raise ValueError('initial multipliers require an initial state')
        mu=np.asarray(initial_multipliers,dtype=float).copy()
        if mu.shape!=(len(objective.constrained),) or not np.all(np.isfinite(mu)):raise ValueError('invalid initial multipliers')
    initial=np.array([(point[0][:,k]@objective.base.s@point[1][:,k])**2 for k in objective.opens])
    fraction=1. if reference_result is not None or np.max(abs(initial-target))<1e-12 else 0.;advance=.1;evaluations=0
    while True:
        trial_fraction=1. if reference_result is not None else min(1.,fraction+advance)
        objective.target=initial+trial_fraction*(target-initial);old=point.copy();oldmu=mu.copy();converged=False
        for iteration in range(maximum_iterations+1):
            e,res,kkt,jac=objective.evaluate(point,mu);evaluations+=1;error=float(np.max(abs(res)))
            if checkpoint is not None:checkpoint(dict(coefficients=point,multipliers=mu,target_squared_overlaps=target,
                fraction=trial_fraction,iteration=iteration,residual=error,energy=e))
            if error<=orbital_tolerance:converged=True;break
            step=np.linalg.lstsq(kkt,-res,rcond=1e-12)[0];norm=np.linalg.norm(step[:objective.dimension])
            if norm>maximum_step:step*=maximum_step/norm
            for scale in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]:
                trial=objective.retract(point,scale*step[:objective.dimension]);mt=mu+scale*step[objective.dimension:]
                _,rr,_,_=objective.evaluate(trial,mt,curvature=False)
                if np.linalg.norm(rr)<np.linalg.norm(res):point,mu=trial,mt;break
            else:break
        if not converged:
            if reference_result is not None:raise RuntimeError(f'relative ROKS continuation residual {error:.3e}')
            point,mu=old,oldmu;advance*=.5
            if advance<1e-5:raise RuntimeError(f'relative ROKS initialization needs branch handling: {error:.3e}')
            continue
        fraction=trial_fraction
        if fraction>=1.-1e-12:break
        advance=min(.2,advance*1.3)
    objective.target=target;e,res,kkt,jac=objective.evaluate(point,mu)
    index,minimum=_certificate(kkt,jac,objective.dimension);descriptors=[]
    for c,ctx,ref,row in zip(point,contexts,responses,objective.base.cache['values']):
        descriptors.append(ROKSSingletResult(c,objective.base.occ,objective.base.triplet,ctx.state.overlap,row['focks'][0],row['focks'][1],
            row['energy'],float(np.max(abs(row['g']))),evaluations,False,reference_state=ctx.state,
            core_electrons=np.asarray(ctx.state.atomic_numbers)-np.asarray(calculator._valence),
            spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=getattr(ref,'frame_clusters',None)))
    result=RelativePairROKSResult(tuple(descriptors),target,mu,float(np.max(abs(res))),index,minimum,evaluations)
    from matrix_apoc import follow_spin_adapted_singlet
    for old,new,ctx in zip(parents,result.components,contexts):
        follow_spin_adapted_singlet(old.to_apoc(),(new.to_apoc(),),delta_cross_overlap(old.reference_state,ctx),minimum_overlap=minimum_state_overlap)
    if gradient:
        g=np.zeros_like(contexts[0].state.coordinates_bohr)
        for c,r,ctx in zip(point,result.components,contexts):
            q=sum(2*(c.T@f@c)*o[None,:] for f,o in zip(r.fock_mixed,r.occupations))-sum((c.T@f@c)*o[None,:] for f,o in zip(r.fock_triplet,r.triplet_occupations))
            weighted=c@(.5*(q+q.T))@c.T
            g+=.5*(2*ctx.analytic_gradient(r.densities(),.5*weighted)-ctx.analytic_gradient(r.densities(triplet=True),np.zeros_like(weighted)))
        result=replace(result,gradient_hartree_per_bohr=g)
    return result


@dataclass(frozen=True)
class RelativePairROKSHessian:
    reference: RelativePairROKSResult
    hessian_hartree_per_bohr2: np.ndarray
    frozen_orbital_hessian_hartree_per_bohr2: np.ndarray
    response_hessian_hartree_per_bohr2: np.ndarray
    orbital_kkt_hessian_hartree: np.ndarray
    orbital_nuclear_coupling: np.ndarray
    orbital_response: np.ndarray
    maximum_response_residual: float
    raw_symmetry_error_hartree_per_bohr2: float
    minimum_absolute_orbital_curvature_hartree: float
    method: str = 'analytic_relative_active_pair_kkt_schur_complement'


def evaluate_relative_pair_roks_analytic_hessian(calculator,coordinates_angstrom,responses,*,reference_result):
    from .calculator import ANGSTROM_TO_BOHR
    result=reference_result
    if not isinstance(result,RelativePairROKSResult) or not np.allclose(result.components[0].reference_state.coordinates_bohr,
            np.asarray(coordinates_angstrom)*ANGSTROM_TO_BOHR,atol=1e-10,rtol=0):raise ValueError('relative Hessian requires a matching stationary KKT state')
    contexts=_contexts(calculator,coordinates_angstrom,responses,result.components)
    objective=_RelativePairObjective(contexts,result.occupations,result.target_squared_overlaps)
    _,res,kkt,jac=objective.evaluate(result.coefficients,result.multipliers)
    if np.max(abs(res))>1e-8:raise ValueError('relative KKT certificate does not match the current functional')
    _,minimum=_certificate(kkt,jac,objective.dimension)
    hrr,hkk,b=_coupled_geometry_blocks(contexts,result.components,objective.base)
    if np.max(abs(hkk-objective.base.cache['h']))>1e-8:raise ValueError('relative orbital/nuclear chart mismatch')
    # Common metric-preserving AO transport conserves ALL cross-parent
    # overlaps at every orbital point. The explicit RR and KR constraint
    # jets vanish in that chart; their KK jets remain in the Lagrangian KKT.
    augmented=np.vstack([b,np.zeros((len(jac),b.shape[1]))]);response=np.linalg.solve(kkt,-augmented)
    correction=augmented.T@response;total=hrr+correction
    return RelativePairROKSHessian(result,.5*(total+total.T),hrr,correction,kkt,augmented,response,
        float(np.max(abs(kkt@response+augmented))),float(np.max(abs(total-total.T))),minimum)


def save_relative_pair_roks(path,result,*,request_hash):
    """Atomic joint KKT checkpoint; parent residuals remain unconstrained."""
    if not isinstance(result,RelativePairROKSResult):raise TypeError('expected a relative-pair ROKS result')
    data=dict(schema='mouse.relative_pair_roks.v1',request_hash=request_hash,
        target_squared_overlaps=result.target_squared_overlaps,multipliers=result.multipliers,
        orbital_residual=result.orbital_residual,orbital_index=result.orbital_saddle_index,
        minimum_curvature=result.minimum_absolute_orbital_curvature_hartree,evaluations=result.evaluations)
    if result.gradient_hartree_per_bohr is not None:data['gradient']=result.gradient_hartree_per_bohr
    for i,parent in enumerate(result.components):
        prefix=f'parent_{i}_'
        for f in fields(parent):
            value=getattr(parent,f.name)
            if value is not None and f.name!='reference_state':data[prefix+'result_'+f.name]=value
        state=parent.reference_state
        for f in fields(state):
            if f.name=='basis':
                for b in fields(state.basis):data[prefix+'basis_'+b.name]=getattr(state.basis,b.name)
            else:data[prefix+'state_'+f.name]=getattr(state,f.name)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**data)
    temporary.replace(path)


def load_relative_pair_roks(path,*,request_hash,require_gradient=True):
    from .electronic_state import GFN2OrbitalState,GFN2OrbitalBasis
    with np.load(path,allow_pickle=False) as data:
        if str(data['schema'])!='mouse.relative_pair_roks.v1' or str(data['request_hash'])!=request_hash:
            raise ValueError('relative-pair checkpoint belongs to another request')
        def value(key):
            a=np.array(data[key],copy=True);return a.item() if a.ndim==0 else a
        parents=[]
        for i in range(2):
            prefix=f'parent_{i}_'
            basis=GFN2OrbitalBasis(**{f.name:value(prefix+'basis_'+f.name) for f in fields(GFN2OrbitalBasis)})
            metadata={f.name:value(prefix+'state_'+f.name) for f in fields(GFN2OrbitalState) if f.name!='basis'}
            metadata['ao_shell']=tuple(metadata['ao_shell']);state=GFN2OrbitalState(**metadata,basis=basis)
            kwargs={f.name:value(prefix+'result_'+f.name) for f in fields(ROKSSingletResult) if prefix+'result_'+f.name in data}
            parents.append(ROKSSingletResult(**kwargs,reference_state=state))
        g=value('gradient') if 'gradient' in data else None
        result=RelativePairROKSResult(tuple(parents),value('target_squared_overlaps'),value('multipliers'),
            float(data['orbital_residual']),int(data['orbital_index']),float(data['minimum_curvature']),int(data['evaluations']),g)
    if require_gradient and g is None:raise ValueError('relative checkpoint has no nuclear gradient')
    return result
