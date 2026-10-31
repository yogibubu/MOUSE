"""Variational 50/50 ROKS with shared active orbitals.

Both parent CSFs are spin-pure. Either the hole or both open orbitals are
constrained equal; the remaining orbitals optimize independently. Only the JOINT
orbital residual is stationary. Nuclear Hessians use the coupled response,
not a mean of separately relaxed parent Hessians. No energy offset or penalty
parameter appears in this functional.
"""
from dataclasses import dataclass,replace
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from .deltascf import prepare_gfn2_deltascf,project_delta_orbitals,delta_cross_overlap
from .roks import ROKSSingletResult
from .roks_orbital_response import orbital_residual_jacobian
from .orbital_saddle import generalized_mode_following


def root_metric(s,power):
    w,v=np.linalg.eigh(s)
    if np.min(w)<=1e-10:raise ValueError('shared ROKS needs a nonsingular AO metric')
    return (v*w**power)@v.T


class _SharedPairObjective:
 def __init__(self,contexts,occupations,shared_orbital_indices=None):
  self.contexts=contexts;self.occ=np.asarray(occupations);self.triplet=np.array([np.maximum(*self.occ),np.minimum(*self.occ)])
  self.n=len(self.occ[0]);self.s=contexts[0].state.overlap;self.left,self.right=np.triu_indices(self.n,1)
  active_opens=np.flatnonzero(self.occ[0]!=self.occ[1]);assert len(active_opens)==2
  self.opens=active_opens if shared_orbital_indices is None else np.asarray(shared_orbital_indices,dtype=int)
  if self.opens.ndim!=1 or not 0<=len(self.opens)<=2 or len(set(self.opens))!=len(self.opens) or not np.all(np.isin(self.opens,active_opens)):
   raise ValueError('share one or both open orbitals')
  shared=np.isin(self.left,self.opens)|np.isin(self.right,self.opens)
  labels=np.vstack([self.occ,self.triplet])
  private=(~shared)&np.any(labels[:,self.left]!=labels[:,self.right],axis=0)
  self.shared=np.flatnonzero(shared);self.private=np.flatnonzero(private);self.ns=len(self.shared);self.np=len(self.private);self.dimension=self.ns+2*self.np
  self.generators=np.zeros((len(self.left),self.n,self.n))
  for a,(i,j) in enumerate(zip(self.left,self.right)):self.generators[a,i,j]=1.;self.generators[a,j,i]=-1.
  self.cache={}
 def seed(self,components):
  root=root_metric(self.s,.5);inverse=root_metric(self.s,-.5);q=[root@c for c in components];opens=self.opens;others=np.array([i for i in range(self.n) if i not in opens])
  cross=q[0][:,opens].T@q[1][:,opens]
  if len(opens)==2 and abs(cross[0,1])+abs(cross[1,0])>abs(cross[0,0])+abs(cross[1,1]):q[1][:,opens]=q[1][:,opens[::-1]]
  for i in range(len(opens)):
   if q[0][:,opens[i]]@q[1][:,opens[i]]<0:q[1][:,opens[i]]*=-1
  mean=q[0][:,opens]+q[1][:,opens];u,_,vt=np.linalg.svd(mean,full_matrices=False);common=u@vt
  result=[]
  for qq in q:
   rest=qq[:,others]-common@(common.T@qq[:,others]);u,_,vt=np.linalg.svd(rest,full_matrices=False)
   qq[:,others]=u@vt;qq[:,opens]=common;result.append(inverse@qq)
  return np.asarray(result)
 def maps(self,point):
  a,b=point;v=a.T@self.s@b
  shared=self.generators[self.shared];hs=[shared,v.T@shared@v]
  maps=[np.zeros((len(self.left),self.dimension)) for _ in range(2)]
  for i in range(2):
   maps[i][:,:self.ns]=hs[i][:,self.left,self.right].T
   maps[i][self.private,self.ns+i*self.np+np.arange(self.np)]=1.
  return maps,hs,v
 def retract(self,point,step):
  a,b=point;_,_,v=self.maps(point);g=np.einsum('a,aij->ij',step[:self.ns],self.generators[self.shared]);common=expm(g)
  pa=expm(np.einsum('a,aij->ij',step[self.ns:self.ns+self.np],self.generators[self.private]));pb=expm(np.einsum('a,aij->ij',step[self.ns+self.np:],self.generators[self.private]))
  return np.asarray([a@common@pa,a@common@v@pb])
 def evaluate(self,point,curvature=True):
  if 'point' not in self.cache or not np.array_equal(point,self.cache['point']):
   values=[]
   for c,ctx in zip(point,self.contexts):
    focks=[];energy=0.;gradient=np.zeros(len(self.left))
    for weight,o in [(2.,self.occ),(-1.,self.triplet)]:
     dm=np.array([(c*x)@c.T for x in o]);f,e=ctx.fock_and_energy(dm);energy+=weight*e;focks.append(f)
     q=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));gradient+=weight*2*(q[self.left,self.right]-q[self.right,self.left])
    values.append(dict(energy=energy,g=gradient,focks=focks))
   maps,hs,v=self.maps(point);g=.5*sum(t.T@row['g'] for t,row in zip(maps,values));self.cache.clear();self.cache.update(point=point.copy(),values=values,maps=maps,hs=hs,v=v,g=g,energy=.5*sum(row['energy'] for row in values))
  if curvature and 'h' not in self.cache:
   total=np.zeros((self.dimension,self.dimension))
   for model,(c,ctx,row,t,hs) in enumerate(zip(point,self.contexts,self.cache['values'],self.cache['maps'],self.cache['hs'])):
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=self.occ
    kernel=ctx.density_response_kernel(state);h=np.zeros((len(self.left),len(self.left)))
    for weight,o,f in [(2.,self.occ,row['focks'][0]),(-1.,self.triplet,row['focks'][1])]:
     h+=weight*orbital_residual_jacobian(c,o,f,kernel,ctx.polarization,self.left,self.right)
    h=.5*(h+h.T);total+=.5*t.T@h@t
    # E(exp(Hs) exp(Hp)): mixed second chart derivative includes 1/2[Hs,Hp].
    gg=np.zeros((self.n,self.n));gg[self.left,self.right]=row['g'];gg[self.right,self.left]=-row['g']
    comm=gg@hs-hs@gg
    correction=.25*comm[:,self.left[self.private],self.right[self.private]]
    columns=self.ns+model*self.np+np.arange(self.np)
    total[np.ix_(np.arange(self.ns),columns)]+=correction
    total[np.ix_(columns,np.arange(self.ns))]+=correction.T
   self.cache['h']=.5*(total+total.T)
  return self.cache['energy'],self.cache['g'],self.cache.get('h')


@dataclass(frozen=True)
class SharedPairROKSResult:
    components: tuple
    orbital_residual: float
    orbital_saddle_index: int | None
    minimum_absolute_orbital_curvature_hartree: float
    evaluations: int
    converged: bool
    gradient_hartree_per_bohr: np.ndarray | None = None
    shared_orbital_indices: tuple | None = None

    def __post_init__(self):
        if len(self.components)!=2 or not all(isinstance(r,ROKSSingletResult) for r in self.components):
            raise ValueError('shared-pair ROKS needs two parent CSF descriptors')
        a,b=self.components
        if (not np.array_equal(a.occupations,b.occupations)
                or not np.allclose(a.overlap,b.overlap,atol=1e-10,rtol=0)
                or not np.array_equal(a.core_electrons,b.core_electrons)
                or a.reference_state is None or b.reference_state is None
                or not np.array_equal(a.reference_state.atomic_numbers,b.reference_state.atomic_numbers)
                or not np.allclose(a.reference_state.coordinates_bohr,b.reference_state.coordinates_bohr,atol=1e-10,rtol=0)):
            raise ValueError('shared-pair parents have different molecular states')
        opens=np.flatnonzero(a.occupations[0]!=a.occupations[1])
        shared=tuple(int(i) for i in opens) if self.shared_orbital_indices is None else tuple(self.shared_orbital_indices)
        if (not 1<=len(shared)<=2 or len(set(shared))!=len(shared)
                or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i not in opens for i in shared)):
            raise ValueError('invalid shared active-orbital constraint')
        object.__setattr__(self,'shared_orbital_indices',tuple(int(i) for i in shared))
        if np.max(abs(a.common_coefficients[:,shared]-b.common_coefficients[:,shared]))>1e-8:
            raise ValueError('parent active orbitals are not shared')
        fa,fb=a.spin_frame_clusters,b.spin_frame_clusters
        if (fa is None)!=(fb is None) or fa is not None and not np.array_equal(fa,fb):
            raise ValueError('parent response frames differ')
        if (not np.isfinite(self.orbital_residual) or self.orbital_residual<0
                or not np.isfinite(self.minimum_absolute_orbital_curvature_hartree)
                or self.minimum_absolute_orbital_curvature_hartree<0):
            raise ValueError('invalid joint orbital certificate')
        if self.converged and (self.orbital_residual>1e-8 or self.orbital_saddle_index is None
                or self.orbital_saddle_index<0 or self.minimum_absolute_orbital_curvature_hartree<=1e-7):
            raise ValueError('shared-pair solution is not tightly stationary and index-resolved')
        if self.gradient_hartree_per_bohr is not None:
            g=np.array(self.gradient_hartree_per_bohr,copy=True)
            if g.shape!=a.reference_state.coordinates_bohr.shape or not np.all(np.isfinite(g)):
                raise ValueError('invalid shared-pair nuclear gradient')
            g.setflags(write=False);object.__setattr__(self,'gradient_hartree_per_bohr',g)

    @property
    def coefficients(self):return np.asarray([r.common_coefficients for r in self.components])
    @property
    def occupations(self):return self.components[0].occupations
    @property
    def energy_hartree(self):return .5*sum(r.energy_hartree for r in self.components)
    @property
    def spin_squared(self):return 0.
    @property
    def component_unconstrained_residuals(self):return tuple(r.orbital_residual for r in self.components)
    @property
    def component_active_pair_overlap(self):
        a,b=self.components;opens=np.flatnonzero(a.occupations[0]!=a.occupations[1])
        v=a.common_coefficients[:,opens].T@a.overlap@b.common_coefficients[:,opens]
        return float(np.clip(abs(v[0,0]*v[1,1]+v[0,1]*v[1,0]),0.,1.))
    @property
    def component_csf_overlap(self):
        a,b=self.components;value=0.
        for oa in [a.occupations,a.occupations[::-1]]:
            for ob in [b.occupations,b.occupations[::-1]]:
                value+=.5*np.prod([np.linalg.det(a.common_coefficients[:,x>.5].T@a.overlap@b.common_coefficients[:,y>.5]) for x,y in zip(oa,ob)])
        return float(abs(value))


class SharedPairROKSConvergenceError(RuntimeError):
    def __init__(self,result):
        self.result=result
        super().__init__(f'joint ROKS stationarity failed: residual={result.orbital_residual:.3e}')


def _contexts(calculator,coordinates,responses,components):
    from .spin_reference import validate_spin_reference
    if len(responses)!=2 or len(components)!=2:
        raise ValueError('provide two ordered full/spectral responses and parent states')
    contexts=[]
    for ref,parent in zip(responses,components):
        validate_spin_reference(parent,ref)
        contexts.append(prepare_gfn2_deltascf(calculator,coordinates,
            orbital_reference=parent.reference_state,spin_reference=ref))
    if not np.allclose(contexts[0].state.overlap,contexts[1].state.overlap,atol=1e-12,rtol=0):
        raise ValueError('parent AO metrics differ')
    return contexts


def evaluate_shared_pair_roks(calculator,coordinates_angstrom,responses,*,
        reference_result=None,initial_components=None,gradient=False,
        freeze_and_release=None,orbital_tolerance=1e-10,maximum_iterations=150,
        maximum_step=.15,minimum_state_overlap=.7,checkpoint=None,shared_orbitals=None):
    """Optimize a joint spin-pure 50/50 surface, without any S0 SCC call.

    On initialization, freeze the two common open orbitals, relax spectators,
    then infer the full saddle index for release. Warm starts use stationary
    orbital Newton continuation and certify the observed index. Parent orbital
    gradients need not vanish; their joint, constrained gradient must vanish.
    ``shared_orbitals='hole'`` shares the beta-only orbital of the promotion,
    allowing the particle to relax separately in the two parent CSFs. A retained
    result fixes this constraint across geometry steps and restart.
    """
    if reference_result is not None and initial_components is not None:
        raise ValueError('provide either a retained joint state or initial parent states')
    if reference_result is not None and not isinstance(reference_result,SharedPairROKSResult):
        raise TypeError('expected a retained shared-pair ROKS state')
    parents=initial_components if reference_result is None else reference_result.components
    if parents is None:raise ValueError('shared ROKS needs molecular orbital seeds')
    if (maximum_iterations<1 or not all(np.isfinite(x) and x>0 for x in [orbital_tolerance,maximum_step])
            or not 0<=minimum_state_overlap<=1):raise ValueError('invalid shared ROKS controls')
    contexts=_contexts(calculator,coordinates_angstrom,responses,parents)
    if not np.array_equal(parents[0].occupations,parents[1].occupations):raise ValueError('parent occupations differ')
    if shared_orbitals not in {None,'active_pair','hole'}:
        raise ValueError('unknown shared active-orbital constraint')
    occ=parents[0].occupations
    if shared_orbitals is None and reference_result is not None:
        shared=reference_result.shared_orbital_indices
    else:
        shared=tuple(np.flatnonzero(occ[0]<occ[1] if shared_orbitals=='hole' else occ[0]!=occ[1]))
    if reference_result is not None and tuple(shared)!=reference_result.shared_orbital_indices:
        raise ValueError('retained state has a different shared active-orbital constraint')
    objective=_SharedPairObjective(contexts,occ,shared)
    guesses=[project_delta_orbitals(parent,ctx)[0] for parent,ctx in zip(parents,contexts)]
    point=objective.seed(guesses);freeze=reference_result is None if freeze_and_release is None else bool(freeze_and_release)
    evaluations=0;converged=False
    def emit(stage,result):
        if checkpoint is not None:checkpoint(dict(phase=stage,coefficients=result.point,
            energy=result.energy,residual=float(np.max(abs(result.gradient),initial=0.)),
            index=result.observed_index,iteration=result.iterations))
    if freeze:
        private=np.arange(objective.ns,objective.dimension)
        def restricted(c,curvature=True):
            e,g,h=objective.evaluate(c,curvature=curvature)
            return e,g[private],None if h is None else h[np.ix_(private,private)]
        def retract_private(c,step):
            full=np.zeros(objective.dimension);full[private]=step
            return objective.retract(c,full)
        frozen=generalized_mode_following(restricted,retract_private,point,saddle_index=0,
            gradient_tolerance=orbital_tolerance,maximum_iterations=maximum_iterations,
            maximum_step=maximum_step,checkpoint=lambda r:emit('freeze',r))
        point=frozen.point;evaluations+=frozen.iterations
        if not frozen.converged:raise RuntimeError('shared ROKS frozen spectator relaxation failed')
        _,_,h=objective.evaluate(point);values=np.linalg.eigvalsh(h)
        if np.min(abs(values))<=1e-7:raise ValueError('post-freeze joint index is unresolved')
        target=int(np.sum(values<-1e-7))
        released=generalized_mode_following(objective.evaluate,objective.retract,point,
            saddle_index=target,gradient_tolerance=orbital_tolerance,
            maximum_iterations=maximum_iterations,maximum_step=maximum_step,
            checkpoint=lambda r:emit('release',r))
        point=released.point;evaluations+=released.iterations;converged=released.converged
    else:
        for iteration in range(maximum_iterations+1):
            e,g,h=objective.evaluate(point);values=np.linalg.eigvalsh(h)
            error=float(np.max(abs(g)));evaluations+=1
            if checkpoint is not None:checkpoint(dict(phase='newton',coefficients=point,
                energy=e,residual=error,index=int(np.sum(values<-1e-7)),iteration=iteration))
            if error<=orbital_tolerance and np.min(abs(values))>1e-7:converged=True;break
            delta=np.linalg.lstsq(h,-g,rcond=1e-10)[0];norm=np.linalg.norm(delta)
            if norm>maximum_step:delta*=maximum_step/norm
            for fraction in [1.,.5,.25,.125,.0625,.03125]:
                trial=objective.retract(point,fraction*delta);_,gg,_=objective.evaluate(trial,curvature=False)
                if np.linalg.norm(gg)<np.linalg.norm(g):point=trial;break
            else:break
    e,g,h=objective.evaluate(point);values=np.linalg.eigvalsh(h);descriptors=[]
    for c,ctx,ref,row in zip(point,contexts,responses,objective.cache['values']):
        descriptors.append(ROKSSingletResult(c,objective.occ,objective.triplet,ctx.state.overlap,
            row['focks'][0],row['focks'][1],row['energy'],float(np.max(abs(row['g']))),
            evaluations,False,reference_state=ctx.state,
            core_electrons=np.asarray(ctx.state.atomic_numbers)-np.asarray(calculator._valence),
            spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=getattr(ref,'frame_clusters',None)))
    result=SharedPairROKSResult(tuple(descriptors),float(np.max(abs(g))),
        int(np.sum(values<-1e-7)) if converged else None,float(np.min(abs(values))),evaluations,converged,
        shared_orbital_indices=tuple(shared))
    if not converged:raise SharedPairROKSConvergenceError(result)
    from matrix_apoc import follow_spin_adapted_singlet
    for old,new,ctx in zip(parents,result.components,contexts):
        follow_spin_adapted_singlet(old.to_apoc(),(new.to_apoc(),),delta_cross_overlap(old.reference_state,ctx),minimum_overlap=minimum_state_overlap)
    if gradient:
        total=np.zeros_like(contexts[0].state.coordinates_bohr)
        for c,r,ctx in zip(point,result.components,contexts):
            q=sum(2*(c.T@f@c)*o[None,:] for f,o in zip(r.fock_mixed,r.occupations))
            q-=sum((c.T@f@c)*o[None,:] for f,o in zip(r.fock_triplet,r.triplet_occupations))
            weighted=c@(.5*(q+q.T))@c.T;zero=np.zeros_like(weighted)
            total+=.5*(2*ctx.analytic_gradient(r.densities(),.5*weighted)-ctx.analytic_gradient(r.densities(triplet=True),zero))
        result=replace(result,gradient_hartree_per_bohr=total)
    return result


@dataclass(frozen=True)
class SharedPairROKSHessian:
    reference: SharedPairROKSResult
    hessian_hartree_per_bohr2: np.ndarray
    frozen_orbital_hessian_hartree_per_bohr2: np.ndarray
    response_hessian_hartree_per_bohr2: np.ndarray
    orbital_hessian_hartree: np.ndarray
    orbital_nuclear_coupling: np.ndarray
    orbital_response: np.ndarray
    maximum_response_residual: float
    raw_symmetry_error_hartree_per_bohr2: float
    minimum_absolute_orbital_curvature_hartree: float
    method: str = 'analytic_shared_active_pair_coupled_schur_complement'


def _coupled_geometry_blocks(contexts,components,objective):
    """Native RR/KK/KR jets in the joint orbital chart, including BCH terms."""
    from .deltascf_analytic_hessian import _fixed_density_geometry,_response_blocks
    point=np.asarray([r.common_coefficients for r in components])
    objective.evaluate(point,curvature=False);maps=objective.cache['maps'];hs=objective.cache['hs']
    nr=contexts[0].state.coordinates_bohr.size;hrr=np.zeros((nr,nr));hkk=np.zeros((objective.dimension,objective.dimension));b=np.zeros((objective.dimension,nr))
    for model,(ctx,r,c,t,shared,row) in enumerate(zip(contexts,components,point,maps,hs,objective.cache['values'])):
        # Mixed/triplet total densities coincide. Reuse the complete native
        # fixed-density geometry jet; only their spin corrections differ.
        kernel,ds,df,d2s,err=_fixed_density_geometry(ctx,r.densities())
        if ctx.charge_correction is not None:
            charge=ctx.charge_correction;dfield,efield=charge.geometry(r.density_ao,ds,d2s)
            df,err=df+dfield,err+efield;base=kernel
            kernel=lambda dp,base=base,charge=charge:base(dp)+charge.response(dp)
        rr,kk,kr=_response_blocks([c],[(2.,r.occupations,r.fock_mixed),(-1.,r.triplet_occupations,r.fock_triplet)],
            ds,d2s,[(df,err)]*2,[kernel]*2,ctx.polarization,orbital_pairs=(objective.left,objective.right))
        hrr+=.5*rr;hkk+=.5*t.T@kk@t;b+=.5*t.T@kr
        gg=np.zeros((objective.n,objective.n));gg[objective.left,objective.right]=row['g'];gg[objective.right,objective.left]=-row['g']
        comm=gg@shared-shared@gg;correction=.25*comm[:,objective.left[objective.private],objective.right[objective.private]]
        columns=objective.ns+model*objective.np+np.arange(objective.np)
        hkk[np.ix_(np.arange(objective.ns),columns)]+=correction
        hkk[np.ix_(columns,np.arange(objective.ns))]+=correction.T
    return hrr,.5*(hkk+hkk.T),b


def evaluate_shared_pair_roks_analytic_hessian(calculator,coordinates_angstrom,responses,*,reference_result):
    from .deltascf_analytic_hessian import _fixed_density_geometry,_response_blocks
    from .calculator import ANGSTROM_TO_BOHR
    result=reference_result
    if (not isinstance(result,SharedPairROKSResult) or not result.converged or result.orbital_residual>1e-8
            or not np.allclose(result.components[0].reference_state.coordinates_bohr,
                np.asarray(coordinates_angstrom)*ANGSTROM_TO_BOHR,atol=1e-10,rtol=0)):
        raise ValueError('coupled Hessian needs a matching tightly stationary joint state')
    contexts=_contexts(calculator,coordinates_angstrom,responses,result.components)
    objective=_SharedPairObjective(contexts,result.occupations,result.shared_orbital_indices)
    hrr,hkk,b=_coupled_geometry_blocks(contexts,result.components,objective)
    hkk=.5*(hkk+hkk.T);minimum=float(np.min(abs(np.linalg.eigvalsh(hkk))))
    if minimum<1e-8:raise ValueError('coupled ROKS orbital response is singular')
    response=np.linalg.solve(hkk,-b);correction=b.T@response;total=hrr+correction
    method='analytic_shared_active_pair_coupled_schur_complement' if len(result.shared_orbital_indices)==2 else 'analytic_shared_active_hole_coupled_schur_complement'
    return SharedPairROKSHessian(result,.5*(total+total.T),hrr,correction,hkk,b,response,
        float(np.max(abs(hkk@response+b))),float(np.max(abs(total-total.T))),minimum,method)


def save_shared_pair_roks(path,result,*,request_hash):
    """Persist the JOINT certificate; parent descriptors remain nonstationary."""
    from dataclasses import fields
    from pathlib import Path
    if not isinstance(result,SharedPairROKSResult) or not result.converged:
        raise ValueError('shared checkpoint requires a stationary joint solution')
    data=dict(schema='mouse.shared_pair_roks.v2',request_hash=request_hash,shared_orbital_indices=result.shared_orbital_indices,
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


def load_shared_pair_roks(path,*,request_hash,require_gradient=True):
    """Load an explicitly coupled state, never a fake stationary parent state."""
    from dataclasses import fields
    from .electronic_state import GFN2OrbitalState,GFN2OrbitalBasis
    with np.load(path,allow_pickle=False) as data:
        if str(data['schema']) not in {'mouse.shared_pair_roks.v1','mouse.shared_pair_roks.v2'} or str(data['request_hash'])!=request_hash:
            raise ValueError('shared-pair checkpoint belongs to another request')
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
        gradient=value('gradient') if 'gradient' in data else None
        shared=None if str(data['schema'])=='mouse.shared_pair_roks.v1' else tuple(value('shared_orbital_indices'))
        result=SharedPairROKSResult(tuple(parents),float(data['orbital_residual']),int(data['orbital_index']),
            float(data['minimum_curvature']),int(data['evaluations']),True,gradient,shared)
    if require_gradient and gradient is None:raise ValueError('shared checkpoint has no nuclear gradient')
    return result
