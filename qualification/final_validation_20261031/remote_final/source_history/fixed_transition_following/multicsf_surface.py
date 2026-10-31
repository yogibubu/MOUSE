"""Opt-in research multi-CSF surface with frozen rank, frame and references.

No DFT calculation or TDA solve occurs during surface evaluation. Preparation
currently needs fixed reference transition weights and projected bare ERIs.
"""
from dataclasses import dataclass
from pathlib import Path
import json,hashlib
import numpy as np
from .deltascf import prepare_gfn2_deltascf,delta_cross_overlap
from .calculator import ANGSTROM_TO_BOHR
from .deltascf_derivatives import load_derivative_point,save_derivative_point
from .spin_reference import save_response_reference,load_response_reference
from .restricted_ground import RestrictedGroundResult,evaluate_restricted_ground,restricted_ground_analytic_hessian
from .multicsf import MultiCSFObjective
from .multicsf_nuclear import FrozenLowdinERI,metric_power,nuclear_response,parent_gradient
from .multicsf_frame import RelativeActiveFrame


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_transition_rank(weights):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not len(w) or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0:raise ValueError('invalid transition weights')
    w=w/w.sum();number=float(1/(w@w));rank=int(np.floor(number+.5))
    if rank>4:raise ValueError('effective transition rank exceeds supported four-CSF space')
    return rank,number


def _load_point(path):
    with np.load(path,allow_pickle=False) as a:xyz=a['coordinates_angstrom'].copy();request=str(a['request_hash'])
    return load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)


class MultiCSFModel:
    """Portable checksummed frozen model. Rank is fixed at preparation."""
    def __init__(self,path):
        self.path=Path(path).expanduser().resolve();self.root=self.path.parent;self.fingerprint=_sha(self.path);self.manifest=json.loads(self.path.read_text());m=self.manifest
        if m.get('schema')!='mouse.multicsf.model.v1' or m.get('weights')!=[.5,.5] or m.get('energy_offset_ev')!=0 or m.get('diagonal_lambda_exchange')!=.9:raise ValueError('unsupported multi-CSF energy contract')
        required={'template.npz','arrays.npz','response_0.npz','response_1.npz','ground_0.npz','ground_1.npz','ground_2.npz'}
        if set(m.get('sha256',{}))!=required or len(m.get('grounds',[]))!=3:raise ValueError('incomplete multi-CSF model contract')
        self.validate_files();self.state_id=m['state_id'];self.rank=m['rank'];self.no=m['noccupied']
        if isinstance(self.state_id,bool) or not isinstance(self.state_id,int) or self.state_id<1:raise ValueError('invalid model state identifier')
        self.template=_load_point(self.root/'template.npz');self.anchor=self.template.reference_state;self.xyz=self.anchor.coordinates_bohr/ANGSTROM_TO_BOHR
        self.responses=tuple(load_response_reference(self.root/f'response_{i}.npz') for i in range(2))
        from .nonlocal_response import FrozenNonlocalResponseReference
        if not all(isinstance(r,FrozenNonlocalResponseReference) for r in self.responses):raise ValueError('multi-CSF requires full nonlocal responses')
        for field in ('atomic_numbers','orbital_shell','anchor_coordinates_bohr','operators','pair_charge_hartree','pair_spin_hartree','pair_reference_moments'):
            if not np.array_equal(getattr(self.responses[0],field),getattr(self.responses[1],field)):raise ValueError('multi-CSF parents have inconsistent response/anchor operators')
        if self.responses[0].fingerprint==self.responses[1].fingerprint:raise ValueError('multi-CSF requires two distinct parent models')
        if any(abs(r.response_exchange_scale-.9)>1e-12 for r in self.responses):raise ValueError('model diagonal lambda differs from 0.9')
        with np.load(self.root/'arrays.npz',allow_pickle=False) as a:self.arrays={k:a[k].copy() for k in a.files}
        a=self.arrays;n=len(self.anchor.overlap);rank,number=effective_transition_rank(a['transition_weights'])
        policy=m.get('rank_policy','effective_pairs_nearest')
        if policy not in ('effective_pairs_nearest','explicit_fixed_rank'):raise ValueError('unsupported active rank policy')
        if (policy=='effective_pairs_nearest' and rank!=self.rank) or isinstance(self.rank,bool) or not isinstance(self.rank,int) or not 1<=self.rank<=min(4,self.no,n-self.no) or self.no!=self.anchor.electron_count//2:raise ValueError('model rank/electron count disagrees with frozen transition')
        for key,shape in [('eri',(n,n,n,n)),('initial_coefficients',(2,n,n)),('reference_coefficients',(2,n,n)),('reference_transitions',(2,self.no,n-self.no))]:
            if key not in a or a[key].shape!=shape or not np.all(np.isfinite(a[key])):raise ValueError(f'invalid model array: {key}')
        for c in np.concatenate([a['initial_coefficients'],a['reference_coefficients']]):
            if not np.allclose(c.T@self.anchor.overlap@c,np.eye(n),atol=1e-8,rtol=0):raise ValueError('model orbitals are not orthonormal')
        self.eri=FrozenLowdinERI(self.anchor,a['eri']);self.grounds=[]
        for i,ref in enumerate([None]+list(self.responses)):
            row=m['grounds'][i];raw=_load_point(self.root/row['path']);g=RestrictedGroundResult(raw,row['minimum_curvature'],row['index'])
            if raw.spin_reference_fingerprint!=(None if ref is None else ref.fingerprint) or not raw.converged or raw.orbital_residual>1e-8 or g.orbital_saddle_index!=0 or g.minimum_orbital_curvature_hartree<=0:raise ValueError('model ground seed lacks a matching certificate')
            if not np.allclose(raw.reference_state.coordinates_bohr,self.anchor.coordinates_bohr,atol=1e-10,rtol=0):raise ValueError('ground seed and anchor geometries differ')
            self.grounds.append(g)
        active=np.r_[np.arange(self.rank),self.no+np.arange(self.rank)];c=a['initial_coefficients'];self.target=c[0][:,active].T@self.anchor.overlap@c[1][:,active]

    def validate_files(self):
        if _sha(self.path)!=self.fingerprint:raise ValueError('multi-CSF model manifest changed')
        for name,digest in self.manifest['sha256'].items():
            p=(self.root/name).resolve()
            if not p.is_relative_to(self.root) or _sha(p)!=digest:raise ValueError('multi-CSF model input checksum mismatch')

    def objectives(self,calculator,xyz,reference=None):
        contexts=[prepare_gfn2_deltascf(calculator,xyz,orbital_reference=self.anchor,spin_reference=r) for r in self.responses]
        state=contexts[0].state;t,_=self.eri.chart(state);transport=metric_power(state.overlap,-.5)@t.u@metric_power(self.anchor.overlap,.5)
        eri=self.eri.ao_eri(state);objects=[]
        for i,ctx in enumerate(contexts):objects.append(MultiCSFObjective(ctx,eri,self.no,self.rank,transport@self.arrays['reference_coefficients'][i],self.arrays['reference_transitions'][i]))
        if reference is None:point=[transport@c for c in self.arrays['initial_coefficients']]
        else:
            if reference.model_fingerprint!=self.fingerprint:raise ValueError('retained state has a different multi-CSF model')
            previous=reference.reference_state;old,_=self.eri.chart(previous);mapping=metric_power(state.overlap,-.5)@t.u@old.u.T@metric_power(previous.overlap,.5);point=[mapping@c for c in reference.coefficients]
        return objects,point


def write_multicsf_model(directory,*,template,responses,eri,initial_coefficients,reference_coefficients,reference_transitions,transition_weights,grounds,state_id=1,provenance=None,active_rank=None):
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    rank,number=effective_transition_rank(transition_weights);policy='effective_pairs_nearest'
    if active_rank is not None:
        no=int(template.reference_state.electron_count)//2;n=len(template.reference_state.overlap)
        if isinstance(active_rank,bool) or not isinstance(active_rank,int) or not 1<=active_rank<=min(4,no,n-no):raise ValueError('invalid explicit active rank')
        rank=active_rank;policy='explicit_fixed_rank'
    xyz=template.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
    if len(responses)!=2 or len(grounds)!=3:raise ValueError('model requires two responses and three certified ground seeds')
    save_derivative_point(root/'template.npz',template,request_hash='multicsf:model:template',coordinates_angstrom=xyz)
    for i,r in enumerate(responses):save_response_reference(root/f'response_{i}.npz',r)
    np.savez_compressed(root/'arrays.npz',eri=eri,initial_coefficients=initial_coefficients,reference_coefficients=reference_coefficients,reference_transitions=reference_transitions,transition_weights=transition_weights)
    rows=[]
    for i,g in enumerate(grounds):
        path=root/f'ground_{i}.npz';save_derivative_point(path,g.reference,request_hash=f'multicsf:model:ground:{i}',coordinates_angstrom=xyz)
        rows.append(dict(path=path.name,minimum_curvature=g.minimum_orbital_curvature_hartree,index=g.orbital_saddle_index))
    manifest=dict(schema='mouse.multicsf.model.v1',experimental=True,rank=rank,rank_policy=policy,effective_pairs=number,noccupied=int(template.reference_state.electron_count)//2,state_id=state_id,weights=[.5,.5],diagonal_lambda_exchange=.9,energy_offset_ev=0,offdiagonal='bare projected Slater-Condon',orbital_manifold='fixed full active cross-overlap matrix',grounds=rows,provenance=provenance or {},sha256={p.name:_sha(p) for p in root.glob('*.npz')})
    path=root/'model.json';path.write_text(json.dumps(manifest,indent=2)+'\n');MultiCSFModel(path);return path


@dataclass(frozen=True)
class MultiCSFSurfaceResult:
    model_fingerprint: str
    reference_state: object
    coefficients: np.ndarray
    ci_coefficients: np.ndarray
    grounds: tuple
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray | None
    orbital_residual: float
    orbital_saddle_index: int
    minimum_absolute_curvature: float
    parent_gaps_hartree: tuple
    fidelity: tuple
    active_cross_overlap: np.ndarray
    selected_states: tuple

    def __post_init__(self):
        for name in ('coefficients','ci_coefficients','gradient_hartree_per_bohr','hessian_hartree_per_bohr2','active_cross_overlap'):
            value=getattr(self,name)
            if value is None:continue
            array=np.array(value,copy=True)
            if not np.all(np.isfinite(array)):raise ValueError('nonfinite multi-CSF result')
            array.setflags(write=False);object.__setattr__(self,name,array)


def _apoc_expansion(obj,c,state_id,energy,calculator):
    from matrix_apoc import DeterminantOrbitals,DeltaSCFState,SpinAdaptedSinglet,SingletCSFExpansion
    obj.evaluate(c,False);state=obj.context.state;csfs=[]
    for occ in obj.occupations:
        wf=DeterminantOrbitals(state.atomic_numbers,state.coordinates_bohr,state.overlap,(c,c),tuple(occ),np.asarray(state.atomic_numbers)-np.asarray(calculator._valence),charge=state.charge,multiplicity=1,model=state.model_id)
        csfs.append(SpinAdaptedSinglet(state_id,DeltaSCFState(state_id,wf)))
    phase=(-1.)**(obj.no-1+np.arange(obj.rank))
    return SingletCSFExpansion(state_id,tuple(csfs),obj.cache['ci']*phase,energy)


def evaluate_multicsf_surface(calculator,coordinates_angstrom,model,*,reference_result=None,compute_hessian=False,minimum_overlap=.9):
    """Analytic excited total energy, gradient and optional Hessian, no SCC/TDA."""
    model.validate_files();xyz=np.asarray(coordinates_angstrom,dtype=float)
    if (reference_result is not None and reference_result.model_fingerprint==model.fingerprint
            and np.allclose(xyz*ANGSTROM_TO_BOHR,reference_result.reference_state.coordinates_bohr,atol=1e-12,rtol=0)
            and (not compute_hessian or reference_result.hessian_hartree_per_bohr2 is not None)):
        return reference_result
    objects,point=model.objectives(calculator,xyz,reference_result);frame=RelativeActiveFrame(objects)
    point=frame.polish(point);e,g,h,*_=frame.evaluate(point);values=np.linalg.eigvalsh(h);residual=float(np.max(abs(g)))
    if residual>1e-9 or np.min(abs(values))<1e-7:raise ValueError('multi-CSF frame stationarity/curvature is unresolved')
    gram=point[0][:,frame.active].T@frame.s@point[1][:,frame.active]
    if not np.allclose(gram,model.target,atol=1e-9,rtol=0):raise ValueError('multi-CSF relative frame differs from frozen anchor')
    fidelities=tuple(o.fidelity(c)[0] for o,c in zip(objects,point))
    if min(fidelities)**2<=.5 or any(o.cache['coefficient_margin']<.05 for o in objects):raise ValueError('multi-CSF fixed-transition identity is unresolved')
    parent_e=[o.evaluate(c,True)[0] for o,c in zip(objects,point)];selected=tuple(_apoc_expansion(o,c,model.state_id,ee,calculator) for o,c,ee in zip(objects,point,parent_e))
    if reference_result is not None:
        from matrix_apoc import multicsf_overlap,StateContinuityError
        cross=delta_cross_overlap(reference_result.reference_state,objects[0].context)
        if any(multicsf_overlap(old,new,cross)[0]<minimum_overlap for old,new in zip(reference_result.selected_states,selected)):raise StateContinuityError('multi-CSF coherent parent continuity failed')
    seeds=model.grounds if reference_result is None else reference_result.grounds;grounds=[];weights=[1.,-.5,-.5]
    for seed,ref in zip(seeds,[None]+list(model.responses)):
        # Completed anchor ground checkpoints are reused, not optimized again.
        if np.allclose(seed.reference_state.coordinates_bohr,xyz*ANGSTROM_TO_BOHR,atol=1e-12,rtol=0):ground=seed
        else:ground=evaluate_restricted_ground(calculator,xyz,template=model.anchor,spin_reference=ref,reference_result=seed,gradient=False)
        grounds.append(ground)
    gradient=sum(.5*parent_gradient(o,c,model.eri) for o,c in zip(objects,point));total=e;hh=None
    if compute_hessian:
        blocks=[nuclear_response(o,c,model.eri) for o,c in zip(objects,point)];hh=frame.nuclear_response(point,blocks)['hessian']
    for weight,ground,ref in zip(weights,grounds,[None]+list(model.responses)):
        ctx=prepare_gfn2_deltascf(calculator,xyz,orbital_reference=model.anchor,spin_reference=ref);c=ground.coefficients[0];f,_=ctx.fock_and_energy(ground.spin_densities_ao);q=sum((c.T@ff@c)*occ[None,:] for ff,occ in zip(f,ground.occupations))
        gradient+=weight*ctx.analytic_gradient(ground.spin_densities_ao,c@(.5*(q+q.T))@c.T);total+=weight*ground.energy_hartree
        if compute_hessian:hh+=weight*restricted_ground_analytic_hessian(calculator,xyz,reference_result=ground,spin_reference=ref).hessian_hartree_per_bohr2
    gaps=tuple(ee-gg.energy_hartree for ee,gg in zip(parent_e,grounds[1:]))
    if min(gaps)<=0:raise ValueError('multi-CSF parent has a nonpositive gap')
    return MultiCSFSurfaceResult(model.fingerprint,objects[0].context.state,np.array(point),np.array([o.cache['ci'].copy() for o in objects]),tuple(grounds),float(total),gradient,hh,residual,int(np.count_nonzero(values< -1e-7)),float(np.min(abs(values))),gaps,fidelities,gram,selected)


def save_multicsf_result(directory,result,model):
    """Save a completed accepted generation; caller atomically publishes it."""
    if result.model_fingerprint!=model.fingerprint:raise ValueError('result/model fingerprint mismatch')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False);xyz=result.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
    data=dict(coordinates_angstrom=xyz,coefficients=result.coefficients,ci_coefficients=result.ci_coefficients,energy=result.energy_hartree,gradient=result.gradient_hartree_per_bohr,frame=result.active_cross_overlap,fidelity=result.fidelity,parent_gaps=result.parent_gaps_hartree)
    if result.hessian_hartree_per_bohr2 is not None:data['hessian']=result.hessian_hartree_per_bohr2
    np.savez_compressed(root/'point.npz',**data);grounds=[]
    for i,g in enumerate(result.grounds):
        p=root/f'ground_{i}.npz';save_derivative_point(p,g.reference,request_hash=f'multicsf:accepted:ground:{i}',coordinates_angstrom=xyz)
        grounds.append(dict(path=p.name,minimum_curvature=g.minimum_orbital_curvature_hartree,index=g.orbital_saddle_index))
    metadata=dict(schema='mouse.multicsf.result.v1',model_fingerprint=model.fingerprint,orbital_residual=result.orbital_residual,orbital_saddle_index=result.orbital_saddle_index,minimum_absolute_curvature=result.minimum_absolute_curvature,grounds=grounds,sha256={p.name:_sha(p) for p in root.glob('*.npz')})
    path=root/'result.json';path.write_text(json.dumps(metadata,indent=2)+'\n');return path


def load_multicsf_result(path,model,calculator):
    """Check hashes and stationary wavefunction; never rerun optimization."""
    path=Path(path);m=json.loads(path.read_text());root=path.parent;model.validate_files()
    if m.get('schema')!='mouse.multicsf.result.v1' or m.get('model_fingerprint')!=model.fingerprint:raise ValueError('retained multi-CSF state belongs to another model')
    for name,digest in m['sha256'].items():
        if (root/name).resolve().parent!=root.resolve() or _sha(root/name)!=digest:raise ValueError('retained multi-CSF checkpoint checksum mismatch')
    with np.load(root/'point.npz',allow_pickle=False) as a:d={k:a[k].copy() for k in a.files}
    objects,_=model.objectives(calculator,d['coordinates_angstrom']);point=d['coefficients'];frame=RelativeActiveFrame(objects);_,g,h,*_=frame.evaluate(point);values=np.linalg.eigvalsh(h)
    gram=point[0][:,frame.active].T@frame.s@point[1][:,frame.active]
    if np.max(abs(g))>1e-9 or np.min(abs(values))<1e-7 or not np.allclose(gram,model.target,atol=1e-9,rtol=0):raise ValueError('retained multi-CSF state fails stationarity/frame certificate')
    selected=tuple(_apoc_expansion(o,c,model.state_id,o.evaluate(c,False)[0],calculator) for o,c in zip(objects,point));grounds=[]
    for row,ref in zip(m['grounds'],[None]+list(model.responses)):
        raw=_load_point(root/row['path']);ground=RestrictedGroundResult(raw,row['minimum_curvature'],row['index'])
        if not ground.converged or ground.orbital_residual>1e-8 or ground.orbital_saddle_index!=0 or ground.minimum_orbital_curvature_hartree<=0 or raw.spin_reference_fingerprint!=(None if ref is None else ref.fingerprint) or not np.allclose(raw.reference_state.coordinates_bohr,objects[0].context.state.coordinates_bohr,atol=1e-10,rtol=0):raise ValueError('retained ground checkpoint certificate mismatch')
        grounds.append(ground)
    if len(grounds)!=3:raise ValueError('retained state lacks three ground references')
    ci=np.array([o.cache['ci'] for o in objects]);fidelity=tuple(o.fidelity(c)[0] for o,c in zip(objects,point))
    gaps=tuple(o.evaluate(c,False)[0]-gg.energy_hartree for o,c,gg in zip(objects,point,grounds[1:]));energy=sum(.5*o.evaluate(c,False)[0] for o,c in zip(objects,point))+grounds[0].energy_hartree-.5*sum(gg.energy_hartree for gg in grounds[1:])
    if min(gaps)<=0 or min(fidelity)**2<=.5 or abs(energy-float(d['energy']))>1e-10 or not np.allclose(ci,d['ci_coefficients'],atol=1e-9,rtol=0):raise ValueError('retained multi-CSF energy/identity mismatch')
    return MultiCSFSurfaceResult(model.fingerprint,objects[0].context.state,point,ci,tuple(grounds),float(d['energy']),d['gradient'],d.get('hessian'),float(np.max(abs(g))),int(np.count_nonzero(values< -1e-7)),float(np.min(abs(values))),gaps,fidelity,gram,selected)
