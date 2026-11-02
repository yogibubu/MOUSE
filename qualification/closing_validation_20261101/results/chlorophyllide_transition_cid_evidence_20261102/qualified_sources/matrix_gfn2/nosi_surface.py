"""Opt-in prepared spin-pure OO-CID surfaces with analytic nuclear derivatives.

The model uses frozen physical couplings and auxiliary orbitals transported
in its anchor's orthonormal AO frame. Its stationary native ground and ROKS
target orbitals respond fully. Preparation is separate: surface evaluation
imports no DFT backend and never solves TDA or calls a ground SCC driver.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import ast,json,hashlib
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from .calculator import ANGSTROM_TO_BOHR
from .deltascf import prepare_gfn2_deltascf
from .spin_reference import load_response_reference
from .nonlocal_response import FrozenNonlocalResponseReference
from .multicsf_surface import _load_point
from .multicsf_nuclear import metric_power
from ._nosi_refine import ground as refine_ground,roks as refine_roks
from ._nosi_surface_response import MatrixResponse
from ._nosi_nuclear import projected_eigen_jet
from ._nosi_spin import csfs as four_singlets
from ._nosi_configurations import from_labels as configuration_records,validate as validate_configurations,basis as configuration_basis


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _closed(c,ids):return [(1.,(c[:,ids],c[:,ids]))]
def _open(c,no):
    a=c[:,list(range(1,no))+[no]];b=c[:,:no]
    return [(2**-.5,(a,b)),(2**-.5,(b,a))]


def _basis(ground,states,auxiliary,labels,no,auxiliary_records=None):
    basis=[_closed(ground,list(range(no)))]+[_open(c,no) for c in states]
    basis.extend(_closed(c,list(range(1,no))+[no]) for c in states)
    if auxiliary_records is not None:return basis+configuration_basis(auxiliary,auxiliary_records)
    core=list(range(no-2))
    for label in labels[1+2*len(states):-2]:
        if label.startswith('pair_'):basis.append(_closed(auxiliary,core+list(ast.literal_eval(label[5:]))))
        elif label.startswith('open_'):
            _,pair,opened=label.split('_',2);pair=int(pair);opened=ast.literal_eval(opened)
            a=auxiliary[:,core+[pair,opened[0]]];b=auxiliary[:,core+[pair,opened[1]]]
            basis.append([(2**-.5,(a,b)),(2**-.5,(b,a))])
        else:raise ValueError('invalid frozen auxiliary CSF label')
    return basis+four_singlets(auxiliary,no)


def _cross_metric(left,right,*,nbase=None,auxiliary_gram=None):
    result=np.zeros((len(left),len(right)))
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            if auxiliary_gram is not None and i>=nbase and j>=nbase:
                result[i,j]=auxiliary_gram[i-nbase,j-nbase];continue
            for wa,(aa,ab) in a:
                for wb,(ba,bb) in b:result[i,j]+=wa*wb*np.linalg.det(aa.T@ba)*np.linalg.det(ab.T@bb)
    return result


@dataclass(frozen=True)
class NOSIStateResult:
    state_id: str
    energy_hartree: float
    excitation_hartree: float
    root: int
    overlap: float
    margin: float
    gradient_hartree_per_bohr: object = None
    hessian_hartree_per_bohr2: object = None
    spin_squared: float = 0.


@dataclass(frozen=True)
class NOSISurfaceResult:
    states: tuple
    ground_energy_hartree: float
    model_fingerprint: str
    coordinates_angstrom: np.ndarray
    overlap: np.ndarray
    transport_u: np.ndarray
    ground_coefficients: np.ndarray
    state_coefficients: np.ndarray
    multipliers: np.ndarray
    ci_coefficients: np.ndarray
    maximum_stationary_residual: float


class PreparedNOSIModel:
    """Checksummed, opt-in selected-CID model; evaluation uses only MOUSE."""
    def __init__(self,path):
        self.path=Path(path).expanduser().resolve();self.root=self.path.parent;self.fingerprint=_sha(self.path)
        self.manifest=json.loads(self.path.read_text());m=self.manifest
        if m.get('schema') not in ('mouse.nosi.selected_cid.v1','mouse.nosi.screened_even_cid.v1','mouse.nosi.screened_cisd.v1') or m.get('weights')!=[.5,.5] or m.get('lambda_exchange')!=.9 or m.get('energy_offset_ev')!=0:raise ValueError('unsupported NOSI model contract')
        if set(m.get('sha256',{}))!={'template.npz','response.npz','arrays.npz'}:raise ValueError('incomplete NOSI model')
        self.validate_files();self.template=_load_point(self.root/'template.npz');self.anchor=self.template.reference_state
        self.response=load_response_reference(self.root/'response.npz')
        if not isinstance(self.response,FrozenNonlocalResponseReference):raise ValueError('NOSI requires its full nonlocal reference')
        if abs(self.response.response_exchange_scale-.9)>1e-12:raise ValueError('NOSI response exchange scale differs')
        with np.load(self.root/'arrays.npz',allow_pickle=False) as z:self.arrays={k:z[k].copy() for k in z.files}
        self.extended=m['schema'] in ('mouse.nosi.screened_even_cid.v1','mouse.nosi.screened_cisd.v1')
        augmentation='frozen_reflection_even_doubles_complete_singles' if m['schema']=='mouse.nosi.screened_cisd.v1' else 'frozen_reflection_even'
        if self.extended and (m.get('active_space_size')!=5 or m.get('interaction')!='static_constrained_RPA' or m.get('augmentation')!=augmentation):raise ValueError('unsupported screened CID preparation')
        self.no=self.anchor.electron_count//2;self.labels=m['labels'];self.state_ids=tuple(m['state_ids']);a=self.arrays;n=len(self.anchor.overlap);ns=len(self.state_ids)
        expected={'h':(n,n),'eri':(n,n,n,n),'ground':(n,n),'states':(ns,n,n),'multipliers':(ns,),'auxiliary_frame':(n,n),'reference_ci':(len(self.labels),ns)}
        if self.no<2 or n-self.no<2:raise ValueError('NOSI frontier requires two occupied and two virtual orbitals')
        if not np.array_equal(self.response.atomic_numbers,self.anchor.atomic_numbers) or not np.allclose(self.response.anchor_coordinates_bohr,self.anchor.coordinates_bohr,atol=1e-12,rtol=0):raise ValueError('NOSI response anchor differs')
        if set(a)!=set(expected) or not ns or len(set(self.state_ids))!=ns:raise ValueError('invalid NOSI state set')
        for name,shape in expected.items():
            if a[name].shape!=shape or not np.all(np.isfinite(a[name])):raise ValueError('invalid NOSI array '+name)
            a[name].setflags(write=False)
        if self.labels[:1+ns]!=['S0',*self.state_ids] or len(self.labels)<1+2*ns+2:raise ValueError('invalid NOSI CSF structure')
        if not self.extended and self.labels[-2:]!=['four_spin_0','four_spin_1']:raise ValueError('invalid NOSI frontier recoupling')
        self.auxiliary_records=validate_configurations(configuration_records(self.labels[1+2*ns:],self.no),self.no,n) if self.extended else None
        for c in [a['ground'],*a['states']]:
            if not np.allclose(c.T@self.anchor.overlap@c,np.eye(n),atol=1e-8,rtol=0):raise ValueError('invalid prepared orbitals')
        if not np.allclose(a['auxiliary_frame'].T@a['auxiliary_frame'],np.eye(n),atol=1e-8,rtol=0):raise ValueError('invalid auxiliary anchor frame')
        if not np.allclose(a['h'],a['h'].T,atol=1e-9,rtol=0):raise ValueError('non-Hermitian one-body operator')
        w=a['eri']
        if not np.allclose(w,w.transpose(2,3,0,1),atol=1e-9,rtol=0) or not np.allclose(w,w.transpose(1,0,3,2),atol=1e-9,rtol=0):raise ValueError('non-Hermitian two-body operator')
        self._anchor_sqrt=metric_power(self.anchor.overlap,.5)
        self.anchor_basis=_basis(self._anchor_sqrt@a['ground'],[self._anchor_sqrt@c for c in a['states']],a['auxiliary_frame'],self.labels,self.no,self.auxiliary_records)
        gram=_cross_metric(self.anchor_basis,self.anchor_basis)
        self._auxiliary_gram=gram[1+2*ns:,1+2*ns:].copy()
        if not np.allclose(a['reference_ci'].T@gram@a['reference_ci'],np.eye(ns),atol=1e-7,rtol=0):raise ValueError('invalid prepared CI normalization')

    def validate_files(self):
        if _sha(self.path)!=self.fingerprint:raise ValueError('NOSI model manifest changed')
        for name,digest in self.manifest['sha256'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or _sha(path)!=digest:raise ValueError('NOSI input checksum mismatch')

    def evaluate(self,calculator,coordinates_angstrom,*,properties=('energy','gradient'),reference_result=None):
        self.validate_files();properties=tuple(properties)
        if not properties or set(properties)-{'energy','gradient','hessian'}:raise ValueError('invalid NOSI properties')
        derivatives=bool(set(properties)&{'gradient','hessian'});hessian='hessian' in properties
        xyz=np.asarray(coordinates_angstrom,dtype=float)
        if xyz.shape!=self.anchor.coordinates_bohr.shape or not np.all(np.isfinite(xyz)):raise ValueError('invalid NOSI coordinates')
        context=prepare_gfn2_deltascf(calculator,xyz,orbital_reference=self.anchor,spin_reference=self.response);s=context.state.overlap;u=context.charge_correction.transport.u
        if reference_result is None:
            mapping=metric_power(s,-.5)@u@self._anchor_sqrt;ground=mapping@self.arrays['ground'];states=np.array([mapping@c for c in self.arrays['states']]);multipliers=self.arrays['multipliers'].copy();reference_basis=self.anchor_basis;reference_ci=self.arrays['reference_ci']
        else:
            if reference_result.model_fingerprint!=self.fingerprint or tuple(r.state_id for r in reference_result.states)!=self.state_ids:raise ValueError('NOSI restart has a different model or state set')
            mapping=metric_power(s,-.5)@u@reference_result.transport_u.T@metric_power(reference_result.overlap,.5)
            ground=mapping@reference_result.ground_coefficients;states=np.array([mapping@c for c in reference_result.state_coefficients]);multipliers=reference_result.multipliers.copy()
            old=reference_result.transport_u.T@metric_power(reference_result.overlap,.5)
            reference_basis=_basis(old@reference_result.ground_coefficients,[old@c for c in reference_result.state_coefficients],self.arrays['auxiliary_frame'],self.labels,self.no,self.auxiliary_records);reference_ci=reference_result.ci_coefficients
        ground,eg,ground_residual,_=refine_ground(context,ground);residuals=[ground_residual]
        for i,c in enumerate(states):
            q=np.outer(ground[:,:self.no].T@s@c[:,0],ground[:,self.no:].T@s@c[:,self.no]);norm=np.linalg.norm(q)
            if norm<1e-8:raise ValueError('lost retained NOSI orbital character')
            c,mu,energy,residual,_=refine_roks(context,self.arrays['eri'],c,ground,q/norm,multipliers[i]);states[i]=c;multipliers[i]=mu;residuals.append(residual)
        response=MatrixResponse(context,self.response,ground,states,multipliers,self.arrays['auxiliary_frame'],self.labels,self.arrays['h'],self.arrays['eri'],derivatives=derivatives,auxiliary_records=self.auxiliary_records)
        zero=np.zeros(xyz.size if derivatives else 0);h,sjet,egjet=response.matrices(zero)
        p=np.eye(len(self.labels))[:,1:];p[0]-=sjet.v[0,1:]/sjet.v[0,0];sp=p.T@sjet.v@p
        if np.linalg.eigvalsh(sp).min()<1e-8:raise ValueError('unresolved NOSI metric linear dependence')
        energies,z=eigh(p.T@h.v@p,sp);ci=p@z
        current_basis=[[(w,(a.v,b.v)) for w,(a,b) in csf] for csf in response.current_basis]
        overlaps=abs(reference_ci.T@_cross_metric(reference_basis,current_basis,nbase=1+2*len(self.state_ids),auxiliary_gram=self._auxiliary_gram)@ci);ii,jj=linear_sum_assignment(-overlaps)
        roots=np.empty(len(self.state_ids),dtype=int);weights=np.empty(len(roots));margins=np.empty(len(roots))
        for i,j in zip(ii,jj):
            ov=float(overlaps[i,j]);margin=ov-float(np.max(np.delete(overlaps[i],j),initial=0.))
            if ov<.7 or margin<.05:raise ValueError('unresolved NOSI state identity')
            roots[i]=j;weights[i]=ov;margins[i]=margin
        gradients={};hessians={}
        if derivatives:
            for i,root in enumerate(roots):gradients[i]=projected_eigen_jet(h,sjet,int(root)).a.copy().reshape(xyz.shape)
        if hessian:
            matrices=[np.empty((xyz.size,xyz.size)) for _ in roots]
            for column,direction in enumerate(np.eye(xyz.size)):
                hh,ss,_=response.matrices(direction)
                for i,root in enumerate(roots):matrices[i][:,column]=projected_eigen_jet(hh,ss,int(root)).ab
            for i,matrix in enumerate(matrices):
                if np.max(abs(matrix-matrix.T))>2e-6:raise ValueError('NOSI Hessian symmetry validation failed')
                hessians[i]=.5*(matrix+matrix.T)
        results=tuple(NOSIStateResult(identity,float(energies[root]),float(energies[root]-eg),int(root),float(weights[i]),float(margins[i]),gradients.get(i),hessians.get(i)) for i,(identity,root) in enumerate(zip(self.state_ids,roots)))
        return NOSISurfaceResult(results,float(eg),self.fingerprint,xyz.copy(),s.copy(),u.copy(),ground.copy(),states.copy(),multipliers.copy(),ci[:,roots].copy(),float(max(residuals)))


    def apoc_states(self,result):
        """Expand the coherent CI wavefunctions and verify S²=0 through APOC."""
        from matrix_apoc import NonOrthogonalSingletCIState
        if result.model_fingerprint!=self.fingerprint:raise ValueError('NOSI state belongs to another model')
        auxiliary=metric_power(result.overlap,-.5)@result.transport_u@self.arrays['auxiliary_frame']
        basis=_basis(result.ground_coefficients,result.state_coefficients,auxiliary,self.labels,self.no,self.auxiliary_records);answer=[]
        for index,state in enumerate(result.states):
            determinants=[];coefficients=[]
            for ci,csf in zip(result.ci_coefficients[:,index],basis):
                for weight,pair in csf:
                    coefficient=float(ci*weight)
                    if abs(coefficient)>1e-14:determinants.append(pair);coefficients.append(coefficient)
            answer.append(NonOrthogonalSingletCIState(index+1,tuple(determinants),np.array(coefficients),result.overlap,state.energy_hartree))
        return tuple(answer)


def save_nosi_result(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(schema='mouse.nosi.result.v1',model_fingerprint=result.model_fingerprint,ground_energy_hartree=result.ground_energy_hartree,maximum_stationary_residual=result.maximum_stationary_residual,states=[dict(state_id=r.state_id,energy_hartree=r.energy_hartree,excitation_hartree=r.excitation_hartree,root=r.root,overlap=r.overlap,margin=r.margin) for r in result.states])
    arrays={name:getattr(result,name) for name in ['coordinates_angstrom','overlap','transport_u','ground_coefficients','state_coefficients','multipliers','ci_coefficients']}
    for i,state in enumerate(result.states):
        if state.gradient_hartree_per_bohr is not None:arrays[f'gradient_{i}']=state.gradient_hartree_per_bohr
        if state.hessian_hartree_per_bohr2 is not None:arrays[f'hessian_{i}']=state.hessian_hartree_per_bohr2
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,metadata=json.dumps(metadata),**arrays)
    temporary.replace(path)


def load_nosi_result(path,model):
    model.validate_files()
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in z.files if k!='metadata'}
    if metadata.get('schema')!='mouse.nosi.result.v1' or metadata.get('model_fingerprint')!=model.fingerprint:raise ValueError('incompatible NOSI restart')
    states=[]
    for i,row in enumerate(metadata['states']):states.append(NOSIStateResult(**row,gradient_hartree_per_bohr=arrays.pop(f'gradient_{i}',None),hessian_hartree_per_bohr2=arrays.pop(f'hessian_{i}',None)))
    if tuple(r.state_id for r in states)!=model.state_ids:raise ValueError('NOSI restart state set differs')
    if not np.all(np.isfinite([metadata['ground_energy_hartree'],metadata['maximum_stationary_residual']])) or metadata['maximum_stationary_residual']<0:raise ValueError('invalid NOSI restart metadata')
    for state in states:
        if not np.all(np.isfinite([state.energy_hartree,state.excitation_hartree,state.overlap,state.margin])) or not 0<=state.root<len(model.labels)-1:raise ValueError('invalid NOSI restart state')
        for value,shape in [(state.gradient_hartree_per_bohr,model.anchor.coordinates_bohr.shape),(state.hessian_hartree_per_bohr2,(model.anchor.coordinates_bohr.size,)*2)]:
            if value is not None and (value.shape!=shape or not np.all(np.isfinite(value))):raise ValueError('invalid NOSI restart derivative')

    n=len(model.anchor.overlap);ns=len(states);shapes={'coordinates_angstrom':model.anchor.coordinates_bohr.shape,'overlap':(n,n),'transport_u':(n,n),'ground_coefficients':(n,n),'state_coefficients':(ns,n,n),'multipliers':(ns,),'ci_coefficients':(len(model.labels),ns)}
    if set(arrays)!=set(shapes):raise ValueError('incomplete NOSI restart')
    for name,shape in shapes.items():
        if arrays[name].shape!=shape or not np.all(np.isfinite(arrays[name])):raise ValueError('invalid NOSI restart array')
    return NOSISurfaceResult(tuple(states),metadata['ground_energy_hartree'],model.fingerprint,maximum_stationary_residual=metadata['maximum_stationary_residual'],**arrays)
