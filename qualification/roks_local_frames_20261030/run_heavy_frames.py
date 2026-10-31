"""Remote-only qualification of the fixed local-frame ROKS 50/50 candidate."""
import os,socket,sys,json,time,hashlib,traceback
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
BASE=Path('/home/enzo/MOUSE-roks-heavy-frames-20261030');DIRECT=Path('/home/enzo/MOUSE-roks-direct-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');OPT=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030')
assert socket.getfqdn()=='oracle.sns.it';os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,str(OLD));import remote_bootstrap
for path in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(path))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import (GFN2Calculator,FrozenNonlocalResponseReference,FrozenLocalFrameResponseReference,save_response_reference,load_response_reference,combine_roks_results)
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.roks import evaluate_gfn2_roks_singlet
from matrix_gfn2.roks_orbital_response import run_roks_gmf
from matrix_gfn2.deltascf_derivatives import save_derivative_point,load_derivative_point
from matrix_gfn2.roks_tracking import active_orbital_character
from matrix_link.scan import QMScanBackend
from matrix_link.qmqm_component import CappedQMComponent
from matrix_link.optimizer import _exact_cartesian_normal_modes
PROTOCOL=json.loads((BASE/'protocol.json').read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def load(path,request):
 with np.load(path) as d:xyz=d['coordinates_angstrom'].copy()
 return load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)

def prepare(name):
 root=BASE/'results'/name;root.mkdir(parents=True,exist_ok=True)
 if (root/'prepared.json').exists():return json.loads((root/'prepared.json').read_text())
 original=[];global_refs=[];sources=[]
 for i,profile in enumerate(['full_response','qualified_spectral_response']):
  cp=OPT/'results'/name/profile/'seed.npz'
  if cp.exists():
   state=load(cp,'retained_vertical');olddata=json.loads((DIRECT/'results'/name/'prepared.json').read_text());ref=load_response_reference(olddata['response_paths'][i]);source_request='retained_vertical'
  else:
   record=json.loads((OLD/'results'/profile/name/'n_pi/result.json').read_text());assert record['qualified']
   cp=Path(record['checkpoint']);source_request=record['request_hash'];state=load(cp,source_request)
   atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in state.reference_state.atomic_numbers]
   calc=GFN2Calculator(atoms,temperature_kelvin=0.);xyz=state.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
   ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=state.reference_state)
   modelpath=OLD/'results'/profile/name/'model.npz'
   with np.load(modelpath) as d:model={k:d[k].copy() for k in ['operators','charge','spin','q0','linear']}
   ref=FrozenNonlocalResponseReference.from_vertical_model(state.reference_state,ctx.kernel.partition,model,sha(modelpath))
  original.append(state);global_refs.append(ref);sources.append(dict(path=str(cp),sha256=sha(cp),request_hash=source_request))
 atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in original[0].reference_state.atomic_numbers]
 calc=GFN2Calculator(atoms,temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
 refs=[FrozenLocalFrameResponseReference.from_reference(r,heavy_first=True) for r in global_refs];paths=[];anchors=[];seeds=[]
 for i,(ref,old) in enumerate(zip(refs,original)):
  path=root/f'response_{i}.npz';save_response_reference(path,ref);paths.append(str(path))
  xyz=old.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR;ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=old.reference_state,spin_reference=ref)
  fm,em=ctx.fock_and_energy(old.densities());ft,et=ctx.fock_and_energy(old.densities(triplet=True))
  if max(np.max(abs(fm-old.fock_mixed)),np.max(abs(ft-old.fock_triplet)))>1e-9 or abs(2*em-et-old.energy_hartree)>1e-9:raise ValueError('local transport changed the vertical Hamiltonian')
  anchor=replace(old,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=ref.frame_clusters,gradient_hartree_per_bohr=None,ground_energy_hartree=None)
  anchors.append(anchor);save_derivative_point(root/f'anchor_{i}.npz',anchor,request_hash='local_anchor_v1',coordinates_angstrom=xyz)
  source=old;sourcepath=Path(sources[i]['path'])
  if name=='unused_local_source':sourcepath=DIRECT/'results'/name/'local_frames'/f'relaxed_{i}.npz';source=load(sourcepath,'local_frame_probe_v1')
  elif name=='nitroxyl':sourcepath=DIRECT/'results'/name/'ground_replacement'/f'excited_{i}.npz';source=load(sourcepath,'ground_replacement_v1')
  xyz=source.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR;ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=source.reference_state,spin_reference=ref)
  from matrix_gfn2.roks_orbital_response import run_roks_orbital_newton
  result=run_roks_orbital_newton(ctx,source.common_coefficients,source.occupations,orbital_tolerance=1e-10,maximum_evaluations=150,maximum_step=.2,certify_index=True)
  result=replace(result,reference_state=ctx.state,core_electrons=source.core_electrons,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=ref.frame_clusters)
  result=evaluate_gfn2_roks_singlet(calc,xyz,*result.occupations,spin_reference=ref,reference_result=result,
    reference_policy='geometry_only',orbital_solver='gmf',saddle_index=result.orbital_saddle_index,orbital_tolerance=1e-10,gradient=True)
  character=active_orbital_character(anchor,result,ref.anchor_coordinates_bohr)
  if character['transition']<.7:raise ValueError('initial local model lost the specified excitation: '+str(character))
  seeds.append(result);save_derivative_point(root/f'seed_{i}.npz',result,request_hash='local_seed_v1',coordinates_angstrom=xyz)
  sources[i]['geometry_seed']=dict(path=str(sourcepath),sha256=sha(sourcepath),local_pair=character)
 initial=combine_roks_results(*seeds,component_tracking='active_pair')
 ground=original[0].reference_state.alpha_occupations;alpha=original[0].occupations[0]
 data=dict(host=socket.getfqdn(),molecule=name,atoms=atoms,response_paths=paths,
  promotion=[int(np.flatnonzero(ground-alpha==1)[0]),int(np.flatnonzero(alpha-ground==1)[0])],
  saddle_indices=[s.orbital_saddle_index for s in original],sources=sources,frames=[r.frame_clusters.tolist() for r in refs],protocol_sha256=sha(BASE/'protocol.json'))
 write(root/'prepared.json',data);return data


def component_for(name):
 data=prepare(name);root=BASE/'results'/name;calc=GFN2Calculator(data['atoms'],temperature_kelvin=0.)
 calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
 backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(data['promotion']),
  native_roks_composite_response_paths=tuple(data['response_paths']),native_roks_composite_saddle_indices=tuple(data['saddle_indices']),
  native_roks_composite_reference_policy='geometry_only',native_roks_composite_component_tracking='active_pair',
  native_roks_composite_following='character_newton_index',native_roks_composite_ground_surface='native_plus_gaps')
 comp=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),root/'optimization')
 if comp.reference is None:
  seeds=[load(root/f'seed_{i}.npz','local_seed_v1') for i in range(2)];anchors=[load(root/f'anchor_{i}.npz','local_anchor_v1') for i in range(2)]
  comp.reference=(combine_roks_results(*seeds,component_tracking='active_pair'),tuple(r.to_apoc(1) for r in seeds))
  comp.anchor=(combine_roks_results(*anchors,component_tracking='active_pair'),tuple(r.to_apoc(1) for r in anchors))
 return data,comp


def run(name):
 root=BASE/'results'/name
 if (root/'complete.json').exists():return
 data,comp=component_for(name);start=time.monotonic();state=comp.reference[0]
 xyz=state.components[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR;x=xyz.ravel()*ANGSTROM_TO_BOHR
 h=np.eye(x.size)*.3;history=[];escape_count=0
 if (root/'optimizer_state.npz').exists():
  with np.load(root/'optimizer_state.npz') as d:
   if np.max(abs(d['coordinates_angstrom']-xyz))<1e-12:h=d['hessian_seed'].copy();escape_count=int(d['escapes'])
 if (root/'trace.json').exists():history=json.loads((root/'trace.json').read_text())
 point=comp.evaluate(xyz);comp.accept(point);status='maximum_steps';freq=None
 for iteration in range(161):
  g=point.gradient_hartree_per_bohr;energy=point.energy_hartree;maxg=float(np.max(abs(g)))
  row=dict(step=len(history),energy=energy,max_gradient=maxg,component_gaps_ev=[v*27.211386245988 for v in point.scf_state[0].component_gaps_hartree],component_pair=point.scf_state[0].component_active_pair_overlap,local_anchor=point.execution['anchor_active_orbital_character'],indices=point.execution['component_orbital_saddle_indices'],escapes=escape_count)
  history.append(row);write(root/'trace.json',history);print(name,json.dumps(row),flush=True)
  tmp=root/'optimizer_state.tmp'
  with tmp.open('wb') as stream:np.savez_compressed(stream,coordinates_angstrom=x.reshape(-1,3)/ANGSTROM_TO_BOHR,hessian_seed=h,escapes=escape_count)
  tmp.replace(root/'optimizer_state.npz')
  if maxg<3e-5:
   retained=root/f'analytic_stationary_{escape_count}.npz'
   same=False
   if retained.exists():
    with np.load(retained) as d:
     same=np.max(abs(d['xyz']-x.reshape(-1,3)/ANGSTROM_TO_BOHR))<1e-12
     if same:old_hessian=d['hessian'].copy()
   final=replace(point,hessian_hartree_per_bohr2=old_hessian) if same else comp.evaluate(x.reshape(-1,3)/ANGSTROM_TO_BOHR,compute_hessian=True)
   modes=_exact_cartesian_normal_modes(data['atoms'],x.reshape(-1,3)/ANGSTROM_TO_BOHR,final.hessian_hartree_per_bohr2);freq=np.asarray(modes.frequencies_cm)
   np.savez_compressed(root/f'analytic_stationary_{escape_count}.npz',xyz=x.reshape(-1,3)/ANGSTROM_TO_BOHR,hessian=final.hessian_hartree_per_bohr2,gradient=g,frequencies_cm=freq,directions=modes.cartesian_directions)
   if freq.min()>=-20:status='qualified_stationary';point=final;break
   if escape_count>=2:status='negative_modes_remaining';point=final;break
   v=np.asarray(modes.cartesian_directions[0]);v/=np.linalg.norm(v);dx=.05*ANGSTROM_TO_BOHR*v.ravel()
   trial=comp.evaluate((x+dx).reshape(-1,3)/ANGSTROM_TO_BOHR)
   if trial.energy_hartree>=energy:trial=comp.evaluate((x-dx).reshape(-1,3)/ANGSTROM_TO_BOHR);dx=-dx
   if trial.energy_hartree>=energy:status='negative_mode_displacement_not_downhill';break
   comp.accept(trial);point=trial;x=x+dx;escape_count+=1
   values,vectors=np.linalg.eigh(final.hessian_hartree_per_bohr2);h=(vectors*np.maximum(abs(values),1e-4))@vectors.T
   continue
  values,vectors=np.linalg.eigh(.5*(h+h.T));delta=-vectors@((vectors.T@g)/np.maximum(values,1e-5));norm=np.linalg.norm(delta)
  if norm>.15:delta*=.15/norm
  accepted=False;errors=[]
  for fraction in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125]:
   dx=fraction*delta
   try:
    trial=comp.evaluate((x+dx).reshape(-1,3)/ANGSTROM_TO_BOHR)
    if trial.energy_hartree<=energy+1e-4*np.dot(g,dx):accepted=True;break
   except Exception as exc:errors.append(str(exc))
  if not accepted:status='line_search_stalled';write(root/'last_rejections.json',errors);break
  y=trial.gradient_hartree_per_bohr-g;hs=h@dx;sy=np.dot(dx,y);shs=np.dot(dx,hs)
  if shs>1e-16:
   if sy<.2*shs:
    theta=.8*shs/(shs-sy);y=theta*y+(1-theta)*hs;sy=np.dot(dx,y)
   h+=np.outer(y,y)/sy-np.outer(hs,hs)/shs
  comp.accept(trial);point=trial;x=x+dx
 out=dict(host=socket.getfqdn(),molecule=name,status=status,seconds=time.monotonic()-start,maximum_gradient=float(np.max(abs(point.gradient_hartree_per_bohr))),energy=point.energy_hartree,xyz=(x.reshape(-1,3)/ANGSTROM_TO_BOHR).tolist(),frequencies_cm=None if freq is None else freq.tolist(),steps=len(history),escapes=escape_count,execution=point.execution)
 if status=='qualified_stationary':
  v=np.sin(np.arange(x.size)+.7).reshape(-1,3);v-=v.mean(axis=0);v/=np.linalg.norm(v);checks=[]
  for step in [2e-4,1e-4]:
   minus=comp.evaluate(x.reshape(-1,3)/ANGSTROM_TO_BOHR-step/ANGSTROM_TO_BOHR*v);plus=comp.evaluate(x.reshape(-1,3)/ANGSTROM_TO_BOHR+step/ANGSTROM_TO_BOHR*v)
   checks.append(dict(step=step,gradient_error=float(abs((plus.energy_hartree-minus.energy_hartree)/(2*step)-point.gradient_hartree_per_bohr@v.ravel())),hvp_error=float(np.max(abs((plus.gradient_hartree_per_bohr-minus.gradient_hartree_per_bohr)/(2*step)-point.hessian_hartree_per_bohr2@v.ravel())))))
  out['checks']=checks;out['passed']=all(c['gradient_error']<1e-7 and c['hvp_error']<1e-6 for c in checks)
 else:out['passed']=False
 write(root/'complete.json',out);print('COMPLETE',name,json.dumps(out),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=2):run(sys.argv[1])
