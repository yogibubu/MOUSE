"""Frozen, label-free MOUSE 50/50 predictions with checkpointed stationary parents.
Runs only on Oracle, serially after reference preparation releases compute.lock.
Pure helpers are extracted from audited earlier campaign sources without executing
their development loops. No numerical reference label is read by this program.
"""
from pathlib import Path
import socket,os,sys,json,hashlib,ast,argparse,traceback,time,subprocess,signal,datetime,re
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
P=Path(__file__).resolve().parent;SNAP=Path('/home/enzo/MOUSE-roks-relative-pair-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');ROOT=P/'blind/mouse';ROOT.mkdir(exist_ok=True)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 def encode(a):return a.tolist() if hasattr(a,'tolist') else a.item()
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=encode)+'\n');tmp.replace(path)
def get_plan():
 sources=[OLD/'prepare_F.py',OLD/'full_response.py',OLD/'full_fast.py',OLD/'qualified_spectral_response.py',SNAP/'source/qualification/remote_bootstrap.py']
 sources+=sorted((SNAP/'source/packages/matrix-gfn2/src/matrix_gfn2').glob('*.py'))
 sources+=sorted((SNAP/'source/packages/matrix-apoc/src/matrix_apoc').glob('*.py'))
 return dict(schema='mouse.blind.prediction.v1',host=socket.getfqdn(),script_sha256=digest(__file__),manifest_sha256=digest(P/'blind/manifest.json'),assignment_catalog_sha256=digest(P/'blind/assignment_catalog.json'),candidate_protocol_sha256=digest(P/'inputs/candidate_source_protocol.json'),weights=[.5,.5],lambda_exchange=.9,orbital_gap_scale=1.,energy_offset_ev=0.,threads=2,saddle_indices=list(range(6)),parent_iterations=100,native_iterations=120,tolerance=1e-9,csf_overlap=.7,transition_overlap=.7,projection_norm=.7,assignment_margin=.05,relative_max_iterations=150,timeout_seconds_per_molecule=3600,calibration='unchanged full AO-pair PBE0 charge/spin response and qualified spectral one-body shift; two-thread execution',runtime_adaptations=['remove reporting of tbe_ev and error_ev; no numerical labels available','reference-file orbital_energies alias; normalize PySCF TDA X vectors to unit norm before projection','freeze-and-release indices 0..5 for all states; no score-based recovery','all imported helper BLAS thread limits capped at two'],source_sha256={str(p):digest(p) for p in sources},read_sealed_labels=False)
def normsym(s):
 return re.sub(r'[{} _]','',s.removeprefix('^1')).replace("''",'"')
def worker(name):
 global BASE,HOST,np,SimpleNamespace,replace,solve_sylvester,linear_sum_assignment,threadpool_limits,EV
 global GFN2Calculator,evaluate_gfn2_deltascf,evaluate_gfn2_roks_singlet,ANGSTROM_TO_BOHR,stationary_response_reference,prepare_gfn2_deltascf,ROKSSingletResult,load_derivative_point,save_derivative_point,singlet_csf_overlap,scf,gto,symm,lib,source_molecule
 BASE=P;HOST=socket.getfqdn();EV=27.211386245988
 os.environ['MOUSE_VALIDATION_ROOT']=str(SNAP);sys.path.insert(0,str(SNAP/'source/qualification'));import remote_bootstrap
 import numpy as np
 from types import SimpleNamespace,ModuleType
 from dataclasses import replace
 from scipy.linalg import solve_sylvester,eigh
 from scipy.optimize import linear_sum_assignment
 from threadpoolctl import threadpool_limits as raw_threadpool_limits
 def threadpool_limits(limits=2,**kwargs):return raw_threadpool_limits(limits=2,**kwargs)
 from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf,evaluate_gfn2_roks_singlet,FrozenResponseReference,FrozenNonlocalResponseReference
 from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
 from matrix_gfn2.response_reference import stationary_response_reference
 from matrix_gfn2.deltascf import prepare_gfn2_deltascf
 from matrix_gfn2.roks import ROKSSingletResult
 from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
 from matrix_apoc.deltascf import singlet_csf_overlap
 from pyscf import scf,gto,symm,lib,dft,ao2mo
 sys.path.append('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');from extent_support import source_molecule
 lib.num_threads(2)
 with threadpool_limits(2):
  manifest=json.loads((P/'blind/manifest.json').read_text());public=json.loads((P/'blind/assignment_catalog.json').read_text());items=[r for r in manifest['states'] if r['molecule']==name]
  reference=P/'blind/references'/name;out=ROOT/name;out.mkdir(exist_ok=True)
  if not (reference/'result.json').exists():raise RuntimeError('DFT reference spectrum unavailable; checkpoints preserved')
  rd=json.loads((reference/'result.json').read_text());targets=[];unassigned=[]
  with np.load(reference/'backend_tda.npz') as f:labels=f['state_symmetries'];converged=f['converged']
  for row in items:
   assignment=next(r for r in public['states'] if r['id']==row['id']);wanted=normsym(row['state_label']);roots=[i+1 for i,l in enumerate(labels) if normsym(str(l))==wanted]
   ordinal=assignment['ordinal_same_singlet_symmetry']
   if len(roots)<ordinal or not converged[roots[ordinal-1]-1]:
    unassigned.append(dict(id=row['id'],status='completed_unassigned',reason='Fixed symmetry/ordinal not covered by converged reference roots or point-group correlation unresolved',target_symmetry=wanted,ordinal=ordinal,point_group=rd['point_group'],available_roots=roots));continue
   targets.append(dict(row,id=row['id'],symmetry=wanted,reference_root=roots[ordinal-1],prior_assignment_present=False))
  write(out/'reference_assignment.json',dict(targets=targets,unassigned=unassigned,assignment_catalog_sha256=digest(P/'blind/assignment_catalog.json'),reference_sha256=digest(reference/'backend_tda.npz'),reference_labels_read=False))
  if not targets:write(out/'result.json',dict(status='completed_unassigned',rows=unassigned));return
  # Execute only the unchanged prepare function, not the old development script.
  prepared=P/'blind/prepared';prepared.mkdir(exist_ok=True);env=dict(globals(),ROOT=prepared,PROTOCOL={'saddle_indices':list(range(6))})
  def extract(path,names,namespace,transform=None):
   tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
   if len(nodes)!=len(names):raise RuntimeError('Source API changed: '+str(path))
   mod=ast.Module(body=nodes,type_ignores=[])
   if transform:mod=transform.visit(mod);ast.fix_missing_locations(mod)
   namespace['__file__']=str(path);exec(compile(mod,str(path),'exec'),namespace)
  extract(OLD/'prepare_F.py',['prepare'],env)
  # PySCF X amplitudes carry norm 1/sqrt(2); archived matching uses unit norm.
  # Keep original spectra immutable and export a certified unit-normalized copy.
  canonical=P/'blind/mouse_reference'/name;canonical.mkdir(parents=True,exist_ok=True)
  for filename in ['reference.npz','backend.chk']:
   link=canonical/filename
   if not link.exists():link.symlink_to(reference/filename)
  vector=canonical/'backend_tda.npz'
  if not vector.exists():
   with np.load(reference/'backend_tda.npz') as f:values={k:f[k].copy() for k in f.files}
   norms=np.linalg.norm(values['amplitudes'],axis=1)
   if np.any(norms<1e-12):raise RuntimeError('Zero reference transition vector')
   values['amplitudes']/=norms[:,None];np.savez_compressed(vector,**values)
   write(canonical/'normalization.json',dict(source_sha256=digest(reference/'backend_tda.npz'),original_norms=norms,export_sha256=digest(vector)))
  case=dict(name=name,cohort='frozen_blind_20261030',reference=str(canonical),targets=targets)
  env['prepare'](case);case=json.loads((prepared/name/'case.json').read_text())
  for r in case['excluded']:
   unassigned.append(dict(id=r['state'],status='not_representable' if any(s in r['error'] for s in ['compact projection','one compact orbital promotion']) else 'nonconverged',reason=r['error']))
  if not case['targets']:write(out/'result.json',dict(status='no_qualified_native_seeds',rows=unassigned));return
  def state_from(target):
   cp=Path(target['checkpoint'])
   with np.load(cp) as f:xyz=f['coordinates_angstrom'].copy()
   return load_derivative_point(cp,request_hash=target['request_hash'],coordinates_angstrom=xyz,require_gradient=False),xyz
  class RemoveReferenceScores(ast.NodeTransformer):
   def visit_Call(self,node):
    self.generic_visit(node)
    if isinstance(node.func,ast.Name) and node.func.id=='dict':node.keywords=[x for x in node.keywords if x.arg not in ['tbe_ev','error_ev']]
    return node
  full=ModuleType('blind_full_response');full.__dict__.update(globals());full.__dict__.update(eigh=eigh,dft=dft,ao2mo=ao2mo,FrozenResponseReference=FrozenResponseReference,state_from=state_from,ROOT=P/'results/full_response')
  full.ROOT.mkdir(parents=True,exist_ok=True)
  extract(OLD/'full_response.py',['symmetric_basis','MetricTerm','FullSpin','VerticalReference','run'],full.__dict__,RemoveReferenceScores())
  fast=dict(globals(),full=full,eigh=eigh,dft=dft,ao2mo=ao2mo)
  extract(OLD/'full_fast.py',['features','invsqrt','calibrate'],fast);full.calibrate=fast['calibrate'];full.run(case)
  spectralroot=P/'results/qualified_spectral_response';spectralroot.mkdir(exist_ok=True)
  spectralprotocol=ROOT/'spectral_protocol.json'
  expected=dict(profile='unchanged qualified spectral shift',source_sha256=digest(OLD/'qualified_spectral_response.py'),blind_protocol_sha256=digest(ROOT/'protocol.json'))
  if spectralprotocol.exists():assert json.loads(spectralprotocol.read_text())==expected
  else:write(spectralprotocol,expected)
  class ReferenceAlias(ast.NodeTransformer):
   def visit_Constant(self,node):
    if node.value=='orbital_energies_hartree':return ast.copy_location(ast.Constant(value='orbital_energies'),node)
    return node
  spectral=dict(globals(),ROOT=spectralroot,p=spectralprotocol,linear_sum_assignment=linear_sum_assignment)
  extract(OLD/'qualified_spectral_response.py',['calibrate'],spectral,ReferenceAlias());full.ROOT=spectralroot;full.calibrate=spectral['calibrate'];full.run(case)
  from matrix_gfn2.relative_pair_roks import evaluate_relative_pair_roks,save_relative_pair_roks,load_relative_pair_roks
  from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
  from matrix_gfn2.restricted_ground import evaluate_restricted_ground,RestrictedGroundResult
  from matrix_gfn2.roks_tracking import active_orbital_character
  rows=list(unassigned)
  for target in case['targets']:
   sid=target['id'];dest=out/sid;dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():rows.append(json.loads((dest/'result.json').read_text()));continue
   try:
    parents=[];responses=[];identity=[]
    with np.load(target['transition_file']) as f:amplitude=f['amplitude'];occupied=f['occupied'];virtual=f['virtual']
    for profile in ['full_response','qualified_spectral_response']:
     folder=P/'results'/profile/name;report=json.loads((folder/sid/'result.json').read_text())
     if not report.get('qualified'):raise RuntimeError('No qualified '+profile+' stationary parent')
     parent,xyz=state_from(report);s=parent.reference_state;calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
     native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);modelpath=folder/'model.npz'
     with np.load(modelpath) as f:model={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
     response=FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,model,digest(modelpath));parent=replace(parent,spin_reference_fingerprint=response.fingerprint)
     h=s.coefficients[:,occupied].T@s.overlap@parent.common_coefficients[:,target['hole']];p=s.coefficients[:,virtual].T@s.overlap@parent.common_coefficients[:,target['particle']];rank=np.outer(h,p)
     score=float(abs(np.sum(amplitude*rank))/np.linalg.norm(rank));identity.append(dict(profile=profile,transition_overlap=score,hole_norm=float(np.linalg.norm(h)),particle_norm=float(np.linalg.norm(p))))
     if min(score,np.linalg.norm(h),np.linalg.norm(p))<.7:raise RuntimeError('Assigned transition lost in '+profile)
     parents.append(parent);responses.append(response)
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC in final direct ground response'))
    def checkpoint(data):
     with (dest/'latest.tmp.npz').open('wb') as f:np.savez_compressed(f,**data)
     (dest/'latest.tmp.npz').replace(dest/'latest.npz')
    if (dest/'stationary.npz').exists():
     joint=load_relative_pair_roks(dest/'stationary.npz',request_hash='blind_relative_v1',require_gradient=False)
    else:
     warm=parents;resume={}
     if (dest/'latest.npz').exists():
      with np.load(dest/'latest.npz') as f:
       warm=[replace(p,common_coefficients=c.copy()) for p,c in zip(parents,f['coefficients'])]
       resume=dict(target_squared_overlaps=f['target_squared_overlaps'].copy(),initial_multipliers=f['multipliers'].copy())
     joint=evaluate_relative_pair_roks(calc,xyz,responses,initial_components=warm,anchor_components=parents,minimum_state_overlap=.7,maximum_iterations=150,checkpoint=checkpoint,**resume)
     save_relative_pair_roks(dest/'stationary.npz',joint,request_hash='blind_relative_v1')
    following=[active_orbital_character(old,new,ref.anchor_coordinates_bohr) for old,new,ref in zip(parents,joint.components,responses)]
    if any(r['transition']<.7 for r in following):raise RuntimeError('Joint solution lost assigned parent character')
    grounds=[]
    for i,response in enumerate([None]+responses):
     gdir=out/'grounds'/str(i);gdir.mkdir(parents=True,exist_ok=True)
     req=key(dict(model=None if response is None else response.fingerprint,geometry=targets[0]['geometry_sha256']))
     if (gdir/'certificate.json').exists():
      cert=json.loads((gdir/'certificate.json').read_text());assert cert['request_hash']==req
      raw=load_derivative_point(gdir/'stationary.npz',request_hash=req,coordinates_angstrom=xyz,require_gradient=False)
      ground=RestrictedGroundResult(raw,cert['minimum_curvature'],cert['index'])
     else:
      ground=evaluate_restricted_ground(calc,xyz,template=parents[0].reference_state,spin_reference=response,gradient=False)
      save_derivative_point(gdir/'stationary.npz',ground.reference,request_hash=req,coordinates_angstrom=xyz)
      write(gdir/'certificate.json',dict(request_hash=req,minimum_curvature=ground.minimum_orbital_curvature_hartree,index=ground.orbital_saddle_index))
     grounds.append(ground)
    total=GroundReferencedROKSResult(joint,tuple(grounds))
    gaps=list(total.component_gaps_hartree);gap=total.excitation_energy_hartree*EV
    qualified=bool(total.spectrum_qualified and gap>0 and joint.orbital_residual<=1e-8 and joint.minimum_absolute_orbital_curvature_hartree>=1e-7)
    for i,g in enumerate(total.grounds):save_derivative_point(dest/f'ground_{i}.npz',g.reference,request_hash='blind_corrected_S0_v1',coordinates_angstrom=xyz)
    row=dict(id=sid,molecule=name,family=target['family'],status='matched' if qualified else 'nonconverged',excitation_ev=gap,parent_gaps_ev=[float(x*EV) for x in gaps],joint_residual=joint.orbital_residual,joint_index=joint.orbital_saddle_index,minimum_curvature_hartree=joint.minimum_absolute_orbital_curvature_hartree,identity=identity,weights=[.5,.5],lambda_exchange=.9,energy_offset=0.,reference_root=target['reference_root'],numerical_reference_labels_read=False)
   except Exception as exc:row=dict(id=sid,molecule=name,status='nonconverged',reason=str(exc),traceback=traceback.format_exc())
   write(dest/'result.json',row);rows.append(row);print(name,sid,row['status'],flush=True)
  assert set(r['id'] for r in rows)==set(r['id'] for r in items)
  write(out/'result.json',dict(status='prediction_pass_complete',host=HOST,molecule=name,rows=rows,numerical_reference_labels_read=False))
def orchestrate():
 import fcntl
 lock=(P/'blind/references/compute.lock').open('a');print('Waiting for serial reference batch to release compute lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
 plan=get_plan();protocol=ROOT/'protocol.json'
 if protocol.exists():assert json.loads(protocol.read_text())==plan
 else:write(protocol,plan)
 manifest=json.loads((P/'blind/manifest.json').read_text());molecules={r['molecule']:r['n_atoms'] for r in manifest['states']}
 for name in sorted(molecules,key=lambda n:(molecules[n],n)):
  d=ROOT/name;d.mkdir(exist_ok=True)
  if (d/'result.json').exists():continue
  with (d/'worker.log').open('a') as log:
   proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(d/'resources.txt'),sys.executable,str(Path(__file__)),'--worker',name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(ROOT/'progress.json',dict(status='running',molecule=name,pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),labels_read=False))
   try:rc=proc.wait(timeout=3600)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=-9
  if rc:write(d/'failure.json',dict(status='timeout' if rc==-9 else 'execution_error',returncode=rc,checkpoint_preserved=True))
 write(ROOT/'progress.json',dict(status='prediction_pass_complete',labels_read=False,scoring='not performed; all-state ledger and method coverage must be audited before unsealing'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
 if a.freeze:
  plan=get_plan();f=ROOT/'protocol.json'
  if f.exists():assert json.loads(f.read_text())==plan
  else:write(f,plan)
  print('Frozen MOUSE prediction protocol',digest(f))
 elif a.worker:worker(a.worker)
 else:orchestrate()
