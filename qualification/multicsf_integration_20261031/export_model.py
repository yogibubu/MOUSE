"""Export existing stationary rank-two campaigns; no new electronic solve."""
from pathlib import Path
import os,sys,socket,json,hashlib,fcntl
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');W=V/'multicsf_integration';D=Path('/home/enzo/MOUSE-state-following-20261031/reduced_validation');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
os.environ.update(MOUSE_VALIDATION_ROOT=str(W),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');sys.path.insert(0,str(W/'source/qualification'));import remote_bootstrap
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference,write_multicsf_model,MultiCSFModel
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.restricted_ground import RestrictedGroundResult,evaluate_restricted_ground
from matrix_gfn2.deltascf_derivatives import save_derivative_point
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p) as a:x=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
 return load_derivative_point(p,request_hash=req,coordinates_angstrom=x,require_gradient=False)

def export(name,sid):
 targetdir=W/'models'/name/sid;manifest=targetdir/'model.json'
 if manifest.exists():MultiCSFModel(manifest);print('Existing model retained',manifest);return
 choices={'B022':['0','1'],'B016':['0','1'],'B024':['0_fr_v1','1_gmf_polish_v1']};attempts=choices[sid]
 cp=D/'prepared'/name/'ground.npz';template=load(cp);s=template.reference_state;xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;hashes={str(cp):sha(cp)}
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);responses=[];refs=[];points=[];projected=[]
 case=json.loads((D/'prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid);transition=Path(target['transition_file']);hashes[str(transition)]=sha(transition)
 with np.load(transition) as a:amp=a['amplitude'].copy();oi=a['occupied'].copy();vi=a['virtual'].copy()
 no=len(oi);weights=np.linalg.svd(amp,compute_uv=False)**2;weights/=weights.sum()
 for i,attempt in enumerate(attempts):
  modelpath=P/'results/full_response'/name/'model.npz' if i==0 else D/'models'/name/'model.npz';hashes[str(modelpath)]=sha(modelpath)
  with np.load(modelpath) as a:data={k:a[k].copy() for k in ['operators','charge','spin','q0','linear']}
  responses.append(FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,data,sha(modelpath)))
  spectrum=D/'tda'/name/str(i)/'spectrum.npz';hashes[str(spectrum)]=sha(spectrum)
  with np.load(spectrum) as a:c=a['coefficients'].copy()
  refs.append(c);x=c[:,:no].T@s.overlap@s.coefficients[:,oi]@amp@s.coefficients[:,vi].T@s.overlap@c[:,no:];projected.append(x/np.linalg.norm(x))
  folder=V/'pi_pi_repair/optimized'/name/sid/attempt;report=json.loads((folder/'result.json').read_text());assert report['status']=='stationary_candidate'
  point=folder/('latest_released.npz' if '_fr_' in attempt else 'latest.npz');hashes[str(point)]=sha(point)
  with np.load(point) as a:points.append(a['coefficients'].copy())
 ep=P/'results/full_response'/name/'projected_eri.npz';hashes[str(ep)]=sha(ep)
 with np.load(ep) as a:eri=a['eri'].copy()
 grounds=[]
 for i,response in enumerate([None]+responses):
  cache=W/'ground_cache'/name/str(i);candidates=[cache/'certificate.json']+[p for root in [V,V.parent/'MOUSE-state-following-20261031',P/'blind/mouse'] for p in root.glob(f'**/{name}/**/grounds/{i}/certificate.json')]
  if i>0:candidates.append(D/'static_response_controls'/name/str(i-1)/'ground/certificate.json')
  ground=None
  for certpath in candidates:
   path=certpath.parent/'stationary.npz'
   if not certpath.exists() or not path.exists():continue
   raw=load(path)
   if raw.spin_reference_fingerprint!=(None if response is None else response.fingerprint):continue
   cert=json.loads(certpath.read_text());ground=RestrictedGroundResult(raw,cert['minimum_curvature'],cert['index']);hashes[str(path)]=sha(path);break
  if ground is None:
   # Only absent certificates are computed, retaining completed F/Q seeds.
   guess=template if i==0 and hasattr(template,'occupations') and np.array_equal(template.occupations[0],template.occupations[1]) else None
   ground=evaluate_restricted_ground(calc,xyz,template=s,spin_reference=response,reference_result=guess,gradient=False)
   cache.mkdir(parents=True,exist_ok=True);path=cache/'stationary.npz';save_derivative_point(path,ground.reference,request_hash='multicsf:missing-ground-certificate',coordinates_angstrom=xyz);hashes[str(path)]=sha(path)
   (cache/'certificate.json').write_text(json.dumps(dict(minimum_curvature=ground.minimum_orbital_curvature_hartree,index=ground.orbital_saddle_index,fingerprint=ground.spin_reference_fingerprint,reason='missing stationary ground certificate; completed checkpoints reused when present'),indent=2)+'\n')
   print('Created missing ground certificate',name,i,flush=True)
  grounds.append(ground)
 modelpath=write_multicsf_model(targetdir,template=template,responses=responses,eri=eri,initial_coefficients=points,reference_coefficients=refs,reference_transitions=projected,transition_weights=weights,grounds=grounds,provenance=dict(host=socket.getfqdn(),state=sid,reference_root=target['reference_root'],input_sha256=hashes,labels_already_exposed=True,energy_inputs_to_rank_selection=False,rank_rule='nearest integer inverse NTO-weight purity'))
 print('Exported',modelpath,flush=True)
if __name__=='__main__':
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 with threadpool_limits(limits=2):export(sys.argv[1],sys.argv[2])
