"""Label-free parent TDA controls on each corrected GFN2 ground's own orbitals."""
from pathlib import Path
import socket,os,sys,json,hashlib,time,resource
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation'
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from scipy.linalg import eigh
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.restricted_ground import evaluate_restricted_ground,RestrictedGroundResult
name=sys.argv[1];EV=27.211386245988
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def load(p):
 with np.load(p) as f:x=f['coordinates_angstrom'].copy();q=str(f['request_hash'])
 return load_derivative_point(p,request_hash=q,coordinates_angstrom=x,require_gradient=False)
with threadpool_limits(limits=2):
 final=D/'tda'/name/'result.json'
 if final.exists():sys.exit(0)
 s=load(D/'prepared'/name/'ground.npz').reference_state;xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
 case=json.loads((D/'prepared'/name/'case.json').read_text());scope=json.loads((R/'reduced_scope.json').read_text());ids={r['id'] for r in scope['primary_states']}|{'B039'};targets=[t for t in case['targets'] if t['id'] in ids]
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
 native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);profiles=[]
 for index,path in enumerate([P/'results/full_response'/name/'model.npz',D/'models'/name/'model.npz']):
  out=D/'tda'/name/str(index);out.mkdir(parents=True,exist_ok=True)
  if (out/'result.json').exists():profiles.append(json.loads((out/'result.json').read_text()));continue
  with np.load(path) as f:model={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
  ref=FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,model,sha(path));ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref)
  gp=None;ground_start=time.perf_counter();reused=False
  caches=[out/'ground',D/'static_response_controls'/name/str(index)/'ground']+list((D/'results'/name).glob(f'*/grounds/{index+1}'))+list((R/'ritz_projection_probe/results'/name).glob(f'*/grounds/{index+1}'))+[P/'blind/mouse'/name/'grounds'/str(index+1)]
  for cache in caches:
   cp=cache/'stationary.npz';cert=cache/'certificate.json'
   if not cp.exists() or not cert.exists():continue
   raw=load(cp)
   if raw.spin_reference_fingerprint!=ref.fingerprint:continue
   metadata=json.loads(cert.read_text());gp=RestrictedGroundResult(raw,metadata['minimum_curvature'],metadata['index']);reused=True;break
  if gp is None:
   gp=evaluate_restricted_ground(calc,xyz,template=s,spin_reference=ref,gradient=False)
   cache=out/'ground';cache.mkdir(exist_ok=True);save_derivative_point(cache/'stationary.npz',gp.reference,request_hash='reduced_parent_tda_ground',coordinates_angstrom=xyz)
   write(cache/'certificate.json',dict(fingerprint=ref.fingerprint,index=gp.orbital_saddle_index,minimum_curvature=gp.minimum_orbital_curvature_hartree,residual=gp.orbital_residual))
  assert gp.converged and gp.orbital_residual<=1e-8 and gp.orbital_saddle_index==0 and gp.minimum_orbital_curvature_hartree>0
  assert gp.spin_reference_fingerprint==ref.fingerprint and np.array_equal(gp.occupations[0],gp.occupations[1])
  ground_seconds=time.perf_counter()-ground_start
  static_cache=D/'static_response_controls'/name/str(index)/'spectrum.npz'
  cached=static_cache.exists()
  if cached:
   oldreport=json.loads((static_cache.parent/'result.json').read_text());assert oldreport['model_sha256']==sha(path) and oldreport['fingerprint']==ref.fingerprint
   with np.load(static_cache) as f:c=f['coefficients'].copy();static=f['matrix'].copy()
   mo_fock=c.T@gp.fock[0]@c;assert np.max(abs(mo_fock-np.diag(np.diag(mo_fock))))<1e-8
   assert np.max(abs(c.T@s.overlap@c-np.eye(len(c))))<1e-8
   energies=np.diag(mo_fock)
  else:energies,c=eigh(gp.fock[0],s.overlap)
  nocc=int(np.sum(s.alpha_occupations));occ=np.arange(nocc);virt=np.arange(nocc,len(c));ii=np.repeat(occ,len(virt));aa=np.tile(virt,len(occ));ci=c[:,ii];ca=c[:,aa]
  state=replace(s,coefficients=c,fock=gp.fock[0],orbital_energies_hartree=energies,total_energy_hartree=gp.energy_hartree);kernel=ctx.density_response_kernel(state)
  start=time.perf_counter()
  if not cached:
   columns=[]
   for i,a in zip(ii,aa):
    density=np.outer(c[:,i],c[:,a]);f=kernel(.5*(density+density.T));columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
   static=np.diag(energies[aa]-energies[ii])+2*np.asarray(columns).T
  # Static real density response contains the symmetrized exchange contraction.
  # TDA A requires -(ij|ab), while the static kernel contains the average of
  # (ij|ab) and (ib|aj). Restore their difference, leaving local kernels intact.
  eripath=P/'results/full_response'/name/'projected_eri.npz'
  with np.load(eripath) as f:eri=f['eri'].copy()
  co=c[:,occ];cv=c[:,virt]
  direct=np.einsum('mnpq,mi,nj,pa,qb->iajb',eri,co,co,cv,cv,optimize=True).reshape(len(ii),len(ii))
  crossed=np.einsum('mnpq,mi,nb,pj,qa->iajb',eri,co,cv,co,cv,optimize=True).reshape(len(ii),len(ii))
  correction=.5*.225*(crossed-direct);matrix=static+correction
  hermiticity=float(np.max(abs(matrix-matrix.T)));assert hermiticity<1e-8
  values,vectors=eigh(matrix);seconds=time.perf_counter()-start
  np.savez_compressed(out/'spectrum.npz',energies_hartree=values,vectors=vectors,coefficients=c,matrix=matrix,static_matrix=static)
  rows=[];oldoi=np.flatnonzero(s.alpha_occupations==1);oldvi=np.flatnonzero(s.alpha_occupations==0)
  for target in targets:
   with np.load(target['transition_file']) as f:amplitude=f['amplitude'].copy()
   projected=(c[:,occ].T@s.overlap@s.coefficients[:,oldoi])@amplitude@(s.coefficients[:,oldvi].T@s.overlap@c[:,virt]);norm=float(np.linalg.norm(projected));total_norm=norm*target['reference_transition_norm']
   scores=abs(projected.ravel()@vectors)/max(norm,1e-300);k=int(np.argmax(scores));margin=float(scores[k]-np.max(np.delete(scores,k),initial=0.));qualified=bool(total_norm>=.7 and scores[k]>=.7 and margin>=.05 and values[k]>0)
   residual=float(np.max(abs(matrix@vectors[:,k]-values[k]*vectors[:,k])));assert residual<1e-8
   rows.append(dict(eigen_residual_hartree=residual,state=target['id'],status='matched' if qualified else 'unassigned',excitation_ev=float(values[k]*EV),root=k+1,overlap=float(scores[k]),margin=margin,projection_norm=total_norm))
  report=dict(host=socket.getfqdn(),profile=index,model_sha256=sha(path),fingerprint=ref.fingerprint,rows=rows,orbitals='canonical orbitals of the corrected GFN2 restricted ground; no DFT molecular orbitals',ground_reused=reused,ground_seconds=ground_seconds,tda_seconds=seconds,maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hermiticity_error=hermiticity,static_matrix_reused=cached,nonlocal_exchange_fraction=.225,nonlocal_exchange_correction_max_hartree=float(np.max(abs(correction))),projected_eri_sha256=sha(eripath),operator='true TDA A: static charge response plus 0.5*0.225*((ib|aj)-(ij|ab))',reference_excitation_labels_read=False)
  write(out/'result.json',report);profiles.append(report)
 rows=[]
 for target in targets:
  parents=[next(x for x in p['rows'] if x['state']==target['id']) for p in profiles];qualified=all(x['status']=='matched' for x in parents)
  rows.append(dict(state=target['id'],status='matched' if qualified else 'unassigned',parent_energies_ev=[x['excitation_ev'] for x in parents],mean_parent_tda_ev=float(np.mean([x['excitation_ev'] for x in parents])) if qualified else None))
 write(final,dict(host=socket.getfqdn(),molecule=name,status='controls_complete',rows=rows,profiles=profiles,mean_definition='arithmetic mean of matched parent TDA excitations; not asserted to equal a coupled composite linear-response Hessian',reference_excitation_labels_read=False))
 print(name,'TDA complete',flush=True)
