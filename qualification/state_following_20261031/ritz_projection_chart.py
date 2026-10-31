"""Diagnose native-sector rejection using the certified relaxed Q ground chart."""
from pathlib import Path
import socket,os,sys,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'ritz_projection_probe';name='Bimane_Syn'
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.restricted_ground import evaluate_restricted_ground,RestrictedGroundResult
from pyscf import scf,gto,lib
sys.path.append('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');from extent_support import source_molecule
lib.num_threads(2)
def load(p):
 with np.load(p) as f:xyz=f['coordinates_angstrom'].copy();request=str(f['request_hash'])
 return load_derivative_point(p,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
with threadpool_limits(limits=2):
 native=load(P/'blind/prepared'/name/'ground.npz');s=native.reference_state;xyz=s.coordinates_bohr/1.8897261254578281
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
 ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);path=D/'models'/name/'model.npz'
 with np.load(path) as f:model={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
 ref=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,model,hashlib.sha256(path.read_bytes()).hexdigest())
 out=D/'ground_chart';out.mkdir(exist_ok=True)
 import shutil
 cached=D/'results'/name/'B015/grounds/2'
 assert (cached/'certificate.json').exists(),'wait for the excited-state worker to finish its Q ground solve'
 shutil.copy2(cached/'stationary.npz',out/'ground.npz');shutil.copy2(cached/'certificate.json',out/'ground.json')
 if (out/'ground.json').exists():
  cert=json.loads((out/'ground.json').read_text());ground=RestrictedGroundResult(load(out/'ground.npz'),cert['minimum_curvature'],cert['index'])
 else:
  ground=evaluate_restricted_ground(calc,xyz,template=s,spin_reference=ref,gradient=False)
  save_derivative_point(out/'ground.npz',ground.reference,request_hash='ordering_probe_Q_ground',coordinates_angstrom=xyz)
  cert=dict(energy_hartree=ground.energy_hartree,minimum_curvature=ground.minimum_orbital_curvature_hartree,index=ground.orbital_saddle_index,residual=ground.orbital_residual,fingerprint=ref.fingerprint);write(out/'ground.json',cert)
 print('GROUND',cert,flush=True)
 reference=P/'blind/mouse_reference'/name;mol=scf.chkfile.load_mol(str(reference/'backend.chk'));small,t=source_molecule(s);cross=gto.intor_cross('int1e_ovlp',mol,small)@t
 with np.load(reference/'reference.npz') as f:cd=f['coefficients'];od=f['occupations']
 oi=np.flatnonzero(s.alpha_occupations==1);vi=np.flatnonzero(s.alpha_occupations==0);di=np.flatnonzero(od==2);da=np.flatnonzero(od==0)
 case=json.loads((P/'blind/prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']=='B015')
 with np.load(reference/'backend_tda.npz') as f:x=f['amplitudes'][target['reference_root']-1].reshape(len(di),len(da)).copy()
 x/=np.linalg.norm(x);accepted=json.loads((D/'results'/name/'B015/accepted.json').read_text());state=load(D/'results'/name/'B015'/accepted['checkpoint'])
 rows=[]
 for label,c in [('native',s.coefficients),('relaxed_Q',ground.coefficients[0])]:
  overlap=c.T@cross.T@cd;q=overlap[np.ix_(oi,di)]@x@overlap[np.ix_(vi,da)].T;norm=float(np.linalg.norm(q));q/=norm;cap=float(np.linalg.svd(q,compute_uv=False)[0]);pair=np.flatnonzero(state.occupations[0]!=state.occupations[1]);scores=[]
  for h,p in [pair,pair[::-1]]:
   u=c[:,oi].T@s.overlap@state.common_coefficients[:,h];v=c[:,vi].T@s.overlap@state.common_coefficients[:,p];rank=np.outer(u,v);score=float(abs(np.sum(q*rank))/np.linalg.norm(rank))
   scores.append(dict(hole_norm=float(np.linalg.norm(u)),particle_norm=float(np.linalg.norm(v)),fidelity=score/cap,raw_overlap=score,physical_projected_overlap=score*norm))
  rows.append(dict(chart=label,projection_norm=norm,rank_one_limit=cap,directions=scores))
 report=dict(host=socket.getfqdn(),accepted_fraction=accepted['fraction'],ground=cert,charts=rows,reference_excitation_labels_read=False);write(out/'chart_comparison.json',report);print(json.dumps(report,indent=2),flush=True)
