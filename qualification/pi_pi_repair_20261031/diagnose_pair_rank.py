"""Oracle-only fixed-orbital diagnostic of rank-one CSF truncation.

Uses completed spectra and ground checkpoints; no new reference calculation.
The 14-state energies have already been exposed: all tests are development.
"""
from pathlib import Path
import socket,os,sys,json,hashlib,time
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation';OUT=V/'pi_pi_repair'
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(V))
sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
EV=27.211386245988

def load(p):
 with np.load(p) as a:x=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
 return load_derivative_point(p,request_hash=req,coordinates_angstrom=x,require_gradient=False),x

def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
 import fcntl
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 selected=[('Cyclopentadiene','B016'),('Furan','B022'),('Phthalimide','B024'),('Adenine','B014'),('Bimane_Syn','B015'),('Phthalimide','B005'),('Nitropyridine_N-oxide','B029')]
 plan=dict(schema='mouse.pi_pi.rank_diagnostic.v1',host=socket.getfqdn(),states=[sid for _,sid in selected],reason='Test whether missing coherent single-excitation pairs cause the pi-pi bias; no empirical energy correction',model_and_weights_unchanged=True,lambda_exchange=.9,weights=[.5,.5],energy_offset_ev=0.,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),orbitals='corrected parent ground orbitals, rotated into assigned NTO basis; no state optimization in this diagnostic',offdiagonal='projection of the already corrected true-TDA A operator between distinct NTO pairs',diagonal='unmodified spin-polarized common-orbital ROKS 2 E_M - E_T',ranks=[1,2,3,4],production_claim=False,accuracy_labels_already_exposed=True)
 OUT.mkdir(exist_ok=True)
 pp=OUT/'rank_diagnostic_protocol.json'
 if pp.exists():assert json.loads(pp.read_text())==plan
 else:write(pp,plan)
 rows=[]
 for name,sid in selected:
  report=OUT/name/(sid+'_rank_diagnostic.json')
  if report.exists():rows.append(json.loads(report.read_text()));continue
  case=json.loads((D/'prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
  ground,xyz=load(D/'prepared'/name/'ground.npz');s=ground.reference_state
  calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
  calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
  native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
  with np.load(target['transition_file']) as f:amplitude=f['amplitude'].copy();oldoi=f['occupied'].copy();oldvi=f['virtual'].copy()
  profiles=[]
  for index,path in enumerate([P/'results/full_response'/name/'model.npz',D/'models'/name/'model.npz']):
   with np.load(path) as f:model={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
   ref=FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,model,hashlib.sha256(path.read_bytes()).hexdigest());ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref)
   with np.load(D/'tda'/name/str(index)/'spectrum.npz') as f:c=f['coefficients'].copy();matrix=f['matrix'].copy()
   no=len(oldoi);nv=len(oldvi);oi=np.arange(no);vi=np.arange(no,no+nv)
   projected=(c[:,oi].T@s.overlap@s.coefficients[:,oldoi])@amplitude@(s.coefficients[:,oldvi].T@s.overlap@c[:,vi]);norm=np.linalg.norm(projected)
   u,values,vh=np.linalg.svd(projected/norm,full_matrices=True);rotated=np.column_stack((c[:,oi]@u,c[:,vi]@vh.T));assert np.max(abs(rotated.T@s.overlap@rotated-np.eye(len(c))))<1e-8
   occ=np.array([s.alpha_occupations,s.beta_occupations]);p0=np.array([(c*o)@c.T for o in occ]);_,e0=ctx.fock_and_energy(p0)
   count=min(4,len(values));vectors=np.column_stack([np.outer(u[:,j],vh[j]).ravel() for j in range(count)]);tda=vectors.T@matrix@vectors
   diag=[]
   for j in range(count):
    o=occ.copy();o[0,j]=0;o[0,no+j]=1;t=np.array([np.maximum(*o),np.minimum(*o)])
    _,em=ctx.fock_and_energy(np.array([(rotated*x)@rotated.T for x in o]));_,et=ctx.fock_and_energy(np.array([(rotated*x)@rotated.T for x in t]));diag.append(2*em-et-e0)
   h=tda.copy();np.fill_diagonal(h,diag);ranks=[]
   for rank in range(1,count+1):
    e,z=eigh(h[:rank,:rank]);target_vector=values[:rank]/np.linalg.norm(values[:rank]);scores=abs(target_vector@z);root=int(np.argmax(scores))
    ranks.append(dict(rank=rank,energy_ev=float(e[root]*EV),coefficient_overlap=float(scores[root]),root=root+1,coefficients=z[:,root].tolist(),retained_target_weight=float(np.sum(values[:rank]**2))))
   profiles.append(dict(profile=index,singular_values=values.tolist(),ROKS_diagonal_ev=(np.array(diag)*EV).tolist(),TDA_projected_ev=(tda*EV).tolist(),effective_matrix_ev=(h*EV).tolist(),rank_results=ranks))
  row=dict(molecule=name,state=sid,profiles=profiles,scope='fixed-ground-orbital configuration-mixing diagnostic, not an optimized excited-state prediction')
  write(report,row);rows.append(row);print(name,sid,json.dumps([p['rank_results'] for p in profiles]),flush=True)
 write(OUT/'rank_diagnostic_summary.json',dict(protocol=plan,rows=rows))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
