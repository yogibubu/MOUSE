"""Resume diagnosis: exact first/charge/spin ablations, no TBE fitting.

Original development plus already exposed B are diagnostic data. No independent
accuracy claim is made. Reuses completed DFT response models and all converged
state checkpoints. No nuclear derivatives for this vertical AO-pair prototype.
"""
from remote_bootstrap import HOST, BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,numpy as np,traceback
from scipy.linalg import solve_sylvester
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,key,EV
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
import full_response as full
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/response_ablation';ROOT.mkdir(exist_ok=True)
VARIANTS={'native_control':(0,0,0),'fock_only':(1,0,0),'charge_only':(0,1,0),'spin_only':(0,0,1),'fock_charge':(1,1,0),'fock_spin':(1,0,1),'charge_spin':(0,1,1)}
CASES=[c for c in plan() if c['name'] in ['nitroxyl','uracil']]
CASES += [json.loads((BASE/'prepared_B'/n/'case.json').read_text()) for n in ['acetaldehyde','acetone','formamide']]
PROTOCOL=dict(host=HOST,script_sha256=digest(__file__),variants=VARIANTS,cases=[dict(name=c['name'],states=[t['id'] for t in c['targets']]) for c in CASES],
 lambda_exchange=.9,energy_offset=0.,scope='development diagnosis only; all B values previously exposed',model_selection='none: report all variants and failed identities',
 nuclear_derivatives=False,source_models={c['name']:digest(BASE/'results/full_response'/c['name']/'model.npz') for c in CASES},
 qualification=dict(csf_minimum=.7,reference_transition_minimum=.7,orbital_residual_maximum=1e-9,saddle_indices=[0,1,2,3]))
p=ROOT/'protocol.json'
# Compare canonical JSON representation, including tuple-to-list normalization.
if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(PROTOCOL))
else:write(p,PROTOCOL)

def setup(case,label):
 old,xyz=state_from(case['targets'][0]);s=old.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 native=prepare_gfn2_deltascf(calc,xyz,ground_state=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=old.ground_energy_hartree))
 with np.load(BASE/'results/full_response'/case['name']/'model.npz') as f:data={k:f[k].copy() for k in f.files if k!='request_hash'}
 a=data['operators'];dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a]);b=.5*(s.overlap@a+a@s.overlap)
 kn=np.array([np.einsum('aij,ij->a',dual,native.polarization.evaluate(np.array([.5*x,-.5*x]))[0][0]) for x in dual]).T
 assert np.max(abs(kn-kn.T))<1e-10
 # Verify exact native spin energy AND Fock representation before ablation.
 m=np.random.default_rng(20261030).normal(size=s.overlap.shape);m=.01*(m+m.T)
 f,e,_=native.polarization.evaluate(np.array([.5*m,-.5*m]));q=np.einsum('aij,ij->a',b,m)
 fe=np.einsum('a,aij->ij',kn@q,b)
 assert np.max(abs(fe-f[0]))<1e-10 and abs(.5*q@kn@q-e)<1e-10
 flags=VARIANTS[label];data['linear']=data['linear']*flags[0];data['charge']=data['charge']*flags[1]
 if not flags[2]:data['spin']=kn
 request=key(dict(protocol=digest(p),source_model=PROTOCOL['source_models'][case['name']],variant=label))
 dest=ROOT/label/case['name'];dest.mkdir(parents=True,exist_ok=True)
 if not (dest/'model.npz').exists():np.savez_compressed(dest/'model.npz',request_hash=request,**data)
 return data,request

with threadpool_limits(limits=2,user_api='blas'):
 for case in CASES:
  for label in VARIANTS:
   full.ROOT=ROOT/label;full.ROOT.mkdir(exist_ok=True)
   full.calibrate=lambda case,s,ctx,label=label:setup(case,label)
   full.run(case)
   for t in case['targets']:
    result=ROOT/label/case['name']/t['id']/'result.json';r=json.loads(result.read_text())
    # Full.run checks CSF and stationarity; add archived DFT transition identity.
    if r.get('qualified') and t.get('transition_file'):
     old,xyz=state_from(t);s=old.reference_state
     solved=load_derivative_point(r['checkpoint'],request_hash=r['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
     with np.load(t['transition_file']) as f:amp=f['amplitude'];occ=f['occupied'];virt=f['virtual']
     h=s.coefficients[:,occ].T@s.overlap@solved.common_coefficients[:,t['hole']]
     part=s.coefficients[:,virt].T@s.overlap@solved.common_coefficients[:,t['particle']];rank=np.outer(h,part)
     overlap=float(abs(np.sum(amp*rank))/np.linalg.norm(rank));r['reference_transition_overlap']=overlap
     r['qualified']=bool(overlap>=.7);write(result,r)
   print('COMPLETED',case['name'],label,flush=True)
rows=[]
for case in CASES:
 for t in case['targets']:
  record=dict(molecule=case['name'],state=t['id'],tbe_ev=t['tbe_ev'],variants={})
  for label in list(VARIANTS)+['full']:
   path=(BASE/'results/full_response' if label=='full' else ROOT/label)/case['name']/t['id']/'result.json'
   r=json.loads(path.read_text());record['variants'][label]=r
  rows.append(record)
summary=dict(host=HOST,protocol_sha256=digest(p),scope=PROTOCOL['scope'],rows=rows,statistics={})
for label in list(VARIANTS)+['full']:
 valid=[r for r in rows if r['variants'][label].get('qualified')]
 summary['statistics'][label]=dict(count=len(valid),mae_ev=float(np.mean([abs(r['variants'][label]['error_ev']) for r in valid])),npi_mae_ev=float(np.mean([abs(r['variants'][label]['error_ev']) for r in valid if r['state']=='n_pi'])))
write(ROOT/'summary.json',summary);print('SUMMARY',summary['statistics'],flush=True)
