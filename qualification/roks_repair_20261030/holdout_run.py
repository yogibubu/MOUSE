"""Run only the preregistered profile on available holdout checkpoints."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import argparse,json,numpy as np,traceback
from types import SimpleNamespace
import spin_kernel_probe as spin
import response_kernel_probe as full
import reference_survey as survey
import frozen_relaxation as baseline
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from frozen_relaxation import write,digest,key,state_from

P=json.loads((BASE/'results/holdout_protocol.json').read_text())
assert P['selected_reference']=='bhandhlyp' and P['selected_response']=='response_kernel'
DATA=BASE/'holdout_data';ROOT=BASE/'results/holdout';ROOT.mkdir(exist_ok=True)
cases=[json.loads((DATA/c['name']/'case.json').read_text()) for c in P['cases'] if (DATA/c['name']/'case.json').exists()]
names=[c['name'] for c in P['cases']]
baseline.OUT=DATA/'frozen_relaxation';baseline.OUT.mkdir(exist_ok=True)
if not (baseline.OUT/'plan.json').exists():write(baseline.OUT/'plan.json',dict(host=HOST,cases=cases))
spin.ROOT=ROOT/'spin_kernel';spin.ROOT.mkdir(exist_ok=True);spin.OLD=DATA
spin.PROTOCOL=dict(spin.PROTOCOL,planned_molecules=names,holdout_protocol_sha256=digest(BASE/'results/holdout_protocol.json'))
full.ROOT=ROOT/'response_kernel';full.ROOT.mkdir(exist_ok=True);full.OLD=DATA
full.PROTOCOL=dict(full.PROTOCOL,planned_molecules=names,holdout_protocol_sha256=digest(BASE/'results/holdout_protocol.json'),
 initialization='canonical GFN2 ground orbitals; cold target-state optimization')
survey.ROOT=ROOT/'bhandhlyp';survey.ROOT.mkdir(exist_ok=True);survey.OLD=DATA
survey.PROTOCOL=dict(survey.PROTOCOL,cases=names,holdout_protocol_sha256=digest(BASE/'results/holdout_protocol.json'))

def response_case(case):
 target=case['targets'][0];r,xyz=state_from(target);s=r.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r.ground_energy_hartree))
 ref=full.reference(case,ctx);request=key(dict(reference=ref.fingerprint,checkpoint=digest(target['checkpoint'])))
 out=full.ROOT/case['name']/target['id'];out.mkdir(exist_ok=True)
 if not (out/'latest.npz').exists() and not (out/'converged.npz').exists():
  np.savez_compressed(out/'latest.npz',request_hash=request,coefficients=s.coefficients)
  write(out/'initialization.json',dict(host=HOST,seed='canonical ground orbitals',source_checkpoint_sha256=digest(target['checkpoint'])))
 full.run(case)
 path=full.ROOT/case['name']/'result.json';report=json.loads(path.read_text())
 for row in report['records']:
  if not row.get('qualified'):continue
  result=load_derivative_point(out/'converged.npz',request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  with np.load(DATA/case['name']/'reference_transition.npz') as f:a=f['amplitude'];oi=f['occupied'];ov=f['virtual']
  h=s.coefficients[:,oi].T@s.overlap@result.common_coefficients[:,target['hole']]
  p=s.coefficients[:,ov].T@s.overlap@result.common_coefficients[:,target['particle']]
  rank=np.outer(h,p);score=float(abs(np.sum(a*rank))/np.linalg.norm(rank))
  row['reference_transition_overlap']=score;row['qualified']=bool(row['qualified'] and score>=.7)
  row['identity_policy']='frozen CSF and transition overlap thresholds >=0.7'
 write(path,report)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',choices=['baseline','response','reference'],required=True);a=p.parse_args()
 for case in cases:
  try:
   path=baseline.OUT/case['name']/'result.json'
   if not path.exists():baseline.run(case)
   b=json.loads(path.read_text())
   if not all(r['qualified'] for r in b['records']):
    write(ROOT/case['name']/'exclusion.json',dict(host=HOST,reason='baseline state/reference mapping failed'));continue
   if a.job=='response':response_case(case)
   elif a.job=='reference':survey.run(case)
  except Exception as exc:
   write(ROOT/case['name']/(a.job+'_failure.json'),dict(host=HOST,error=str(exc),traceback=traceback.format_exc()))
   print(traceback.format_exc(),flush=True)
 write(ROOT/(a.job+'_status.json'),dict(host=HOST,protocol_sha256=digest(BASE/'results/holdout_protocol.json'),
  prepared_cases=[c['name'] for c in cases],registered_cases=names))
