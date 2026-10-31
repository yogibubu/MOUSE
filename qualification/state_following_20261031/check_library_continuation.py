"""Remote molecular/restart verification of the public continuation API."""
from pathlib import Path
import os,socket,sys,json,argparse,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import (GFN2Calculator,FrozenNonlocalResponseReference,continue_roks_one_body_response,ROKSContinuationState)
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.roks_tracking import common_orbital_singlet_overlap

def load(p):
 with np.load(p) as d:x=d['coordinates_angstrom'].copy();req=str(d['request_hash'])
 return load_derivative_point(p,request_hash=req,coordinates_angstrom=x,require_gradient=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,sid):
 out=R/'library_validation'/name/sid;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 full=P/'results/full_response'/name;spectral=P/'results/qualified_spectral_response'/name
 report=json.loads((full/sid/'result.json').read_text());seed=load(Path(report['checkpoint']));s=seed.reference_state;xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
 ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);refs=[]
 for folder in [full,spectral]:
  path=folder/'model.npz'
  with np.load(path) as f:model={k:f[k].copy() for k in ['operators','charge','spin','linear','q0']}
  refs.append(FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,model,sha(path)))
 case=json.loads((P/'blind/prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
 with np.load(target['transition_file']) as f:amplitude=f['amplitude'].copy();occupied=f['occupied'].copy();virtual=f['virtual'].copy()
 options=dict(reference_result=seed,reference_coefficients=s.coefficients,occupied=occupied,virtual=virtual,reference_transition=amplitude,projected_reference_norm=target['reference_transition_norm'])
 class StopAfterAcceptedCheckpoint(Exception):pass
 saved=[]
 def save(state):
  cp=out/'accepted.npz';save_derivative_point(cp,state.result,request_hash='library_checkpoint',coordinates_angstrom=xyz)
  (out/'accepted.json').write_text(json.dumps(dict(fraction=state.fraction,origin_fingerprint=state.origin_fingerprint,target_fingerprint=state.target_fingerprint,history=state.history),indent=2)+'\n')
  saved.append(state.fraction)
  if state.fraction>=.4:raise StopAfterAcceptedCheckpoint()
 try:continue_roks_one_body_response(calc,xyz,refs,checkpoint=save,**options)
 except StopAfterAcceptedCheckpoint:pass
 else:raise AssertionError('checkpoint interruption did not occur')
 metadata=json.loads((out/'accepted.json').read_text());restart=ROKSContinuationState(metadata['fraction'],load(out/'accepted.npz'),metadata['origin_fingerprint'],metadata['target_fingerprint'],tuple(metadata['history']))
 continued=[];final=continue_roks_one_body_response(calc,xyz,refs,restart=restart,checkpoint=lambda state:continued.append(state.fraction),**options)
 assert final.fraction==1 and continued[0]>restart.fraction and final.result.spin_reference_fingerprint==refs[1].fingerprint
 original=R/'recovery_rank_one'/name/sid;accepted=json.loads((original/'accepted.json').read_text());old=load(original/accepted['checkpoint'])
 energy_error=abs(final.result.energy_hartree-old.energy_hartree);score=common_orbital_singlet_overlap(old.common_coefficients,final.result.common_coefficients,s.overlap,old.occupations)
 assert energy_error<1e-9 and score>.999999 and final.result.orbital_residual<=1e-9
 # A changed target Hamiltonian must not accept the saved restart.
 bad=replace(refs[1],pair_linear_hartree=refs[1].pair_linear_hartree+1e-6)
 try:continue_roks_one_body_response(calc,xyz,[refs[0],bad],restart=restart,**options)
 except ValueError as exc:assert 'another problem' in str(exc)
 else:raise AssertionError('foreign restart accepted')
 result=dict(host=socket.getfqdn(),molecule=name,state=sid,passed=True,saved_fractions=saved,resumed_fractions=continued,accepted_fractions_not_recomputed=True,endpoint_fingerprint_exact=True,endpoint_energy_error_hartree=energy_error,endpoint_csf_overlap=score,index=final.result.orbital_saddle_index,residual=final.result.orbital_residual,foreign_restart_rejected=True,source_sha256=sha(R/'source/packages/matrix-gfn2/src/matrix_gfn2/roks_continuation.py'),reference_labels_read=False)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--molecule',default='Diazirine');ap.add_argument('--state',default='B001');a=ap.parse_args()
 with threadpool_limits(limits=2):run(a.molecule,a.state)
