"""Independent D: two frozen profiles, independently optimized; fixed energy mean."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,numpy as np,argparse,traceback
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,state_from
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from pyscf import lib
import prepare_D as prepare
import full_response as full
import full_fast
import qualified_spectral_response as profile
from reference_td_checkpoint import reference
lib.num_threads(4)
PROTO=BASE/'ensemble_holdout_D_protocol.json';PROTOCOL=json.loads(PROTO.read_text())

def run(case):
 name=case['name'];reference(case,PROTO);prepare.prepare(case)
 prepared=json.loads((BASE/'prepared_D'/name/'case.json').read_text());out=BASE/'results/holdout_D'/name;out.mkdir(parents=True,exist_ok=True)
 if len(prepared['targets'])!=1:
  write(out/'result.json',dict(host=HOST,molecule=name,qualified=False,exclusions=prepared['excluded']));return
 t=prepared['targets'][0];old,xyz=state_from(t);s=old.reference_state
 full.ROOT=BASE/'results/full_response';full.calibrate=full_fast.calibrate;full.run(prepared)
 full.ROOT=BASE/'results/qualified_spectral_response';full.calibrate=profile.calibrate;full.run(prepared)
 with np.load(t['transition_file']) as f:a=f['amplitude'];occ=f['occupied'];virt=f['virtual']
 rows={};qualified=True
 for label,folder in [('raw_full',BASE/'results/full_response'),('spectral',BASE/'results/qualified_spectral_response')]:
  r=json.loads((folder/name/'n_pi/result.json').read_text());score=None
  good=bool(r.get('qualified') and r.get('excitation_ev',-1)>0)
  if good:
   solved=load_derivative_point(r['checkpoint'],request_hash=r['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
   h=s.coefficients[:,occ].T@s.overlap@solved.common_coefficients[:,t['hole']];p=s.coefficients[:,virt].T@s.overlap@solved.common_coefficients[:,t['particle']];rank=np.outer(h,p)
   score=float(abs(np.sum(a*rank))/np.linalg.norm(rank));good=bool(score>=.7)
  r['reference_transition_overlap']=score;r['paired_qualified']=good;rows[label]=r;qualified=bool(qualified and good)
 mapping=json.loads((BASE/'results/qualified_spectral_response'/name/'mapping.json').read_text());goodmap=bool(mapping['qualified'][t['hole']] and mapping['qualified'][t['particle']]);qualified=bool(qualified and goodmap)
 mean=.5*(rows['raw_full']['excitation_ev']+rows['spectral']['excitation_ev']) if qualified else None
 result=dict(host=HOST,molecule=name,qualified=qualified,components=rows,weights=[.5,.5],ensemble_ev=mean,ensemble_error_ev=None if mean is None else mean-t['tbe_ev'],
  native_roks_ev=t['native_roks_ev'],tbe_ev=t['tbe_ev'],target_orbital_mapping_qualified=goodmap,protocol_sha256=digest(PROTO),blind_model_selection=True,no_fitted_weights=True,energy_offset=0.,nuclear_derivatives_supported=False)
 write(out/'result.json',result);print('D COMPLETE',name,qualified,mean,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);args=p.parse_args();case=PROTOCOL['cases'][args.index]
 try:run(case)
 except Exception as exc:
  out=BASE/'results/holdout_D'/case['name'];out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));raise
