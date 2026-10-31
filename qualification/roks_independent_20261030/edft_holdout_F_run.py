"""New F states, old controls and the unchanged scoped ensemble candidate."""
from remote_bootstrap import BASE,HOST
import sys,json,traceback
from pyscf import lib
from reference_td_checkpoint import reference
import prepare_F
import full_response as full
import full_fast
import qualified_spectral_response as spectral
import edft_statefollow_response as candidate
from edft_pbe0_response import calibrate
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
lib.num_threads(4)
protocol=BASE/'edft_holdout_F_protocol.json';case=json.loads(protocol.read_text())['cases'][int(sys.argv[1])]
root=BASE/'results/holdout_F'/case['name'];root.mkdir(parents=True,exist_ok=True)
try:
    print('F HOST',HOST,case['name'],flush=True);reference(case,protocol);prepare_F.prepare(case)
    prepared=json.loads((BASE/'prepared_F'/case['name']/'case.json').read_text())
    if len(prepared['targets'])!=1:
        write(root/'result.json',dict(host=HOST,molecule=case['name'],qualified=False,exclusions=prepared['excluded'],protocol_sha256=digest(protocol)));sys.exit(0)
    full.ROOT=BASE/'results/full_response';full.calibrate=full_fast.calibrate;full.run(prepared)
    full.ROOT=BASE/'results/qualified_spectral_response';full.calibrate=spectral.calibrate;full.run(prepared)
    candidate.ROOT=BASE/'results/edft_pbe0_statefollow';candidate.calibrate=calibrate;candidate.CONSISTENT_REFERENCE=False;candidate.RESUME_ROOT=BASE/'results/edft_pbe0';lib.num_threads(2)
    with candidate.threadpool_limits(limits=2,user_api='blas'):candidate.run(prepared,'qualified_spectral_response','n_pi')
    name=case['name'];t=prepared['targets'][0];models={}
    for label,folder in [('raw_full','full_response'),('old_spectral','qualified_spectral_response'),('candidate','edft_pbe0_statefollow/qualified_spectral_response')]:
        models[label]=json.loads((BASE/'results'/folder/name/'n_pi/result.json').read_text())
    good=all(d.get('qualified') for d in models.values())
    result=dict(host=HOST,molecule=name,qualified=good,models=models,native_roks_ev=t['native_roks_ev'],tbe_ev=t['tbe_ev'],
                old_fixed_mean_ev=.5*(models['raw_full']['excitation_ev']+models['old_spectral']['excitation_ev']) if good else None,
                candidate_scope='first singlet npi, not full GX24 or a universal all-state method',protocol_sha256=digest(protocol),weights_of_old_composite=[.5,.5],no_fitted_weights=True)
    write(root/'result.json',result);print('F COMPLETE',name,good,result,flush=True)
except Exception as exc:
    write(root/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));raise
