"""Resume the same RSH energy model using excited-state/transition tracking."""
from remote_bootstrap import BASE,HOST
import edft_statefollow_response as driver
import edft_rsh_consistent_calibration as calibration
from reks_probe import cases
from pathlib import Path
import json,argparse,fcntl,sys
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest

ROOT=BASE/'results/edft_rsh_statefollow';ROOT.mkdir(exist_ok=True)
link=ROOT/'calibration'
if not link.exists():link.symlink_to('../edft_rsh_consistent/calibration',target_is_directory=True)

def calibrate(case,s,ctx):
    with (ROOT/'calibration.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return calibration.calibrate(case,s,ctx)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--molecule',required=True);ap.add_argument('--state',default='n_pi');args=ap.parse_args()
    protocol=dict(host=HOST,script_sha256=digest(__file__),solver_sha256=digest(BASE/'edft_statefollow_response.py'),identity_sha256=digest(BASE/'edft_transition_identity.py'),
                  energy_functional_changed=False,calibration='verified unchanged consistent RSH model, checkpointed ground DFT reference reused',
                  xi=.32,lambda_exchange=.9,identity='warm qualified excited CSF >=.7, step CSF >=.7, fixed DFT transition projection >=.7; canonical native CSF is a diagnostic',
                  previous_protocol='results/edft_rsh_consistent/protocol.json',scope='exploratory tracking revision; original canonical-overlap failures remain recorded',nuclear_derivatives=False)
    file=ROOT/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol
    else:write(file,protocol)
    driver.ROOT=ROOT;driver.RESUME_ROOT=calibration.ROOT;driver.CONSISTENT_REFERENCE=True;driver.calibrate=calibrate;driver.lib.num_threads(2)
    with driver.threadpool_limits(limits=2,user_api='blas'):driver.run(next(c for c in cases() if c['name']==args.molecule),'qualified_spectral_response',args.state)
