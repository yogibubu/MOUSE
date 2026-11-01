"""One-off checkpoint-preserving migration to matched large-system cutoffs."""
from pathlib import Path
import socket,os,signal,subprocess,time,sys,json
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
P=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101');out=P/'pigment_full_response/Beta_carotene';oldpid=int((out/'worker.lock').read_text());command=subprocess.check_output(['ps','-p',str(oldpid),'-o','command='],text=True);assert 'pigment_full_response.py Beta_carotene' in command
while not (out/'coulomb_eri_complete.json').exists():
 os.kill(oldpid,0);time.sleep(10)
os.kill(oldpid,signal.SIGTERM)
for unused in range(100):
 if subprocess.run(['ps','-p',str(oldpid)],stdout=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Old worker has not stopped; do not duplicate it')
rejected=out/'diagnostic_cutoff24';rejected.mkdir(exist_ok=True)
for name in ['native.npy','native_complete.json','native_progress.json']:
 source=out/name
 if source.exists():source.rename(rejected/name)
(out/'worker.lock').unlink(missing_ok=True)
(out/'cutoff_reconciliation.json').write_text(json.dumps(dict(reason='Match native ground evaluator: D4 coordination 40 bohr, pair 50 bohr; anisotropic graph 64 bohr',retained='All converged DFT, embedding, compact anchor, projected integrals, complete Coulomb ERI and exchange checkpoints',recomputed='Native response only; previous 24-bohr diagnostic preserved'),indent=2)+'\n')
with (P/'beta_carotene_response.log').open('a') as log:
 log.write('\nResuming with native evaluator cutoffs and preserved integral checkpoints\n');log.flush();code=subprocess.call([sys.executable,str(P/'pigment_full_response.py'),'Beta_carotene','--wait-for-anchor','--ri-xc'],stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(code)
