"""Finalize conserved checkpoints, corrected observables and convergence evidence."""
from pathlib import Path
import socket,time,json,hashlib,subprocess,sys,os
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent;os.chdir(HERE)
while not (HERE/'pigment_cid_expansion/Beta_carotene/screening_13/frontier_13/result.json').exists():time.sleep(10)
def invoke(*args):subprocess.run([sys.executable,'-u',*args],check=True,cwd=HERE)
invoke('refresh_pigment_observables.py')
paths=list((HERE/'pigment_cid_expansion/Beta_carotene').glob('**/result.json'))
for path in paths:
 r=json.loads(path.read_text());a=path.with_name('analysis.json')
 if a.exists():
  old=json.loads(a.read_text());assert all(abs(x['excitation_ev']-y['excitation_ev'])<1e-8 for x,y in zip(old['states'],r['states']))
  old['states']=[dict(new,double_subspace_weight=previous['double_subspace_weight']) for previous,new in zip(old['states'],r['states'])];old['result_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();old['observable_phase_refreshed_without_repeating_electronic_work']=True;a.write_text(json.dumps(old,indent=2)+'\n')
 else:invoke('analyze_pigment_expansion.py','Beta_carotene',str(r['frontier_occupied']),'--screening-frontier',str(r['screening_active_frontier']))
for n in [11,12,13]:invoke('analyze_pigment_expansion.py','Beta_carotene',str(n),'--screening-frontier',str(n),'--coherent-parent')
for w,ns in [(8,[6,7,8]),(10,[8,9,10]),(12,[10,11,12]),(16,[11,12,13])]:invoke('summarize_pigment_convergence.py','--reuse-analysis','--screening-frontier',str(w),'--frontiers',*map(str,ns))
invoke('assess_beta_convergence.py','--window','16','--frontiers','11','12','13','--screening-control','10','--previous-window','14')
(HERE/'beta_convergence_assessment.json').replace(HERE/'beta_convergence_assessment_v1.json')
invoke('assess_beta_coherent_convergence.py')
print('All numerical audits completed; inspect the convergence verdict before claiming success.',flush=True)
