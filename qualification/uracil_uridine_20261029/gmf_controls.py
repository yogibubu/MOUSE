from remote_bootstrap import HOST
from pathlib import Path
import subprocess,sys,json,time
base=Path(__file__).resolve().parent;records=[]
for name,target,index in [('uracil_s0','pi_pi',2),('uracil_n_pi','n_pi',1),('uracil_pi_pi','pi_pi',2),('capped_s0','n_pi',1),('capped_s0','pi_pi',2),('capped_n_pi','n_pi',1),('capped_pi_pi','pi_pi',2),('uridine_n_pi','n_pi',1)]:
 with (base/f'gmf_control_{name}_{target}.log').open('a') as log:
  t=time.time();p=subprocess.run([sys.executable,str(base/'gmf_probe.py'),name,target,str(index),'--seed','ground','--freeze'],stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(name=name,target=target,index=index,returncode=p.returncode,seconds=time.time()-t));(base/'gmf_controls_status.json').write_text(json.dumps(records,indent=2))
