from remote_bootstrap import HOST
from pathlib import Path
import subprocess,sys,json,time
base=Path(__file__).resolve().parent
records=[]
for name,target in [('uracil_s0','pi_pi'),('uracil_n_pi','n_pi'),('uracil_pi_pi','pi_pi'),('uridine_s0','pi_pi')]:
 with (base/f'derivatives_{name}_{target}.log').open('a') as log:
  t=time.time();p=subprocess.run([sys.executable,str(base/'derivatives.py'),name,target],stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(name=name,target=target,returncode=p.returncode,seconds=time.time()-t));(base/'derivative_status.json').write_text(json.dumps(records,indent=2))
