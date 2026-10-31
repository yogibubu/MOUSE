from remote_bootstrap import HOST
from pathlib import Path
import subprocess,sys,json,time,shutil
base=Path(__file__).resolve().parent
# Reuse already converged uracil states whose promoted orbitals match the target.
for name,target in [('uracil_s0','n_pi'),('uracil_s0','pi_pi'),('uracil_n_pi','n_pi'),('uracil_n_pi','pi_pi'),('uracil_pi_pi','pi_pi')]:
 src=base/'results'/name/f'roks_{target}.npz';dst=src.parent/'qualified'/src.name;dst.parent.mkdir(exist_ok=True)
 if not dst.exists():shutil.copy2(src,dst)
records=[]
for name in ['uridine_s0','uridine_n_pi','uridine_pi_pi','capped_s0','capped_n_pi','capped_pi_pi']:
 if name.startswith('capped'):
  with (base/f'{name}.log').open('a') as log:subprocess.run([sys.executable,str(base/'benchmark.py'),name,'--mode','inspect'],stdout=log,stderr=subprocess.STDOUT)
 for target in ['n_pi','pi_pi']:
  if name=='uridine_s0' and target=='n_pi':continue # independent pilot owns this point
  with (base/f'qualified_{name}_{target}.log').open('a') as log:
   t=time.time();p=subprocess.run([sys.executable,str(base/'refine.py'),name,target],stdout=log,stderr=subprocess.STDOUT)
  records.append(dict(name=name,target=target,returncode=p.returncode,seconds=time.time()-t));(base/'qualified_status.json').write_text(json.dumps(records,indent=2))
