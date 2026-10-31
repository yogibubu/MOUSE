from remote_bootstrap import HOST
from pathlib import Path
import subprocess,sys,os,json,time
base=Path(__file__).resolve().parent
records=[]
for name in ['uracil_s0','uracil_n_pi','uracil_pi_pi','uridine_s0','uridine_n_pi','uridine_pi_pi']:
 with (base/f'{name}.log').open('a') as log:
  start=time.time();proc=subprocess.run([sys.executable,str(base/'benchmark.py'),name,'--mode','roks'],stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(name=name,returncode=proc.returncode,seconds=time.time()-start));(base/'status.json').write_text(json.dumps(records,indent=2))
