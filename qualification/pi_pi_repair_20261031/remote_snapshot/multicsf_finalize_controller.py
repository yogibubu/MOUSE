from pathlib import Path
import subprocess,sys,os,time,json
v=Path('/home/enzo/MOUSE-orbital-performance-20261031')
while True:
 try:os.kill(818027,0)
 except ProcessLookupError:break
 time.sleep(5)
name='Adenine';sid='B014';i=1
for worker,attempt in [('optimize_multicsf_retained.py','1_retained_v1'),('optimize_multicsf_fr.py','1_fr_v1')]:
 with (v/('multicsf_B014_'+attempt+'.log')).open('a') as f:subprocess.run([sys.executable,str(v/worker),name,sid,str(i)],stdout=f,stderr=subprocess.STDOUT,timeout=240)
 p=v/'pi_pi_repair/optimized'/name/sid/attempt/'result.json'
 if p.exists() and json.loads(p.read_text())['status']=='stationary_candidate':break
with (v/'audit_multicsf.log').open('w') as f:subprocess.run([sys.executable,str(v/'audit_multicsf.py')],stdout=f,stderr=subprocess.STDOUT,timeout=120)
