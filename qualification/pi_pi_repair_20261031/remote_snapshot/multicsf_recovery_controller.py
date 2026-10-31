from pathlib import Path
import subprocess,sys,os,time,json
v=Path('/home/enzo/MOUSE-orbital-performance-20261031')
while True:
 try:os.kill(812680,0)
 except ProcessLookupError:break
 time.sleep(5)
for i in [0,1]:
 p=v/'pi_pi_repair/optimized/Phthalimide/B024'/str(i)/'result.json'
 if p.exists() and json.loads(p.read_text())['status']=='stationary_candidate':continue
 for worker,attempt in [('optimize_multicsf_retained.py',str(i)+'_retained_v1'),('optimize_multicsf_fr.py',str(i)+'_fr_v1')]:
  with (v/('multicsf_B024_'+attempt+'.log')).open('a') as f:subprocess.run([sys.executable,str(v/worker),'Phthalimide','B024',str(i)],stdout=f,stderr=subprocess.STDOUT,timeout=240)
  p=v/'pi_pi_repair/optimized/Phthalimide/B024'/attempt/'result.json'
  if p.exists() and json.loads(p.read_text())['status']=='stationary_candidate':break
with (v/'audit_multicsf.log').open('w') as f:subprocess.run([sys.executable,str(v/'audit_multicsf.py')],stdout=f,stderr=subprocess.STDOUT,timeout=120)
