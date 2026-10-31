from pathlib import Path
import subprocess,sys,json
v=Path('/home/enzo/MOUSE-orbital-performance-20261031');w=v/'pi_pi_repair/trusted_newton_runtime'
with (w/'regression.log').open('w') as f:r=subprocess.run([sys.executable,str(w/'run_regression.py')],stdout=f,stderr=subprocess.STDOUT,timeout=120)
if r.returncode:raise RuntimeError('trusted Newton regression failed')
for name,sid in [('Aza-naphthalene','B011'),('Benzoquinone','B039')]:
 for i in [0,1]:
  for worker,attempt in [('optimize_multicsf.py',str(i)),('optimize_multicsf_fr_v2.py',str(i)+'_fr_v2')]:
   with (v/('regression_'+worker+'_'+sid+'_'+str(i)+'.log')).open('a') as f:subprocess.run([sys.executable,str(v/worker),name,sid,str(i)],stdout=f,stderr=subprocess.STDOUT,timeout=240)
   p=v/'pi_pi_repair/optimized'/name/sid/attempt/'result.json'
   if p.exists() and json.loads(p.read_text())['status']=='stationary_candidate':break
with (v/'audit_multicsf.log').open('w') as f:subprocess.run([sys.executable,str(v/'audit_multicsf.py')],stdout=f,stderr=subprocess.STDOUT,timeout=120)
