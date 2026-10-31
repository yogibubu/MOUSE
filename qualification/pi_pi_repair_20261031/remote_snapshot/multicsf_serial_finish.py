from pathlib import Path
import subprocess,sys,json
v=Path('/home/enzo/MOUSE-orbital-performance-20261031')
def run(worker,name,sid,i):
 with (v/('repair_'+worker+'_'+sid+'_'+str(i)+'.log')).open('a') as f:subprocess.run([sys.executable,str(v/worker),name,sid,str(i)],stdout=f,stderr=subprocess.STDOUT,timeout=240)
run('optimize_multicsf_polish.py','Phthalimide','B024',1)
for worker,attempt in [('optimize_multicsf_retained.py','1_retained_v1'),('optimize_multicsf_fr_v2.py','1_fr_v2')]:
 p=v/'pi_pi_repair/optimized/Adenine/B014'/attempt/'result.json'
 if not p.exists():run(worker,'Adenine','B014',1)
 if p.exists() and json.loads(p.read_text())['status']=='stationary_candidate':break
with (v/'audit_multicsf.log').open('w') as f:subprocess.run([sys.executable,str(v/'audit_multicsf.py')],stdout=f,stderr=subprocess.STDOUT,timeout=120)
