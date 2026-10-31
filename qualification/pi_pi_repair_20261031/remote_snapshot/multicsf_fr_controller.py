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
 with (v/('multicsf_B024_'+str(i)+'_fr.log')).open('a') as f:subprocess.run([sys.executable,str(v/'optimize_multicsf_fr.py'),'Phthalimide','B024',str(i)],stdout=f,stderr=subprocess.STDOUT,timeout=240)
