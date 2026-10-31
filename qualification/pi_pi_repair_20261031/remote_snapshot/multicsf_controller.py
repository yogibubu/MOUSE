from pathlib import Path
import subprocess,sys
v=Path('/home/enzo/MOUSE-orbital-performance-20261031')
for name,sid in [('Furan','B022'),('Cyclopentadiene','B016'),('Phthalimide','B024'),('Adenine','B014')]:
 for i in [0,1]:
  with (v/('multicsf_'+sid+'_'+str(i)+'.log')).open('a') as f:subprocess.run([sys.executable,str(v/'optimize_multicsf.py'),name,sid,str(i)],stdout=f,stderr=subprocess.STDOUT,timeout=240)
