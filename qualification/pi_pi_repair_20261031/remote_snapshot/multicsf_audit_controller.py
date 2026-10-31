from pathlib import Path
import subprocess,sys,os,time
v=Path('/home/enzo/MOUSE-orbital-performance-20261031')
while True:
 try:os.kill(816351,0)
 except ProcessLookupError:break
 time.sleep(5)
with (v/'audit_multicsf.log').open('w') as f:subprocess.run([sys.executable,str(v/'audit_multicsf.py')],stdout=f,stderr=subprocess.STDOUT,timeout=120)
