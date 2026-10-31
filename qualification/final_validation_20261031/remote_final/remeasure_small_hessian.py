"""Correct the property scope of three small-system timing measurements."""
from pathlib import Path
import sys,subprocess,socket
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent
for repeat in range(3):
 with (R/f'remeasure_hno_hessian_{repeat}.log').open('a') as log:
  subprocess.run(['/usr/bin/time','-v','-o',str(R/f'remeasure_hno_hessian_{repeat}.resources'),sys.executable,str(R/'cost.py'),'nitroxyl','hessian',str(repeat)],stdout=log,stderr=subprocess.STDOUT,check=True)
