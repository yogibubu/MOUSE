"""Measure corrected true-TDA costs; old static-response timings are preserved."""
from pathlib import Path
import socket,sys,subprocess
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent
for name,repeats in [('nitroxyl',range(3)),('acetaldehyde',[0])]:
 for repeat in repeats:
  with (R/f'correct_tda_{name}_{repeat}.log').open('a') as log:subprocess.run(['/usr/bin/time','-v','-o',str(R/f'correct_tda_{name}_{repeat}.resources'),sys.executable,str(R/'cost.py'),name,'tda',str(repeat)],stdout=log,stderr=subprocess.STDOUT,check=True)
