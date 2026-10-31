from pathlib import Path
import subprocess,sys,socket
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent
for rank in [2,3]:subprocess.run([sys.executable,str(R/'csf_rank_convergence.py'),'Furofuran',str(rank)],cwd=R,check=True)
