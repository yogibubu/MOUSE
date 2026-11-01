"""Bounded remote parallelism; retain every completed analytic column."""
from pathlib import Path
import sys,subprocess,concurrent.futures,socket,argparse
assert socket.getfqdn()=='oracle.sns.it'
HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
p=argparse.ArgumentParser();p.add_argument('--criegee',action='store_true');a=p.parse_args()
cases=[('independent','Criegee_Intermediate')] if a.criegee else [('last_two','Furan'),('last_two','uracil')]
def run(job):
 cohort,name,worker=job
 with (HERE/f'hessian_{name}_worker{worker}.log').open('a') as log:subprocess.run([sys.executable,str(HERE/'hessian_columns.py'),cohort,name,'--worker',str(worker),'--workers','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,worker,'complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(run,[(cohort,name,worker) for cohort,name in cases for worker in range(2)]))
