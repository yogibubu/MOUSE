"""Explicit execution-thread override; scientific request and checkpoints unchanged."""
import argparse,traceback
import dft_relaxation as job
from pyscf import lib
p=argparse.ArgumentParser();p.add_argument('--molecule',required=True);p.add_argument('--state');p.add_argument('--threads',type=int,default=8);a=p.parse_args()
lib.num_threads(a.threads)
runtime=dict(host=job.HOST,actual_openmp_threads=lib.num_threads(),
 protocol_threads_mean='default; execution override recorded here',
 wrapper_sha256=job.digest(__file__))
job.OUT.mkdir(exist_ok=True);job.write(job.OUT/'protocol.json',job.PROTOCOL)
for state in job.PROTOCOL['states']:
 if a.state and a.state!=state:continue
 folder=job.OUT/a.molecule/state;folder.mkdir(parents=True,exist_ok=True)
 job.write(folder/'runtime.json',runtime)
 try:
  r=job.run(a.molecule,state)
  r['execution']=runtime;job.write(folder/'result.json',r)
 except Exception as exc:
  job.write(folder/'failure.json',dict(host=job.HOST,error=str(exc),traceback=traceback.format_exc()))
  print('FAILURE',a.molecule,state,traceback.format_exc(),flush=True)
