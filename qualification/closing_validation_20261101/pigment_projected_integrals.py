"""Checkpointed RI Coulomb factors in the pigment core/native-valence chart.

Reuse the converged full reference and native embedding; no new SCF. Separate
full- and long-range factors support the unchanged CAM response coefficients.
The expensive full-AO factors are streamed, never expanded to a full-AO ERI.
"""
from pathlib import Path
import socket,subprocess,time,json,os,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
from prepare_fresh import HERE
import numpy as np
from pyscf import scf,df,ao2mo
from threadpoolctl import threadpool_limits

def write(path,data):
 temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

def run(name):
 out=HERE/'pigment_integrals'/name;out.mkdir(parents=True,exist_ok=True)
 folder=HERE/'pigment_dft_reference'/name;embedded=HERE/'pigment_embedding'/name
 assert (embedded/'certificate.json').exists()
 with np.load(embedded/'embedding.npz') as z:lift=z['embedding'].copy()
 with np.load(folder/'PBE0/reference.npz') as z:cd=z['coefficients'].copy()
 cert=json.loads((embedded/'certificate.json').read_text());nc=cert['frozen_core_orbitals'];n=lift.shape[1]
 coefficient=np.asfortranarray(np.column_stack([cd[:,:nc],lift]));mol=scf.chkfile.load_mol(str(folder/'PBE0/backend.chk'))
 base=df.DF(mol);base.auxbasis='def2-universal-jkfit';base.max_memory=64000;base._cderi=str(folder/'PBE0/df_integrals.h5')
 for label,omega in [('coulomb',0.),('long_range',.33)]:
  done=out/(label+'_complete.json')
  if done.exists():continue
  start=time.monotonic()
  manager=base.range_coulomb(omega) if omega else None
  target=manager.__enter__() if manager else base
  try:
   if omega:
    target._cderi_to_save=str(out/'long_range_full_DF.h5')
    if (out/'long_range_full_DF.h5').exists():target._cderi=str(out/'long_range_full_DF.h5')
    else:target.build()
   count=target.get_naoaux();shape=(count,nc+n,nc+n);path=out/(label+'_factors.npy');progress=out/(label+'_progress.json')
   completed=int(json.loads(progress.read_text())['completed']) if progress.exists() else 0
   array=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=np.float64)
   assert array.shape==shape
   offset=0
   for block in target.loop(blksize=16):
    hi=offset+len(block)
    if hi>completed:
     transformed=ao2mo._ao2mo.nr_e2(block,coefficient,(0,nc+n,0,nc+n),aosym='s2',mosym='s1')
     array[offset:hi]=transformed.reshape(len(block),nc+n,nc+n);array.flush()
     write(progress,dict(completed=hi,total=count,wall_seconds=time.monotonic()-start))
     print(name,label,hi,'/',count,flush=True)
    offset=hi
   assert offset==count
   write(done,dict(host=socket.getfqdn(),shape=shape,core_orbitals=nc,native_orbitals=n,range_parameter=omega,auxbasis=base.auxbasis,wall_seconds=time.monotonic()-start,full_AO_ERI_allocated=False,new_SCF_calls=0,source_embedding=cert['output_sha256']))
  finally:
   if manager:manager.__exit__(None,None,None)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-embedding',action='store_true');a=p.parse_args();out=HERE/'pigment_integrals'/a.name;out.mkdir(parents=True,exist_ok=True);lock=out/'worker.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  while not (HERE/'pigment_embedding'/a.name/'certificate.json').exists():
   if not a.wait_for_embedding:raise RuntimeError('Embedding incomplete')
   time.sleep(20)
  with threadpool_limits(8):run(a.name)
 finally:lock.unlink(missing_ok=True)
