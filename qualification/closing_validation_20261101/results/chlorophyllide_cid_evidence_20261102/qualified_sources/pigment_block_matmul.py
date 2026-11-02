"""Restartable block products for large frozen reference matrices, remote only."""
import socket
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
from pathlib import Path
import json,os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

def _rows(task):
 left_path,right_path,out_path,lo,hi,marker=task
 left=np.load(left_path,mmap_mode='r');right=np.load(right_path,mmap_mode='r');out=np.load(out_path,mmap_mode='r+');np.matmul(left[:,lo:hi].T,right,out=out[lo:hi]);out.flush();p=Path(marker);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(start=lo,stop=hi))+'\n');tmp.replace(p);return lo,hi

def product(left_path,right_path,out_path,*,workers=4,block=1024):
 left_path=Path(left_path);right_path=Path(right_path);out_path=Path(out_path);left=np.load(left_path,mmap_mode='r');right=np.load(right_path,mmap_mode='r');assert left.shape[0]==right.shape[0];shape=(left.shape[1],right.shape[1]);journal=out_path.with_suffix('.blocks');journal.mkdir(exist_ok=True)
 if not out_path.exists():np.lib.format.open_memmap(out_path,mode='w+',shape=shape,dtype=np.float64).flush()
 else:assert np.load(out_path,mmap_mode='r').shape==shape
 tasks=[]
 for lo in range(0,shape[0],block):
  hi=min(lo+block,shape[0]);marker=journal/f'{lo}_{hi}.json'
  if not marker.exists():tasks.append((str(left_path),str(right_path),str(out_path),lo,hi,str(marker)))
 previous=os.environ.get('VECLIB_MAXIMUM_THREADS');os.environ['VECLIB_MAXIMUM_THREADS']='1'
 try:
  with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
   for lo,hi in pool.map(_rows,tasks):print(out_path.name,hi,'/',shape[0],flush=True)
 finally:
  if previous is None:os.environ.pop('VECLIB_MAXIMUM_THREADS',None)
  else:os.environ['VECLIB_MAXIMUM_THREADS']=previous
 return np.load(out_path,mmap_mode='r+')
