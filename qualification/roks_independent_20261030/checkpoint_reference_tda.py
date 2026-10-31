"""Preserve our own running Davidson trial space before changing only the solver.

Reads explicitly named scientific arrays from our paused Oracle process, no
injection. Completed SCF and all completed matrix-vector products are retained.
Any error resumes the original job; it is stopped only after verified storage.
"""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import os,signal,subprocess,re,struct,json,numpy as np,sys,argparse
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args();proc=Path('/proc')/str(a.pid)
out=BASE/'references_C/pyrazine';checkpoint=out/'tda_progress.npz'
if (out/'backend_tda.npz').exists():raise SystemExit('reference root already complete; unchanged')
assert proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==BASE.resolve()
args=(proc/'cmdline').read_bytes().split(b'\0');assert len(args)>3 and args[1]==b'holdout_C_run.py' and args[2:4]==[b'--index',b'1']
assert not checkpoint.exists();stored=False
os.kill(a.pid,signal.SIGSTOP)
try:
 dump=subprocess.check_output([str(BASE/'profiling/bin/py-spy'),'dump','--pid',str(a.pid),'--locals','--nonblocking'],text=True,timeout=15)
 frame=dump[dump.index('    eigh (pyscf/tdscf/_lr_eig.py:'):];frame=frame[:frame.index('    kernel (pyscf/tdscf/rhf.py:')]
 assert '_lr_eig.py:111)' in frame
 ptrs={k:int(re.search(r'\b'+k+r': <numpy.ndarray at (0x[0-9a-f]+)>',frame).group(1),16) for k in ['xs','ax','w','e','xt']}
 with (proc/'mem').open('rb',buffering=0) as mem:
  def read(address,n):mem.seek(address);b=mem.read(n);assert len(b)==n;return b
  def array(address):
   data=struct.unpack('<Q',read(address+16,8))[0];nd=struct.unpack('<i',read(address+24,4))[0];assert 1<=nd<=2
   shp=struct.unpack('<Q',read(address+32,8))[0];st=struct.unpack('<Q',read(address+40,8))[0]
   shape=struct.unpack('<'+'q'*nd,read(shp,nd*8));strides=struct.unpack('<'+'q'*nd,read(st,nd*8));assert min(shape)>0 and min(strides)>0
   size=sum((n-1)*s for n,s in zip(shape,strides))+8;assert size<=200_000_000
   value=np.ndarray(shape,dtype=np.float64,buffer=read(data,size),strides=strides).copy();assert np.all(np.isfinite(value));return value
  data={k:array(v) for k,v in ptrs.items()}
 assert data['xs'].shape==data['ax'].shape and len(data['w'])==len(data['xs'])
 assert np.max(abs(data['xs'][:8]@data['xs'][:8].T-np.eye(min(8,len(data['xs'])))))<1e-8
 residual=float(np.linalg.norm(data['ax'][0]-data['w'][0]*data['xs'][0]));assert residual<1e-5
 tmp=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(tmp,reference_sha256=digest(out/'reference.npz'),**data);tmp.replace(checkpoint)
 with np.load(checkpoint) as f:assert np.array_equal(f['xs'],data['xs']) and np.array_equal(f['ax'],data['ax'])
 write(out/'tda_progress.json',dict(host=HOST,pid=a.pid,source_sha256=digest(__file__),reference_sha256=digest(out/'reference.npz'),checkpoint_sha256=digest(checkpoint),completed_matvecs=len(data['xs']),ritz_residual=residual,
  iterations=int(re.search(r'icyc: (\d+)',frame).group(1)),reason='reference TDA Davidson stagnation near requested tolerance; preserve trial space and resume the same eigenpair with stricter linear-dependence threshold'))
 (out/'tda_checkpoint_stack.txt').write_text(dump);stored=True;print('TDA PRESERVED',len(data['xs']),residual,flush=True)
finally:
 if stored:os.kill(a.pid,signal.SIGTERM)
 os.kill(a.pid,signal.SIGCONT)
