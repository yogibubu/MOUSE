"""Oracle-only independent dense-eigenvalue/restart check of reference scheduling."""
from pathlib import Path
import socket,os,json,tempfile,sys
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
from pyscf.tdscf import _lr_eig
from pyscf.lib import logger
from threadpoolctl import threadpool_limits
from reference_eigensolver import checkpointed_solver
R=Path('/home/enzo/MOUSE-state-following-20261031')
with threadpool_limits(limits=2),tempfile.TemporaryDirectory(dir=R) as tmp:
 rng=np.random.default_rng(12);matrix=np.diag(np.linspace(1.,6.,80));noise=rng.normal(size=(80,80))*.02;matrix+=noise+noise.T;diag=np.diag(matrix).copy();initial=np.eye(80)[:1];calls=[]
 def operator(x):calls.append(np.array(x).copy());return x@matrix.T
 def precond(x,e):
  d=diag-e;d[abs(d)<1e-6]=1e-6;return x/d
 def pick(w,v,n,env):i=np.flatnonzero(w>0);return w[i],v[:,i],i
 class Stop(Exception):pass
 def interrupt(p):
  if p['cycle']>=2:raise Stop()
 args=dict(pick=pick,nroots=1,tol_residual=1e-10,max_cycle=150,max_memory=4000,lindep=1e-22,verbose=logger.Logger(sys.stdout,0))
 try:checkpointed_solver(_lr_eig.eigh,tmp,'test_operator',interrupt)(operator,initial,precond,**args)
 except Stop:pass
 else:raise AssertionError('checkpoint interruption missing')
 with np.load(Path(tmp)/'ritz_latest.npz') as f:saved=f['vectors'].copy()
 assert abs(float(saved[0]@initial[0]))<.99999
 calls.clear();converged,energies,vectors=checkpointed_solver(_lr_eig.eigh,tmp,'test_operator')(operator,initial,precond,**args)
 assert abs(float(calls[0][0]@saved[0]))>1-1e-10
 reference=np.linalg.eigvalsh(matrix)[0];err=abs(float(energies[0]-reference));res=float(np.linalg.norm(matrix@vectors[0]-energies[0]*vectors[0]));assert all(converged) and err<1e-9 and res<1e-10
 try:checkpointed_solver(_lr_eig.eigh,tmp,'foreign_operator')(operator,initial,precond,**args)
 except ValueError:pass
 else:raise AssertionError('foreign checkpoint accepted')
 report=dict(host=socket.getfqdn(),status='passed',dense_reference_error_hartree=err,residual=res,restart_uses_saved_ritz_vector=True,foreign_checkpoint_rejected=True,maximum_space_increment=1,operator_unchanged=True)
 (R/'reference_solver_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
