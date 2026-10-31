"""Checkpointed one-vector Davidson; preserves the exact PySCF response operator."""
from pathlib import Path
import json,time
import numpy as np
from pyscf.tdscf import _lr_eig

def checkpointed_solver(base,folder,fingerprint,hook=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 def solve(aop,x0,precond,**kwargs):
  checkpoint=folder/'ritz_latest.npz';pick=kwargs['pick'];nroots=kwargs.get('nroots',1);started=time.perf_counter()
  if checkpoint.exists():
   with np.load(checkpoint) as f:
    if str(f['fingerprint'])!=fingerprint:raise ValueError('reference Ritz checkpoint belongs to another operator')
    x0=f['vectors'].copy()
   if kwargs.get('x0sym') is not None:
    labels=np.unique(kwargs['x0sym']);assert len(labels)==1
    kwargs['x0sym']=np.repeat(labels[0],len(x0))
  def save_pick(w,v,n,env):
   w,v,idx=pick(w,v,n,env);count=min(nroots,len(w));vectors=v[:,:count].T@env['xs'];av=v[:,:count].T@env['ax'];residual=np.linalg.norm(av-w[:count,None]*vectors,axis=1)
   temp=folder/'ritz_latest.tmp.npz';np.savez_compressed(temp,fingerprint=fingerprint,vectors=vectors,energies_hartree=w[:count],residuals=residual,cycle=env['icyc']);temp.replace(checkpoint)
   progress=dict(cycle=int(env['icyc']),vectors=int(len(env['xs'])),residuals=residual.tolist(),seconds_current_run=time.perf_counter()-started,ritz_checkpoint_saved=True)
   temp=folder/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(folder/'progress.json');print('REFERENCE_RITZ',str(folder),progress,flush=True)
   if hook is not None:hook(progress)
   return w,v,idx
  kwargs['pick']=save_pick;previous=_lr_eig.MAX_SPACE_INC;_lr_eig.MAX_SPACE_INC=1
  try:return base(aop,x0,precond,**kwargs)
  finally:_lr_eig.MAX_SPACE_INC=previous
 return solve
