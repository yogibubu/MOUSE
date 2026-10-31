"""TDA control of the complete vertical response functional on its own GF MOs."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,json,numpy as np,time
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,key,digest,EV
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf,run_deltascf
from full_response import VerticalReference
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from pyscf import lib
lib.num_threads(4)
with threadpool_limits(limits=2,user_api='blas'):
 for case in [c for c in plan() if c['name'] in ['nitroxyl','uracil']]:
  out=BASE/'results/full_response'/case['name'];final=out/'tda.json'
  if final.exists():continue
  r0,xyz=state_from(case['targets'][0]);s=r0.reference_state
  with np.load(out/'model.npz') as f:data={k:f[k].copy() for k in f.files};request=str(f['request_hash'])
  calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
  ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
  native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground);ref=VerticalReference(s,native.kernel.partition,data,request)
  ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
  start=time.perf_counter();gp=run_deltascf(s.overlap,np.array([s.coefficients]*2),np.array([s.alpha_occupations,s.beta_occupations]),ctx.fock_and_energy,
   selection='aufbau',density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
  gt=time.perf_counter()-start;energies,c=eigh(gp.fock[0],s.overlap)
  state=replace(s,coefficients=c,fock=gp.fock[0],orbital_energies_hartree=energies,total_energy_hartree=gp.energy_hartree)
  occ=np.flatnonzero(s.alpha_occupations==1);virt=np.flatnonzero(s.alpha_occupations==0);ii=np.repeat(occ,len(virt));aa=np.tile(virt,len(occ));ci=c[:,ii];ca=c[:,aa]
  kernel=ctx.density_response_kernel(state);columns=[];start=time.perf_counter()
  for i,a in zip(ii,aa):
   p=np.outer(c[:,i],c[:,a]);f=kernel(.5*(p+p.T));columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
  matrix=np.diag(energies[aa]-energies[ii])+2*np.array(columns).T;assert np.max(abs(matrix-matrix.T))<1e-8
  values,vectors=np.linalg.eigh(matrix);seconds=time.perf_counter()-start
  np.savez_compressed(out/'tda.npz',matrix=matrix,energies=values,vectors=vectors,coefficients=c,ground_coefficients=gp.coefficients,ground_energy=gp.energy_hartree)
  rows=[]
  for t in case['targets']:
   a=np.zeros((len(occ),len(virt)));a[np.flatnonzero(occ==t['hole'])[0],np.flatnonzero(virt==t['particle'])[0]]=1.
   projected=(c[:,occ].T@s.overlap@s.coefficients[:,occ])@a@(s.coefficients[:,virt].T@s.overlap@c[:,virt])
   norm=np.linalg.norm(projected);score=abs(projected.ravel()@vectors)/norm;k=int(np.argmax(score));margin=float(score[k]-np.max(np.delete(score,k),initial=0.))
   rows.append(dict(state=t['id'],energy_ev=float(values[k]*EV),tbe_ev=t['tbe_ev'],error_ev=float(values[k]*EV-t['tbe_ev']),
    overlap=float(score[k]),margin=margin,projection_norm=float(norm),qualified=bool(norm>=.7 and score[k]>=.7 and margin>=.05)))
  report=dict(host=HOST,request_hash=request,orbitals='same corrected GFN2 functional; no DFT orbitals',solver_seconds=seconds,ground_seconds=gt,ground_energy_hartree=gp.energy_hartree,rows=rows,scope='vertical development control')
  write(final,report);print(report,flush=True)
