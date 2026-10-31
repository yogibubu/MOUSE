"""Charge+spin response matching; frozen variational molecular parameters."""
from remote_bootstrap import HOST,BASE
import spin_kernel_probe as spin_job
from spin_kernel_probe import (plan,state_from,prepare_gfn2_deltascf,GFN2Calculator,
 evaluate_gfn2_roks_singlet,load_derivative_point,save_derivative_point,
 singlet_csf_overlap,OLD,EV,key,write,digest)
from matrix_gfn2 import FrozenResponseReference
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import argparse,numpy as np,json,time,traceback
from pyscf import dft,scf,lib
lib.num_threads(8);ROOT=BASE/'results/response_kernel';ROOT.mkdir(parents=True,exist_ok=True)
PROTOCOL=dict(spin_job.PROTOCOL,charge_response_matching=True,
 charge_origin='fixed ground-reference Mulliken shell populations',
 charge_correction='projected DFT charge Hessian minus projected full GFN2 charge Hessian',
 frozen_anchor_energy='DFT frozen excitation minus corrected-SE frozen excitation, constant under displacement')

def reference(case,ctx):
 spin=spin_job.coupling(case,ctx);s=ctx.state;out=ROOT/case['name'];out.mkdir(exist_ok=True)
 request=key(dict(protocol=PROTOCOL,spin=spin.fingerprint,script=digest(__file__)))
 path=out/'kernel.npz'
 pop=np.bincount(spin.orbital_shell,weights=np.sum(((s.coefficients*(s.alpha_occupations+s.beta_occupations))@s.coefficients.T)*s.overlap,axis=1),minlength=len(spin.coupling_hartree))
 if path.exists():
  with np.load(path) as f:assert str(f['request_hash'])==request;u=f['charge_correction'].copy()
 else:
  with np.load(spin_job.ROOT/case['name']/'kernel.npz') as f:dual=f['dual_density_gfn2'].copy();lift=f['embedding'].copy()
  folder=Path(case['reference']);mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=20000
  with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=np.array([f['occupations']/2]*2)
  mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=20000;mf.small_rho_cutoff=0
  mf.grids.level=3;mf.grids.build(with_non0tab=True);mf._eri=mol.intor('int2e',aosym='s8')
  response=mf.gen_response(mo_coeff=np.array([c,c]),mo_occ=occ,hermi=1,with_j=True)
  rho=np.array([lift@d@lift.T for d in dual]);ns=len(dual);wd=np.empty((ns,ns));ws=np.empty_like(wd)
  for b,(r,d) in enumerate(zip(rho,dual)):
   col=out/f'column_{b:03d}.npz'
   if col.exists():
    with np.load(col) as f:assert str(f['request_hash'])==request;wd[:,b]=f['dft'];ws[:,b]=f['se']
    continue
   t0=time.perf_counter();dm=np.array([.5*r,.5*r]);v=response(dm)
   v+=(1-.9)*.25*mf.get_k(mol,dm,hermi=1)
   wd[:,b]=.5*np.einsum('aij,ij->a',rho,v[0]+v[1])
   ws[:,b]=np.einsum('aij,ij->a',dual,ctx.kernel(d))
   np.savez_compressed(col,request_hash=request,dft=wd[:,b],se=ws[:,b])
   print('CHARGE',case['name'],b+1,ns,time.perf_counter()-t0,flush=True)
  ad=float(np.max(abs(wd-wd.T)));ass=float(np.max(abs(ws-ws.T)))
  assert ad<1e-6 and ass<1e-8
  wd=.5*(wd+wd.T);ws=.5*(ws+ws.T);u=wd-ws
  np.savez_compressed(path,request_hash=request,charge_correction=u,dft_charge=wd,se_charge=ws,
   spin_coupling=spin.coupling_hartree,reference_shell_populations=pop,orbital_shell=spin.orbital_shell,
   atomic_numbers=s.atomic_numbers)
  write(out/'kernel.json',dict(host=HOST,request_hash=request,spin_reference_fingerprint=spin.fingerprint,
   dft_symmetry_error=ad,se_symmetry_error=ass,dft_charge_diagonal=np.diag(wd),se_charge_diagonal=np.diag(ws),
   charge_correction_eigenvalues=np.linalg.eigvalsh(u),lambda_exchange=.9))
 return FrozenResponseReference(s.atomic_numbers,spin.orbital_shell,spin.coupling_hartree,
  f'PBE0/frozen-charge-spin-response/{request}',.9,u,pop)

def run(case):
 r0,xyz=state_from(next(t for t in case['targets'] if 'checkpoint' in t));s=r0.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
 ctx0=prepare_gfn2_deltascf(calc,xyz,ground_state=ground);ref=reference(case,ctx0)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
 prior=json.loads((OLD/'frozen_relaxation'/case['name']/'result.json').read_text());records=[]
 for target in case['targets']:
  row=dict(id=target['id'],tbe_ev=target['tbe_ev']);out=ROOT/case['name']/target['id'];out.mkdir(exist_ok=True)
  try:
   old=next(x for x in prior['records'] if x['id']==target['id'])
   if not old['qualified']:raise ValueError('prior ambiguous reference remains excluded')
   r,_=state_from(target);req=key(dict(reference=ref.fingerprint,checkpoint=digest(target['checkpoint'])))
   path=out/'converged.npz'
   if path.exists():result=load_derivative_point(path,request_hash=req,coordinates_angstrom=xyz,require_gradient=False)
   else:
    c=r.common_coefficients.copy();latest=out/'latest.npz'
    if latest.exists():
     with np.load(latest) as f:assert str(f['request_hash'])==req;c=f['coefficients'].copy()
    fm,em=ctx.fock_and_energy(np.array([(c*o)@c.T for o in r.occupations]));ft,et=ctx.fock_and_energy(np.array([(c*o)@c.T for o in r.triplet_occupations]))
    seed=replace(r,common_coefficients=c,fock_mixed=fm,fock_triplet=ft,energy_hartree=2*em-et,
     converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None,
     spin_reference_fingerprint=ref.fingerprint,gradient_hartree_per_bohr=None)
    def checkpoint(point):
     temp=latest.with_suffix('.tmp.npz');np.savez_compressed(temp,request_hash=req,coefficients=point.point);temp.replace(latest)
     write(out/'history.json',list(point.history))
    t0=time.perf_counter()
    result=evaluate_gfn2_roks_singlet(calc,xyz,*r.occupations,ground_state=ground,reference_result=seed,
     spin_reference=ref,orbital_solver='gmf',saddle_index=r.orbital_saddle_index,
     maximum_iterations=120,orbital_tolerance=1e-9,maximum_step=.2,checkpoint=checkpoint,gradient=False)
    row['optimization_seconds']=time.perf_counter()-t0;save_derivative_point(path,result,request_hash=req,coordinates_angstrom=xyz)
   frozen=[]
   for o in (r.occupations,r.triplet_occupations):
    _,e=ctx.fock_and_energy(np.array([(s.coefficients*oo)@s.coefficients.T for oo in o]));frozen.append(e)
   ef=2*frozen[0]-frozen[1]-result.ground_energy_hartree
   relaxation=result.excitation_energy_hartree-ef;hybrid=old['dft_frozen_ev']+relaxation*EV
   score=singlet_csf_overlap(replace(result,common_coefficients=s.coefficients).to_apoc(),result.to_apoc(),s.overlap)[0]
   row.update(qualified=score>=.7,canonical_csf_overlap=score,orbital_residual=result.orbital_residual,
    index=result.orbital_saddle_index,spin_squared=result.spin_squared,request_hash=req,
    raw_roks_ev=result.excitation_energy_hartree*EV,frozen_ev=ef*EV,relaxation_ev=relaxation*EV,
    hybrid_ev=hybrid,hybrid_error_ev=hybrid-target['tbe_ev'],previous_hybrid_ev=old['hybrid_ev'],
    previous_hybrid_error_ev=old['hybrid_error_ev'],spin_reference_fingerprint=ref.fingerprint,
    frozen_anchor_offset_hartree=(old['dft_frozen_ev']-ef*EV)/EV)
  except Exception as exc:row.update(qualified=False,error=str(exc),traceback=traceback.format_exc())
  records.append(row);write(ROOT/case['name']/'result.json',dict(host=HOST,protocol=PROTOCOL,records=records));print('STATE',case['name'],row,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--molecule');a=p.parse_args();write(ROOT/'protocol.json',PROTOCOL)
 for case in plan():
  if a.molecule and case['name']!=a.molecule:continue
  try:run(case)
  except Exception as exc:write(ROOT/case['name']/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
