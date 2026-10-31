"""Project the ground-DFT spin Hessian onto dual GFN2 Mulliken shell moments.

No excited-state target energies enter the coupling. Exact exchange is scaled
once by the same lambda=0.9 as the uracil TDA response. GGA fxc is unchanged.
The molecular W is then frozen as a parameter of the variational GFN2 model.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,argparse,json,numpy as np,time,traceback
from types import SimpleNamespace
from dataclasses import replace
from scipy.linalg import eigh
from pyscf import dft,scf,gto,lib
from pyscf.scf import _response_functions
OLD=Path('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029')
sys.path.insert(0,str(OLD))
from frozen_relaxation import plan,state_from,digest,write,key,EV
from extent_support import source_molecule
from matrix_gfn2 import GFN2Calculator,FrozenSpinReference,evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_apoc.deltascf import singlet_csf_overlap
lib.num_threads(8);ROOT=BASE/'results/spin_kernel';ROOT.mkdir(parents=True,exist_ok=True)
PROTOCOL=dict(host=HOST,xc='PBE0',grid_level=3,lambda_exchange=.9,
 frozen_molecular_spin_coupling=True,parameters_fitted=0,
 dual='Mulliken shell moments dual to orthonormal shell projectors',
 embedding='cross-AO isometry after removing occupied DFT cores',
 planned_molecules=[c['name'] for c in plan()],orbital_tolerance=1e-9,
 maximum_iterations=120,minimum_csf_overlap=.70,saddle_index='retained SE index',
 excluded='prior ambiguous cytosine n_pi; no reassignment',
 new_default=False)

def invsqrt(a):
 w,u=eigh(a)
 if min(w)<1e-9:raise ValueError('singular embedding metric')
 return (u*w**-.5)@u.T

def coupling(case,ctx):
 out=ROOT/case['name'];out.mkdir(exist_ok=True);folder=Path(case['reference']);s=ctx.state
 request=key(dict(protocol=PROTOCOL,reference=digest(folder/'reference.npz'),
  checkpoint=digest(folder/'backend.chk'),source=digest(__file__)))
 final=out/'kernel.npz'
 if final.exists():
  with np.load(final) as f:
   assert str(f['request_hash'])==request
   return FrozenSpinReference(s.atomic_numbers,ctx.polarization.orbital_shell,f['coupling'],f'PBE0/spin-response/{request}',.9)
 with np.load(folder/'reference.npz') as f:ref={k:f[k] for k in f.files}
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=20000
 mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=20000;mf.small_rho_cutoff=0;mf.grids.level=3;mf.grids.build(with_non0tab=True)
 mf._eri=mol.intor('int2e',aosym='s8')
 c=ref['coefficients'];occ=np.array([ref['occupations']/2]*2)
 mf.mo_coeff=np.array([c,c]);mf.mo_occ=occ
 respond=mf.gen_response(hermi=1,with_j=False)
 small,t=source_molecule(s);cross=gto.intor_cross('int1e_ovlp',mol,small)@t
 core=int(np.sum(np.asarray(s.atomic_numbers)-np.asarray(ctx.calculator._valence))/2)
 assert core>=0
 ccore=c[:,np.flatnonzero(ref['occupations']==2)[:core]]
 x=np.linalg.solve(ref['overlap'],cross)-ccore@(ccore.T@cross)
 ss=s.overlap;ws,us=eigh(ss);sqrt=(us*np.sqrt(ws))@us.T
 lift=x@invsqrt(x.T@ref['overlap']@x)@sqrt
 iso=float(np.max(abs(lift.T@ref['overlap']@lift-ss)))
 assert iso<1e-8
 shells=ctx.polarization.orbital_shell;ns=len(ctx.polarization.coupling_hartree)
 half=invsqrt(ss);trial=[];moments=[]
 for k in range(ns):
  mask=(shells==k).astype(float);trial.append((half*(mask/mask.sum()))@half)
  moments.append(.5*(mask[:,None]*ss+ss*mask[None,:]))
 trial=np.array(trial);moments=np.array(moments)
 dualmap=np.einsum('aij,bij->ab',moments,trial);dual=np.einsum('bij,ba->aij',trial,np.linalg.inv(dualmap))
 error=float(np.max(abs(np.einsum('aij,bij->ab',moments,dual)-np.eye(ns))))
 assert error<1e-9
 rho=np.array([lift@d@lift.T for d in dual]);w=np.zeros((ns,ns));times=[]
 fraction=float(mf._numint.rsh_and_hybrid_coeff(mf.xc,mol.spin)[2])
 for b,r in enumerate(rho):
  point=out/f'column_{b:03d}.npz'
  if point.exists():
   with np.load(point) as f:assert str(f['request_hash'])==request;w[:,b]=f['column']
   continue
  t0=time.perf_counter();dm=np.array([.5*r,-.5*r]);v=respond(dm)
  v+=(1-.9)*fraction*mf.get_k(mol,dm,hermi=1)
  w[:,b]=.5*np.einsum('aij,ij->a',rho,v[0]-v[1])
  np.savez_compressed(point,request_hash=request,column=w[:,b]);times.append(time.perf_counter()-t0)
  print('KERNEL',case['name'],b+1,ns,times[-1],flush=True)
 asym=float(np.max(abs(w-w.T)));assert asym<1e-6
 w=.5*(w+w.T)
 np.savez_compressed(final,request_hash=request,coupling=w,source_coupling=ctx.polarization.coupling_hartree,
  dual_density_gfn2=dual,embedding=lift,reference_sha256=digest(folder/'reference.npz'))
 write(out/'kernel.json',dict(host=HOST,request_hash=request,shells=ns,core_orbitals_removed=core,
  isometry_error=iso,dual_moment_error=error,raw_symmetry_error=asym,
  source_diagonal=np.diag(ctx.polarization.coupling_hartree),dft_diagonal=np.diag(w),
  lambda_exchange=.9,reference_sha256=digest(folder/'reference.npz'),seconds_new_columns=sum(times)))
 return FrozenSpinReference(s.atomic_numbers,shells,w,f'PBE0/spin-response/{request}',.9)

def run(case):
 valid=[t for t in case['targets'] if 'checkpoint' in t];r0,xyz=state_from(valid[0]);s=r0.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
 oldctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground);spin=coupling(case,oldctx)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=spin)
 previous=json.loads((OLD/'frozen_relaxation'/case['name']/'result.json').read_text());records=[]
 for target in case['targets']:
  row=dict(id=target['id'],tbe_ev=target['tbe_ev']);out=ROOT/case['name']/target['id'];out.mkdir(exist_ok=True)
  try:
   prior=next(x for x in previous['records'] if x['id']==target['id'])
   if not prior['qualified']:raise ValueError('prior reference mapping excluded: '+prior['exclusion_reason'])
   r,_=state_from(target);request=key(dict(spin=spin.fingerprint,checkpoint=digest(target['checkpoint'])))
   path=out/'converged.npz'
   if path.exists():result=load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)
   else:
    c=r.common_coefficients.copy();latest=out/'latest.npz'
    if latest.exists():
     with np.load(latest) as f:assert str(f['request_hash'])==request;c=f['coefficients'].copy()
    fm,em=ctx.fock_and_energy(np.array([(c*o)@c.T for o in r.occupations]));ft,et=ctx.fock_and_energy(np.array([(c*o)@c.T for o in r.triplet_occupations]))
    seed=replace(r,common_coefficients=c,fock_mixed=fm,fock_triplet=ft,energy_hartree=2*em-et,
     converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None,
     spin_reference_fingerprint=spin.fingerprint,gradient_hartree_per_bohr=None)
    def checkpoint(point):
     temp=latest.with_suffix('.tmp.npz');np.savez_compressed(temp,request_hash=request,coefficients=point.point);temp.replace(latest)
     write(out/'history.json',list(point.history))
    t0=time.perf_counter()
    result=evaluate_gfn2_roks_singlet(calc,xyz,*r.occupations,ground_state=ground,reference_result=seed,
     spin_reference=spin,orbital_solver='gmf',saddle_index=r.orbital_saddle_index,
     maximum_iterations=120,orbital_tolerance=1e-9,maximum_step=.2,checkpoint=checkpoint,gradient=False)
    row['optimization_seconds']=time.perf_counter()-t0
    save_derivative_point(path,result,request_hash=request,coordinates_angstrom=xyz)
   frozen=[]
   for o in (r.occupations,r.triplet_occupations):
    _,e=ctx.fock_and_energy(np.array([(s.coefficients*oo)@s.coefficients.T for oo in o]));frozen.append(e)
   ef=2*frozen[0]-frozen[1]-result.ground_energy_hartree
   relaxation=result.excitation_energy_hartree-ef
   hybrid=prior['dft_frozen_ev']+relaxation*EV
   score=singlet_csf_overlap(replace(result,common_coefficients=s.coefficients).to_apoc(),result.to_apoc(),s.overlap)[0]
   row.update(qualified=score>=.7,canonical_csf_overlap=score,orbital_residual=result.orbital_residual,
    index=result.orbital_saddle_index,spin_squared=result.spin_squared,
    raw_roks_ev=result.excitation_energy_hartree*EV,frozen_ev=ef*EV,relaxation_ev=relaxation*EV,
    hybrid_ev=hybrid,hybrid_error_ev=hybrid-target['tbe_ev'],previous_hybrid_ev=prior['hybrid_ev'],
    previous_hybrid_error_ev=prior['hybrid_error_ev'],spin_reference_fingerprint=spin.fingerprint)
  except Exception as exc:row.update(qualified=False,error=str(exc),traceback=traceback.format_exc())
  records.append(row);write(ROOT/case['name']/'result.json',dict(host=HOST,protocol=PROTOCOL,records=records));print('STATE',case['name'],row,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--molecule');a=p.parse_args();write(ROOT/'protocol.json',PROTOCOL)
 for case in plan():
  if a.molecule and case['name']!=a.molecule:continue
  try:run(case)
  except Exception as exc:write(ROOT/case['name']/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
