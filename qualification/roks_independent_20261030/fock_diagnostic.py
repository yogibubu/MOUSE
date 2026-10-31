"""Development-only first-derivative matching, not an independent validation.

Fit a scalar shell potential to the DFT-minus-GFN2 Fock operator in an
orthonormal metric. This fits a ground-state operator, never excitation TBEs.
The uniform potential gauge is removed before the projection.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,key,digest,invsqrt
from matrix_gfn2 import FrozenFockResponseReference,load_response_reference,save_response_reference
from pyscf import scf,dft,lib
import evaluate
lib.num_threads(4);evaluate.LABELS=['fock_shell_charge_spin']
for case in [c for c in plan() if c['name'] in ['nitroxyl','uracil']]:
 name=case['name'];out=BASE/'results/models'/name;source=out/'shell_charge_spin.npz'
 if not source.exists():continue
 ref=load_response_reference(source);path=out/'fock_shell_charge_spin.npz'
 if not path.exists():
  r,xyz=state_from(case['targets'][0]);s=r.reference_state;folder=Path(case['reference'])
  previous=Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/name/'kernel.npz'
  with np.load(previous) as f:lift=f['embedding'];assert str(f['reference_sha256'])==digest(folder/'reference.npz')
  with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=f['occupations']
  mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mf=dft.RKS(mol);mf.xc='PBE0';mf.verbose=0
  mf.grids.level=3;mf.small_rho_cutoff=0;mf.grids.build(with_non0tab=True);mf.max_memory=20000;mf._eri=mol.intor('int2e',aosym='s8')
  dm=(c*occ)@c.T;fock=mf.get_hcore()+mf.get_veff(mol,dm)
  half=invsqrt(s.overlap);difference=half@(lift.T@fock@lift-s.fock)@half;n=len(half);eye=np.eye(n)
  difference-=eye*np.trace(difference)/n
  operators=[]
  for k in range(len(ref.coupling_hartree)):
   mask=(ref.orbital_shell==k).astype(float);b=.5*s.overlap*(mask[:,None]+mask[None,:]);z=half@b@half
   operators.append(z-eye*np.trace(z)/n)
  operators=np.asarray(operators);v=np.linalg.lstsq(operators.reshape(len(operators),-1).T,difference.ravel(),rcond=1e-10)[0]
  projected=np.einsum('a,aij->ij',v,operators)
  request=key(dict(source=digest(source),reference=digest(folder/'reference.npz'),script=digest(__file__)))
  model=FrozenFockResponseReference(ref.atomic_numbers,ref.orbital_shell,ref.coupling_hartree,
   ref.source_model+'/first-derivative/'+request,.9,ref.charge_correction_hartree,ref.reference_shell_populations,v)
  save_response_reference(path,model)
  write(out/'fock_projection.json',dict(host=HOST,request_hash=request,shell_potential_hartree=v,
   fock_difference_norm=float(np.linalg.norm(difference)),residual_norm=float(np.linalg.norm(difference-projected)),
   excited_reference_energies_used=False,energy_offset=0.,development_only=True))
 evaluate.run(case)
