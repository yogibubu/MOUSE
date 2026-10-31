"""No fitted excitation energies: projected, invariant tensor response coefficients."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,numpy as np,argparse,traceback
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
import spin_kernel_probe as spin
import response_kernel_probe as charge
from spin_kernel_probe import state_from,write,digest,key,EV
from extent_support import source_molecule
from matrix_gfn2 import GFN2Calculator,FrozenTensorResponseReference,FrozenResponseReference,save_response_reference,load_response_reference
from matrix_gfn2.tensor_spin import onsite_spin_generators
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from pyscf import dft,scf,lib
lib.num_threads(8)
spin.ROOT=BASE/'results/shell_spin';spin.ROOT.mkdir(exist_ok=True)
charge.ROOT=BASE/'results/shell_charge';charge.ROOT.mkdir(exist_ok=True)
ROOT=BASE/'results/models';ROOT.mkdir(exist_ok=True)

def calibrate(case):
 name=case['name'];out=ROOT/name;out.mkdir(exist_ok=True)
 if not case['targets']:return
 if all((out/(label+'.npz')).exists() for label in ['shell_charge_spin','tensor_spin','tensor_charge_spin']):
  for label in ['shell_charge_spin','tensor_spin','tensor_charge_spin']:load_response_reference(out/(label+'.npz'))
  print('REUSE COMPLETE CALIBRATION',name,flush=True);return out
 r,xyz=state_from(case['targets'][0]);s=r.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r.ground_energy_hartree)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 previous=Path('/home/enzo/MOUSE-roks-repair-20261030/results')
 if (previous/'response_kernel'/name/'kernel.npz').exists():
  kernel=previous/'response_kernel'/name/'kernel.npz'
  with np.load(kernel) as f:
   shell=FrozenResponseReference(f['atomic_numbers'],f['orbital_shell'],f['spin_coupling'],'PBE0/frozen-charge-spin-response/'+str(f['request_hash']),.9,f['charge_correction'],f['reference_shell_populations'])
  embedding_file=previous/'spin_kernel'/name/'kernel.npz'
  with np.load(embedding_file) as f:assert str(f['reference_sha256'])==digest(Path(case['reference'])/'reference.npz')
  write(out/'reused_shell_reference.json',dict(kernel=str(kernel),sha256=digest(kernel)))
 else:
  shell=charge.reference(case,ctx);embedding_file=spin.ROOT/name/'kernel.npz'
 shellfile=out/'shell_charge_spin.npz'
 if not shellfile.exists():save_response_reference(shellfile,shell)
 request=key(dict(protocol=digest(BASE/'protocol.json'),shell=shell.fingerprint,script=digest(__file__)))
 final=out/'coefficients.npz'
 if final.exists():
  with np.load(final) as f:assert str(f['request_hash'])==request;ws=f['spin'];uc=f['charge']
 else:
  a,groups=onsite_spin_generators(s);n=len(s.overlap);ns=len(shell.coupling_hartree)
  shellops=np.array([np.diag((shell.orbital_shell==i).astype(float)) for i in range(ns)])
  allops=np.concatenate([shellops,a]);b=.5*(s.overlap@allops+allops@s.overlap)
  half=spin.invsqrt(s.overlap);trial=half@allops@half
  gram=np.einsum('aij,bij->ab',b,trial);dual=np.einsum('bij,ba->aij',trial,np.linalg.inv(gram))[ns:]
  error=float(np.max(abs(np.einsum('aij,bij->ab',b,dual)-np.eye(len(allops))[:,ns:])))
  assert error<1e-8
  with np.load(embedding_file) as f:lift=f['embedding']
  folder=Path(case['reference']);mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=20000
  with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=np.array([f['occupations']/2]*2)
  mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=20000;mf.small_rho_cutoff=0
  mf.grids.level=3;mf.grids.build(with_non0tab=True);mf._eri=mol.intor('int2e',aosym='s8')
  respond=mf.gen_response(mo_coeff=np.array([c,c]),mo_occ=occ,hermi=1,with_j=True)
  ds=[];dc=[];native=[]
  for i,d in enumerate(dual):
   path=out/f'column_{i:03d}.npz'
   if path.exists():
    with np.load(path) as f:assert str(f['request_hash'])==request;cs=float(f['spin']);cc=float(f['charge']);cn=float(f['native'])
   else:
    start=time.perf_counter();rho=lift@d@lift.T
    pm=np.array([.5*rho,-.5*rho]);v=respond(pm)+(1-.9)*.25*mf.get_k(mol,pm,hermi=1)
    cs=.5*float(np.sum(rho*(v[0]-v[1])))
    pm=np.array([.5*rho,.5*rho]);v=respond(pm)+(1-.9)*.25*mf.get_k(mol,pm,hermi=1)
    cc=.5*float(np.sum(rho*(v[0]+v[1])));cn=float(np.sum(d*ctx.kernel(d)))
    np.savez_compressed(path,request_hash=request,spin=cs,charge=cc,native=cn,seconds=time.perf_counter()-start)
    print('TENSOR COLUMN',name,i+1,len(dual),cs,cc-cn,flush=True)
   ds.append(cs);dc.append(cc);native.append(cn)
  ws=np.zeros((len(s.atomic_numbers),2));uc=np.zeros_like(ws)
  for atom,g in set(map(tuple,groups)):
   ids=np.flatnonzero((groups[:,0]==atom)&(groups[:,1]==g));ws[atom,g]=np.mean(np.asarray(ds)[ids]);uc[atom,g]=np.mean((np.asarray(dc)-native)[ids])
  np.savez_compressed(final,request_hash=request,spin=ws,charge=uc,groups=groups,
   spin_diagonal=ds,charge_diagonal=dc,native_charge_diagonal=native,dual_density=dual,dual_error=error)
 for label,cc in [('tensor_spin',np.zeros_like(uc)),('tensor_charge_spin',uc)]:
  ref=FrozenTensorResponseReference(shell.atomic_numbers,shell.orbital_shell,shell.coupling_hartree,
   shell.source_model+'/isotropic-onsite-tensor/'+request,.9,shell.charge_correction_hartree,shell.reference_shell_populations,ws,cc)
  path=out/(label+'.npz')
  if not path.exists():save_response_reference(path,ref)
 write(out/'calibration.json',dict(host=HOST,request_hash=request,spin_coefficients=ws,charge_coefficients=uc,
  scalar_energy_offset=0.,excited_reference_energies_used=False,lambda_exchange=.9,script_sha256=digest(__file__)))
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',type=int,default=0);p.add_argument('--groups',type=int,default=1);p.add_argument('--development',action='store_true');args=p.parse_args()
 if args.development:
  cases=[c for c in spin.plan() if c['name'] in ['nitroxyl','uracil']]
 else:cases=[json.loads(p.read_text()) for p in sorted((BASE/'prepared').glob('*/case.json'))]
 for i,c in enumerate(cases):
  if i%args.groups!=args.group:continue
  try:calibrate(c)
  except Exception as exc:write(ROOT/c['name']/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
