"""Independent C predictions for one frozen profile, with archived SCF reuse."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,traceback,numpy as np,argparse
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,key,digest,state_from
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from pyscf import gto,dft,scf,lib
import prepare_C as prepare
import full_response as full
import full_fast
import qualified_spectral_response as profile
lib.num_threads(4)
PROTOCOL=json.loads((BASE/'qualified_spectral_holdout_C_protocol.json').read_text())
def reference(case):
 out=Path(case['reference']);out.mkdir(parents=True,exist_ok=True)
 request=key(dict(protocol=digest(BASE/'qualified_spectral_holdout_C_protocol.json'),geometry=digest(case['geometry']),model='PBE0/aug-cc-pVDZ/cart/grid3'))
 marker=out/'request.json'
 if marker.exists():assert json.loads(marker.read_text())['request_hash']==request
 else:write(marker,dict(host=HOST,request_hash=request))
 if (out/'reference.npz').exists() and (out/'backend_tda.npz').exists():return
 text=Path(case['geometry']).read_text().splitlines();nat=int(text[0]);mol=gto.M(atom='\n'.join(text[2:2+nat]),basis='aug-cc-pvdz',unit='Angstrom',cart=True,verbose=0)
 mol.max_memory=40000;mf=dft.RKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=40000;mf.small_rho_cutoff=0;mf.grids.level=3
 mf.conv_tol=1e-10;mf.conv_tol_grad=3e-8;mf.max_cycle=120;mf.chkfile=str(out/'backend.chk')
 mf=scf.addons.remove_linear_dep_(mf,threshold=1e-7,lindep=1e-9)
 cached=out/'reference.npz'
 if cached.exists():
  with np.load(cached) as f:mf.mo_coeff=f['coefficients'];mf.mo_occ=f['occupations'];mf.mo_energy=f['orbital_energies_hartree'];mf.e_tot=float(f['energy_hartree'])
  mf.converged=True
 else:
  dm=None
  if (out/'backend.chk').exists():
   previous=scf.chkfile.load(str(out/'backend.chk'),'scf')
   if previous and 'mo_coeff' in previous:dm=(previous['mo_coeff']*previous['mo_occ'])@previous['mo_coeff'].T
  start=time.perf_counter();mf.kernel(dm0=dm)
  if not mf.converged:
   density=mf.make_rdm1();mf=mf.newton();mf.conv_tol=1e-10;mf.conv_tol_grad=3e-8;mf.max_cycle=80;mf.kernel(dm0=density)
  assert mf.converged
  s=mf.get_ovlp();c=mf.mo_coeff;fock=mf.get_fock(dm=mf.make_rdm1())
  assert np.max(abs(c.T@s@c-np.eye(c.shape[1])))<2e-7
  grad=float(np.max(abs(mf.get_grad(c,mf.mo_occ,fock))));assert grad<1e-7
  np.savez_compressed(cached,atomic_numbers=mol.atom_charges(),coordinates_bohr=mol.atom_coords(),overlap=s,
   coefficients=c,occupations=mf.mo_occ,orbital_energies_hartree=mf.mo_energy,fock=fock,energy_hartree=mf.e_tot)
  write(out/'scf.json',dict(host=HOST,request_hash=request,converged=True,energy_hartree=mf.e_tot,orbital_residual=grad,seconds=time.perf_counter()-start))
 if not (out/'backend_tda.npz').exists():
  td=mf.TDA();td.nstates=1;td.conv_tol=1e-8;td.max_cycle=160
  previous=scf.chkfile.load(str(out/'backend.chk'),'tddft');recovered=False;guess=None
  if previous and 'e' in previous and 'xy' in previous:
   energies=np.asarray(previous['e']);vectors=np.array([np.asarray(x).ravel()/np.linalg.norm(x) for x,y in previous['xy']])
   np.savez_compressed(out/'previous_tda_roots.npz',energies_hartree=energies,amplitudes=vectors)
   vind,_=td.gen_vind();guess=vectors[:1];residual=float(np.max(abs(vind(guess)[0]-energies[0]*guess[0])))
   if residual<1e-8:
    np.savez_compressed(out/'backend_tda.npz',amplitudes=guess,energies_hartree=energies[:1]);recovered=True
    write(out/'tda_recovery.json',dict(host=HOST,reference_root=1,residual=residual,recovered_without_new_solver=True,other_roots_preserved=True))
  if not recovered:
   td.kernel(x0=guess);assert bool(td.converged[0])
   amplitudes=np.array([x.ravel()/np.linalg.norm(x) for x,y in td.xy])
   np.savez_compressed(out/'backend_tda.npz',amplitudes=amplitudes,energies_hartree=td.e)



def run(case):
 name=case['name'];reference(case);prepare.prepare(case)
 prepared=json.loads((BASE/'prepared_C'/name/'case.json').read_text());out=BASE/'results/holdout_C'/name;out.mkdir(parents=True,exist_ok=True)
 if len(prepared['targets'])!=1:
  write(out/'result.json',dict(host=HOST,molecule=name,qualified=False,exclusions=prepared['excluded']));return
 t=prepared['targets'][0];old,xyz=state_from(t);s=old.reference_state
 full.ROOT=BASE/'results/full_response';full.calibrate=full_fast.calibrate;full.run(prepared)
 full.ROOT=BASE/'results/qualified_spectral_response';full.calibrate=profile.calibrate;full.run(prepared)
 row=json.loads((full.ROOT/name/'n_pi/result.json').read_text());baseline=json.loads((BASE/'results/full_response'/name/'n_pi/result.json').read_text())
 mapping=json.loads((full.ROOT/name/'mapping.json').read_text());goodmap=bool(mapping['qualified'][t['hole']] and mapping['qualified'][t['particle']]);transition=None
 qualified=bool(row.get('qualified') and goodmap)
 if qualified:
  r=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  with np.load(t['transition_file']) as f:a=f['amplitude'];occ=f['occupied'];virt=f['virtual']
  h=s.coefficients[:,occ].T@s.overlap@r.common_coefficients[:,t['hole']];p=s.coefficients[:,virt].T@s.overlap@r.common_coefficients[:,t['particle']];rank=np.outer(h,p)
  transition=float(abs(np.sum(a*rank))/np.linalg.norm(rank));qualified=bool(transition>=.7)
 write(out/'result.json',dict(host=HOST,molecule=name,qualified=qualified,target_orbital_mapping_qualified=goodmap,reference_transition_overlap=transition,
  spectral_response=row,raw_full_response=baseline,native_roks_ev=t['native_roks_ev'],tbe_ev=t['tbe_ev'],protocol_sha256=digest(BASE/'qualified_spectral_holdout_C_protocol.json'),
  blind_model_selection=True,energy_offset=0.,scope='unused three-molecule npi holdout; profile unchanged'))
 print('C COMPLETE',name,qualified,row.get('excitation_ev'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);args=p.parse_args();case=PROTOCOL['cases'][args.index]
 try:run(case)
 except Exception as exc:
  out=BASE/'results/holdout_C'/case['name'];out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));raise
