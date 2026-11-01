"""Checkpointed one-time full DFT references, verified MAC-STUDIO only.

Uses the existing CAM/PBE0 functional and aug-cc-pVDZ orbital basis with
RI-JK def2-universal-jkfit auxiliary basis. Density fitting is an explicitly
recorded numerical preparation approximation, not an excitation-energy fit.
Source structures are fixed starting geometries, not claimed as optimized.
"""
from pathlib import Path
import socket,subprocess,argparse,time,json,os,hashlib
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
import numpy as np
from pyscf import gto,dft,lib
from threadpoolctl import threadpool_limits
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--input');p.add_argument('--threads',type=int,default=8);a=p.parse_args();input=Path(a.input) if a.input else HERE/'pigment_inputs'/(a.name+'.xyz');out=HERE/'pigment_dft_reference'/a.name;out.mkdir(parents=True,exist_ok=True);lines=input.read_text().splitlines();n=int(lines[0]);atoms='\n'.join(' '.join([r.split()[0].capitalize(),*r.split()[1:4]]) for r in lines[2:2+n]);mol=gto.M(atom=atoms,basis='aug-cc-pVDZ',unit='Angstrom',charge=0,spin=0,verbose=4,output=str(out/'molecule.log'),max_memory=64000)
summary=[]
with threadpool_limits(a.threads):
 lib.num_threads(a.threads)
 for label,xc in [('PBE0','PBE0'),('CAM','CAM-B3LYP')]:
  folder=out/label;folder.mkdir(exist_ok=True);done=folder/'reference.npz'
  if done.exists():summary.append(json.loads((folder/'certificate.json').read_text()));continue
  t=time.monotonic();mf=dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit');mf.with_df._cderi_to_save=str(folder/'df_integrals.h5');mf.xc=xc;mf.grids.level=3;mf.small_rho_cutoff=0.;mf.max_memory=64000;mf.conv_tol=1e-10;mf.conv_tol_grad=1e-7;mf.max_cycle=150;mf.chkfile=str(folder/'backend.chk');dm=None
  if (folder/'backend.chk').exists():dm=mf.init_guess_by_chkfile(str(folder/'backend.chk'))
  elif label=='CAM' and (out/'PBE0/backend.chk').exists():dm=mf.init_guess_by_chkfile(str(out/'PBE0/backend.chk'))
  def callback(env):
   data=dict(cycle=int(env['cycle']),energy_hartree=float(env['e_tot']),gradient_norm=float(env.get('norm_gorb',np.nan)),density_norm=float(env.get('norm_ddm',np.nan)),elapsed_seconds=time.monotonic()-t)
   (folder/'progress.json').write_text(json.dumps(data,indent=2)+'\n');print(a.name,label,data,flush=True)
  mf.callback=callback;mf.kernel(dm0=dm)
  if not mf.converged:
   last=mf.make_rdm1();mf=mf.newton();mf.max_cycle=80;mf.callback=callback;mf.kernel(dm0=last)
  assert mf.converged,(a.name,label,'reference SCF incomplete; checkpoint preserved')
  np.savez_compressed(done,coefficients=mf.mo_coeff,occupations=mf.mo_occ,orbital_energies=mf.mo_energy,overlap=mf.get_ovlp(),atomic_numbers=mol.atom_charges(),coordinates_bohr=mol.atom_coords())
  certificate=dict(molecule=a.name,host=socket.getfqdn(),xc=xc,basis='aug-cc-pVDZ',auxiliary_basis='def2-universal-jkfit',density_fitting=True,nao=mol.nao_nr(),electrons=mol.nelectron,energy_hartree=float(mf.e_tot),wall_seconds=time.monotonic()-t,geometry_sha256=hashlib.sha256(input.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(done.read_bytes()).hexdigest(),converged=True,excitation_energies_read=False,geometry_status='Fixed source starting geometry; not an optimized MOUSE geometry');(folder/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n');summary.append(certificate)
 (out/'completion.json').write_text(json.dumps(summary,indent=2)+'\n')
