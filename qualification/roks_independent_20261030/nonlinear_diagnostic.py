"""Nonstationary exact-DFT checks in the fixed GFN2 valence subspace.

These energies are diagnostics on the already optimized quadratic-model
orbitals; they are not an energy-offset prescription or relaxed DFT predictions.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,numpy as np,time
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,EV
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from pyscf import dft,scf,lib
lib.num_threads(8)
for case in [c for c in plan() if c['name'] in ['nitroxyl','uracil']]:
 out=BASE/'results/full_response'/case['name'];final=out/'nonlinear_diagnostic.json'
 if final.exists():continue
 r0,xyz=state_from(case['targets'][0]);s=r0.reference_state;folder=Path(case['reference'])
 with np.load(folder/'reference.npz') as f:cr=f['coefficients'];occ=f['occupations']
 with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz') as f:lift=f['embedding']
 with np.load(out/'tda.npz') as f:cg=f['ground_coefficients'];model_ground=float(f['ground_energy'])
 # CHNO only, all have a 1s core on each non-hydrogen atom.
 ncore=int(np.count_nonzero(s.atomic_numbers>1));core=cr[:,np.flatnonzero(occ==2)[:ncore]];pcore=core@core.T
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=40000;mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=40000
 mf.small_rho_cutoff=0;mf.grids.level=3;mf.grids.build(with_non0tab=True);mf._eri=mol.intor('int2e',aosym='s8');h=mf.get_hcore()
 def evaluate(c,o):
  p=np.array([(x*z)@x.T for x,z in zip(c,o)]);dm=np.array([pcore+lift@x@lift.T for x in p])
  v=mf.get_veff(mol,dm);energy=float(mf.energy_tot(dm=dm,h1e=h,vhf=v))
  fock=np.array([lift.T@(h+x)@lift for x in v]);return energy,fock
 eg,fg=evaluate(cg,np.array([s.alpha_occupations,s.beta_occupations]));ground_density=np.array([(c*o)@c.T for c,o in zip(cg,[s.alpha_occupations,s.beta_occupations])])
 gres=max(float(np.max(abs(f@p@s.overlap-s.overlap@p@f))) for f,p in zip(fg,ground_density));rows=[]
 for target in case['targets']:
  rp=out/target['id']/'result.json'
  if not rp.exists():continue
  row=json.loads(rp.read_text())
  if not row['qualified']:continue
  r=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  em,fm=evaluate(np.array([r.common_coefficients]*2),r.occupations);et,ft=evaluate(np.array([r.common_coefficients]*2),r.triplet_occupations)
  fmo=np.array([r.common_coefficients.T@f@r.common_coefficients for f in fm]);fto=np.array([r.common_coefficients.T@f@r.common_coefficients for f in ft])
  q=2*sum(f*o[None,:] for f,o in zip(fmo,r.occupations))-sum(f*o[None,:] for f,o in zip(fto,r.triplet_occupations))
  residual=float(np.max(abs(2*(q-q.T))))
  exact=(2*em-et-eg)*EV
  rows.append(dict(state=target['id'],quadratic_model_ev=row['excitation_ev'],exact_embedded_dft_on_same_orbitals_ev=exact,
   nonlinear_and_kernel_scaling_difference_ev=exact-row['excitation_ev'],exact_dft_orbital_residual=residual,
   tbe_ev=target['tbe_ev'],relaxed_exact_dft=False))
 report=dict(host=HOST,molecule=case['name'],ground_exact_dft_orbital_residual=gres,ground_model_energy_hartree=model_ground,
  ground_exact_embedded_dft_hartree=eg,rows=rows,
  scope='nonstationary diagnostic on frozen quadratic-model orbitals; includes XC nonlinearity and the 0.9-versus-1 HF-response difference; not a corrected prediction')
 write(final,report);print(report,flush=True)
