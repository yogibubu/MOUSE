"""Independent PySCF response check of the complete projected AO-pair tensor."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np,json,argparse
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,EV
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from full_response import VerticalReference
from pyscf import dft,scf,lib
from pyscf.scf import _response_functions
lib.num_threads(8)
p=argparse.ArgumentParser();p.add_argument('--molecule',default='nitroxyl');args=p.parse_args()
case=next(c for c in plan() if c['name']==args.molecule);r,xyz=state_from(case['targets'][0]);s=r.reference_state
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r.ground_energy_hartree);native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
out=BASE/'results/edft_rsh_consistent/calibration'/case['name']
with np.load(out/'model.npz') as f:data={k:f[k].copy() for k in f.files};request=str(f['request_hash'])
ref=VerticalReference(s,native.kernel.partition,data,request);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
folder=BASE/'results/edft_rsh_consistent/references'/case['name']
with np.load(out/'embedding.npz') as f:lift=f['embedding']
with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=f['occupations']
mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mf=dft.UKS(mol);mf.xc='RSH(1;-.625;.2)+.625*GGA_X_PBE+GGA_C_PBE';mf.verbose=0;mf.small_rho_cutoff=0
mf.grids.level=3;mf.grids.build(with_non0tab=True);mf.max_memory=40000;mf._eri=mol.intor('int2e',aosym='s8')
ncore=int(np.sum(s.atomic_numbers-np.asarray(calc._valence))/2);core=c[:,np.flatnonzero(occ==2)[:ncore]]
cref=np.column_stack([core,lift@s.coefficients]);oref=np.r_[np.ones(ncore),s.alpha_occupations]
respond=mf.gen_response(mo_coeff=np.array([cref]*2),mo_occ=np.array([oref]*2),hermi=1,with_j=True)
rng=np.random.default_rng(20261030);checks=[]
for k in range(3):
 d=rng.normal(size=s.overlap.shape);d=(d+d.T)/2;d/=np.linalg.norm(d);rho=lift@d@lift.T
 pc=np.array([.5*rho,.5*rho]);ps=np.array([.5*rho,-.5*rho])
 vc=respond(pc)+.1*(.375*mf.get_k(mol,pc,hermi=1)+.625*mf.get_k(mol,pc,hermi=1,omega=.2));vs=respond(ps)+.1*(.375*mf.get_k(mol,ps,hermi=1)+.625*mf.get_k(mol,ps,hermi=1,omega=.2))
 expected_charge=lift.T@(.5*(vc[0]+vc[1]))@lift;expected_spin=lift.T@(.5*(vs[0]-vs[1]))@lift
 actual_charge=ctx.charge_correction.response(d)+native.kernel(d)
 actual_spin=ctx.polarization.evaluate(np.array([.5*d,-.5*d]))[0][0]
 checks.append(dict(charge_error=float(np.max(abs(actual_charge-expected_charge))),spin_error=float(np.max(abs(actual_spin-expected_spin)))))
# First derivative of the complete variational energy, tested independently.
p0=np.array([(s.coefficients*o)@s.coefficients.T for o in [s.alpha_occupations,s.beta_occupations]])
dp=rng.normal(size=p0.shape);dp=(dp+dp.transpose(0,2,1))/2;dp/=np.linalg.norm(dp)
f,_=ctx.fock_and_energy(p0);h=1e-5;ep=ctx.fock_and_energy(p0+h*dp)[1];em=ctx.fock_and_energy(p0-h*dp)[1]
first=abs((ep-em)/(2*h)-np.sum(f*dp))
report=dict(host=HOST,molecule=case['name'],request_hash=request,checks=checks,energy_fock_direction_error=float(first),scope='independent full PySCF RSH UKS response at embedded native valence plus new RSH core, lambda .9 once')
report['passed']=bool(all(max(c.values())<1e-6 for c in checks) and first<2e-7)
write(out/'independent_audit.json',report);print(report);assert report['passed']
