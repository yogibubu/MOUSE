"""Resume retained native pigment density to strict functional stationarity."""
from pigment_compact_reference import HERE,load_state
from pigment_reference_embedding import write,sha
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from dataclasses import fields,replace
from pyscf import scf
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import numpy as np,argparse,time,socket

def run(name):
 folder=HERE/'pigment_preflight'/name;cert=folder/'stationary_native_certificate.json'
 if cert.exists():return
 s=load_state(name);calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O',12:'Mg'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);ctx=prepare_gfn2_deltascf(calc,s.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=s,allow_large=True);no=s.electron_count//2;n=len(s.overlap);p=2*s.coefficients[:,:no]@s.coefficients[:,:no].T;start=time.monotonic();core=calc._model(s.coordinates_bohr,relocate_basis=True)[4]
 mol=scf.chkfile.load_mol(str(HERE/'pigment_dft_reference'/name/'PBE0/backend.chk'));mf=scf.RHF(mol);mf.verbose=0;mf.chkfile=None;mf.get_hcore=lambda *x,**k:core;mf.get_ovlp=lambda *x,**k:s.overlap;mf.get_veff=lambda mol=None,dm=None,*x,**k:ctx.fock_and_energy(np.array([.5*dm,.5*dm]))[0][0]-core;mf.energy_elec=lambda dm=None,h1e=None,vhf=None:(ctx.fock_and_energy(np.array([.5*dm,.5*dm]))[1],0.);mf.energy_nuc=lambda *x,**k:0.
 def occ(e,c=None):
  o=np.zeros(n);o[np.argsort(e)[:no]]=2;return o
 mf.get_occ=occ;mf.max_cycle=150;mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.diis_space=12
 checkpoint=folder/'native_refinement_density.npz'
 if checkpoint.exists():
  with np.load(checkpoint) as z:p=z['density'].copy()
 history=[]
 def callback(env):
  np.savez_compressed(checkpoint,density=env['dm']);history.append(dict(cycle=env['cycle'],energy=float(env['e_tot']),gradient_norm=float(env['norm_gorb'])));write(folder/'native_refinement_history.json',history)
 mf.callback=callback;mf.kernel(dm0=p);assert mf.converged
 p=mf.make_rdm1();f,e=ctx.fock_and_energy(np.array([.5*p,.5*p]));levels,c=eigh(f[0],s.overlap);residual=float(np.max(abs((c.T@f[0]@c)[:no,no:])));fixed=2*c[:,:no]@c[:,:no].T;assert residual<1e-9 and np.max(abs(fixed-p))<1e-8
 occupation=np.zeros(n);occupation[:no]=1;s=replace(s,coefficients=c,fock=f[0],orbital_energies_hartree=levels,alpha_occupations=occupation,beta_occupations=occupation,total_energy_hartree=float(e));output=folder/'stationary_native.npz';np.savez_compressed(output,**{f.name:np.asarray(getattr(s,f.name)) for f in fields(s) if f.name!='basis'},**{'basis_'+f.name:np.asarray(getattr(s.basis,f.name)) for f in fields(s.basis)})
 write(cert,dict(host=socket.getfqdn(),source_checkpoint_sha256=sha(folder/'native_ground.npz'),stationary_checkpoint_sha256=sha(output),residual=residual,wall_seconds=time.monotonic()-start,history=history,initial_checkpoint_preserved=True,native_orbital_tolerance=1e-9,ground_SCC_driver_calls=0,status='Retained native density refined under the same complete large-system functional'))
 print(name,residual,e,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
 with threadpool_limits(2):run(a.name)
