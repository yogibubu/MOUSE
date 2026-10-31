"""Restricted-ground real singlet/triplet orbital stability, independently checked.

H/4 = diag(e_a-e_i) + 4 K_ia,jb for half-symmetrized real transition densities.
Both spin channels rotate together (singlet) or oppositely (triplet). Direct
energy curvatures validate the normalization and analytic response. This is
an electronic stability test, not a nuclear Hessian of the AO-pair prototype.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,json,numpy as np
from scipy.linalg import eigh,expm
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,EV
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from full_response import VerticalReference
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/response_ablation';account=json.loads((ROOT/'decomposition.json').read_text());rows=[]
cases={c['name']:c for c in plan() if c['name'] in ['nitroxyl','uracil']}
cases.update({n:json.loads((BASE/'prepared_B'/n/'case.json').read_text()) for n in ['acetaldehyde','acetone','formamide']})
with threadpool_limits(limits=2,user_api='blas'):
 for name,case in cases.items():
  old,xyz=state_from(case['targets'][0]);s=old.reference_state
  calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
  ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=old.ground_energy_hartree);native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
  for label in ['native_control','spin_only','full']:
   previous=next(r for r in account['rows'] if r['molecule']==name and r['variant']==label)
   folder=(BASE/'results/full_response' if label=='full' else ROOT/label)/name
   with np.load(folder/'model.npz') as f:data={k:f[k].copy() for k in f.files};request=str(f['request_hash'])
   with np.load(previous['ground_checkpoint']) as f:cg=f['coefficients'];ge=float(f['energy_hartree'])
   ref=VerticalReference(s,native.kernel.partition,data,request);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
   p=np.array([(c*o)@c.T for c,o in zip(cg,[s.alpha_occupations,s.beta_occupations])]);assert np.max(abs(p[0]-p[1]))<1e-8
   fs,e0=ctx.fock_and_energy(p);eps,c=eigh(fs[0],s.overlap)
   state=replace(s,coefficients=c,fock=fs[0],orbital_energies_hartree=eps);charge=ctx.density_response_kernel(state)
   occ=np.flatnonzero(s.alpha_occupations==1);virt=np.flatnonzero(s.alpha_occupations==0);ii=np.repeat(occ,len(virt));aa=np.tile(virt,len(occ));ci=c[:,ii];ca=c[:,aa]
   for channel in ['singlet','triplet']:
    cols=[]
    for i,a in zip(ii,aa):
     t=.5*(np.outer(c[:,i],c[:,a])+np.outer(c[:,a],c[:,i]))
     f=charge(t) if channel=='singlet' else ctx.polarization.evaluate(np.array([.5*t,-.5*t]))[0][0]
     cols.append(np.einsum('mi,mn,ni->i',ci,f,ca))
    h=np.diag(eps[aa]-eps[ii])+4*np.array(cols).T;assert np.max(abs(h-h.T))<1e-8
    values,vectors=eigh(h);v=vectors[:,0];k=np.zeros(s.overlap.shape);k[aa,ii]=v;k[ii,aa]=-v
    energies=[];step=1e-4
    for sign in [-1,1]:
     cp=c@expm(sign*step*k);cm=cp if channel=='singlet' else c@expm(-sign*step*k)
     dp=np.array([(cp*s.alpha_occupations)@cp.T,(cm*s.beta_occupations)@cm.T]);energies.append(ctx.fock_and_energy(dp)[1])
    fd=(sum(energies)-2*e0)/step**2;analytic=4*values[0];err=abs(fd-analytic)
    assert err<3e-5
    rows.append(dict(molecule=name,variant=label,channel=channel,lowest_hessian_over4_hartree=float(values[0]),negative_modes=int(np.sum(values<-1e-8)),finite_curvature=fd,analytic_curvature=float(analytic),error_hartree=err,step=step))
write(ROOT/'ground_stability.json',dict(host=HOST,script_sha256=digest(__file__),scope='real electronic orbital stability, not nuclear Hessian',rows=rows,passed=True))
print('STABILITY',len(rows),'max error',max(r['error_hartree'] for r in rows),flush=True)
