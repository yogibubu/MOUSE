"""Diagnostic TDA of the analytic GFN2 charge functional; no production changes.

A singlet = orbital gaps + 2 (ia | dF/dP | jb). Symmetric half transition
matrices supply the real symmetric density kernel. The shell spin term cancels
between alpha-alpha and alpha-beta singlet response channels. This diagnostic
is not the empirical Grimme/Ohno screened-exchange sTDA model.
"""
from remote_bootstrap import HOST
from benchmark import BASE,write,EV,characterize
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf_derivatives import load_derivative_point
import numpy as np,json,time
from scipy.linalg import expm
from types import SimpleNamespace
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
name='quest_s0';out=BASE/'results'/name;meta=json.loads((out/'summary.json').read_text());xyz=np.array(meta['input']['coordinates_angstrom']);g=load_derivative_point(out/'ground.npz',request_hash=name,coordinates_angstrom=xyz);s=g.reference_state
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in meta['input']['atomic_numbers']],temperature_kelvin=0.)
kernel=calc.density_response_kernel(s);occupied=np.flatnonzero(s.alpha_occupations>.5);virtual=np.flatnonzero(s.alpha_occupations<.5);ii=np.repeat(occupied,len(virtual));aa=np.tile(virtual,len(occupied));ci=s.coefficients[:,ii];ca=s.coefficients[:,aa]
t=time.perf_counter();cols=[]
for i,a in zip(ii,aa):
 tr=np.outer(s.coefficients[:,i],s.coefficients[:,a]);f=kernel(.5*(tr+tr.T));cols.append(np.einsum('mi,mn,ni->i',ci,f,ca))
k=np.array(cols).T;a=np.diag(s.orbital_energies_hartree[aa]-s.orbital_energies_hartree[ii])+2*k
error=float(np.max(abs(a-a.T)));assert error<1e-10
values,vectors=np.linalg.eigh(.5*(a+a.T));p,n,_,_=characterize(s,meta['input']['base_atoms']);roots=[]
for root in range(10):
 w=vectors[:,root]**2
 roots.append(dict(root=root+1,energy_ev=float(values[root]*EV),hole_pi=float(w@p[ii]),particle_pi=float(w@p[aa]),hole_n=float(w@n[ii])))
write(BASE/'parent_response_diagnostic.json',dict(host=HOST,model='unparametrized analytic GFN2 charge-functional singlet TDA diagnostic',seconds=time.perf_counter()-t,matrix_asymmetry=error,roots=roots))
print(json.dumps(roots,indent=2))

ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=SimpleNamespace(converged=True,orbital_state=s))
p0=np.array([(s.coefficients*o)@s.coefficients.T for o in [s.alpha_occupations,s.beta_occupations]])
fd_errors=[]
for b in [0,len(ii)//2,len(ii)-1]:
 tr=np.outer(s.coefficients[:,ii[b]],s.coefficients[:,aa[b]]);tr=.5*(tr+tr.T);step=1e-4
 fm=ctx.fock_and_energy(p0-step*np.array([tr,tr]))[0][0];fp=ctx.fock_and_energy(p0+step*np.array([tr,tr]))[0][0]
 vals=np.einsum('mi,mn,ni->i',ci,(fp-fm)/(2*step),ca)
 fd_errors.append(float(np.max(abs(vals-2*k[:,b]))))
rng=np.random.default_rng(2718);v=rng.normal(size=len(ii));v/=np.linalg.norm(v);rotation=np.zeros((s.nmo,s.nmo));rotation[ii,aa]=v;rotation[aa,ii]=-v
step=.001;energies=[]
for sign in [-1,0,1]:
 c=s.coefficients@expm(sign*step*rotation);p=np.array([(c*o)@c.T for o in [s.alpha_occupations,s.beta_occupations]]);energies.append(ctx.fock_and_energy(p)[1])
fd=(energies[2]-2*energies[1]+energies[0])/step**2
exact=float(v@(4*np.diag(s.orbital_energies_hartree[aa]-s.orbital_energies_hartree[ii])+16*k)@v)
validation=dict(host=HOST,maximum_fock_density_derivative_error=max(fd_errors),finite_rotation_curvature=fd,analytic_rotation_curvature=exact,curvature_error=abs(fd-exact))
assert max(fd_errors)<1e-8 and abs(fd-exact)<1e-5,validation
write(BASE/'parent_response_validation.json',validation)
print(json.dumps(validation,indent=2))
