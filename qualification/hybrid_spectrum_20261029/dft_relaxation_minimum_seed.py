"""Resume a difficult ROKS root using a Cs-preserving variational seed.

Only an initializer: the original full-space Newton equations still certify
the final state. No geometry symmetrization, basis truncation or fit.
"""
from dft_relaxation import Objective,OUT,HOST,EV,write,digest
import argparse,json,numpy as np
from scipy.linalg import expm,expm_frechet
from scipy.optimize import minimize
from pyscf import lib,symm

p=argparse.ArgumentParser();p.add_argument('--molecule',required=True);p.add_argument('--state',required=True);a=p.parse_args()
lib.num_threads(8);obj=Objective(a.molecule,a.state);folder=OUT/a.molecule/a.state;out=folder/'minimum_seed';out.mkdir(exist_ok=True)
original=folder/'latest.npz';original_hash=digest(original)
with np.load(original) as f:
 assert str(f['request_hash'])==obj.request
 c=f['coefficients'].copy()
obj.mol.symmetry=True;obj.mol.build(False,False)
assert obj.mol.groupname=='Cs'
labels=symm.label_orb_symm(obj.mol,obj.mol.irrep_name,obj.mol.symm_orb,obj.C0,s=obj.S,check=True)
transform=obj.C0.T@obj.S@c
off=labels[:,None]!=labels[None,:];leak=float(np.max(abs(transform[off])))
assert leak<1e-5
for label in set(labels):
 idx=np.flatnonzero(labels==label);u,_,vt=np.linalg.svd(transform[np.ix_(idx,idx)])
 transform[np.ix_(idx,idx)]=u@vt
transform[off]=0;c=obj.C0@transform
active=np.flatnonzero(labels[obj.i]==labels[obj.j]);i,j=obj.i[active],obj.j[active]
v=obj.evaluate(c);diag=np.zeros(obj.size)
for factor,occ,f in zip((2.,-1.),(obj.occ,obj.trip),v['f']):
 for o,f0 in zip(occ,f):
  d=np.diag(f0);diag+=2*factor*(d[obj.i]-d[obj.j])*(o[obj.j]-o[obj.i])
scale=np.sqrt(np.clip(abs(diag[active]),.05,200));cache={};history=[]

def fun(y):
 if 'y' in cache and np.array_equal(y,cache['y']):return cache['energy'],cache['gradient']
 k=np.zeros((obj.n,obj.n));k[i,j]=y/scale;k[j,i]=-y/scale;u=expm(k)
 val=obj.evaluate(c@u);g=np.zeros((obj.n,obj.n))
 g[i,j]=val['g'][active]/2;g[j,i]=-val['g'][active]/2
 chart=expm_frechet(k.T,u@g,compute_expm=False)
 grad=(chart[i,j]-chart[j,i])/scale
 cache.update(y=y.copy(),energy=val['energy'],gradient=grad,value=val)
 return val['energy'],grad

# Verify the exact exponential-chart gradient at a nonzero chart coordinate.
rng=np.random.default_rng(20261030);test=rng.normal(size=len(i));test*=.03/np.linalg.norm(test)
direction=rng.normal(size=len(i));direction/=np.linalg.norm(direction);h=1e-4
_,g=fun(test);ep,_=fun(test+h*direction);em,_=fun(test-h*direction)
error=abs((ep-em)/(2*h)-g@direction)
write(out/'chart_gradient_audit.json',dict(host=HOST,error_hartree=error,step=h,passed=error<2e-7))
assert error<2e-7

def callback(y):
 fun(y);val=cache['value'];score=obj.overlap(obj.C0,val['c'])
 record=dict(iteration=len(history),energy_hartree=val['energy'],excitation_ev=(val['energy']-obj.ground)*EV,
  seed_csf_overlap=score,full_max_residual=float(np.max(abs(val['g']))))
 history.append(record);write(out/'history.json',history);print('MINIMUM SEED',record,flush=True)
 if score>=.7:
  temp=out/'latest.tmp.npz';np.savez_compressed(temp,request_hash=obj.request,coefficients=val['c'],
   occupations=obj.occ,energy_hartree=val['energy'],gradient=val['g']);temp.replace(out/'latest.npz')

solved=minimize(fun,np.zeros(len(i)),method='L-BFGS-B',jac=True,callback=callback,
 options=dict(gtol=2e-8,ftol=2e-16,maxiter=150,maxls=16,maxcor=12))
fun(solved.x);val=cache['value'];score=obj.overlap(obj.C0,val['c'])
result=dict(host=HOST,request_hash=obj.request,source_checkpoint_sha256=original_hash,
 method='Cs-preserving L-BFGS initializer; full-space Newton certification follows',
 group=obj.mol.groupname,initial_wrong_irrep_leak=leak,rotations=len(i),
 seed_csf_overlap=score,full_max_residual=float(np.max(abs(val['g']))),
 success=bool(solved.success),message=str(solved.message),iterations=solved.nit,
 script_sha256=digest(__file__),actual_openmp_threads=lib.num_threads())
write(out/'result.json',result)
assert score>=.7 and np.max(abs(val['g']))<1e-4
# Retain the original saddle-search checkpoint and promote the improved seed.
backup=folder/'before_minimum_seed.npz'
if not backup.exists():backup.write_bytes(original.read_bytes())
tmp=folder/'latest.tmp.npz';np.savez_compressed(tmp,request_hash=obj.request,coefficients=val['c'],
 occupations=obj.occ,energy_hartree=val['energy'],gradient=val['g']);tmp.replace(original)
print('PROMOTED',result,flush=True)
