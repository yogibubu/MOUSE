"""Compare parameter-free GFN2 screened wavefunction couplings at fixed NTOs.
All cases are exposed development data; no model is adopted from this test.
"""
from parent_geometry_model import V,D,P
import socket,os,sys,json,hashlib,time,fcntl
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
EV=27.211386245988;OUT=V/'pi_pi_repair'
with threadpool_limits(limits=2):
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 selected=[('Furan','B022'),('Cyclopentadiene','B016'),('Phthalimide','B024'),('Adenine','B014'),('Bimane_Syn','B015'),('Aza-naphthalene','B011'),('Phthalimide','B005')]
 rows=[]
 for name,sid in selected:
  gp=D/'prepared'/name/'ground.npz'
  with np.load(gp) as a:xyz=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
  s=load_derivative_point(gp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False).reference_state
  calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
  # Freeze the physical native-S0 charge kernel, not the geometry-only core seed.
  kernel=calc.density_response_kernel(s);folder=OUT/'native_screened_models'/name;folder.mkdir(parents=True,exist_ok=True);cache=folder/'model.npz'
  if cache.exists():
   with np.load(cache) as a:variants={k:a[k].copy() for k in ['monopole','full_kernel']}
  else:
   n=len(s.overlap);shell=np.asarray(kernel.partition.orbital_shell);selector=np.eye(kernel.partition.nshells)[shell].T
   q=.5*s.overlap[None]*(selector[:,:,None]+selector[:,None,:]);gamma=np.asarray(kernel.shell_interaction)
   mono=np.einsum('aij,ab,bkl->ijkl',q,gamma,q,optimize=True);full=np.empty((n,n,n,n))
   for i in range(n):
    for j in range(i+1):
     dp=np.zeros((n,n));dp[i,j]=.5;dp[j,i]+=.5;value=kernel(dp);full[:,:,i,j]=full[:,:,j,i]=value
   symmetry=float(np.max(abs(full-full.transpose(2,3,0,1))));assert symmetry<1e-8
   full=.5*(full+full.transpose(2,3,0,1));variants=dict(monopole=mono,full_kernel=full);np.savez_compressed(cache,**variants)
  case=json.loads((D/'prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
  with np.load(target['transition_file']) as a:amp=a['amplitude'].copy();oi=a['occupied'].copy();vi=a['virtual'].copy()
  no=len(oi);rank=int(np.searchsorted(np.cumsum(np.linalg.svd(amp/np.linalg.norm(amp),compute_uv=False)**2),.95)+1);rank=min(rank,4);profiles=[]
  for index,path in enumerate([P/'results/full_response'/name/'model.npz',D/'models'/name/'model.npz']):
   with np.load(path) as a:data={k:a[k].copy() for k in ['operators','charge','spin','q0','linear']}
   ref=FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,data,hashlib.sha256(path.read_bytes()).hexdigest());ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref)
   with np.load(D/'tda'/name/str(index)/'spectrum.npz') as a:c=a['coefficients'].copy()
   projected=c[:,:no].T@s.overlap@s.coefficients[:,oi]@amp@s.coefficients[:,vi].T@s.overlap@c[:,no:];projected/=np.linalg.norm(projected);u,sv,vh=np.linalg.svd(projected,full_matrices=True);cn=np.column_stack([c[:,:no]@u,c[:,no:]@vh.T]);co=cn[:,:rank];cv=cn[:,no:no+rank]
   o=np.array([s.alpha_occupations,s.beta_occupations]);_,e0=ctx.fock_and_energy(np.array([(c*x)@c.T for x in o]));diagonal=[]
   for i in range(rank):
    occ=o.copy();occ[0,i]=0;occ[0,no+i]=1;triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
    _,em=ctx.fock_and_energy(np.array([(cn*x)@cn.T for x in occ]));_,et=ctx.fock_and_energy(np.array([(cn*x)@cn.T for x in triplet]));diagonal.append(2*em-et-e0)
   results=[]
   for label,eri in variants.items():
    j=np.einsum('mnpq,mk,nk,pl,ql->kl',eri,co,cv,co,cv,optimize=True);k=np.einsum('mnpq,mk,nl,pk,ql->kl',eri,co,co,cv,cv,optimize=True);h=2*j-k;np.fill_diagonal(h,diagonal);es,z=eigh(h);scores=abs(sv[:rank]@z);root=int(np.argmax(scores))
    results.append(dict(variant=label,excitation_ev=float(es[root]*EV),ci=z[:,root].tolist(),matrix_ev=(h*EV).tolist()))
   profiles.append(dict(profile=index,rank=rank,variants=results))
  row=dict(molecule=name,state=sid,profiles=profiles);rows.append(row);print(name,[(p['profile'],[(a['variant'],round(a['excitation_ev'],4)) for a in p['variants']]) for p in profiles],flush=True)
 (OUT/'native_screening_diagnostic.json').write_text(json.dumps(dict(host=socket.getfqdn(),rows=rows,scope='fixed-orbital diagnostic only; no fitted screening factor'),indent=2)+'\n')
