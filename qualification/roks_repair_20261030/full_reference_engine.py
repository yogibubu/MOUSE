"""Oracle-only common-orbital PBE0 singlet relaxation diagnostic.

The ROKS objective is 2 E_UKS(Ms=0) - E_UKS(Ms=1) on ONE set of
orbitals. Full AO space, no frozen cores, no fitted scaling. Newton solves
orbital stationarity (including saddles), with analytic UKS response HVPs.
This orbital Hessian is not a nuclear Hessian of the hybrid model.
"""
from remote_bootstrap import HOST
from pathlib import Path
import argparse,json,time,traceback,hashlib
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator,gmres
from pyscf import dft,scf,lib
from pyscf.scf import _response_functions  # register gen_response
from frozen_relaxation import digest,write,key,EV
from matrix_apoc.deltascf import (DeterminantOrbitals,DeltaSCFState,
                                SpinAdaptedSinglet,singlet_csf_overlap)

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'dft_relaxation'
PROTOCOL=dict(schema='common-orbital-dft-relaxation-v1',host_required='oracle.sns.it',
 xc='PBE0',grid_level=3,small_rho_cutoff=0.,threads=2,max_memory_mb=20000,
 orbital_tolerance=1e-7,maximum_iterations=45,trust_radius=.35,
 minimum_seed_csf_overlap=.70,minimum_step_csf_overlap=.90,
 gmres_rtol=.08,gmres_restart=24,gmres_maxiter=2,
 orbital_space='full AO; all core and valence orbitals may relax',
 objective='2*UKS(mixed)-UKS(triplet), common orbitals',
 solver='analytic-response Newton-GMRES with residual line search',
 cases=['uracil','thymine'],states=['n_pi','pi_pi'],parameters_fitted=0)
lib.num_threads(PROTOCOL['threads'])

class Objective:
 def __init__(self,case,state):
  self.case=case;self.state=state
  source=ROOT/'frozen_relaxation'/case
  self.previous=json.loads((source/'result.json').read_text())
  self.row=next(r for r in self.previous['records'] if r['id']==state)
  plan=json.loads((ROOT/'frozen_relaxation/plan.json').read_text())
  folder=Path(next(c for c in plan['cases'] if c['name']==case)['reference'])
  self.request=key(dict(protocol=PROTOCOL,reference_sha256=digest(folder/'reference.npz'),
   checkpoint_sha256=digest(folder/'backend.chk'),state=state,
   hole=self.row['dft_hole'],particle=self.row['dft_particle']))
  ref=np.load(folder/'reference.npz');self.ref={k:ref[k] for k in ref.files};ref.close()
  self.C0=self.ref['coefficients'];self.S=self.ref['overlap'];self.n=self.C0.shape[1]
  self.occ=np.array([self.ref['occupations']/2]*2)
  self.occ[0,self.row['dft_hole']]=0;self.occ[0,self.row['dft_particle']]=1
  self.trip=np.array([np.maximum(*self.occ),np.minimum(*self.occ)])
  i,j=np.triu_indices(self.n,1);labels=np.vstack((self.occ,self.trip))
  active=np.any(labels[:,i]!=labels[:,j],axis=0);self.i=i[active];self.j=j[active];self.size=len(self.i)
  self.mol=scf.chkfile.load_mol(str(folder/'backend.chk'));self.mol.max_memory=PROTOCOL['max_memory_mb']
  assert np.max(abs(self.mol.intor('int1e_ovlp')-self.S))<1e-9
  self.mf=dft.UKS(self.mol);self.mf.xc=PROTOCOL['xc'];self.mf.verbose=0
  self.mf.max_memory=PROTOCOL['max_memory_mb'];self.mf.small_rho_cutoff=0
  self.mf.grids.level=PROTOCOL['grid_level'];self.mf.grids.build(with_non0tab=True)
  self.h=self.mf.get_hcore()
  # One shared exact in-core ERI tensor; no density-fitting approximation.
  self.mf._eri=self.mol.intor('int2e',aosym='s8')
  self.ground=self.previous['ground']['energy_hartree'];self.calls=0
 def K(self,x):
  k=np.zeros((self.n,self.n));k[self.i,self.j]=x;k[self.j,self.i]=-x;return k
 def rotate(self,c,x):return c@expm(self.K(x))
 def csf(self,c,energy=0.):
  w=DeterminantOrbitals(self.ref['atomic_numbers'],self.ref['coordinates_bohr'],self.S,
   (c,c),tuple(self.occ),np.zeros(len(self.ref['atomic_numbers']),dtype=int),
   charge=self.mol.charge,multiplicity=1,total_energy_hartree=energy,model='PBE0/aug-cc-pVDZ ROKS')
  return SpinAdaptedSinglet(1,DeltaSCFState(1,w),energy)
 def overlap(self,c,d):return singlet_csf_overlap(self.csf(c),self.csf(d),self.S)[0]
 def evaluate(self,c):
  self.calls+=1;fs=[];es=[];dens=[]
  for occ in (self.occ,self.trip):
   dm=np.array([(c*o)@c.T for o in occ]);v=self.mf.get_veff(self.mol,dm)
   es.append(float(self.mf.energy_tot(dm=dm,h1e=self.h,vhf=v)))
   fs.append(np.array([c.T@(self.h+f)@c for f in v]));dens.append(dm)
  q=sum(2*f*o[None,:] for f,o in zip(fs[0],self.occ))
  q-=sum(f*o[None,:] for f,o in zip(fs[1],self.trip))
  g=2*(q[self.i,self.j]-q[self.j,self.i])
  return dict(c=c,energy=2*es[0]-es[1],mixed_energy=es[0],triplet_energy=es[1],
              g=g,f=fs,dm=dens)
 def response(self,v):
  c=v['c'];responses=[]
  for occ in (self.occ,self.trip):
   responses.append(self.mf.gen_response(mo_coeff=np.array([c,c]),mo_occ=occ,hermi=1))
  calls=[0]
  def hv(x):
   calls[0]+=1;k=self.K(x);dq=np.zeros((self.n,self.n))
   for factor,occ,f,respond in zip((2.,-1.),(self.occ,self.trip),v['f'],responses):
    dd=np.array([c@(k*o[None,:]-o[:,None]*k)@c.T for o in occ])
    df=respond(dd)
    for o,f0,df0 in zip(occ,f,df):
     dfmo=c.T@df0@c+f0@k-k@f0
     dq+=factor*dfmo*o[None,:]
   return 2*(dq[self.i,self.j]-dq[self.j,self.i])
  diag=np.zeros(self.size)
  for factor,occ,f in zip((2.,-1.),(self.occ,self.trip),v['f']):
   for o,f0 in zip(occ,f):
    d=np.diag(f0);diag+=2*factor*(d[self.i]-d[self.j])*(o[self.j]-o[self.i])
  diag=np.where(abs(diag)<.02,np.where(diag<0,-.02,.02),diag)
  return hv,diag,calls
