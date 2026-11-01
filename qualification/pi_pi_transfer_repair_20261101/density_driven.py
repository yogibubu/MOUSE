"""Published ensemble density-driven ansatz as a variational matrix correction.

This is not GX24: the native F/Q diagonal functional is retained. xi=.32 is
external (Gould et al., arXiv:2406.18105v2); no local benchmark fit is performed.
The ground-to-CSF transition density yields a quadratic CI energy contribution,
so the correction is inserted before, not after, state diagonalization.
"""
from pathlib import Path
import sys,json,socket
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101');BASE=Path('/home/enzo/MOUSE-frontier-cisd-20261101')
sys.path.insert(0,str(ROOT))
from basis import *
from transition import csf_transition
from nosi_spectrum import spectrum
from screening import screened
from pauli import normal_ordered_one_body

def run(item):
 cohort,name=item['cohort'],item['molecule'];out=HERE/'density_driven'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 d=prepare_basis(cohort,name);no=d['no'];c=d['canonical'];s=d['case']['s']
 with np.load(ROOT/'selected_cid'/cohort/name/'checkpoint.npz') as z:keep=z['indices'].copy();h=z['matrix'].copy();metric=z['metric'].copy()
 basis=[d['basis'][i] for i in keep];ground=d['basis'][0][0][1]
 rho=np.array([csf_transition(ground,b) for b in basis]);rho=.5*(rho+rho.transpose(0,2,1))
 # U[n_0a] = 1/2 (rho_0a|rho_0a); H_dd=-2 xi U.
 interaction=np.einsum('apq,pqrs,brs->ab',rho,d['eri'],rho,optimize=True);dd=-.32*interaction
 variants={'bare':h}
 p=ROOT/'constrained_screening'/cohort/name/'normal_ordered.npz'
 if p.exists():
  with np.load(p) as z:variants['screened']=z['matrix'][np.ix_(keep,keep)].copy()
 else:
  holes=[];particles=[];co=c[:,:no];cv=c[:,no:]
  for row in d['rows']:
   with np.load(d['source']/'coherent_states'/row['id']/'final/final.npz') as z:cs=z['coefficients'].copy()
   holes.append(co.T@s.overlap@cs[:,0]);particles.append(cv.T@s.overlap@cs[:,no])
  op=out/'screening_operators.npz'
  if op.exists():
   with np.load(op) as z:hh=z['h'].copy();w=z['eri'].copy()
  else:
   fg,_=d['ctx'].fock_and_energy(np.array([co@co.T]*2));levels=np.diag(c.T@fg[0]@c);w,certificate=screened(d['eri'],d['sr']@c,levels,no,holes,particles);pg=2*(d['sr']@co)@(d['sr']@co).T;hh=normal_ordered_one_body(w,d['si']@fg[0]@d['si'],pg);np.savez_compressed(op,h=hh,eri=w);write(out/'screening.json',certificate)
  ham=Hamiltonian(hh,w);new=np.zeros_like(h)
  for i in range(len(basis)):
   for j in range(i+1):ov,value=ham.csf(basis[i],basis[j]);assert abs(ov-metric[i,j])<1e-8;new[i,j]=new[j,i]=value
  new-=new[0,0]*metric;resid=(np.diag(h)-np.diag(new))/np.diag(metric);new+=.5*metric*(resid[:,None]+resid[None,:]);new[-2:,-2:]=h[-2:,-2:];variants['screened']=new
 source=d['old'];report=json.loads((source/'result.json').read_text());rows=report['predictions']
 with np.load(source/'checkpoint.npz') as z:ref=np.column_stack([z['coefficients'][:,r['root']] for r in rows])
 pad=np.zeros((len(h),len(rows)));pad[:len(ref)]=ref;results=[]
 for label,matrix in variants.items():
  for corrected in [False,True]:
   new=matrix+dd if corrected else matrix;primary,e,v=spectrum(new,metric,pad,rows,True);variant=label+('_dd' if corrected else '')
   np.savez_compressed(out/(variant+'.npz'),matrix=new,metric=metric,energies=e,coefficients=v,indices=keep)
   results.append(dict(variant=variant,primary=primary));print(name,variant,primary['predictions'],flush=True)
 np.savez_compressed(out/'transition_operators.npz',rho=rho,dd=dd)
 write(out/'result.json',results)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--molecule');args=p.parse_args()
 with threadpool_limits():
  for item in json.loads((HERE/'cases.json').read_text()):
   if args.molecule is None or item['molecule']==args.molecule:run(item)
