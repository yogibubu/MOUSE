"""Energy-label-free fixed-orbital multiplet and interaction diagnostic."""
from pathlib import Path
import sys,json,socket
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
sys.path.insert(0,str(ROOT))
from basis import *
with threadpool_limits():
 for item in json.loads((HERE/'cases.json').read_text()):
  cohort,name=item['cohort'],item['molecule'];out=HERE/'multiplets'/cohort/name;out.mkdir(parents=True,exist_ok=True)
  if (out/'result.json').exists():continue
  d=prepare_basis(cohort,name);ctx=d['ctx'];no=d['no'];n=len(d['cg']);occ=np.zeros((2,n));occ[:,:no]=1
  def energy(c,o):return float(ctx.fock_and_energy(np.array([(c*x)@c.T for x in o]))[1])
  eg=energy(d['cg'],occ);rows=[]
  for row in d['rows']:
   with np.load(d['source']/'coherent_states'/row['id']/'final/final.npz') as z:c=z['coefficients'].copy()
   mix=occ.copy();mix[0,0]=0;mix[0,no]=1;trip=np.array([np.maximum(*mix),np.minimum(*mix)])
   em,et,e0=energy(c,mix),energy(c,trip),energy(c,occ)
   hh,ll=c[:,0],c[:,no];hl=np.outer(hh,ll);hden=np.outer(hh,hh);lden=np.outer(ll,ll)
   k=float(np.einsum('pqrs,pq,rs',d['case']['eri'],hl,hl,optimize=True));klr=float(np.einsum('pqrs,pq,rs',d['case']['eri_lr'],hl,hl,optimize=True))
   jhh=float(np.einsum('pqrs,pq,rs',d['case']['eri'],hden,hden,optimize=True));jll=float(np.einsum('pqrs,pq,rs',d['case']['eri'],lden,lden,optimize=True));jhl=float(np.einsum('pqrs,pq,rs',d['case']['eri'],hden,lden,optimize=True))
   rows.append(dict(id=row['id'],roks_ev=(2*em-et-eg)*EV,triplet_ev=(et-eg)*EV,native_splitting_ev=2*(em-et)*EV,bare_exchange_ev=k*EV,CAM_exchange_ev=(.19*k+.46*klr)*EV,gx24_analogy_ev=(et-eg+1.36*k)*EV,closed_ground_on_excited_ev=(e0-eg)*EV,jhh_ev=jhh*EV,jll_ev=jll*EV,jhl_ev=jhl*EV))
  write(out/'result.json',rows);print(name,rows,flush=True)
