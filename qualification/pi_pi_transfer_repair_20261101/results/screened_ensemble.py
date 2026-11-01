"""Uniform constrained-screened ensemble interaction diagnostic.

xi is the externally published .32, with no benchmark refit. Screening of
transition-density self energies is exploratory, not the published GX24 model.
"""
from pathlib import Path
import sys,json,socket
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-pi-pi-transfer-repair-20261101');NATIVE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-native-reference-20261101')
sys.path.insert(0,str(ROOT))
from nosi_spectrum import spectrum
import numpy as np
from threadpoolctl import threadpool_limits
with threadpool_limits(2):
 for item in json.loads((HERE/'cases.json').read_text()):
  cohort,name=item['cohort'],item['molecule'];src=HERE/'density_driven'/cohort/name;out=HERE/'screened_ensemble'/cohort/name;out.mkdir(parents=True,exist_ok=True)
  if not (src/'result.json').exists():continue
  if (out/'result.json').exists():continue
  op=src/'screening_operators.npz'
  if not op.exists():op=ROOT/'constrained_screening'/cohort/name/'operators.npz'
  with np.load(op) as z:w=z['eri'].copy()
  with np.load(src/'transition_operators.npz') as z:rho=z['rho'].copy()
  dd=-.32*np.einsum('apq,pqrs,brs->ab',rho,w,rho,optimize=True)
  with np.load(src/'screened.npz') as z:h=z['matrix'].copy();s=z['metric'].copy()
  old=NATIVE/'nosi_coherent'/cohort/name;rows=json.loads((old/'result.json').read_text())['predictions']
  with np.load(old/'checkpoint.npz') as z:ref=np.column_stack([z['coefficients'][:,r['root']] for r in rows])
  pad=np.zeros((len(h),len(rows)));pad[:len(ref)]=ref
  primary,e,v=spectrum(h+dd,s,pad,rows,True)
  np.savez_compressed(out/'checkpoint.npz',matrix=h+dd,metric=s,energies=e,coefficients=v,dd=dd)
  (out/'result.json').write_text(json.dumps(dict(primary=primary),indent=2,default=lambda x:x.tolist())+'\n');print(name,primary['predictions'],flush=True)
