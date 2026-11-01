"""Reflection-even valence-space augmentation of the original frontier CID.

One common frozen configuration space contains the original complete frontier
rank-two block in both parities and the expanded reflection-even block. No
state energy or state-family label enters this symmetry selection. A molecular
plane must be certified geometrically and all canonical MO parities resolved.
The interaction is the same prepared m=5 screened operator in every block.
"""
from pathlib import Path
import sys,json,socket,re
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-pi-pi-transfer-repair-20261101')
sys.path.insert(0,str(ROOT))
from basis import *
from nosi_spectrum import spectrum
from scipy.spatial.distance import cdist

def parity(d):
 s=d['case']['s'];xyz=s.coordinates_bohr;z=np.array(s.atomic_numbers);heavy=xyz[z>1];origin=heavy.mean(0);_,values,axes=np.linalg.svd(heavy-origin,full_matrices=True)
 if len(values)<2 or values[1]<1e-5:return None,dict(available=False,reason='no molecular plane')
 normal=axes[-1];r=np.eye(3)-2*np.outer(normal,normal);ref=(xyz-origin)@r.T+origin;distance=cdist(ref,xyz);distance[z[:,None]!=z[None,:]]=1e6;ii,jj=linear_sum_assignment(distance);error=float(np.max(distance[ii,jj]))
 if error>1e-4:return None,dict(available=False,reason='molecular reflection not certified',geometry_error_bohr=error)
 small,t=source_molecule(s);other=small.copy();other.set_geom_(ref,unit='Bohr');rotation=np.eye(small.nao_nr());loc=small.ao_loc_nr()
 for shell in range(small.nbas):
  l=small.bas_angular(shell)
  if l==1:rotation[loc[shell]:loc[shell+1],loc[shell]:loc[shell+1]]=r
  elif l!=0:raise ValueError('unsupported angular reflection')
 cross=gto.intor_cross('int1e_ovlp',small,other);op=t.T@cross@rotation@t;c=d['canonical'];representation=c.T@op@c;characters=np.diag(representation)
 if np.max(abs(abs(characters)-1))>1e-5:return None,dict(available=False,reason='unresolved canonical reflection parity',geometry_error_bohr=error,characters=characters)
 return np.sign(characters).astype(int),dict(available=True,geometry_error_bohr=error,normal=normal,mo_characters=characters,offdiagonal_error=float(np.max(abs(representation-np.diag(characters)))))

def run(item):
 cohort,name=item['cohort'],item['molecule'];out=HERE/'even_sector_5'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 src=HERE/'expanded_cid_fast_5'/cohort/name
 if not (src/'result.json').exists():return
 report=json.loads((src/'result.json').read_text());labels=report['labels'];d=prepare_basis(cohort,name);no=d['no'];nbase=1+2*len(d['rows']);signs,certificate=parity(d);keep=list(range(nbase))
 for i,label in enumerate(labels[nbase:],nbase):
  match=re.fullmatch(r'pair_h(\d+)_p(\d+)',label)
  if match:hi=int(match[1]);hj=hi;a=int(match[2]);b=a
  else:
   match=re.fullmatch(r'(?:open|four)_h(\d+)_(\d+)_p(\d+)_(\d+)(?:_[01])?',label)
   if not match:raise ValueError(label)
   hi,hj,a,b=map(int,match.groups())
  original=hi>=no-2 and hj>=no-2 and a<=no+1 and b<=no+1
  if original or (signs is not None and signs[hi]*signs[hj]*signs[a]*signs[b]>0):keep.append(i)
 with np.load(src/'checkpoint.npz') as z:h=z['matrix'][np.ix_(keep,keep)].copy();metric=z['metric'][np.ix_(keep,keep)].copy()
 old=d['old'];rows=json.loads((old/'result.json').read_text())['predictions']
 with np.load(old/'checkpoint.npz') as z:ref=np.column_stack([z['coefficients'][:,r['root']] for r in rows])
 pad=np.zeros((len(keep),len(rows)));pad[:len(ref)]=ref;primary,e,v=spectrum(h,metric,pad,rows,True)
 np.savez_compressed(out/'checkpoint.npz',matrix=h,metric=metric,energies=e,coefficients=v,indices=keep)
 write(out/'result.json',dict(primary=primary,labels=[labels[i] for i in keep],reflection=certificate,source_sha256=sha(__file__),parent_sha256=sha(src/'checkpoint.npz'),single_common_interaction=True,no_state_family_branch=True));print(name,len(keep),certificate['available'],primary['predictions'],flush=True)
if __name__=='__main__':
 with threadpool_limits():
  for item in json.loads((HERE/'cases.json').read_text()):run(item)
