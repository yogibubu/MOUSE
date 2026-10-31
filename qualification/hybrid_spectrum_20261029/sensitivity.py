from remote_bootstrap import HOST
from pathlib import Path
from dataclasses import replace
from scipy.optimize import linear_sum_assignment
import argparse,numpy as np,json
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';EV=27.211386245988
parser=argparse.ArgumentParser()
parser.add_argument('--reference-energy-scale',type=float,required=True)
args=parser.parse_args()
if not np.isfinite(args.reference_energy_scale) or args.reference_energy_scale<=0:
 parser.error('reference energy scale must be finite and positive')
RUN=OUT/f'scale_{args.reference_energy_scale:.12g}'
with np.load(OUT/'source_matrices.npz') as f:parent=f['parent'];empirical=f['empirical'];ii=f['occupied'];aa=f['virtual']
records=[]
for name in ['b3lyp_avdz','pbe0_avdz','pbe0_avtz']:
 with np.load(RUN/f'{name}_mapped_spectrum.npz') as f:
  assert float(f['reference_energy_scale'])==args.reference_energy_scale
  source=f['source_energies'];ref=f['reference_energies'];over=f['overlaps'];margins=f['margins']
 for threshold in [.7,.6,.5]:
  allowed=(over>=threshold)&(margins>=.05);ids=np.flatnonzero(allowed[ii]&allowed[aa]);shift=args.reference_energy_scale*(ref[aa[ids]]-ref[ii[ids]])-(source[aa[ids]]-source[ii[ids]])
  for kernel,a in [('parent',parent),('empirical',empirical)]:
   old,v0=np.linalg.eigh(a);sub=a[np.ix_(ids,ids)]+np.diag(shift);vals,v=np.linalg.eigh(sub);embedded=np.zeros((len(ii),len(vals)));embedded[ids]=v
   anchors=[]
   for h,p in [(20,21),(19,21)]:
    idx=int(np.flatnonzero((ii==h)&(aa==p))[0]);anchors.append(int(np.argmax(abs(v0[idx])**2)))
   score=abs(v0[:,anchors].T@embedded);r,c=linear_sum_assignment(-score**2)
   for x,y in zip(r,c):records.append(dict(reference=name,kernel=kernel,minimum_overlap=threshold,configurations=len(ids),state=['n_pi','pi_pi'][x],energy_ev=float(vals[y]*EV),state_overlap=float(score[x,y])))
(RUN/'sensitivity.json').write_text(json.dumps(dict(host=HOST,reference_energy_scale=args.reference_energy_scale,rows=records),indent=2)+'\n')
for row in records:
 if row['reference']=='b3lyp_avdz' and row['kernel']=='parent':print(row)
