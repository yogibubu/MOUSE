"""Project completed response-consistent matrices into the frozen selected space."""
from pathlib import Path
import sys,json,socket,importlib.util
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101')
sys.path.insert(0,str(ROOT))
from nosi_spectrum import spectrum
import numpy as np
from threadpoolctl import threadpool_limits

def select(case,variant='pauli_common'):
 cohort,name=case['cohort'],case['molecule'];out=HERE/variant/cohort/name
 if (out/'result.json').exists():return
 parts=variant.split(':');src=ROOT/parts[0]/cohort/name;checkpoint=src/(parts[1]+'.npz' if len(parts)>1 else 'checkpoint.npz')
 if not checkpoint.exists():return
 with np.load(ROOT/'selected_cid'/cohort/name/'checkpoint.npz') as z:keep=z['indices'].copy()
 with np.load(checkpoint) as z:h=z['matrix'][np.ix_(keep,keep)].copy();s=z['metric'][np.ix_(keep,keep)].copy()
 old=NATIVE/'nosi_coherent'/cohort/name;rows=json.loads((old/'result.json').read_text())['predictions']
 with np.load(old/'checkpoint.npz') as z:reference=np.column_stack([z['coefficients'][:,r['root']] for r in rows])
 pad=np.zeros((len(keep),len(rows)));pad[:len(reference)]=reference
 primary,e,c=spectrum(h,s,pad,rows,True);control,fe,fc=spectrum(h,s,pad,rows,False)
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'checkpoint.npz',matrix=h,metric=s,energies=e,coefficients=c,full_energies=fe,full_coefficients=fc,indices=keep)
 (out/'result.json').write_text(json.dumps(dict(primary=primary,full_control=control,variant=variant,cohort=cohort,molecule=name),indent=2,default=lambda x:x.tolist())+'\n');print(variant,name,primary['predictions'],flush=True)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--variant',default='pauli_common');args=parser.parse_args()
 cases=json.loads((HERE/'cases.json').read_text())
 with threadpool_limits(2):
  if args.prepare:
   spec=importlib.util.spec_from_file_location('kernel_pauli_driver',ROOT/'run_pauli.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   for case in cases:module.run(case['cohort'],case['molecule']);select(case,args.variant)
  else:
   for case in cases:select(case,args.variant)
