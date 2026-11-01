"""Copy immutable legacy molecular inputs; never export target energies."""
from pathlib import Path
import socket,json,hashlib,sys
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
ROOT=Path('/home/enzo/MOUSE-frontier-cisd-20261101');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');DATA=ROOT/'data';rows=[]
def link(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 if not b.exists():b.symlink_to(a.resolve())
for row in json.loads((ROOT/'protocol.json').read_text())['prospective_cases']:
 name=row['molecule'];prepared=OLD/row['prepared_group']/name;metadata=json.loads((prepared/'case.json').read_text());target=next(t for t in metadata['targets'] if t['id']=='n_pi')
 src=DATA/'prospective_cisd/predictions'/name;ref=DATA/'prospective_cisd/references'/name;ref.mkdir(parents=True,exist_ok=True)
 link(prepared/'ground.npz',src/'native_ground.npz');link(prepared/'ground.npz',src/'grounds/0/stationary.npz')
 for filename in ['embedding.npz','projected_eri.npz','model.npz']:link(OLD/'results/full_response'/name/filename,src/'full_response/anchor'/filename)
 link(OLD/'results/qualified_spectral_response'/name/'model.npz',src/'qualified_spectral_response/anchor/model.npz')
 oldref=Path(metadata['reference']);link(oldref/'backend.chk',ref/'backend.chk');link(oldref/'backend_tda.npz',ref/(row['id']+'.npz'))
 with np.load(oldref/'reference.npz') as z:data={k:z[k].copy() for k in z.files}
 if 'orbital_energies' not in data:data['orbital_energies']=data['orbital_energies_hartree']
 np.savez_compressed(ref/'reference.npz',**data)
 result=json.loads((OLD/'results/full_response'/name/'n_pi/result.json').read_text())
 with np.load(result['checkpoint']) as z:c=z['result_common_coefficients'].copy();occ=z['result_occupations'].copy()
 if c.ndim==3:c=c[0]
 no=int(np.sum(occ[0]));opened=np.flatnonzero(occ[0]!=occ[1]);hole=int(opened[opened<no][0]);particle=int(opened[opened>=no][0]);order=[hole]+[i for i in range(no) if i!=hole]+[particle]+[i for i in range(no,len(c)) if i!=particle]
 np.savez_compressed(src/'retained_F_seed.npz',coefficients=c[:,order])
 rows.append(dict(id=row['id'],molecule=name,family='n_pi',symmetry=target['symmetry'],ordinal_same_symmetry=target['reference_root']))
(DATA/'prospective_cisd/manifest.json').write_text(json.dumps(dict(states=rows,energy_labels_included=False),indent=2))
