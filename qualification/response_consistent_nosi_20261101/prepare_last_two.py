"""Export archived native inputs and fixed pi-pi target anchors, no target energies."""
from pathlib import Path
import socket,json,numpy as np
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');PUB=Path('/home/enzo/MOUSE-publication-validation-20261030');D=Path('/home/enzo/MOUSE-state-following-20261031/reduced_validation');DATA=ROOT/'data/last_two';rows=[]
def link(a,b):
 assert a.exists(),a
 b.parent.mkdir(parents=True,exist_ok=True)
 if not b.exists():b.symlink_to(a.resolve())
for row in json.loads((ROOT/'last_two_protocol.json').read_text())['cases']:
 name=row['molecule']
 if name=='Furan':
  metadata=json.loads((D/'prepared/Furan/case.json').read_text());ground=D/'prepared/Furan/ground.npz';full=PUB/'results/full_response/Furan';q=D/'models/Furan/model.npz';reference=Path(metadata['reference']);tda=reference/'backend_tda.npz'
 else:
  ground=Path('/home/enzo/MOUSE-deltascf-validation-20261029/uracil_uridine_20261029/results/quest_s0/ground.npz');full=OLD/'results/full_response/uracil';q=OLD/'results/qualified_spectral_response/uracil/model.npz';reference=Path('/home/enzo/MATRIX-runs/tda-uracil-application-20261022-v1/docs/reports/tda_uracil_application_20261022/results/pbe0_avdz');tda=reference/'selected_full.npz'
 src=DATA/'predictions'/name;ref=DATA/'references'/name
 link(ground,src/'native_ground.npz');link(ground,src/'grounds/0/stationary.npz')
 for file in ['embedding.npz','projected_eri.npz','model.npz']:link(full/file,src/'full_response/anchor'/file)
 link(q,src/'qualified_spectral_response/anchor/model.npz');link(reference/'backend.chk',ref/'backend.chk');link(tda,ref/(row['id']+'.npz'))
 with np.load(reference/'reference.npz') as z:data={k:z[k].copy() for k in z.files}
 if 'orbital_energies' not in data:data['orbital_energies']=data['orbital_energies_hartree']
 np.savez_compressed(ref/'reference.npz',**data)
 result=json.loads((full/row['legacy_state']/'result.json').read_text())
 with np.load(result['checkpoint']) as z:c=z['result_common_coefficients'].copy();occ=z['result_occupations'].copy()
 if c.ndim==3:c=c[0]
 no=int(np.sum(occ[0]));opened=np.flatnonzero(occ[0]!=occ[1]);hole=int(opened[opened<no][0]);particle=int(opened[opened>=no][0]);order=[hole]+[i for i in range(no) if i!=hole]+[particle]+[i for i in range(no,len(c)) if i!=particle]
 np.savez_compressed(src/'retained_F_seed.npz',coefficients=c[:,order])
 rows.append(dict(id=row['id'],molecule=name,family='pi_pi',symmetry=row['symmetry'],ordinal_same_symmetry=row['reference_root']))
(DATA/'manifest.json').write_text(json.dumps(dict(states=rows,target_energy_labels_included=False),indent=2)+'\n')
