"""Adapter for immutable old carbonyl/diazine inputs; no labels reach builders."""
from pathlib import Path
import sys
import anchor as anchor_module
from anchor import *

DATA=DEST/'extra_dataset';BASE=Path('/home/enzo/MOUSE-roks-independent-20261030')
PROTOCOL=json.loads((DEST/'npi_extension_protocol.json').read_text())

def link(source,target):
    source=Path(source);target=Path(target);assert source.exists(),source
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve())

def build():
    rows=[]
    for row in PROTOCOL['cases']:
        name=row['molecule'];prepared=BASE/row['prepared_group']/name
        old=json.loads((prepared/'case.json').read_text());target=next(t for t in old['targets'] if t['id']=='n_pi')
        source=DATA/'extra_npi/predictions'/name;reference=DATA/'extra_npi/references'/name;reference.mkdir(parents=True,exist_ok=True)
        link(prepared/'ground.npz',source/'native_ground.npz')
        link(prepared/'ground.npz',source/'grounds/0/stationary.npz')
        for filename in ['embedding.npz','projected_eri.npz','model.npz']:
            link(BASE/'results/full_response'/name/filename,source/'full_response/anchor'/filename)
        link(BASE/'results/qualified_spectral_response'/name/'model.npz',source/'qualified_spectral_response/anchor/model.npz')
        ref=Path(old['reference']);link(ref/'backend.chk',reference/'backend.chk')
        if not (reference/'reference.npz').exists():
            with np.load(ref/'reference.npz') as z:data={k:z[k].copy() for k in z.files}
            if 'orbital_energies' not in data:data['orbital_energies']=data['orbital_energies_hartree']
            np.savez_compressed(reference/'reference.npz',**data)
        link(ref/'backend_tda.npz',reference/(row['id']+'.npz'))
        # Convert the retained legacy promotion into the fixed h=0,p=no chart.
        seed=source/'retained_F_seed.npz'
        if not seed.exists():
            result=json.loads((BASE/'results/full_response'/name/'n_pi/result.json').read_text())
            with np.load(result['checkpoint']) as z:
                c=z['result_common_coefficients'].copy();occ=z['result_occupations'].copy()
                if c.ndim==3:c=c[0]
            no=int(np.sum(occ[0]));opened=np.flatnonzero(occ[0]!=occ[1]);h=int(opened[opened<no][0]);p=int(opened[opened>=no][0])
            order=[h]+[i for i in range(no) if i!=h]+[p]+[i for i in range(no,len(c)) if i!=p]
            np.savez_compressed(seed,coefficients=c[:,order])
        rows.append(dict(id=row['id'],molecule=name,family='n_pi',symmetry=target['symmetry'],ordinal_same_symmetry=target['reference_root']))
    write(DATA/'extra_npi/manifest.json',dict(states=rows,energy_labels_included=False,protocol_sha256=sha(DEST/'npi_extension_protocol.json')))

anchor_module.OUT=DATA
original_prepare=anchor_module.prepare
def prepare_extra(cohort,name):
    assert cohort=='extra_npi'
    result=original_prepare(cohort,name)
    result['retained_seed']=result['src']/'retained_F_seed.npz'
    return result
anchor_module.prepare=prepare_extra

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        build()
        from calibrate import calibrate
        from full_reference import full_reference
        from heldout_predict import predict_molecule
        from nosi_coherent import run as nosi
        from nosi_projected import projected
        # All DFT stages precede the guards in state-only preparation.
        for row in PROTOCOL['cases']:
            name=row['molecule'];anchor_module.anchor('extra_npi',name);calibrate('extra_npi',name);full_reference('extra_npi',name)
        for row in PROTOCOL['cases']:
            name=row['molecule'];predict_molecule('extra_npi',name);nosi('extra_npi',name);projected('extra_npi',name)
