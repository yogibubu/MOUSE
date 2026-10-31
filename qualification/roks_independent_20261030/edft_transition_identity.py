"""Fixed DFT transition anchors for cross-functional orbital continuation.

Determinant overlap with the original native canonical frame is a diagnostic,
not a cross-functional state label. Eligibility requires overlap with the
previously qualified excited CSF AND the unchanged DFT transition anchor.
"""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import numpy as np
from pyscf import gto,scf
from extent_support import source_molecule

def anchor(case,target,state,out,write,digest):
    if target.get('transition_file'):
        file=Path(target['transition_file'])
        with np.load(file) as f:a=f['amplitude'];occupied=f['occupied'];virtual=f['virtual']
        origin=dict(kind='existing qualified DFT transition projected on GFN2',source=str(file),sha256=digest(file),root=target.get('reference_root'))
    else:
        if case['name'] not in ['nitroxyl','uracil']:raise ValueError('no frozen DFT transition root for this case')
        root=2 if case['name']=='uracil' and target['id']=='pi_pi' else 1
        folder=Path(case['reference']);file=next(folder/name for name in ['backend_tda.npz','selected_full.npz'] if (folder/name).exists())
        with np.load(folder/'reference.npz') as f:cd=f['coefficients'];od=f['occupations']
        mol=scf.chkfile.load_mol(str(folder/'backend.chk'));small,transform=source_molecule(state)
        cross=(gto.intor_cross('int1e_ovlp',mol,small)@transform).T
        di=np.flatnonzero(od==2);dv=np.flatnonzero(od==0)
        occupied=np.flatnonzero(state.alpha_occupations==1);virtual=np.flatnonzero(state.alpha_occupations==0)
        with np.load(file) as f:amplitude=f['amplitudes'][root-1].reshape(len(di),len(dv))
        oi=state.coefficients[:,occupied].T@cross@cd[:,di];ov=state.coefficients[:,virtual].T@cross@cd[:,dv]
        a=oi@amplitude@ov.T;norm=np.linalg.norm(a)
        assert norm>=.7,'the frozen DFT reference does not project into the compact valence space'
        a/=norm
        origin=dict(kind='original qualified root, no root substitution',source=str(file),sha256=digest(file),root=root,projection_norm=float(norm))
    np.savez_compressed(out/'identity_anchor.npz',amplitude=a,occupied=occupied,virtual=virtual)
    write(out/'identity_anchor.json',dict(host=HOST,**origin,minimum_transition_overlap=.7,minimum_tracked_csf_overlap=.7,
          native_canonical_csf_overlap='diagnostic only; may decrease under a change of functional'))
    return a,occupied,virtual

def transition_overlap(c,hole,particle,state,prepared):
    a,occupied,virtual=prepared
    h=state.coefficients[:,occupied].T@state.overlap@c[:,hole]
    p=state.coefficients[:,virtual].T@state.overlap@c[:,particle];rank=np.outer(h,p);norm=np.linalg.norm(rank)
    return 0. if norm<1e-12 else float(abs(np.sum(a*rank))/norm)
