"""Separate transported-anchor and relaxed-native subspace effects on rank.

No new SCF or response solve. Reproduce the original compact transition first.
"""
from pathlib import Path
import os,sys,socket,json,fcntl,hashlib
assert socket.getfqdn()=='oracle.sns.it'
W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
os.environ.update(MOUSE_VALIDATION_ROOT=str(W),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(W/'source/qualification'));import remote_bootstrap
sys.path.insert(0,str(W))
from anchor_rank_symmetry import participation
import numpy as np
from pyscf import gto,scf,lib
from threadpoolctl import threadpool_limits
from matrix_gfn2 import MultiCSFModel
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.multicsf_nuclear import metric_power
sys.path.insert(0,'/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029')
from extent_support import source_molecule

with threadpool_limits(limits=2):
    lib.num_threads(2)
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    summary=json.loads((W/'anchor_rank_symmetry/summary.json').read_text());assert summary['complete']
    audits=[]
    for name,sid in [('Phthalimide','B024'),('Furan','B022'),('Cyclopentadiene','B016')]:
        model=MultiCSFModel(W/f'models/{name}/{sid}/model.json')
        ref=P/'blind/mouse_reference'/name
        oldmol=scf.chkfile.load_mol(str(ref/'backend.chk'))
        with np.load(ref/'reference.npz') as a:cd=a['coefficients'].copy();occ=a['occupations'].copy()
        no=int(np.count_nonzero(occ==2));nc=model.no
        with np.load(ref/'backend_tda.npz') as a:x=a['amplitudes'][model.manifest['provenance']['reference_root']-1].reshape(no,-1).copy()
        x/=np.linalg.norm(x)
        small,t=source_molecule(model.anchor)
        cross=t.T@gto.intor_cross('int1e_ovlp',small,oldmol)@cd
        ov=model.anchor.coefficients.T@cross
        projected=ov[:nc,:no]@x@ov[nc:,no:].T;baseline_norm=float(np.linalg.norm(projected));projected/=baseline_norm
        transition=next(Path(p) for p in model.manifest['provenance']['input_sha256'] if p.endswith('/transition.npz'))
        with np.load(transition) as a:original=a['amplitude'].copy()
        error=float(min(np.max(abs(projected-original)),np.max(abs(projected+original))))
        assert error<1e-7, f'original projection not reproduced: {name} {error}'
        relaxed=model.grounds[0].coefficients[0].T@cross
        rx=relaxed[:nc,:no]@x@relaxed[nc:,no:].T
        rows=[]
        for scale in [.98,1.02]:
            folder=W/'anchor_rank_symmetry'/name/f'{scale:.2f}'
            row=json.loads((folder/'result.json').read_text())
            with np.load(folder/'native.npz') as a:xyz=a['coordinates_angstrom'].copy()
            raw=load_derivative_point(folder/'native.npz',request_hash='anchor_rank_native',coordinates_angstrom=xyz,require_gradient=False)
            state=raw.reference_state;sm,t=source_molecule(state)
            mol=scf.chkfile.load_mol(str(folder/'backend.chk'))
            with np.load(folder/'reference.npz') as a:cdnew=a['coefficients'].copy()
            with np.load(folder/'tda.npz') as a:xnew=a['amplitudes'][row['root']-1].copy()
            xnew/=np.linalg.norm(xnew)
            chart,_=model.eri.chart(state)
            transport=metric_power(state.overlap,-.5)@chart.u@metric_power(model.anchor.overlap,.5)
            c=transport@model.anchor.coefficients
            ov=c.T@t.T@gto.intor_cross('int1e_ovlp',sm,mol)@cdnew
            px=ov[:nc,:no]@xnew@ov[nc:,no:].T
            rows.append(dict(scale=scale,transported_anchor_compact=participation(px),
                             projection_norm=float(np.linalg.norm(px)),
                             relaxed_native_compact=row['compact_transition']))
        audits.append(dict(molecule=name,state=sid,original_projection_max_error=error,
                           original_compact=participation(original),original_projection_norm=baseline_norm,
                           original_relaxed_native_compact=participation(rx),rows=rows))
    out=dict(host=socket.getfqdn(),rows=audits,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             scope='Reproduced original compact projection; co-rotating Lowdin transport of fixed anchor orbitals separates DFT-transition changes from native occupied-subspace relaxation; no full response-model recalibration')
    (W/'anchor_rank_symmetry/projection_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
