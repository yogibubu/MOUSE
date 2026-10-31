"""Seeded continuation: recompute DFT transitions and native S0 at fixed displaced anchors on Oracle.

This tests reference/rank stability, not recalibrated OO excitation energies.
Completed SCF, response, and native certificates are individually reusable.
"""
from pathlib import Path
import os, sys, socket, json, hashlib, time, fcntl, traceback

assert socket.getfqdn() == 'oracle.sns.it', 'Electronic jobs require Oracle'
V = Path('/home/enzo/MOUSE-orbital-performance-20261031')
W = V / 'multicsf_integration'
P = V.parent / 'MOUSE-publication-validation-20261030'
os.environ.update(MOUSE_VALIDATION_ROOT=str(W), OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.path.insert(0, str(W / 'source/qualification'))
import remote_bootstrap
import numpy as np
from pyscf import gto, dft, scf, lib
from pyscf.tdscf import _lr_eig
# Request-sized Davidson blocks instead of the default extra Ritz directions.
_lr_eig.MAX_SPACE_INC = None
from threadpoolctl import threadpool_limits
from matrix_gfn2 import MultiCSFModel, GFN2Calculator
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.restricted_ground import evaluate_restricted_ground, RestrictedGroundResult
from matrix_gfn2.deltascf_derivatives import save_derivative_point, load_derivative_point
sys.path.insert(0, '/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029')
from extent_support import source_molecule

OUT = W / 'anchor_rank_converged'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)

def participation(x):
    w = np.linalg.svd(x, compute_uv=False)**2
    w /= w.sum()
    n = float(1 / (w @ w))
    return dict(weights=w.tolist(), effective_pairs=n, rank=int(np.floor(n + .5)))

def run(name, sid, scale):
    base = MultiCSFModel(W / f'models/{name}/{sid}/model.json')
    ref = P / 'blind/mouse_reference' / name
    folder = OUT / name / f'{scale:.2f}'
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / 'result.json'
    request = dict(protocol_sha256=sha(OUT/'protocol.json'), model_fingerprint=base.fingerprint,
                   scale=scale, reference_sha256=sha(ref/'reference.npz'),
                   spectrum_sha256=sha(ref/'backend_tda.npz'))
    if result.exists():
        old = json.loads(result.read_text())
        assert old['request'] == request
        return old
    start = time.monotonic()
    oldmol = scf.chkfile.load_mol(str(ref/'backend.chk'))
    with np.load(ref/'reference.npz') as a:
        oldc=a['coefficients'].copy(); oldocc=a['occupations'].copy()
    with np.load(ref/'backend_tda.npz') as a:
        root=base.manifest['provenance']['reference_root']-1
        oldx=a['amplitudes'][root].reshape(np.count_nonzero(oldocc==2), -1).copy()
        symmetry=str(a['state_symmetries'][root])
    oldx /= np.linalg.norm(oldx)
    center = np.mean(oldmol.atom_coords(), axis=0)
    xyz = center + scale*(oldmol.atom_coords()-center)
    mol=oldmol.copy(); mol.output=str(folder/'pyscf.log'); mol.verbose=4
    mol.set_geom_([(oldmol.atom_symbol(i), r) for i,r in enumerate(xyz)], unit='Bohr', symmetry=True)
    assert mol.groupname == oldmol.groupname
    mf=dft.RKS(mol); mf.xc='PBE0'; mf.grids.level=3; mf.small_rho_cutoff=0
    mf.conv_tol=1e-10; mf.max_cycle=150; mf.max_memory=24000; mf.chkfile=str(folder/'backend.chk')
    ground=folder/'reference.npz'
    if ground.exists():
        with np.load(ground) as a:
            mf.mo_coeff=a['coefficients'].copy(); mf.mo_occ=a['occupations'].copy()
            mf.mo_energy=a['orbital_energies'].copy(); mf.e_tot=float(a['energy_hartree']); mf.converged=True
    else:
        density=scf.addons.project_dm_nr2nr(oldmol, (oldc*oldocc)@oldc.T, mol)
        if (folder/'backend.chk').exists():
            try: density=mf.init_guess_by_chkfile(str(folder/'backend.chk'))
            except Exception: pass
        mf.kernel(dm0=density)
        if not mf.converged:
            original=mf; mf=mf.newton(); mf.kernel(original.mo_coeff, original.mo_occ)
        assert mf.converged, 'DFT SCF not converged'
        np.savez_compressed(ground, coefficients=mf.mo_coeff, occupations=mf.mo_occ,
                            orbital_energies=mf.mo_energy, energy_hartree=mf.e_tot,
                            coordinates_bohr=xyz, overlap=mf.get_ovlp())
    response=folder/'tda.npz'
    if not response.exists():
        td=mf.TDA(); td.wfnsym=symmetry; td.nstates=2; td.conv_tol=1e-7; td.max_cycle=150; td.lindep=1e-20
        # Transport the completed fixed-reference transition into the new MO
        # spaces; augment it with ordinary lowest-gap guesses for both roots.
        cross0=gto.intor_cross('int1e_ovlp',oldmol,mol)
        co0=oldc[:,:np.count_nonzero(oldocc==2)].T@cross0@mf.mo_coeff[:,:np.count_nonzero(oldocc==2)]
        cv0=oldc[:,np.count_nonzero(oldocc==2):].T@cross0@mf.mo_coeff[:,np.count_nonzero(oldocc==2):]
        guess=(co0.T@oldx@cv0).ravel()
        conventional=td.get_init_guess(mf,nstates=2,wfnsym=symmetry)
        td.verbose=5
        checkpoint=folder/'ritz_checkpoint.npz'
        fingerprint=sha(ground)+':PBE0:aug-cc-pVDZ:grid3:singlet:'+symmetry
        initial=np.vstack([guess/np.linalg.norm(guess),conventional])
        restart=dict(kind='original fixed-reference transition and gap guesses')
        if checkpoint.exists():
            with np.load(checkpoint) as a:
                assert str(a['fingerprint'])==fingerprint
                initial=a['ritz_vectors'].copy()
            restart=dict(kind='same-geometry saved Ritz vectors',source_sha256=sha(checkpoint))
        else:
            neighbors=[p for p in (OUT/name).glob('*/tda.npz') if p.parent!=folder and (p.parent/'reference.npz').exists()]
            if neighbors:
                neighbor=min(neighbors,key=lambda p:abs(float(p.parent.name)-scale))
                nm=scf.chkfile.load_mol(str(neighbor.parent/'backend.chk'))
                with np.load(neighbor.parent/'reference.npz') as a:nc=a['coefficients'].copy();nn=int(np.count_nonzero(a['occupations']==2))
                ncross=gto.intor_cross('int1e_ovlp',nm,mol)
                oo=nc[:,:nn].T@ncross@mf.mo_coeff[:,:np.count_nonzero(oldocc==2)]
                vv=nc[:,nn:].T@ncross@mf.mo_coeff[:,np.count_nonzero(oldocc==2):]
                with np.load(neighbor) as a:previous=a['amplitudes'].copy()
                initial=np.array([(oo.T@x@vv).ravel() for x in previous])
                initial/=np.linalg.norm(initial,axis=1)[:,None]
                restart=dict(kind='both completed neighboring-anchor roots transported by cross-AO overlap',source=str(neighbor),source_sha256=sha(neighbor))
        write(folder/'response_restart.json',restart)
        original_vind,hdiag=td.gen_vind(mf)
        trial_vectors=[];trial_images=[]
        def saved_vind(vectors):
            images=original_vind(vectors)
            trial_vectors.extend(np.atleast_2d(vectors).copy())
            trial_images.extend(np.atleast_2d(images).copy())
            xx=np.array(trial_vectors);yy=np.array(trial_images)
            metric=xx@xx.T;v,u=np.linalg.eigh(.5*(metric+metric.T));keep=v>1e-9
            transform=u[:,keep]/np.sqrt(v[keep])
            xo=transform.T@xx;yo=transform.T@yy
            projected=xo@yo.T;energies,ci=np.linalg.eigh(.5*(projected+projected.T))
            count=min(2,len(energies));ritz=ci[:,:count].T@xo
            residuals=np.linalg.norm(ci[:,:count].T@yo-energies[:count,None]*ritz,axis=1)
            tmp=checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,ritz_vectors=ritz,ritz_energies=energies[:count],residuals=residuals,
                                fingerprint=fingerprint,trial_count=len(xx))
            tmp.replace(checkpoint)
            return images
        td.gen_vind=lambda mf=None:(saved_vind,hdiag)
        td.kernel(x0=initial)
        assert len(td.e)==2 and np.all(td.converged), 'two same-symmetry roots required'
        np.savez_compressed(response, amplitudes=np.array([x for x,y in td.xy]),
                            energies_hartree=td.e, converged=td.converged)
    with np.load(response) as a: xs=a['amplitudes'].copy(); energies=a['energies_hartree'].copy()
    xs /= np.linalg.norm(xs.reshape(len(xs),-1),axis=1)[:,None,None]
    no=np.count_nonzero(oldocc==2)
    cross=gto.intor_cross('int1e_ovlp',oldmol,mol)
    co=oldc[:,:no].T@cross@mf.mo_coeff[:,:no]
    cv=oldc[:,no:].T@cross@mf.mo_coeff[:,no:]
    scores=np.array([abs(np.sum(oldx*(co@x@cv.T))) for x in xs])
    chosen=int(np.argmax(scores)); margin=float(scores[chosen]-np.max(np.delete(scores,chosen)))
    assert scores[chosen]>.7 and margin>.05, f'ambiguous state identity: {scores}'
    assert chosen==0, 'ordinal and coherent assignment disagree'
    x=xs[chosen]
    calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in base.anchor.atomic_numbers],temperature_kelvin=0.)
    native=folder/'native.npz'
    if native.exists():
        with np.load(native) as a: coords=a['coordinates_angstrom'].copy()
        raw=load_derivative_point(native,request_hash='anchor_rank_native',coordinates_angstrom=coords,require_gradient=False)
    else:
        solved=evaluate_restricted_ground(calc,xyz/ANGSTROM_TO_BOHR,template=base.anchor,
                                         reference_result=base.grounds[0],gradient=False)
        raw=solved.reference
        save_derivative_point(native,raw,request_hash='anchor_rank_native',coordinates_angstrom=xyz/ANGSTROM_TO_BOHR)
        write(folder/'native_certificate.json',dict(minimum_curvature=solved.minimum_orbital_curvature_hartree,
                                                 index=solved.orbital_saddle_index))
    state=raw.reference_state; small,t=source_molecule(state)
    overlap=raw.coefficients[0].T@(gto.intor_cross('int1e_ovlp',small,mol).T@t).T@mf.mo_coeff
    nc=base.no
    projected=overlap[:nc,:no]@x@overlap[nc:,no:].T
    norm=float(np.linalg.norm(projected))
    assert norm>.7, f'insufficient compact projection {norm}'
    original_transition=base.manifest['provenance']['input_sha256']
    transition_path=next(Path(k) for k in original_transition if k.endswith('/transition.npz'))
    with np.load(transition_path) as a: baseline=a['amplitude'].copy()
    row=dict(request=request,host=socket.getfqdn(),molecule=name,state=sid,scale=scale,
             maximum_displacement_angstrom=float(np.max(np.linalg.norm(xyz-oldmol.atom_coords(),axis=1)))/ANGSTROM_TO_BOHR,
             point_group=mol.groupname,state_symmetry=symmetry,root=chosen+1,
             coherent_reference_overlaps=scores.tolist(),competitor_margin=margin,
             reference_tda_ev=float(energies[chosen]*27.211386245988),
             dft_transition=participation(x),compact_transition=participation(projected),
             original_compact_transition=participation(baseline),compact_projection_norm=norm,
             source_sha256={str(p):sha(p) for p in [ground,response,native]},
             seconds_current_run=time.monotonic()-start,
             scope='Actual DFT and native-ground reoptimization plus compact projection; no OO-response recalibration or excitation-energy prediction')
    write(result,row); print(name,scale,row['compact_transition']['effective_pairs'],flush=True)
    return row

def main():
    protocol=dict(schema='mouse.anchor.rank.v4',host='oracle.sns.it',states=[['Phthalimide','B024'],['Furan','B022'],['Cyclopentadiene','B016']],scales=[.98,1.02],
                  displacement='uniform internal Cartesian dilation about geometric center; same point group',
                  xc='PBE0',basis='aug-cc-pVDZ',grid=3,scf_tolerance=1e-10,tda_tolerance=1e-7,
                  same_symmetry_roots=2,tda_linear_dependence_threshold=1e-20,minimum_overlap=.7,minimum_margin=.05,
                  assignment='maximum coherent transition overlap with original fixed DFT state; require first ordinal unchanged',
                  native='restricted GFN2 direct S0 minimization, freshly projected DFT transition',
                  rank='floor(inverse normalized NTO-weight purity + .5)',
                  production_lambda=.9,production_weights=[.5,.5],threads=2,
                  solver_amendment='two-vector Davidson blocks and lindep=1e-20 retain corrections needed for the unchanged 1e-7 residual target; atomic Ritz restart and neighboring-root transport; operator and identity unchanged; prior attempts retained',independent_accuracy_validation=False,source_sha256=sha(Path(__file__)))
    path=OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol
    else: write(path,protocol)
    lock=(P/'blind/references/compute.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX)
    rows=[]
    for name,sid in protocol['states']:
        for scale in protocol['scales']:
            write(OUT/'progress.json',dict(status='running',molecule=name,state=sid,scale=scale,completed=len(rows),host=socket.getfqdn()))
            try: rows.append(run(name,sid,scale))
            except Exception:
                write(OUT/name/f'{scale:.2f}'/'failure.json',dict(traceback=traceback.format_exc(),checkpoints_preserved=True))
                raise
            write(OUT/'summary.json',dict(protocol=protocol,rows=rows,complete=len(rows)==6))
    write(OUT/'progress.json',dict(status='complete',completed=len(rows),host=socket.getfqdn()))

if __name__=='__main__':
    lib.num_threads(2)
    with threadpool_limits(limits=2): main()
