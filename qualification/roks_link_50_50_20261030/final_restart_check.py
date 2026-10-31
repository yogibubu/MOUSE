"""Exercise final accepted-state guards on completed integration checkpoints."""
import socket,sys,json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
assert socket.getfqdn()=='oracle.sns.it'
sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030');import remote_bootstrap
BASE=Path('/home/enzo/MOUSE-roks-link50-20261030');OLD=Path('/home/enzo/MOUSE-roks-50-50-20261030')
for p in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(p))
from matrix_link.scan import QMScanBackend,PointEvaluationResult
from matrix_link.qmqm_component import CappedQMComponent
checks=[]
for name in ['nitroxyl','acetaldehyde']:
    folder=OLD/'results'/name;meta=json.loads((BASE/'results'/name/'integration.json').read_text())
    manifest=BASE/'results'/name/'scan/roks_50_50/roks_50_50_accepted/accepted.json'
    # Restore the promotion from the numerical contract recorded by the scan spec.
    point=json.loads((BASE/'results'/name/'scan/point_0000/point.json').read_text())
    refs=tuple(folder/p/'response.npz' for p in ['full_response','qualified_spectral_response'])
    # Actual promoted occupations and the native reference reconstruct the exact same pair.
    import numpy as np
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    cp=folder/'full_response/central.npz'
    with np.load(cp) as d:r=load_derivative_point(cp,request_hash=str(d['request_hash']),coordinates_angstrom=d['coordinates_angstrom'])
    ground=r.reference_state.alpha_occupations;alpha=r.occupations[0]
    pair=(int(np.flatnonzero(ground-alpha==1)[0]),int(np.flatnonzero(alpha-ground==1)[0]))
    backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=pair,
        native_roks_composite_response_paths=refs,native_roks_composite_saddle_indices=tuple(meta['saddle_indices']))
    component=CappedQMComponent(backend,SimpleNamespace(_calculator=SimpleNamespace()),manifest.parent.parent)
    retained=component.reference
    valid=PointEvaluationResult(0,0.,scf_state=retained,execution={'selected_electronic_state':1,'gradient_electronic_state':1})
    raw,spectral=retained[0].components
    bad=replace(raw,spin_reference_fingerprint='another Hamiltonian')
    forged=replace(valid,scf_state=(replace(retained[0],components=(bad,spectral)),retained[1]))
    saved=manifest.read_bytes()
    try:component.accept(forged)
    except ValueError as e:assert 'another response' in str(e)
    else:raise AssertionError('wrong model accepted')
    assert component.reference is retained and manifest.read_bytes()==saved
    component.accept(valid)
    resumed=CappedQMComponent(backend,SimpleNamespace(_calculator=SimpleNamespace()),manifest.parent.parent)
    assert all(np.array_equal(a.common_coefficients,b.common_coefficients) for a,b in zip(retained[0].components,resumed.reference[0].components))
    checks.append(dict(molecule=name,wrong_hamiltonian_rejected=True,final_accepted_roundtrip=True))
result=dict(host=socket.getfqdn(),checks=checks,passed=True)
(BASE/'results/final_restart_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
