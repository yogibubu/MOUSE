"""Oracle-only integration: scans, accepted restart and optimizer service."""
import socket,sys,json,hashlib
from pathlib import Path
assert socket.getfqdn()=='oracle.sns.it'
sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030');import remote_bootstrap
BASE=Path('/home/enzo/MOUSE-roks-link50-20261030');OLD=Path('/home/enzo/MOUSE-roks-50-50-20261030')
for p in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(p))
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_link.scan import QMScanBackend,ScanPoint,run_qm_scan_points
from matrix_link.resident_calculators import create_resident_calculator
from matrix_link.qmqm_component import CappedQMComponent
from matrix_link.optimizer import GeometryEvaluationService,OptimizerCoordinateModel
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def run(name):
    root=BASE/'results'/name;root.mkdir(exist_ok=True)
    folder=OLD/'results'/name
    with np.load(folder/'full_response/central.npz') as d:xyz=d['coordinates_angstrom'].copy();req=str(d['request_hash'])
    state=load_derivative_point(folder/'full_response/central.npz',request_hash=req,coordinates_angstrom=xyz)
    atoms=tuple({1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in state.reference_state.atomic_numbers)
    groundocc=state.reference_state.alpha_occupations;alpha=state.occupations[0]
    hole=int(np.flatnonzero(groundocc-alpha==1)[0]);particle=int(np.flatnonzero(alpha-groundocc==1)[0])
    paths=tuple(folder/p/'response.npz' for p in ['full_response','qualified_spectral_response'])
    indices=tuple(json.loads((folder/p/'validation.json').read_text())['saddle_index'] for p in ['full_response','qualified_spectral_response'])
    spec=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=(hole,particle),
        native_roks_composite_response_paths=paths,native_roks_composite_saddle_indices=indices,processors=2,properties=('energy','gradient','hessian'))
    geom=root/'molecule.xyz';geom.write_text(str(len(atoms))+'\nROKS 50/50 LINK validation\n'+''.join(f'{a} {r[0]:.15f} {r[1]:.15f} {r[2]:.15f}\n' for a,r in zip(atoms,xyz)))
    v=np.arange(1,xyz.size+1,dtype=float).reshape(xyz.shape);v-=v.mean(axis=0);v/=np.linalg.norm(v);step=1e-4
    points=(ScanPoint(2,step,xyz+step/ANGSTROM_TO_BOHR*v),ScanPoint(0,0.,xyz),ScanPoint(1,-step,xyz-step/ANGSTROM_TO_BOHR*v))
    # The real public scan entry orders the center first and retains it for both sides.
    scans=run_qm_scan_points(geom,points,spec,run_dir=root/'scan')
    for r in scans:assert r.status=='completed',r.message
    with np.load(folder/'composite.npz') as d:
        oldh=d['hessian'];oldg=d['gradient'];olde=float(d['energy_hartree'])
    center=scans[0];plus=scans[2];minus=scans[1]
    check=dict(scan_energy_error=abs(center.energy_hartree-olde),scan_gradient_error=float(np.max(abs(center.gradient_hartree_per_bohr-oldg.ravel()))),
        scan_hessian_error=float(np.max(abs(center.hessian_hartree_per_bohr2-oldh))),
        scan_hessian_vector_error=float(np.max(abs((plus.gradient_hartree_per_bohr-minus.gradient_hartree_per_bohr)/(2*step)-oldh@v.ravel()))))
    accepted=root/'scan/roks_50_50/roks_50_50_accepted/accepted.json';before=hashlib.sha256(accepted.read_bytes()).hexdigest()
    session=create_resident_calculator('matrix_gfn2',atoms,xyz,processors=2)
    component=CappedQMComponent(spec,session,root/'scan/roks_50_50')
    assert component.reference is not None and component.anchor is not None
    check['restart_csf_exact']=all(np.array_equal(a.common_coefficients,b.common_coefficients) for a,b in zip(component.reference[0].components,center.scf_state[0].components))
    trial=component.evaluate(points[0].coordinates_angstrom,compute_hessian=True)
    check['restart_trial_energy_error']=abs(trial.energy_hartree-plus.energy_hartree)
    check['trial_did_not_advance_checkpoint']=before==hashlib.sha256(accepted.read_bytes()).hexdigest()
    # Resident geometry optimizer dispatch + explicit acceptance + recreated service.
    coords=OptimizerCoordinateModel('sycart',tuple('q'+str(i) for i in range(xyz.size)),np.eye(xyz.size),np.ones(xyz.size))
    service=GeometryEvaluationService(xyzin_path=geom,run_dir=root/'optimizer',coordinate_model=coords,backend=spec)
    evaluation=service.evaluate(np.zeros(xyz.size),tag='center',requested_properties=('energy','gradient','hessian'))
    check['optimizer_energy_error']=abs(evaluation.result.energy_hartree-center.energy_hartree)
    check['optimizer_hessian_error']=float(np.max(abs(evaluation.result.hessian_hartree_per_bohr2-oldh)))
    service.accept_electronic_state(evaluation)
    restarted=GeometryEvaluationService(xyzin_path=geom,run_dir=root/'optimizer',coordinate_model=coords,backend=spec)
    assert restarted._orbital_component.reference is not None
    evaluation2=restarted.evaluate(np.zeros(xyz.size),tag='resumed-center',requested_properties=('energy','gradient','hessian'))
    check['optimizer_restart_energy_error']=abs(evaluation2.result.energy_hartree-evaluation.result.energy_hartree)
    check['optimizer_disk_cache_restored_csfs']=evaluation2.result.scf_state is not None
    # Corrupt accepted bytes: recreation must fail before launching an SCF.
    data=json.loads(accepted.read_text());file=accepted.parent/data['reference'][0]['path'];saved=file.read_bytes()
    try:
        file.write_bytes(saved+b'corrupt')
        try:CappedQMComponent(spec,session,root/'scan/roks_50_50')
        except ValueError as e:check['corrupt_restart_rejected']='checksum' in str(e)
        else:check['corrupt_restart_rejected']=False
    finally:file.write_bytes(saved)
    record=dict(host=socket.getfqdn(),molecule=name,checks=check,weights=[.5,.5],saddle_indices=indices,
        excitation_ev=center.scf_state[0].excitation_energy_hartree*27.211386245988)
    record['passed']=all(v is True if isinstance(v,bool) else v<2e-6 for v in check.values())
    (root/'integration.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True);assert record['passed']
with threadpool_limits(limits=2):
    for name in sys.argv[1:]:run(name)
