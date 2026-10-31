"""Remote-only molecular minima and analytic frequencies of frozen ROKS 50/50."""
import socket,sys,json,time,hashlib,traceback,shutil
from pathlib import Path
HOST=socket.getfqdn()
if HOST!='oracle.sns.it':raise RuntimeError('unauthorized compute host '+HOST)
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
PARENT=Path('/home/enzo/MOUSE-roks-following-20261030')
VERT=Path('/home/enzo/MOUSE-roks-independent-20261030');DER=Path('/home/enzo/MOUSE-roks-50-50-20261030');LINK=Path('/home/enzo/MOUSE-roks-link50-20261030')
sys.path.insert(0,str(VERT));import remote_bootstrap
for p in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(p))
import numpy as np
from dataclasses import replace
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference,save_response_reference,combine_roks_results
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_link.scan import QMScanBackend,PointEvaluationResult,read_point_result
from matrix_link.optimizer import (GeometryEvaluationService,OptimizerCoordinateModel,OptimizerSettings,optimize_geometry,
    optimizer_result_to_json,optimizer_hessian_from_cartesian,_exact_cartesian_normal_modes,OptimizerEvaluation)
PROTOCOL=json.loads((BASE/'protocol.json').read_text())

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def geomfile(path,atoms,xyz):
    path.write_text(str(len(atoms))+'\nROKS 50/50 frozen-response optimization\n'+''.join(f'{a} {r[0]:.16f} {r[1]:.16f} {r[2]:.16f}\n' for a,r in zip(atoms,xyz)))

def prepare(name):
    root=BASE/'results'/name;root.mkdir(parents=True,exist_ok=True)
    if (root/'prepared.json').exists():return json.loads((root/'prepared.json').read_text())
    start=time.monotonic();states=[];paths=[];records=[]
    for profile in ['full_response','qualified_spectral_response']:
        folder=root/profile;folder.mkdir(exist_ok=True)
        if name in ['nitroxyl','acetaldehyde']:
            origin=DER/'results'/name/profile;refpath=origin/'response.npz';cp=origin/'central.npz'
            with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy();req=str(d['request_hash'])
            old=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz)
            shutil.copy2(refpath,folder/'response.npz')
            save_derivative_point(folder/'seed.npz',old,request_hash='retained_vertical',coordinates_angstrom=xyz)
        else:
            report=VERT/'results'/profile/name/'n_pi/result.json';row=json.loads(report.read_text());assert row['qualified']
            cp=Path(row['checkpoint'])
            with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
            old=load_derivative_point(cp,request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
            s=old.reference_state;atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers]
            calc=GFN2Calculator(atoms,temperature_kelvin=0.);ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=old.ground_energy_hartree)
            ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
            with np.load(VERT/'results'/profile/name/'model.npz') as d:data={k:d[k].copy() for k in ['operators','charge','spin','q0','linear']}
            ref=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,data,sha(VERT/'results'/profile/name/'model.npz'))
            save_response_reference(folder/'response.npz',ref);old=replace(old,spin_reference_fingerprint=ref.fingerprint)
            save_derivative_point(folder/'seed.npz',old,request_hash='retained_vertical',coordinates_angstrom=xyz)
        states.append(old);paths.append(str(folder/'response.npz'));records.append(dict(profile=profile,checkpoint=str(cp),checkpoint_sha256=sha(cp),index=old.orbital_saddle_index))
    initial=combine_roks_results(*states)
    ground=states[0].reference_state.alpha_occupations;alpha=states[0].occupations[0]
    hole=int(np.flatnonzero(ground-alpha==1)[0]);particle=int(np.flatnonzero(alpha-ground==1)[0])
    atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in states[0].reference_state.atomic_numbers]
    geomfile(root/'initial.xyz',atoms,xyz)
    data=dict(host=HOST,molecule=name,atoms=atoms,coordinates_angstrom=xyz.tolist(),promotion=[hole,particle],response_paths=paths,
        saddle_indices=[r.orbital_saddle_index for r in states],inputs=records,seconds_current_preparation=time.monotonic()-start,
        original_vertical_excitation_ev=initial.excitation_energy_hartree*27.211386245988)
    write(root/'prepared.json',data);return data

def service_for(name,branch='main',geometry=None):
    prepared=prepare(name);root=BASE/'results'/name;xyz=np.asarray(prepared['coordinates_angstrom']);atoms=prepared['atoms']
    spec=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(prepared['promotion']),
        native_roks_composite_response_paths=tuple(prepared['response_paths']),native_roks_composite_saddle_indices=tuple(prepared['saddle_indices']),
        native_roks_composite_following='character_adaptive_index' if name=='nitroxyl' else 'character',
        processors=2,properties=('energy','gradient'))
    model=OptimizerCoordinateModel('sycart',tuple(f'q{i}' for i in range(xyz.size)),np.eye(xyz.size),np.ones(xyz.size))
    work=root/branch;work.mkdir(exist_ok=True)
    inputfile=root/'initial.xyz'
    if geometry is not None:inputfile=work/'branch_initial.xyz';geomfile(inputfile,atoms,geometry)
    settings=OptimizerSettings(max_steps=80,trust_radius=.1,max_trust_radius=.15,gradient_tolerance=3e-5,
        max_force_tolerance=3e-5,rms_force_tolerance=1e-5,energy_tolerance=1e-8,step_tolerance=2e-4,
        max_displacement_tolerance=2e-4,rms_displacement_tolerance=1e-4,compute_final_hessian=True,
        require_exact_final_hessian=True,min_hessian_eigenvalue=.01,two_sided=True,resume=(work/'optimizer_trace.jsonl').exists())
    service=GeometryEvaluationService(xyzin_path=inputfile,run_dir=work,coordinate_model=model,backend=spec,settings=settings)
    if service._orbital_component.reference is None:
        seeds=[load_derivative_point(root/p/'seed.npz',request_hash='retained_vertical',coordinates_angstrom=xyz,require_gradient=False) for p in ['full_response','qualified_spectral_response']]
        # Retain the completed electronic branch before evaluating any gradient.
        composite=combine_roks_results(*seeds)
        service._orbital_component.reference=(composite,tuple(r.to_apoc(1) for r in seeds))
        service._orbital_component.anchor=service._orbital_component.reference
    return service,settings,model,spec,inputfile

def optimize(name,branch='main',geometry=None):
    root=BASE/'results'/name;work=root/branch
    if (work/'complete.json').exists():return json.loads((work/'complete.json').read_text())
    service,settings,model,spec,inputfile=service_for(name,branch,geometry)
    start=time.monotonic();print('START',name,branch,flush=True)
    initial=service.evaluate(np.zeros(len(model.labels)),tag='resumed-initial-gradient',requested_properties=('energy','gradient'))
    service.accept_electronic_state(initial)
    origin_branch='main' if name=='nitroxyl' else 'negative_mode_minus'
    retained_hessian=PARENT/'results'/name/'resume/optimizer_hessian.json'
    seed=json.loads(retained_hessian.read_text())
    if seed['units']!='hartree per optimizer-coordinate squared':raise ValueError('optimizer seed units changed')
    hessian=np.array(seed['hessian'])
    write(work/'optimizer_seed_provenance.json',dict(path=str(retained_hessian),sha256=sha(retained_hessian),units=seed['units'],parent_source=seed['source']))
    np.savez_compressed(work/'initial_seed.npz',hessian=hessian,gradient=initial.result.gradient_hartree_per_bohr)
    result=optimize_geometry(inputfile,run_dir=work,coordinate_model=model,coordinate_kind='sycart',allow_non_sonic_coordinates=True,
        backend=spec,settings=settings,evaluation_service=service,initial_hessian=hessian,
        initial_hessian_source='retained parent optimizer Hessian; no completed phase restarted',allow_native_sycart=False)
    summary=optimizer_result_to_json(result);write(work/'optimizer_result.json',summary)
    accepted=service._orbital_component.reference[0]
    if not result.converged and np.max(abs(accepted.gradient_hartree_per_bohr))>3e-5:
        row=dict(host=HOST,molecule=name,branch=branch,converged=False,status=result.status,passed=False,
            seconds_current_run=time.monotonic()-start,steps=len(result.iterations),qm_evaluations=result.qm_evaluations,
            maximum_gradient=float(np.max(abs(accepted.gradient_hartree_per_bohr))),
            final_hessian_not_requested='nonstationary terminated branch; accepted checkpoints retained')
        write(work/'complete.json',row);print('TERMINATED',name,branch,json.dumps(row),flush=True);return row
    # Retain an exact final analysis also if the optimizer reports a non-minimum.
    final=service.evaluate(result.final_q,tag='analytic-final-audit',requested_properties=('energy','gradient','hessian'))
    xyz=final.coordinates_angstrom;h=final.result.hessian_hartree_per_bohr2
    modes=_exact_cartesian_normal_modes(result.atoms,xyz,h);freq=np.asarray(modes.frequencies_cm)
    np.savez_compressed(work/'final.npz',coordinates_angstrom=xyz,energy_hartree=final.result.energy_hartree,
        gradient=final.result.gradient_hartree_per_bohr,hessian=h,frequencies_cm=freq)
    geomfile(work/'final.xyz',result.atoms,xyz)
    check=[];rng=np.random.default_rng(20261030);v=rng.normal(size=xyz.shape);v-=v.mean(axis=0);v/=np.linalg.norm(v)
    for step in [2e-4,1e-4]:
        points=[]
        for sign in [-1,1]:
            point=service._orbital_component.evaluate(xyz+sign*step/ANGSTROM_TO_BOHR*v,compute_hessian=False);points.append(point)
        dg=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr)/(2*step)
        check.append(dict(step_bohr=step,hessian_vector_max_error=float(np.max(abs(dg-h@v.ravel())))))
    row=dict(host=HOST,molecule=name,branch=branch,converged=result.converged,status=result.status,
        seconds_current_run=time.monotonic()-start,steps=len(result.iterations),qm_evaluations=result.qm_evaluations,
        energy_hartree=final.result.energy_hartree,maximum_gradient=float(np.max(abs(final.result.gradient_hartree_per_bohr))),
        frequencies_cm=freq.tolist(),imaginary_modes_below_minus20=int(np.sum(freq < -20.)),
        gradient_checks=check,maximum_atom_shift_angstrom=float(np.max(np.linalg.norm(xyz-np.asarray(prepare(name)['coordinates_angstrom']),axis=1))),
        component_csf_overlap=final.result.scf_state[0].component_csf_overlap,execution=final.result.execution)
    row['passed']=bool(result.converged and row['maximum_gradient']<=3e-5 and not row['imaginary_modes_below_minus20'] and check[-1]['hessian_vector_max_error']<3e-6)
    write(work/'complete.json',row);print('COMPLETE',name,branch,json.dumps(row),flush=True);return row

def migrate(name):
    from matrix_link.composite_roks_component import restore_accepted,persist_accepted,_contract
    if name!='nitroxyl':raise ValueError('frozen HNO-only continuation')
    root=BASE/'results'/name;root.mkdir(parents=True,exist_ok=True)
    parent=PARENT/'results'/name;p=json.loads((parent/'prepared.json').read_text())
    write(root/'prepared.json',p);shutil.copy2(parent/'initial.xyz',root/'initial.xyz')
    backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(p['promotion']),
        native_roks_composite_response_paths=tuple(p['response_paths']),native_roks_composite_saddle_indices=tuple(p['saddle_indices']),
        native_roks_composite_following='character_adaptive_index',processors=2,properties=('energy','gradient'))
    origin=parent/'resume/roks_50_50';old=SimpleNamespace(backend=backend,run_dir=origin,reference=None,anchor=None);restore_accepted(old)
    dest=root/'resume/roks_50_50';new=SimpleNamespace(backend=backend,run_dir=dest,reference=old.reference,anchor=old.anchor)
    if not (dest/'roks_50_50_accepted/accepted.json').exists():persist_accepted(new,new.reference)
    else:restore_accepted(new)
    xyz=new.reference[0].components[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
    manifest=origin/'roks_50_50_accepted/accepted.json'
    write(root/'migration.json',dict(host=HOST,parent_manifest=str(manifest),parent_manifest_sha256=sha(manifest),
        backend_contract=_contract(backend),starting_coordinates_angstrom=xyz.tolist(),original_anchor_preserved=True,
        numerical_correction='Newton recertification at requested temperature; do not replay initial annealing'))
    return xyz

if __name__=='__main__':
    with threadpool_limits(limits=2):
        for name in sys.argv[1:]:
            try:
                xyz=migrate(name)
                optimize(name,'resume',xyz)
            except Exception as exc:
                write(BASE/'results'/name/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()))
                print('FAILURE',name,traceback.format_exc(),flush=True)
