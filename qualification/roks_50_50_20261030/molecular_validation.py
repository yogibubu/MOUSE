"""Oracle-only, checkpointed nuclear derivative validation of fixed 50/50."""
import sys,socket,os
from pathlib import Path
HOST=socket.getfqdn()
if HOST!='oracle.sns.it':raise RuntimeError('unauthorized compute host '+HOST)
OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');BASE=Path('/home/enzo/MOUSE-roks-50-50-20261030')
sys.path.insert(0,str(OLD));import remote_bootstrap
for p in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(p))
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
import json,hashlib,time,argparse
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from threadpoolctl import threadpool_limits
from spin_kernel_probe import plan,state_from
from matrix_gfn2 import (GFN2Calculator,FrozenNonlocalResponseReference,save_response_reference,load_response_reference,
    evaluate_gfn2_roks_singlet,evaluate_gfn2_deltascf_analytic_hessian,combine_roks_results,
    evaluate_composite_roks_analytic_hessian)
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import save_derivative_point,load_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.nonlocal_response import TransportedOperators

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def key(data):return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
def run(name):
    out=BASE/'results'/name;out.mkdir(exist_ok=True)
    case=next(c for c in plan() if c['name']==name) if name in ['nitroxyl','uracil'] else json.loads((OLD/'prepared_B'/name/'case.json').read_text())
    target=next(t for t in case['targets'] if t['id']=='n_pi');native,xyz=state_from(target);s=native.reference_state
    atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers]
    components=[];refs=[];calcs=[];hs=[];metadata=[]
    protocol=json.loads((BASE/'run_protocol.json').read_text())
    for p in ['nonlocal_response.py','deltascf.py']:
        assert sha(BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/p)==protocol['model_source'][p]
    request=key(dict(**protocol,molecule=name))
    for profile in ['full_response','qualified_spectral_response']:
        folder=out/profile;folder.mkdir(exist_ok=True);calc=GFN2Calculator(atoms,temperature_kelvin=0.);calcs.append(calc)
        ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=native.ground_energy_hartree)
        ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
        source=OLD/'results'/profile/name/'model.npz'
        with np.load(source) as f:data={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
        ref=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,data,sha(source));refs.append(ref)
        save_response_reference(folder/'response.npz',ref);assert load_response_reference(folder/'response.npz').fingerprint==ref.fingerprint
        meta=json.loads((OLD/'results'/profile/name/target['id']/'result.json').read_text());assert meta['qualified']
        previous=load_derivative_point(meta['checkpoint'],request_hash=meta['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
        cp=folder/'central.npz';req=key(dict(run=request,model=ref.fingerprint,kind='central'))
        options=dict(orbital_solver='gmf',saddle_index=previous.orbital_saddle_index,orbital_tolerance=1e-11,maximum_iterations=150,maximum_step=.15)
        if cp.exists():central=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz)
        else:
            # Same functional at the anchor; retain the completed vertical solution.
            seed=replace(previous,spin_reference_fingerprint=ref.fingerprint)
            central=evaluate_gfn2_roks_singlet(calc,xyz,*seed.occupations,ground_state=ground,spin_reference=ref,reference_result=seed,gradient=True,**options)
            save_derivative_point(cp,central,request_hash=req,coordinates_angstrom=xyz)
        components.append(central)
        energy_change=abs(central.excitation_energy_hartree*27.211386245988-meta['excitation_ev'])
        hfile=folder/'analytic.npz'
        start=time.monotonic()
        if hfile.exists():
            with np.load(hfile) as f:
                assert str(f['request_hash'])==req;h=f['hessian'].copy();res=float(f['response_residual']);asym=float(f['asymmetry'])
        else:
            ah=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*central.occupations,mode='roks',spin_reference=ref,reference_result=central)
            h=ah.hessian_hartree_per_bohr2;res=ah.maximum_response_residual;asym=ah.raw_symmetry_error_hartree_per_bohr2
            np.savez_compressed(hfile,request_hash=req,hessian=h,response_residual=res,asymmetry=asym)
        hs.append(h)
        print(name,profile,'analytic Hessian ready',time.monotonic()-start,flush=True)
        # All columns of the nuclear Hessian; two steps reveal truncation errors.
        rows=[]
        for step in [2e-4,1e-4]:
            numeric=np.empty_like(h);grad_error=[]
            for column in range(xyz.size):
                v=np.zeros(xyz.size);v[column]=1.;points=[]
                for sign in [-1,1]:
                    geom=xyz+sign*step/ANGSTROM_TO_BOHR*v.reshape(-1,3)
                    dest=folder/f'fd_{step:.0e}_{column:03d}_{sign:+d}.npz';dr=key(dict(parent=req,step=step,column=column,sign=sign))
                    if dest.exists():point=load_derivative_point(dest,request_hash=dr,coordinates_angstrom=geom)
                    else:
                        point=evaluate_gfn2_roks_singlet(calc,geom,*central.occupations,spin_reference=ref,reference_result=central,gradient=True,**options)
                        save_derivative_point(dest,point,request_hash=dr,coordinates_angstrom=geom)
                    assert point.orbital_saddle_index==central.orbital_saddle_index
                    points.append(point)
                numeric[:,column]=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr).ravel()/(2*step)
                grad_error.append(abs((points[1].energy_hartree-points[0].energy_hartree)/(2*step)-central.gradient_hartree_per_bohr.ravel()[column]))
            np.savez_compressed(folder/f'fd_hessian_{step:.0e}.npz',hessian=numeric)
            rows.append(dict(step_bohr=step,gradient_max_error=max(grad_error),hessian_max_error=float(np.max(abs(h-numeric)))))
            print(name,profile,rows[-1],flush=True)
        # Proper rotation of the full molecular energy, Fock and analytical gradient.
        from scipy.linalg import expm
        rot=expm(np.array([[0.,-.19,.11],[.19,0.,-.07],[-.11,.07,0.]]));geom=xyz@rot.T+np.array([.1,-.2,.3])
        cctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
        p=central.densities();f,e=cctx.fock_and_energy(p)
        w=np.zeros_like(s.overlap);g=cctx.analytic_gradient(p,w)
        rotcalc=GFN2Calculator(atoms,temperature_kelvin=0.)
        rotctx=prepare_gfn2_deltascf(rotcalc,geom,spin_reference=ref)
        u=TransportedOperators(rotctx.state,s.coordinates_bohr,data['operators']).u
        pr=u@p@u.T;fr,er=rotctx.fock_and_energy(pr)
        gr=rotctx.analytic_gradient(pr,w)
        cov=dict(energy_error=abs(e-er),fock_error=float(np.max(abs(fr-u@f@u.T))),gradient_error=float(np.max(abs(gr-g@rot.T))))
        md=dict(profile=profile,anchor_energy_change_ev=energy_change,excitation_ev=central.excitation_energy_hartree*27.211386245988,
            saddle_index=central.orbital_saddle_index,orbital_residual=central.orbital_residual,
            response_residual=res,analytic_asymmetry=asym,derivative_checks=rows,rigid_rotation=cov,
            translation_gradient_error=float(np.max(abs(central.gradient_hartree_per_bohr.sum(axis=0)))))
        metadata.append(md);write(folder/'validation.json',md)
    combined=combine_roks_results(*components);mean_h=.5*(hs[0]+hs[1]);np.savez_compressed(out/'composite.npz',energy_hartree=combined.energy_hartree,
        excitation_hartree=combined.excitation_energy_hartree,gradient=combined.gradient_hartree_per_bohr,hessian=mean_h)
    numeric=[]
    for step in [2e-4,1e-4]:
        with np.load(out/'full_response'/f'fd_hessian_{step:.0e}.npz') as a,np.load(out/'qualified_spectral_response'/f'fd_hessian_{step:.0e}.npz') as b:
            numeric.append(dict(step_bohr=step,hessian_max_error=float(np.max(abs(mean_h-.5*(a['hessian']+b['hessian']))))))
    passed=all(m['anchor_energy_change_ev']<2e-7 and m['orbital_residual']<1e-9 and m['response_residual']<1e-9 and m['derivative_checks'][-1]['hessian_max_error']<3e-6
        and m['derivative_checks'][-1]['gradient_max_error']<2e-7 and max(m['rigid_rotation'].values())<2e-7 for m in metadata)
    record=dict(host=HOST,molecule=name,request_hash=request,components=metadata,component_csf_overlap=combined.component_csf_overlap,
        composite_excitation_ev=combined.excitation_energy_hartree*27.211386245988,composite_checks=numeric,passed=passed,
        scope='local frozen-parameter co-rotating PES; analytic nuclear derivatives, independent scalar/gradient finite-difference validation')
    write(out/'validation.json',record);print('COMPLETED',name,passed,flush=True);assert passed

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--molecule',required=True);args=ap.parse_args()
    with threadpool_limits(limits=2):run(args.molecule)
