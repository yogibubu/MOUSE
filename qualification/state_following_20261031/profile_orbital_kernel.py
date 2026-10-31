"""Compare analytic kernel timings at a retained molecular checkpoint on Oracle."""
import os, socket, sys, pathlib, json, cProfile, pstats, time
assert socket.getfqdn() == 'oracle.sns.it'
R = pathlib.Path('/home/enzo/MOUSE-state-following-20261031')
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
os.environ['MOUSE_VALIDATION_ROOT'] = sys.argv[1] if len(sys.argv)>1 else str(R)
sys.path.insert(0, str(R/'source/qualification'))
import remote_bootstrap
import numpy as np
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator, FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

with threadpool_limits(limits=2):
    folder=R/'reduced_validation/results/Nitropyridine_N-oxide/B004'
    meta=json.loads((folder/'accepted.json').read_text())
    with np.load(folder/meta['checkpoint']) as f:
        xyz=f['coordinates_angstrom'].copy(); request=str(f['request_hash'])
    seed=load_derivative_point(folder/meta['checkpoint'], request_hash=request,
                               coordinates_angstrom=xyz, require_gradient=False)
    s=seed.reference_state
    calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers], temperature_kelvin=0.)
    ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
    p=pathlib.Path('/home/enzo/MOUSE-publication-validation-20261030/results')
    models=[]
    for profile in ['full_response','qualified_spectral_response']:
        with np.load(p/profile/'Nitropyridine_N-oxide/model.npz') as f:
            models.append({k:f[k].copy() for k in ['operators','charge','spin','linear','q0']})
    model=models[0];t=meta['fraction']
    model['linear']=(1-t)*model['linear']+t*models[1]['linear']
    ref=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,model,'performance-checkpoint')
    ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref)
    c=seed.common_coefficients; occ=seed.occupations
    triplet=seed.triplet_occupations
    left,right=np.triu_indices(len(c),1)
    active=np.any(np.vstack((occ,triplet))[:,left]!=np.vstack((occ,triplet))[:,right],axis=0)
    left,right=left[active],right[active]
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c
    state.alpha_occupations,state.beta_occupations=occ
    profiler=cProfile.Profile();start=time.monotonic();profiler.enable()
    kernel=ctx.density_response_kernel(state)
    result=orbital_residual_jacobian(c,occ,seed.fock_mixed,kernel,ctx.polarization,left,right)
    profiler.disable();elapsed=time.monotonic()-start
    out=pathlib.Path(os.environ['MOUSE_VALIDATION_ROOT'])/'performance_kernel';out.mkdir(exist_ok=True)
    np.save(out/'jacobian.npy',result)
    with (out/'profile.txt').open('w') as f:
        pstats.Stats(profiler,stream=f).strip_dirs().sort_stats('cumulative').print_stats(25)
    report=dict(host=socket.getfqdn(),elapsed_seconds=elapsed,rotations=len(left),basis=len(c),source_root=os.environ['MOUSE_VALIDATION_ROOT'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
