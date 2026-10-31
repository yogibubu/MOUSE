"""Complete second-molecule analytic Hessian qualification using public API.

Oracle only. Each independently reoptimized displacement is a checkpoint.
This verifies derivatives of a frozen model, not physical frequency accuracy.
"""
from pathlib import Path
import os,sys,socket,json,time,hashlib,fcntl,resource
assert socket.getfqdn()=='oracle.sns.it'
W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
os.environ.update(MOUSE_VALIDATION_ROOT=str(W),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(W/'source/qualification'))
import remote_bootstrap
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import (MultiCSFModel,GFN2Calculator,evaluate_multicsf_surface,
                         save_multicsf_result,load_multicsf_result)
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

with threadpool_limits(limits=2):
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    model=MultiCSFModel(W/'models/Cyclopentadiene/B016/model.json')
    folder=W/'cpd_hessian';folder.mkdir(exist_ok=True)
    protocol=dict(schema='mouse.public.multicsf.hessian.v1',host=socket.getfqdn(),model_fingerprint=model.fingerprint,
                  molecule='Cyclopentadiene',state='B016',rank=model.rank,threads=2,
                  columns=list(range(model.xyz.size)),step_bohr=2e-4,
                  maximum_hessian_error=2e-5,maximum_gradient_error=2e-6,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='full total excited-energy analytic Hessian of fixed portable model; no physical frequency accuracy claim')
    path=folder/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:write(path,protocol)
    calc=GFN2Calculator([{1:'H',6:'C'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.)
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC call'))
    start=time.monotonic();anchor=folder/'anchor'
    if anchor.exists():point=load_multicsf_result(anchor/'result.json',model,calc)
    else:
        point=evaluate_multicsf_surface(calc,model.xyz,model,compute_hessian=True)
        save_multicsf_result(anchor,point,model)
        write(folder/'analytic_cost.json',dict(seconds=time.monotonic()-start,
              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              cold=True,stored_hessian_reused=False))
    assert point.hessian_hartree_per_bohr2 is not None
    reportpath=folder/'report.json'
    report=json.loads(reportpath.read_text()) if reportpath.exists() else dict(protocol=protocol,checks=[])
    done={r['coordinate'] for r in report['checks']}
    for column in protocol['columns']:
        if column in done:continue
        values=[];gradients=[];costs=[]
        for sign in [-1,1]:
            cache=folder/f'column_{column}_{sign}.npz'
            x=model.xyz.copy();x.ravel()[column]+=sign*protocol['step_bohr']/ANGSTROM_TO_BOHR
            if cache.exists():
                with np.load(cache) as a:
                    assert np.array_equal(a['coordinates'],x)
                    values.append(float(a['energy']));gradients.append(a['gradient'].copy());costs.append(float(a['seconds']))
            else:
                started=time.monotonic()
                q=evaluate_multicsf_surface(calc,x,model,reference_result=point,compute_hessian=False)
                seconds=time.monotonic()-started
                np.savez_compressed(cache,coordinates=x,energy=q.energy_hartree,
                                    gradient=q.gradient_hartree_per_bohr,seconds=seconds,
                                    residual=q.orbital_residual,fidelity=q.fidelity)
                values.append(q.energy_hartree);gradients.append(q.gradient_hartree_per_bohr);costs.append(seconds)
        step=protocol['step_bohr']
        row=dict(coordinate=column,energy_gradient_error=float(abs((values[1]-values[0])/(2*step)-point.gradient_hartree_per_bohr.ravel()[column])),
                 hessian_column_max_error=float(np.max(abs((gradients[1]-gradients[0]).ravel()/(2*step)-point.hessian_hartree_per_bohr2[:,column]))),
                 displacement_seconds=costs)
        report['checks'].append(row)
        report.update(complete=len(report['checks'])==model.xyz.size,
                      maximum_hessian_column_error=max(r['hessian_column_max_error'] for r in report['checks']),
                      maximum_energy_gradient_error=max(r['energy_gradient_error'] for r in report['checks']),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(reportpath,report);print(row,flush=True)
        assert row['energy_gradient_error']<protocol['maximum_gradient_error']
        assert row['hessian_column_max_error']<protocol['maximum_hessian_error']
    print('COMPLETE',report['maximum_hessian_column_error'],flush=True)
