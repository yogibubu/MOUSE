"""Oracle-only constrained 50/50 excited-parent mean response check.

Does not validate the full excitation PES ground-reference subtraction, nor
singular unit-overlap constraints. Earlier parent checkpoints are reused.
"""
from parent_geometry_model import ParentGeometryModel,V
import socket,json,time,fcntl
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from multicsf_nuclear_response import nuclear_response,parent_gradient
from multicsf_relative_response import anchor_overlaps,constraint_jet
from multicsf_frame_response import RelativeActiveFrame

with threadpool_limits(limits=2):
    compute=(V.parent/'MOUSE-publication-validation-20261030/blind/references/compute.lock').open('a');fcntl.flock(compute,fcntl.LOCK_EX|fcntl.LOCK_NB)
    models=[ParentGeometryModel('Furan','B022',i,str(i)) for i in range(2)]
    pairs=[m.objective(m.xyz) for m in models];objects=[p[0] for p in pairs];coefficients=[p[1] for p in pairs]
    target=anchor_overlaps(objects,coefficients);q,j,qh,_=constraint_jet(objects,coefficients,target)
    # Independent finite rotations check the complete constraint Hessian.
    rng=np.random.default_rng(311);d=rng.normal(size=j.shape[1]);d/=np.linalg.norm(d);eps=2e-4;vals=[];cursor=0;rotations=[]
    for o,c in zip(objects,coefficients):
        k=np.zeros_like(c);k[o.left,o.right]=d[cursor:cursor+len(o.left)];k[o.right,o.left]=-k[o.left,o.right];rotations.append(k);cursor+=len(o.left)
    for sign in [-1,1]:vals.append(constraint_jet(objects,[c@expm(sign*eps*k) for c,k in zip(coefficients,rotations)],target)[0])
    constraint_first=float(np.max(abs((vals[1]-vals[0])/(2*eps)-j@d)))
    constraint_second=float(np.max(abs((vals[1]+vals[0]-2*q)/eps**2-np.einsum('i,aij,j->a',d,qh,d))))
    assert constraint_first<1e-7 and constraint_second<1e-6
    out=V/'pi_pi_repair/nuclear_validation';start=time.monotonic();responses=[]
    # Reuse the completed certified first-parent Hessian; its energy is checked.
    with np.load(out/'Furan_full_analytic.npz') as a:first={k:a[k].copy() for k in a.files}
    assert abs(objects[0].evaluate(coefficients[0],True)[0]-float(first['energy']))<1e-11
    responses.append(first)
    qcache=out/'Furan_Q_analytic.npz'
    if qcache.exists():
        with np.load(qcache) as a:second={k:a[k].copy() for k in a.files}
        assert abs(objects[1].evaluate(coefficients[1],True)[0]-float(second['energy']))<1e-11
    else:
        second=nuclear_response(objects[1],coefficients[1],models[1].eri);np.savez_compressed(qcache,**second)
    responses.append(second)
    frame=RelativeActiveFrame(objects)
    response=frame.anchor_response(coefficients,responses)
    np.savez_compressed(out/'Furan_frame_parent_mean_analytic.npz',**response)
    report=dict(host=socket.getfqdn(),scope='constrained excited parent total-energy 50/50 mean only; full excitation PES ground terms not yet qualified',target_overlaps=target.tolist(),relative_manifold='full active cross-overlap matrix preserved exactly',constraint_gradient_error=constraint_first,constraint_hessian_error=constraint_second,analytic_seconds=time.monotonic()-start,maximum_response_residual=response['maximum_response_residual'],minimum_absolute_curvature=response['minimum_absolute_curvature'],dimension=response['dimension'],checks=[])
    for coordinate in [0,8]:
        step=2e-4;values=[];gradients=[]
        for sign in [-1,1]:
            xyz=models[0].xyz.copy();xyz.ravel()[coordinate]+=sign*step/1.8897261254578281
            pairs=[m.objective(xyz) for m in models];oo=[p[0] for p in pairs];cc=[p[1] for p in pairs]
            cc=RelativeActiveFrame(oo).polish(cc)
            values.append(sum(.5*o.evaluate(c,False)[0] for o,c in zip(oo,cc)))
            gradients.append(sum(.5*parent_gradient(o,c,m.eri).ravel() for o,c,m in zip(oo,cc,models)))
        row=dict(coordinate=coordinate,energy_gradient_error=abs((values[1]-values[0])/(2*step)-response['gradient'].ravel()[coordinate]),hessian_column_max_error=float(np.max(abs((gradients[1]-gradients[0])/(2*step)-response['hessian'][:,coordinate]))));report['checks'].append(row)
        (out/'Furan_frame_parent_mean_report.json').write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True)
        assert row['energy_gradient_error']<2e-6 and row['hessian_column_max_error']<2e-5
    print(json.dumps(report,indent=2),flush=True)
