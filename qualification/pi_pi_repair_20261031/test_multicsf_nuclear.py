"""Oracle molecular analytic nuclear gradient and relaxed Hessian checks."""
from parent_geometry_model import ParentGeometryModel,V
import socket,json,time,fcntl
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from multicsf_nuclear_response import nuclear_response,parent_gradient

def polish(obj,c):
    for i in range(12):
        e,g,h=obj.evaluate(c,True)
        if np.max(abs(g))<1e-10:return c
        step=np.linalg.solve(h,-g);step*=min(1.,.05/max(np.linalg.norm(step),1e-300));accepted=False
        for fraction in [1.,.5,.25,.125]:
            k=np.zeros_like(c);k[obj.left,obj.right]=fraction*step;k[obj.right,obj.left]=-fraction*step;trial=c@expm(k);_,gg,_=obj.evaluate(trial,False)
            if np.linalg.norm(gg)<np.linalg.norm(g):c=trial;accepted=True;break
        if not accepted:raise RuntimeError('displaced stationary solve failed')
    raise RuntimeError('displaced stationary solve budget')

with threadpool_limits(limits=2):
    lock=(V/'pi_pi_repair/nuclear_qualification.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Also serialize against every electronic campaign worker.
    compute=(V.parent/'MOUSE-publication-validation-20261030/blind/references/compute.lock').open('a');fcntl.flock(compute,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=ParentGeometryModel('Furan','B022',0,'0');obj,c=model.objective(model.xyz);c=polish(obj,c);start=time.monotonic()
    response=nuclear_response(obj,c,model.eri);grad=parent_gradient(obj,c,model.eri)
    assert np.max(abs(grad-response['gradient']))<1e-8
    out=V/'pi_pi_repair/nuclear_validation';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'Furan_full_analytic.npz',**response)
    report=dict(host=socket.getfqdn(),method='single-parent multi-CSF; analytic nuclear/orbital/CI response; not the constrained 50/50 Hessian',maximum_response_residual=response['maximum_response_residual'],gradient_implementation_difference=float(np.max(abs(grad-response['gradient']))),analytic_seconds=time.monotonic()-start,checks=[])
    for coordinate in [0,8]:
        h=2e-4;values=[];gradients=[]
        for sign in [-1,1]:
            xyz=model.xyz.copy();xyz.ravel()[coordinate]+=sign*h/1.8897261254578281
            displaced,initial=model.objective(xyz);point=polish(displaced,initial);values.append(displaced.evaluate(point,False)[0]);gradients.append(parent_gradient(displaced,point,model.eri).ravel())
        first=(values[1]-values[0])/(2*h);column=(gradients[1]-gradients[0])/(2*h)
        row=dict(coordinate=coordinate,energy_gradient_error=abs(first-grad.ravel()[coordinate]),hessian_column_max_error=float(np.max(abs(column-response['hessian'][:,coordinate]))));report['checks'].append(row)
        (out/'Furan_full_report.json').write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True)
        assert row['energy_gradient_error']<2e-6 and row['hessian_column_max_error']<2e-5,row
    print(json.dumps(report,indent=2),flush=True)
