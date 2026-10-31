"""Retain failed directional audit and check finite-difference step convergence."""
from dft_relaxation import Objective,OUT,HOST,digest,write
import numpy as np,argparse,json

p=argparse.ArgumentParser();p.add_argument('--molecule',required=True);p.add_argument('--state',required=True);a=p.parse_args()
obj=Objective(a.molecule,a.state);folder=OUT/a.molecule/a.state
old=json.loads((folder/'derivative_audit.json').read_text())
assert old['request_hash']==obj.request
write(folder/'derivative_audit_initial.json',old)
c=obj.C0.copy();v=obj.evaluate(c)
rng=np.random.default_rng(20261030);x=rng.normal(size=obj.size);x/=np.linalg.norm(x)
hv,_,_=obj.response(v);analytic=hv(x);rows=[]
for h in [1e-4,5e-5]:
 vp=obj.evaluate(obj.rotate(c,h*x));vm=obj.evaluate(obj.rotate(c,-h*x))
 num=(vp['g']-vm['g'])/(2*h)
 row=dict(step=h,energy_direction_error=float(abs((vp['energy']-vm['energy'])/(2*h)-v['g']@x)),
  response_max_error=float(np.max(abs(analytic-num))),
  response_relative_error=float(np.linalg.norm(analytic-num)/np.linalg.norm(num)))
 row['passed']=row['energy_direction_error']<2e-7 and row['response_max_error']<2e-6 and row['response_relative_error']<2e-5
 rows.append(row);print(row,flush=True)
data=dict(host=HOST,request_hash=obj.request,initial_check=old,refinements=rows,
 passed=all(r['passed'] for r in rows),script_sha256=digest(__file__))
write(folder/'derivative_audit.json',data)
assert data['passed'],'directional refinement still fails'
