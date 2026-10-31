from remote_bootstrap import HOST
from benchmark import BASE,write
import numpy as np,json
records=[]
for name,target,qualified in [('uracil_s0','n_pi',False),('uracil_s0','pi_pi',False),('uracil_n_pi','n_pi',False),('uracil_pi_pi','pi_pi',False),('uridine_s0','pi_pi',True)]:
 out=BASE/'results'/name
 if qualified:out=out/'qualified'
 with np.load(out/f'roks_hessian_{target}.npz') as f:h=f['orbital_hessian']
 values=np.linalg.eigvalsh(h)
 records.append(dict(name=name,target=target,index=int(np.count_nonzero(values<-1e-7)),minimum=float(values[0]),near_zero=int(np.count_nonzero(abs(values)<1e-7)),first=values[:8]))
write(BASE/'orbital_indices.json',records)
print(json.dumps(records,default=lambda a:a.tolist(),indent=2))
