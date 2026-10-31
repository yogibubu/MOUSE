from remote_bootstrap import HOST
from benchmark import BASE,characterize,write,EV
import numpy as np,json
from matrix_gfn2.deltascf_derivatives import load_derivative_point
records=[]
name='uridine_s0';out=BASE/'results'/name;d=json.loads((out/'summary.json').read_text());x=np.array(d['input']['coordinates_angstrom']);gs=load_derivative_point(out/'ground.npz',request_hash=name,coordinates_angstrom=x);s=gs.reference_state
for folder in sorted(out.glob('gmf_*')):
 path=folder/'latest.npz'
 if not path.exists():continue
 with np.load(path) as f:c=f['coefficients'];occ=f['occupations']
 p,n,low,_=characterize(s,d['input']['base_atoms'],c);hi=int(np.flatnonzero(occ[1]-occ[0]==1)[0]);pa=int(np.flatnonzero(occ[0]-occ[1]==1)[0]);ids=[hi,pa];block=s.coefficients[:,ids].T@s.overlap@c[:,ids];ov=abs(block[0,0]*block[1,1]+block[0,1]*block[1,0]);pop=np.sum(low[np.isin(s.ao_atom,[i for i in range(len(x)) if i not in d['input']['base_atoms']])]**2,axis=0)
 hist=json.loads((folder/'history.json').read_text())[-1]
 records.append(dict(folder=folder.name,hole_pi=p[hi],particle_pi=p[pa],hole_n=n[hi],active_pair_overlap=ov,sugar_hole=pop[hi],sugar_particle=pop[pa],last=hist))
write(BASE/'gmf_character.json',records);print(json.dumps(records,indent=2))
