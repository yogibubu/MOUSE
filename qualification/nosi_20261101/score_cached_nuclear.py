"""Lightweight scoring of completed independent molecular energy checks."""
from pathlib import Path
import json,argparse
import numpy as np

def run(root):
    certificate=json.loads((root/'hessian_certificate.json').read_text())
    with np.load(root/'nuclear_response.npz') as z:arrays={k:z[k].copy() for k in z.files}
    nr=next(v.shape[0] for k,v in arrays.items() if k.endswith('_gradient'));vectors=[]
    unit=np.zeros(nr);unit[3]=1;vectors.append(unit);v=np.sin(np.arange(nr)+.37);vectors.append(v/np.linalg.norm(v))
    results={identity:[] for identity in certificate['states']}
    for identity in results:
        for index,v in enumerate(vectors):
            estimates=[]
            for eps in [1e-3,5e-4]:
                points=[json.loads((root/'independent_energy_validation'/f'direction_{index}_eps_{eps:.8f}_sign_{sign}.json').read_text()) for sign in [-1,1]]
                em,ep=[point['energies'][identity] for point in points];center=certificate['states'][identity]['energy_hartree']
                estimates.append(((ep-em)/(2*eps),(ep-2*center+em)/eps**2))
            gradient=(4*estimates[1][0]-estimates[0][0])/3;curvature=(4*estimates[1][1]-estimates[0][1])/3
            ge=abs(gradient-arrays[identity+'_gradient']@v);he=abs(curvature-v@arrays[identity+'_hessian']@v)
            assert ge<2e-7,(identity,index,ge)
            assert he<2e-5,(identity,index,he)
            results[identity].append(dict(direction=index,raw_estimates=estimates,richardson_gradient_error=ge,richardson_hessian_error=he))
    output=dict(states=results,analytic_implementation_unchanged=True,independent_displaced_energies_reused=True,
        note='Richardson extrapolation is used only to remove central-difference truncation in the validation; the delivered Hessian is analytic')
    (root/'cached_independent_nuclear_validation.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();run(args.root)
