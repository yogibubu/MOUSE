"""Resume completed columns; schedule only missing analytic columns remotely."""
from pathlib import Path
import os,sys,socket,json,time,subprocess,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');assert socket.getfqdn()=='oracle.sns.it'
cohort,name=sys.argv[1:3];folder=ROOT/'selected_cid_analytic'/cohort/name;start=time.monotonic();progress=folder/'hessian_progress.npz'
if progress.exists():
 with np.load(progress) as z:
    completed=int(z['completed']);ground=z['ground'].copy();nr=len(ground);identities=[x[:-8] for x in z.files if x.endswith('_hessian')]
    energies={i:float(z[i+'_energy']) for i in identities};gradients={i:z[i+'_gradient'].copy() for i in identities};hessians={i:z[i+'_hessian'].copy() for i in identities}
else:
 meta=json.loads((folder/'column_000.json').read_text());identities=list(meta['states']);completed=1
 with np.load(folder/'column_000.npz') as z:
    nr=len(z['ground_gradient']);ground=np.full((nr,nr),np.nan);ground[:,0]=z['ground_hessian_column']
    energies={i:float(meta['states'][i]['energy_hartree']) for i in identities};gradients={i:z[i+'_gradient'].copy() for i in identities};hessians={i:np.full((nr,nr),np.nan) for i in identities}
    for i in identities:hessians[i][:,0]=z[i+'_hessian_column']
source_hash=hashlib.sha256((ROOT/'cid_nuclear.py').read_bytes()).hexdigest()
missing=[j for j in range(completed,nr) if not (folder/f'column_{j:03d}.npz').exists()]
def execute(j):
    with (folder/f'column_{j:03d}.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'cid_nuclear.py'),cohort,name,'--column',str(j)],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
    print('completed',j,flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(execute,missing))
for j in range(completed,nr):
    meta=json.loads((folder/f'column_{j:03d}.json').read_text());assert meta['source_sha256']==source_hash
    with np.load(folder/f'column_{j:03d}.npz') as z:
        ground[:,j]=z['ground_hessian_column']
        for i in identities:
            assert np.max(abs(gradients[i]-z[i+'_gradient']))<1e-9
            hessians[i][:,j]=z[i+'_hessian_column']
translation=np.tile(np.eye(3),(nr//3,1));states={}
for i,h in hessians.items():
    asym=float(np.max(abs(h-h.T)));tr=float(np.max(abs(h@translation)));assert asym<2e-6 and tr<2e-6,(i,asym,tr)
    hessians[i]=.5*(h+h.T);states[i]=dict(energy_hartree=energies[i],raw_symmetry_error=asym,translation_error=tr)
np.savez_compressed(folder/'nuclear_response.npz',ground_hessian=ground,**{i+'_gradient':v for i,v in gradients.items()},**{i+'_hessian':v for i,v in hessians.items()})
(folder/'hessian_certificate.json').write_text(json.dumps(dict(host=socket.getfqdn(),states=states,wall_seconds=time.monotonic()-start,source_sha256=source_hash,parallel_workers=8,BLAS_threads_per_worker=2,preserved_sequential_columns=completed,parallel_missing_columns=missing,scope='parallel completion wall time; initial sequential prefix recorded separately'),indent=2)+'\n')
