"""Resume analytic columns with at most twelve two-thread remote processes."""
from pathlib import Path
import sys,subprocess,concurrent.futures,socket,json,hashlib,runpy
assert socket.getfqdn()=='oracle.sns.it'
HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
runpy.run_path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/source/qualification/remote_bootstrap.py')
import matrix_gfn2
source_root=Path(matrix_gfn2.__file__).parent
if source_root.exists():
 files=[p for p in source_root.glob('*.py') if p.name.startswith('_nosi') or p.name in ['_mixed_jet.py','_einsum_cached.py','nosi_surface.py']]
 (HERE/'compiled_runtime_source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
cases=[('independent','Criegee_Intermediate',15),('last_two','Furan',27),('last_two','uracil',36)]
jobs=[(cohort,name,nr,column) for column in range(36) for cohort,name,nr in cases if column<nr]
def run(job):
 cohort,name,nr,column=job;destination=HERE/'analytic'/cohort/name;path=destination/f'column_{column:03d}.npz'
 if path.exists() and path.with_suffix('.json').exists():return
 with (HERE/f'column_{name}_{column:03d}.log').open('w') as log:subprocess.run([sys.executable,str(HERE/'hessian_columns.py'),cohort,name,'--worker',str(column),'--workers',str(nr)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,column,'complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:list(executor.map(run,jobs))
