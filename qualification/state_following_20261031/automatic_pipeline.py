"""Serial, checkpoint-preserving v2 benchmark; original prediction ledger retained."""
from pathlib import Path
import os,socket,sys,json,hashlib,ast,subprocess,signal,fcntl,datetime,argparse
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'automatic_v2'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(name):
 # Execute the original label-free reference/seed/F preparation. Replace only
 # the guessed-index spectral search and final old prediction loop by a pure
 # calibration of the exact same Q endpoint. Existing cached F work is reused.
 sys.path.insert(0,str(P));import blind_mouse as original
 tree=ast.parse((P/'blind_mouse.py').read_text())
 worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
 block=worker.body[-1];assert isinstance(block,ast.With)
 cut=next(i for i,n in enumerate(block.body) if isinstance(n,ast.ImportFrom) and n.module=='matrix_gfn2.relative_pair_roks')
 tail=block.body[cut-1];assert 'full.run(case)' in ast.unparse(tail)
 text=ast.unparse(tail).replace('full.run(case)', '''parent, xyz = state_from(case['targets'][0])
   s = parent.reference_state
   calc = GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
   ctx = prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
   spectral['calibrate'](case,s,ctx)''')
 # Parse replacement with no artificial indentation from the string template.
 import textwrap
 text=text.replace('\n   ','\n')
 block.body=block.body[:cut-1]+ast.parse(text).body
 ast.fix_missing_locations(worker)
 env=original.__dict__
 exec(compile(ast.Module(body=[worker],type_ignores=[]),str(P/'blind_mouse.py'),'exec'),env)
 env['worker'](name)
def launch(kind,name,sid=None):
 out=D/'runs'/name/(sid or kind);out.mkdir(parents=True,exist_ok=True)
 if kind=='reference':args=[str(P/'targeted_reference.py'),name]
 elif kind=='prepare':args=[str(R/'automatic_pipeline.py'),'--prepare',name]
 elif kind=='state':args=[str(R/'automatic_worker.py'),name,sid]
 else:args=[str(R/'automatic_audit.py')]
 with (out/'worker.log').open('a') as log:
  proc=subprocess.Popen([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(D/'progress.json',dict(host=socket.getfqdn(),status='running',stage=kind,molecule=name,state=sid,pid=proc.pid,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),numerical_labels_read=False))
  try:rc=proc.wait(timeout=3600)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=-9
 if rc:write(out/'failure.json',dict(returncode=rc,checkpoint_preserved=True))
 if kind=='state' and rc:
  write(D/'results'/name/sid/'result.json',dict(molecule=name,state=sid,status='timeout' if rc==-9 else 'execution_error',returncode=rc,checkpoint_preserved=True))
 return rc

def main(ready_only=False):
 D.mkdir(exist_ok=True)
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sources=[R/n for n in ['automatic_pipeline.py','automatic_worker.py','automatic_audit.py']]+[P/n for n in ['blind_mouse.py','targeted_reference.py']]
 sources+=sorted((R/'source/packages/matrix-gfn2/src/matrix_gfn2').glob('*.py'))
 protocol=dict(schema='mouse.automatic.continuation.v2',host=socket.getfqdn(),source_sha256={str(f):sha(f) for f in sources},manifest_sha256=sha(P/'blind/manifest.json'),assignment_sha256=sha(P/'blind/assignment_catalog.json'),weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,threads=2,concurrent_workers=1,seconds_per_stage=3600,minimum_projection=.7,minimum_single_pair_amplitude=.7,minimum_single_pair_separation=.05,minimum_transition_fidelity=.9,minimum_assignment_margin=.05,original_baseline_preserved=True,previous_five_exploratory_cases_reused_without_recalculation=True,reference_labels_read=False,controls='public continuation API defaults; final joint residual <=1e-8 and absolute curvature >=1e-7; target identity checked after joint solve against all available converged same-symmetry roots; incomplete coverage disclosed',preparation='original native seed and full-response search; Q calibration only, replacing fixed-index Q search; targeted reference recovery reuses ground checkpoints')
 f=D/'protocol.json'
 if f.exists():assert read(f)==protocol,'Protocol/source changed; new campaign directory required'
 else:write(f,protocol)
 manifest=read(P/'blind/manifest.json');groups={}
 for row in manifest['states']:groups.setdefault(row['molecule'],[]).append(row)
 oldaudit={r['state']:r for r in read(R/'recovered_assignment_audit.json')['rows']}
 for name,items in sorted(groups.items(),key=lambda x:(not (P/'blind/references'/x[0]/'result.json').exists(),x[1][0]['n_atoms'],x[0])):
  for row in items:
   sid=row['id'];old=R/'recovery_rank_one'/name/sid/'result.json';dest=D/'results'/name/sid/'result.json'
   if old.exists() and not dest.exists():
    v=read(old);v['adopted_completed_exploratory_result']=True;v['assignment_audit']=oldaudit[sid];v['status']='matched' if oldaudit[sid]['uniquely_assigned'] else 'assignment_ambiguous';v['adopted_source_sha256']=sha(old);write(dest,v)
  if all((D/'results'/name/r['id']/'result.json').exists() for r in items):continue
  reference=P/'blind/references'/name/'result.json'
  if not reference.exists():
   if ready_only:continue
   launch('reference',name)
  if not reference.exists():
   for r in items:write(D/'results'/name/r['id']/'result.json',dict(molecule=name,state=r['id'],status='reference_unavailable',checkpoint_preserved=True))
   continue
  if launch('prepare',name):raise RuntimeError('Preparation failed; preserved checkpoints: '+name)
  assignment=P/'blind/mouse'/name/'reference_assignment.json'
  for row in read(assignment).get('unassigned',[]):write(D/'results'/name/row['id']/'result.json',dict(row,molecule=name,state=row['id']))
  casepath=P/'blind/prepared'/name/'case.json'
  if not casepath.exists():continue
  case=read(casepath)
  for excluded in case['excluded']:
   sid=excluded['state'];write(D/'results'/name/sid/'result.json',dict(molecule=name,state=sid,status='not_representable' if any(s in excluded['error'] for s in ['compact projection','one compact orbital promotion']) else 'native_seed_failure',reason=excluded['error']))
  for target in case['targets']:
   if not (D/'results'/name/target['id']/'result.json').exists():launch('state',name,target['id'])
  if launch('audit',name):raise RuntimeError('Assignment audit failed: '+name)
 rows=[read(f) for f in sorted((D/'results').glob('*/*/result.json'))]
 write(D/'ledger.json',dict(host=socket.getfqdn(),expected_states=len(manifest['states']),recorded_states=len(rows),rows=rows,reference_labels_read=False,accuracy_scoring_performed=False))
 write(D/'progress.json',dict(host=socket.getfqdn(),status='ready_pass_complete' if ready_only else 'pass_complete',recorded_states=len(rows),expected_states=len(manifest['states']),reference_labels_read=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare');ap.add_argument('--ready-only',action='store_true');a=ap.parse_args()
 if a.prepare:prepare(a.prepare)
 else:main(a.ready_only)
