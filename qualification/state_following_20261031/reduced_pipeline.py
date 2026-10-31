"""Serial 10-molecule/14-state continuation and GFN2-TDA campaign, no label fitting."""
from pathlib import Path
import os,socket,sys,json,hashlib,subprocess,signal,fcntl,datetime,argparse,traceback
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation';D.mkdir(exist_ok=True)
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def progress(stage,name=None,**extra):write(D/'progress.json',dict(host=socket.getfqdn(),stage=stage,molecule=name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
def selected():
 scope=read(R/'reduced_scope.json');rows=list(scope['primary_states']);manifest=read(P/'blind/manifest.json')['states'];rows.append(next(r for r in manifest if r['id']=='B039'));assert len(rows)==14 and len({r['molecule'] for r in rows})==10;return rows

def ledger():
 targets=selected();rows=[]
 for item in targets:
  f=D/'results'/item['molecule']/item['id']/'result.json';v=read(f) if f.exists() else dict(status='pending');rows.append(dict(item,result=v))
 complete=all(r['result']['status']=='matched' for r in rows)
 write(D/'ledger.json',dict(host=socket.getfqdn(),expected_states=14,recorded_states=sum(r['result']['status']!='pending' for r in rows),closed=complete,rows=rows,accuracy_scoring_performed=False))
 byid={r['id']:r['result']['status'] for r in rows}
 write(D/'scope_ledger.json',dict(original_manifest_sha256=sha(P/'blind/manifest.json'),selected_scope_sha256=sha(R/'reduced_scope.json'),rows=[dict(id=r['id'],molecule=r['molecule'],status=byid.get(r['id'],'deferred_user_scope_reduction')) for r in read(P/'blind/manifest.json')['states']],original_results_preserved=True))
 return complete

def launch(stage,name,args):
 folder=D/'runs'/name/stage;folder.mkdir(parents=True,exist_ok=True)
 with (folder/'worker.log').open('a') as log:
  proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(folder/'resources.txt'),sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  progress(stage,name,status='running',pid=proc.pid,numerical_reference_labels_read=False)
  try:rc=proc.wait(timeout=3600)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=-9
 if rc:write(folder/'failure.json',dict(returncode=rc,checkpoint_preserved=True))
 return rc,folder/'worker.log'

def adopt():
 oldaudit={r['state']:r for r in read(R/'recovered_assignment_audit.json')['rows']}
 for row in read(R/'consistent_reference/report.json')['rows']:
  name=row['molecule'];sid=row['state'];out=D/'results'/name/sid/'result.json'
  if not out.exists():
   result=read(Path(row['prediction_source']));result.update(adopted_completed_exploratory_result=True,adopted_source_sha256=sha(Path(row['prediction_source'])),selected_profile=row['selected_profile'])
   if 'assignment_audit' not in result:result['assignment_audit']=oldaudit[sid]
   write(out,result)
  prepared=D/'prepared'/name;prepared.parent.mkdir(exist_ok=True)
  if not prepared.exists():prepared.symlink_to(P/'blind/prepared'/name,target_is_directory=True)
  model=D/'models'/name/'model.npz';model.parent.mkdir(parents=True,exist_ok=True)
  if not model.exists():model.symlink_to(Path(row['selected_model']))
  choice={key:row[key] for key in ['molecule','original_canonical_gap_hartree','selected_profile','selected_model','selected_model_sha256','response_unchanged']};choice['reference_excitation_labels_read']=False
  f=D/'selection'/f'{name}.json'
  if f.exists():assert read(f)==choice
  else:write(f,choice)
 ledger()

def main(preflight=False):
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 files=[R/n for n in ['reduced_pipeline.py','reduced_prepare.py','reduced_reference.py','reduced_projected_model.py','reduced_spectral_diagnostic.py','reduced_worker.py','reduced_audit.py','reduced_tda.py','reduced_score.py','verify_tda_exchange.py','reference_eigensolver.py','checkpointed_reference.py','verify_reference_eigensolver.py']]+[P/n for n in ['blind_mouse.py','targeted_reference.py']]
 files+=sorted((R/'source/packages/matrix-gfn2/src/matrix_gfn2').glob('*.py'))
 plan=dict(schema='mouse.reduced.consistent.reference.v3',host=socket.getfqdn(),source_sha256={str(f):sha(f) for f in files},scope_sha256=sha(R/'reduced_scope.json'),manifest_sha256=sha(P/'blind/manifest.json'),assignment_sha256=sha(P/'blind/assignment_catalog.json'),states=[r['id'] for r in selected()],molecules=10,threads=2,concurrent_workers=1,timeout_per_stage_seconds=3600,weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,orbital_gap_scaling=1.,selection_rule='retain original qualified spectral operator if canonical occupied/virtual gap >0; otherwise complete projected DFT ground Fock; selection once per molecular frozen anchor, independent of target/error',numerical_criteria='unchanged public continuation, final joint residual/curvature, reference transition guard and available competing-root audit',reference_solver_validation_sha256=sha(R/'reference_solver_validation.json'),reference_solver='same exact PBE0 response, count/symmetry and 1e-7 residual; single-vector Davidson and 1e-18 linear dependence threshold; Ritz checkpoints each iteration and reuse completed symmetry roots',tda_exchange_validation_sha256=sha(R/'tda_exchange_validation.json'),tda='nonlocal exchange corrected and independently verified against PySCF A; same corrected GFN2 ground orbitals and full response for both parents; report parent TDA and their arithmetic mean as separate controls, not a coupled-composite linear-response identity',scoring='only after all 14 selected OO states are matched, all 10 TDA result files are finalized and the prediction lock is written; retain all 44 original dispositions and report 6 solver-development states separately from 8 remaining states; no post-score refitting',scope_is_reduced_after_baseline_outcomes=True,original_baseline_and_excluded_results_preserved=True)
 f=D/'protocol.json'
 if f.exists():assert read(f)==plan,'source/protocol changed; use a new campaign'
 else:write(f,plan)
 adopt()
 if preflight:
  for stage,args in [('prepare',[str(R/'reduced_prepare.py'),'Diazirine']),('select',[str(R/'reduced_reference.py'),'Diazirine']),('tda',[str(R/'reduced_tda.py'),'Diazirine'])]:
   rc,_=launch(stage,'Diazirine',args);assert rc==0,stage
  tda=read(D/'tda/Diazirine/result.json');assert all(x['status']=='matched' for x in tda['rows'])
  progress('preflight_passed','Diazirine',status='passed');return
 groups={}
 for row in selected():groups.setdefault(row['molecule'],[]).append(row)
 ordered=sorted(groups,key=lambda n:(not all((D/'results'/n/r['id']/'result.json').exists() for r in groups[n]),not (P/'blind/references'/n/'result.json').exists(),groups[n][0]['n_atoms'],n))
 for name in ordered:
  items=groups[name]
  if not all((D/'results'/name/r['id']/'result.json').exists() for r in items):
   if not (P/'blind/references'/name/'result.json').exists():
    rc,_=launch('reference',name,[str(R/'checkpointed_reference.py'),name]);assert rc==0 and (P/'blind/references'/name/'result.json').exists(),'reference requires repair: '+name
   rc,log=launch('prepare',name,[str(R/'reduced_prepare.py'),name])
   if rc:
    assert 'spectral transfer loses the native occupied/virtual gap' in log.read_text(),'preparation requires repair: '+name
    rc,_=launch('ordering_diagnostic',name,[str(R/'reduced_spectral_diagnostic.py'),name]);assert rc==0
   rc,_=launch('select',name,[str(R/'reduced_reference.py'),name]);assert rc==0
   case=read(D/'prepared'/name/'case.json');available={t['id'] for t in case['targets']}
   for r in items:
    assert r['id'] in available,'native preparation requires repair: '+r['id']
    out=D/'results'/name/r['id']/'result.json'
    if not out.exists():
     rc,_=launch(r['id'],name,[str(R/'reduced_worker.py'),name,r['id']]);assert rc==0
    rc,_=launch('audit',name,[str(R/'reduced_audit.py')]);assert rc==0
    assert read(out)['status']=='matched','state requires repair: '+r['id']
    ledger()
  if not (D/'tda'/name/'result.json').exists():
   rc,_=launch('tda',name,[str(R/'reduced_tda.py'),name]);assert rc==0,'TDA execution requires repair: '+name
  ledger()
 assert ledger()
 lockfiles=[D/'results'/r['molecule']/r['id']/'result.json' for r in selected()]+[D/'tda'/name/'result.json' for name in groups]
 write(D/'prediction_lock.json',dict(protocol_sha256=sha(D/'protocol.json'),files_sha256={str(f):sha(f) for f in lockfiles},all_selected_oo_states_matched=True,all_parent_tda_controls_finalized=True))
 rc,_=launch('scoring','all',[str(R/'reduced_score.py')]);assert rc==0,'scoring requires inspection'
 progress('complete',status='complete',selected_states=14,molecules=10,accuracy_scoring_performed=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 try:main(a.preflight)
 except Exception as exc:
  previous=read(D/'progress.json') if (D/'progress.json').exists() else {}
  write(D/'progress.json',dict(previous,status='needs_attention',reason=str(exc),checkpoint_preserved=True));print(traceback.format_exc(),flush=True);sys.exit(1)
