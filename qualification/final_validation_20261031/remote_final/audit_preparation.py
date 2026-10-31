"""Read-only accounting of preparation and whole-worker resource measurements."""
from pathlib import Path
import json,re,socket,hashlib
import numpy as np
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent;rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text()) if p.exists() else None
def resource(p):
 if not p.exists():return None
 text=p.read_text();d={}
 for label,key in [('User time (seconds)','cpu_user_seconds'),('System time (seconds)','cpu_system_seconds'),('Maximum resident set size (kbytes)','peak_rss_kib')]:
  m=re.search(re.escape(label)+r':\s*([0-9.]+)',text)
  if m:d[key]=float(m.group(1))
 m=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)',text)
 if m:
  parts=list(map(float,m.group(1).strip().split(':')));d['elapsed_including_queue_seconds']=sum(v*60**i for i,v in enumerate(parts[::-1]))
 return d
for subset in ['independent','independent_challenge']:
 root=R/subset;manifest=load(root/'manifest.json')
 if manifest is None:continue
 for name in dict.fromkeys(r['molecule'] for r in manifest['states']):
  out=root/'predictions'/name;execution=root/'executions'/f'{name}.json';final=out/'pass_complete.json';reference=root/'references'/name;calibration=out/'full_response/anchor/calibration.json'
  resources=R/f'tasks/independent_{name}/resources.txt'
  if name=='Ammonia':resources=R/'independent_ammonia_recovery.resources'
  if subset=='independent_challenge':resources=R/'challenge.resources'
  native=out/'native_ground.npz';compact_nao=None
  if native.exists():
   with np.load(native) as a:compact_nao=int(a['state_overlap'].shape[0])
  row=dict(molecule=name,subset=subset,compact_nao=compact_nao,complete=final.exists(),whole_process_resources=resource(resources),df_t_ground_cost=load(reference/'ground_cost.json'),reference_tda_costs=[load(p) for p in reference.glob('*_cost.json') if p.name!='ground_cost.json'],full_response_calibration=load(calibration),reference_labels_used=False)
  if execution.exists() and final.exists():
   row['wall_after_lock_and_import_seconds']=final.stat().st_mtime-execution.stat().st_mtime
   row['clock_scope']='filesystem timestamps of execution record immediately after compute-lock acquisition and final prediction ledger; excludes import and queue time'
   row['dft_ground_reused_from_earlier_attempt']=(reference/'reference.npz').stat().st_mtime<execution.stat().st_mtime
  rows.append(row)
legacy=[]
OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
for name in ['nitroxyl','acetaldehyde','acetone','Furan','Cyclopentadiene','Phthalimide']:
 full=(OLD if name.islower() else P)/'results/full_response'/name/'calibration.json';dft=P/'blind/references'/name/'ground_result.json'
 legacy.append(dict(molecule=name,full_response_calibration=load(full),dft_ground=load(dft),total_end_to_end_preparation_seconds=None,limitation='archived checkpoints reused; missing whole-preparation timing is unknown, never zero; calibration timings may originate from earlier runtime/threads and are not rescaled'))
report=dict(whole_prediction_workers=rows,existing_cost_cohort_preparation=legacy,accounting='Whole-process RSS includes reference preparation; per-property prepared-surface timings exclude it and must be shown separately. Queue wait is not electronic calculation time.',scope='same common DFT/FQ preparation may be shared; OO reference-transition preparation is an additional stage compared with energy-only TDA-Q')
(R/'preparation_cost_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('Audited preparation for',len(rows),'new molecular workers; no absent timing replaced by zero')
