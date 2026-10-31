from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
protocol=json.loads((BASE/'protocol.json').read_text());rows=[];excluded=[]
for case in protocol['cases']:
 cp=BASE/'prepared'/case['name']/'case.json'
 if not cp.exists():continue
 prepared=json.loads(cp.read_text());excluded.extend(dict(molecule=case['name'],**e) for e in prepared['excluded'])
 for t in prepared['targets']:
  path=BASE/'results/predictions'/case['name']/'shell_charge_spin'/t['id']/'result.json'
  if not path.exists():continue
  row=json.loads(path.read_text())
  if row['qualified']:rows.append(dict(row,native_roks_ev=t['native_roks_ev']))
def metric(rr,key):
 vals=[r[key]-r['tbe_ev'] for r in rr]
 return dict(count=len(vals),mae_ev=sum(map(abs,vals))/len(vals),maxabs_ev=max(map(abs,vals)),mean_signed_ev=sum(vals)/len(vals)) if vals else None
stats={}
for family in ['all','n_pi','pi_pi']:
 rr=[r for r in rows if family=='all' or r['state']==family]
 paired=[dict(r,tda_ev=r['tda']['energy_ev']) for r in rr if r['tda']['qualified']]
 stats[family]=dict(native=metric(rr,'native_roks_ev'),corrected=metric(rr,'excitation_ev'),
  tda_paired=metric(paired,'tda_ev'),corrected_tda_paired=metric(paired,'excitation_ev'))
result=dict(host=HOST,protocol_sha256=digest(BASE/'protocol.json'),rows=rows,exclusions=excluded,
 qualified_molecules=len(set(r['molecule'] for r in rows)),qualified_states=len(rows),planned_molecules=len(protocol['cases']),planned_states=sum(len(c['targets']) for c in protocol['cases']),
 statistics=stats,energy_offset=0.,default_changed=False,
 tensor_development='rejected: HNO and uracil npi worsened; no tensor holdout tuning',
 fock_shell_development='rejected: HNO and uracil worsened; full Fock residual too large',
 universal_npi_correction_demonstrated=False)
write(BASE/'results/cohort_summary.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['rows']},indent=2))
