"""Close the resumed experiment without selecting a profile on C."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,state_from
from matrix_gfn2.deltascf_derivatives import load_derivative_point
R=BASE/'results';Q=R/'qualified_spectral_response'
checks=[];crows=[]
for group,prepared,names in [('B','prepared_B',['acetaldehyde','acetone','formamide']),('C','prepared_C',['glyoxal','pyrazine','tetrazine'])]:
 for name in names:
  case=json.loads((BASE/prepared/name/'case.json').read_text())
  if not case['targets']:
   if group=='C':crows.append(json.loads((R/'holdout_C'/name/'result.json').read_text()))
   continue
  t=case['targets'][0];old,xyz=state_from(t);s=old.reference_state
  with np.load(t['transition_file']) as f:a=f['amplitude'];occ=f['occupied'];virt=f['virtual']
  scores={}
  for label,folder in [('spectral',Q),('raw_full',R/'full_response')]:
   row=json.loads((folder/name/'n_pi/result.json').read_text())
   if not row.get('qualified'):scores[label]=dict(qualified=False);continue
   solved=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
   h=s.coefficients[:,occ].T@s.overlap@solved.common_coefficients[:,t['hole']];p=s.coefficients[:,virt].T@s.overlap@solved.common_coefficients[:,t['particle']];rank=np.outer(h,p)
   tr=float(abs(np.sum(a*rank))/np.linalg.norm(rank));scores[label]=dict(transition_overlap=tr,qualified=bool(tr>=.7))
  checks.append(dict(group=group,molecule=name,scores=scores))
  if group=='C':
   row=json.loads((R/'holdout_C'/name/'result.json').read_text());row['paired_qualified']=bool(row.get('qualified') and all(v['qualified'] for v in scores.values()))
   tda=Q/name/'tda.json';row['tda']=json.loads(tda.read_text()) if tda.exists() else None;crows.append(row)
write(R/'resumed_identity_checks.json',dict(host=HOST,rows=checks,source_sha256=digest(__file__)))
valid=[r for r in crows if r.get('paired_qualified')];stats={}
for label in ['native','raw_full','spectral']:
 errors=[(r['native_roks_ev']-r['tbe_ev']) if label=='native' else r['raw_full_response' if label=='raw_full' else 'spectral_response']['error_ev'] for r in valid]
 stats[label]=dict(count=len(errors),mae_ev=float(np.mean(np.abs(errors))) if errors else None,maxabs_ev=max(map(abs,errors),default=None))
gate=bool(len(valid)==3 and stats['spectral']['mae_ev']<stats['native']['mae_ev'] and stats['spectral']['mae_ev']<stats['raw_full']['mae_ev'] and stats['spectral']['maxabs_ev']<=.5)
write(R/'holdout_C_summary.json',dict(host=HOST,protocol_sha256=digest(BASE/'qualified_spectral_holdout_C_protocol.json'),rows=crows,statistics=stats,passed=gate,profile_changed=False))
print('C SUMMARY',stats,'PASSED',gate,flush=True)
