"""Fit one global convex weight using exposed training data only.

Primary target is the eight npi states; all-nine fit is a sensitivity control.
D is never opened here. Keep the original prospective 50/50 test unchanged.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,sys,datetime
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
training=BASE/'results/combined_summary.json';d=json.loads(training.read_text());rows=d['rows']
def fit(rr):
 candidates={0.,.5,1.}
 for r in rr:
  a=r['models']['raw_full']['excitation_ev'];b=r['models']['spectral']['excitation_ev'];delta=a-b
  if abs(delta)>1e-14:
   w=(r['tbe_ev']-b)/delta
   if 0<=w<=1:candidates.add(float(w))
 def error(w):return sum(abs(w*r['models']['raw_full']['excitation_ev']+(1-w)*r['models']['spectral']['excitation_ev']-r['tbe_ev']) for r in rr)/len(rr)
 values={w:error(w) for w in candidates};best=min(values.values());w=min((w for w in candidates if values[w]<=best+1e-12),key=lambda v:(abs(v-.5),v))
 return dict(raw_full_weight=w,spectral_weight=1-w,training_count=len(rr),training_mae_ev=values[w],half_half_mae_ev=error(.5))
npi=[r for r in rows if r['state']=='n_pi'];primary=fit(npi);sensitivity=fit(rows)
folds=[]
for name in sorted({r['molecule'] for r in npi}):
 train=[r for r in npi if r['molecule']!=name];test=[r for r in npi if r['molecule']==name];f=fit(train);w=f['raw_full_weight']
 folds.append(dict(held_molecule=name,raw_full_weight=w,held_mae_ev=sum(abs(w*r['models']['raw_full']['excitation_ev']+(1-w)*r['models']['spectral']['excitation_ev']-r['tbe_ev']) for r in test)/len(test)))
result=dict(host=HOST,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_sha256=digest(training),original_D_protocol_sha256=digest(BASE/'ensemble_holdout_D_protocol.json'),
 primary_npi_global_fit=primary,all_nine_global_fit_sensitivity=sensitivity,original_control=dict(raw_full_weight=.5,spectral_weight=.5),
 objective='unweighted mean absolute energy error; one convex scalar weight shared by all molecules; no intercept; exact piecewise-linear minimization on [0,1]; ties closest to 1/2',
 selection='primary npi fit fixed before reading any D outputs; report all-nine fit as sensitivity, never select on D',D_results_read_during_weight_fit=False,component_jobs_may_be_running=True,original_50_50_protocol_preserved=True,
 training_states=[dict(molecule=r['molecule'],state=r['state'],tbe_ev=r['tbe_ev']) for r in rows],lambda_exchange=.9,energy_offset=0.,leave_one_molecule_out=folds)
p=BASE/'ensemble_holdout_D_weight_protocol.json'
if p.exists():
 previous=json.loads(p.read_text());result['frozen_utc']=previous['frozen_utc'];assert previous==result
else:write(p,result)
print('FROZEN WEIGHTS',digest(p),primary,sensitivity,flush=True)
