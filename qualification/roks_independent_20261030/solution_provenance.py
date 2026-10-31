"""Freeze new D, stationary-density and REKS artifacts after jobs finish."""
from remote_bootstrap import BASE,HOST
import sys
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
R=BASE/'results';files=[]
for root in ['results/holdout_D','results/stationary_response','results/reks_probe','results/sa_reks_response','results/sa_reks_response_index1','holdout_D_inputs','prepared_D','references_D']:
    files.extend(p for p in (BASE/root).rglob('*') if p.is_file())
for name in ['acrolein','pyrimidine','nitrosomethane']:
    for folder in ['full_response','qualified_spectral_response']:
        files.extend(p for p in (R/folder/name).rglob('*') if p.is_file())
for stem in ['holdout_D_protocol','prepare_D','reference_td_checkpoint','holdout_D_run','holdout_D_tda','holdout_D_finalize','fit_ensemble_weights','stationary_response_calibration','stationary_response_run','stationary_response_audit','reks_probe','sa_reks_response','sa_reks_response_index1','solution_finalize','solution_provenance']:
    files.append(BASE/(stem+'.py'))
files.extend([BASE/'executed_sources/sa_reks_response_initial.py',BASE/'HOLDOUT_D_REPORT_20261030.md',BASE/'SOLUTION_AUDIT_20261030.md',
              BASE/'ensemble_holdout_D_protocol.json',BASE/'ensemble_holdout_D_weight_protocol.json',R/'holdout_D_summary.json',R/'ensemble_weight_decision.json'])
files.extend(p for p in BASE.glob('*.log') if p.name.startswith(('holdout_D','prepare_D','reference_D','stationary_','sa_reks_','reks_probe','fit_ensemble')))
files=sorted({p for p in files if p.exists() and '__pycache__' not in str(p) and not p.name.startswith('._')})
write(R/'solution_provenance.json',dict(host=HOST,previous_manifests={name:digest(R/name) for name in ['provenance.json','resumed_provenance.json','combined_provenance.json']},
       artifacts={str(p.relative_to(BASE)):dict(size=p.stat().st_size,sha256=digest(p)) for p in files},
       user_weights=[.5,.5],fitted_weights_used=False,default_changed=False,new_nuclear_derivatives_supported=False,
       verdict='no universal n-pi correction demonstrated; independent D gate failed; stationary-density and tested SA-REKS alternatives rejected'))
print('SOLUTION HASHED',len(files),flush=True)
