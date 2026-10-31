"""Freeze all new functional attempts, holdouts, controls and failure evidence."""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
R=BASE/'results';files=[]
for path in ['results/explicit_coupling','results/edft_rsh','results/edft_rsh_consistent','results/edft_rsh_statefollow','results/edft_pbe0','results/edft_pbe0_statefollow',
             'results/holdout_F','prepared_E','prepared_F','references_E','references_F','holdout_E_inputs','holdout_F_inputs']:
    files.extend(p for p in (BASE/path).rglob('*') if p.is_file() and not p.is_symlink())
for name in ['cyclopentadienone','maleimide','pyranone']:
    for folder in ['full_response','qualified_spectral_response']:
        files.extend(p for p in (R/folder/name).rglob('*') if p.is_file())
files.extend(BASE.glob('edft*.py'));files.extend(BASE.glob('edft*.json'));files.extend(BASE.glob('edft*.log'))
files.extend([BASE/'explicit_coupling_probe.py',BASE/'explicit_coupling_probe.log',BASE/'prepare_E.py',BASE/'prepare_F.py',BASE/'EDFT_FUNCTIONAL_REPORT_20261030.md',
              R/'edft_functional_final_summary.json',R/'holdout_F_summary.json'])
files.extend(p for p in (BASE/'executed_sources').glob('*.py') if p.name.startswith(('edft','prepare_E')))
files=sorted({p for p in files if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('._')})
audits=[]
for folder in ['edft_rsh','edft_rsh_consistent','edft_pbe0','edft_pbe0_statefollow','edft_rsh_statefollow']:
    for p in (R/folder).rglob('response_audit.json'):
        for row in json.loads(p.read_text())['checks']:audits.append(row)
summary=json.loads((R/'edft_functional_final_summary.json').read_text())
write(R/'edft_provenance.json',dict(host=HOST,previous_manifests={name:digest(R/name) for name in ['provenance.json','resumed_provenance.json','combined_provenance.json','solution_provenance.json']},
       artifacts={str(p.relative_to(BASE)):dict(size=p.stat().st_size,sha256=digest(p)) for p in files},
       scalar_response_audits=dict(directions=len(audits),max_gradient_error=max(x['gradient_error'] for x in audits),max_hessian_error=max(x['hessian_error'] for x in audits)),
       shared_calibration_link='results/edft_rsh_statefollow/calibration -> ../edft_rsh_consistent/calibration',
       F_accuracy_gate_passed=summary['F_passed'],universal_correction_demonstrated=False,old_composite_weights=[.5,.5],default_changed=False,
       model='EDFT-inspired GFN2 response research adapters, not full GX24',energy_offsets_added=False,new_nuclear_derivatives_supported=False))
print('EDFT HASHED',len(files),flush=True)
