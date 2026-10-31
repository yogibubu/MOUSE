"""Hash resumed evidence separately; preserve the previous campaign manifest."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,platform
import numpy,scipy,pyscf
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
R=BASE/'results';files=[]
for folder in ['response_ablation','spectral_response','qualified_spectral_response','qualified_spectral_continuation','holdout_C']:
 files += [p for p in (R/folder).rglob('*') if p.is_file()]
for name in ['glyoxal','pyrazine','tetrazine']:files += [p for p in (R/'full_response'/name).rglob('*') if p.is_file()]
for folder in ['prepared_C','references_C','holdout_C_inputs','executed_sources']:files += [p for p in (BASE/folder).rglob('*') if p.is_file()]
names=['response_ablation','response_decomposition','response_stability','spectral_response','spectral_mapping_audit','qualified_spectral_response','qualified_spectral_stability','spectral_continuation','dft_ground_stability','qualified_spectral_carbonyls','qualified_spectral_tda','holdout_C_protocol','prepare_C','holdout_C_run','holdout_C_tda','all_full_tda','combined_response','checkpoint_reference_tda','resume_reference_tda','resumed_collect','resumed_report','resumed_provenance']
files += [BASE/(n+'.py') for n in names]
files += [p for p in BASE.glob('*.log') if any(p.name.startswith(n) for n in names+['holdout_C'])]
files += [BASE/'qualified_spectral_holdout_C_protocol.json',BASE/'RESUMED_REPORT_20261030.md',R/'resumed_identity_checks.json',R/'holdout_C_summary.json']
files += [BASE/n for n in ['full_fast.py','full_response.py','remote_bootstrap.py']]
files=[p for p in sorted(set(files)) if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('._')]
manifest=dict(host=HOST,previous_provenance_sha256=digest(R/'provenance.json'),protocol_C_sha256=digest(BASE/'qualified_spectral_holdout_C_protocol.json'),
 artifacts={str(p.relative_to(BASE)):dict(size=p.stat().st_size,sha256=digest(p)) for p in files},
 environment=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,pyscf=pyscf.__version__,platform=platform.platform()),
 no_energy_offset=True,lambda_exchange=.9,profile_changed_after_C_predictions=False,default_changed=False,nonlocal_nuclear_derivatives=False,
 checkpoint_policy='prior completed SCF references, response models, all converged and failed-identity states, latest atomic checkpoints retained and reused')
write(R/'resumed_provenance.json',manifest);print('RESUMED HASHED',len(files),flush=True)
