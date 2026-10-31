"""Hash fixed-combination experiments after all jobs finish."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
R=BASE/'results';files=[p for p in (R/'combined_response').rglob('*') if p.is_file()]
files += [BASE/(s+'.py') for s in ['combined_response','combined_collect','combined_tda','all_full_tda','combined_report','combined_provenance']]
files += [p for p in BASE.glob('*.log') if p.name.startswith(('combined_','all_full_tda'))]
files += [R/'combined_summary.json',BASE/'COMBINATION_REPORT_20261030.md']
# Additional raw-full TDA controls for previously completed molecular models.
for name in ['acetaldehyde','acetone','formamide','glyoxal','pyrazine','tetrazine']:
 files += [p for p in (R/'full_response'/name).glob('tda.*')]
files=[p for p in sorted(set(files)) if p.is_file() and not p.name.startswith('._') and '__pycache__' not in str(p)]
write(R/'combined_provenance.json',dict(host=HOST,previous_resumed_provenance_sha256=digest(R/'resumed_provenance.json'),protocol_sha256=digest(R/'combined_response/protocol.json'),
 artifacts={str(p.relative_to(BASE)):dict(size=p.stat().st_size,sha256=digest(p)) for p in files},no_energy_offset=True,no_fitted_weights=True,lambda_exchange=.9,
 default_changed=False,nuclear_derivatives_supported=False,scope='subsequent combination comparison on already exposed molecules'))
print('COMBINATION HASHED',len(files),flush=True)
