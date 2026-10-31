"""Preserve both cytosine optimizer branches; promote only certified stationarity."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import shutil,json
source=BASE/'results/full_reference_canonical_retry/cytosine/pi_pi'
result=json.loads((source/'result.json').read_text())
assert result['qualified'] and result['full_orbital_residual']<=1e-7 and result['seed_csf_overlap']>=.7
final=BASE/'results/full_reference/cytosine/pi_pi'
if (final/'result.json').exists():
 assert json.loads((final/'result.json').read_text())['request_hash']==result['request_hash']
else:
 archive=BASE/'results/full_reference_previous_attempts/cytosine/pi_pi';archive.parent.mkdir(parents=True,exist_ok=True)
 assert not archive.exists();final.rename(archive)
 shutil.copytree(source,final)
print('PROMOTED',HOST,result['excitation_ev'],result['full_orbital_residual'],result['seed_csf_overlap'])
